use std::future::poll_fn;
use std::task::{Context, Poll};

use anyhow::{Result, bail};
use jiff::Timestamp;

use crate::view::{MessageEntry, UncommittedMessages};
use crate::{
    AssistantMessage, Backend, BackendError, BackendFuture, BackendRequest, BackendStream, History,
    Message, MessageView, TokenUsage, ToolCall, ToolCallKind, ToolDefinition, ToolResult,
};

/// Represents the resulting state from a transition as the agent's state machine progresses.
#[derive(Clone, Debug, PartialEq)]
pub enum StepEvent {
    /// Streaming-only: bytes appended to the last assistant's reasoning and content. Both are zero for a
    /// usage-only update. A terminal chunk is reported by `Finished` or `ToolCalls` instead.
    ///
    /// Modern LLMs cannot contribute more reasoning once content has been first contributed, which is to say
    /// that it is strictly always reasoning followed by content (though a single delta at most may update both
    /// when crossing the reasoning-content boundary).
    Updated {
        n_added_reasoning: usize,
        n_added_content: usize,
    },
    /// The backend completed and committed an assistant response without tool calls.
    Finished,
    /// The assistant is awaiting results for newly made tool calls
    /// The handler must append the corresponding results in order of appearance.
    ToolCalls,
    /// The request failed. Any transient assistant message being streamed by the backend was discarded.
    Error(BackendError),
}

enum ResponseState {
    /// Non-streaming response
    Waiting(BackendFuture),
    Streaming {
        events: BackendStream,
        tool_calls: Vec<ToolCall>,
    },
}

pub struct Agent<B> {
    backend: B,
    history: History,
    /// Contains, at most, an assistant message with tool calls and potentially its partial tool results, or
    /// a transient assistant message (i.e. being streamed).
    uncommitted: UncommittedMessages,
    pending_response: Option<ResponseState>,
    usage: TokenUsage,
    version: u64,
}

impl<B: Backend> Agent<B> {
    pub fn new(backend: B) -> Self {
        Self::with_messages(backend, [])
    }

    pub fn with_messages(backend: B, messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            backend,
            history: History::with_messages(messages),
            uncommitted: UncommittedMessages::default(),
            pending_response: None,
            usage: TokenUsage::default(),
            version: 0,
        }
    }

    pub fn messages(&self) -> MessageView<'_> {
        self.history.view(&self.uncommitted)
    }

    /// A monotonic counter for changes to visible messages.
    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn has_active_request(&self) -> bool {
        self.pending_response.is_some()
    }

    pub fn usage(&self) -> TokenUsage {
        self.usage
    }

    /// Returns the tool calls from any pending assistant message wand the number of calls already resolved.
    /// Returns `None` if no such assistant message exists.
    fn uncommitted_tool_calls(&self) -> Option<(&[ToolCall], usize)> {
        let Message::Assistant(assistant) = &self.uncommitted.assistant.as_ref()?.message else {
            unreachable!()
        };
        if assistant.tool_calls.is_empty() {
            return None;
        }
        let completed = if let Some(tool_results) = &self.uncommitted.tool_results {
            let Message::Tool(results) = &tool_results.message else {
                unreachable!()
            };
            results.len()
        } else {
            0
        };
        Some((&assistant.tool_calls, completed))
    }

    /// Cancels a backend response and discards its transient streamed assistant message.
    pub fn cancel_response(&mut self) -> bool {
        if matches!(
            self.pending_response.take(),
            Some(ResponseState::Streaming { .. })
        ) {
            // Discard the transient assistant message to which the stream was writing text.
            self.uncommitted.clear();
            self.version += 1;
            true
        } else {
            false
        }
    }

    /// Appends a user message to the selected history path.
    pub fn start_turn(&mut self, content: String) {
        assert!(self.pending_response.is_none() && self.uncommitted.is_empty());
        self.history
            .commit([(Message::User(content), Timestamp::now())]);
        self.version += 1;
    }

    /// Appends one result for the next unresolved call, committing the assistant/toolresult messages
    /// to the history of all calls were resolved.
    pub fn append_tool_result(&mut self, new_result: ToolResult) {
        let (calls, completed) = self
            .uncommitted_tool_calls()
            .expect("this method should be called if and only if there are pending tool calls");

        // Current tool execution design forbids this from ever happening; but just in case.
        debug_assert_eq!(new_result.tool_call_id, calls[completed].id);

        let total_calls = calls.len();
        if let Some(MessageEntry {
            message: Message::Tool(results),
            ..
        }) = &mut self.uncommitted.tool_results
        {
            results.push(new_result);
        } else {
            let mut results = Vec::with_capacity(total_calls);
            results.push(new_result);
            self.uncommitted.tool_results = Some(MessageEntry {
                message: Message::Tool(results),
                timestamp: None,
            });
        }
        self.version += 1;

        // check if all tool calls were handled (+1 is the result we just added)
        if completed + 1 == total_calls {
            self.uncommitted.tool_results.as_mut().unwrap().timestamp = Some(Timestamp::now());
            self.history.commit(self.uncommitted.take_completed());
        }
    }

    /// Resolves all pending tool calls, if any, with the provided result message.
    pub fn resolve_remaining_tool_calls(&mut self, result: &str) {
        let (calls, completed) = self
            .uncommitted_tool_calls()
            .expect("there should be an assistant message with tool calls");
        let remaining_ids = calls[completed..]
            .iter()
            .map(|call| call.id.clone())
            .collect::<Vec<_>>();
        for tool_call_id in remaining_ids {
            self.append_tool_result(ToolResult {
                tool_call_id,
                content: result.into(),
            });
        }
    }

    /// Appends a finalized assistant response, committing it when it has no tool calls.
    /// The [`Agent`] MUST be idle (i.e. not running a request nor holds pending tool calls).
    // TODO: potentially remove this because this is only in one production call site, with remaining calls
    // going to tests? Or, be clear about its semantics since there's no necessary reason to reject appending
    // arbitrary assistant messages as a part of the agent interface.
    pub fn submit_assistant_msg(&mut self, assistant: AssistantMessage) -> StepEvent {
        assert!(
            !self.has_active_request() && self.uncommitted.is_empty(),
            "An assistant message may be appended to an agent if and only if the agent is idle."
        );

        self.version += 1;
        if assistant.tool_calls.is_empty() {
            self.history
                .commit([(Message::Assistant(assistant), Timestamp::now())]);
            StepEvent::Finished
        } else {
            // Tool calls must be resolved before the assistant message can be committed to history, or
            // else it would lead to unpredictable model behavior (and inference engines likely reject
            // such requests anyway).
            self.uncommitted.assistant = Some(MessageEntry {
                message: Message::Assistant(assistant),
                timestamp: Some(Timestamp::now()),
            });
            StepEvent::ToolCalls
        }
    }

    /// Removes complete turns (a turn starts from a user message) and returns the prompt from the
    /// oldest removed turn.
    pub fn undo(&mut self, count: usize) -> Option<&str> {
        assert!(
            self.uncommitted.is_empty() && !self.has_active_request(),
            "cannot undo while the agent is running"
        );

        let mut content: Option<*const str> = None;
        for _ in 0..count {
            match self.history.undo() {
                Some(Message::User(prompt)) => content = Some(prompt.as_str() as *const str),
                Some(_) => unreachable!(),
                None => break,
            }
        }
        let content = content?;
        self.version += 1;

        // SAFETY: undo only changes the History's spine, and so the underlying messages don't change memory
        // location.
        // TODO: this can be eliminated once we switch Revisions to hold an Arc to the message (which is necessary
        // for other refactors anyway).
        Some(unsafe { &*content })
    }

    /// Selects the newest child turn of the current revision.
    pub fn redo(&mut self) -> Result<bool> {
        if !self.uncommitted.is_empty() || self.has_active_request() {
            bail!("cannot redo while the agent is running");
        }

        if !self.history.redo() {
            return Ok(false);
        }
        self.version += 1;
        Ok(true)
    }

    /// Runs one backend step to completion with the provided tools.
    pub async fn step(&mut self, tools: &[ToolDefinition]) -> StepEvent {
        self.start_step(tools, false);
        loop {
            match poll_fn(|cx| self.poll_step(cx)).await {
                StepEvent::Updated { .. } => {}
                event => return event,
            }
        }
    }

    /// Construct a request with the provided tools to be resolved by the backend.
    pub fn start_step(&mut self, tools: &[ToolDefinition], stream: bool) {
        assert!(self.pending_response.is_none() && self.uncommitted.is_empty());

        let request = BackendRequest {
            messages: self.history.view(&self.uncommitted),
            tools,
        };
        self.pending_response = Some(if stream {
            let events = self.backend.request_stream(request);
            self.uncommitted.assistant = Some(MessageEntry {
                message: Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: Vec::new(),
                }),
                timestamp: None,
            });
            self.version += 1;
            ResponseState::Streaming {
                events,
                tool_calls: Vec::new(),
            }
        } else {
            ResponseState::Waiting(self.backend.request(request))
        });
    }

    /// If there is an in-flight request, this function polls until the visible transcript changes, the response
    /// finishes, or the backend reports that no more work is currently available.
    pub fn poll_step(&mut self, cx: &mut Context<'_>) -> Poll<StepEvent> {
        let Some(mut response) = self.pending_response.take() else {
            return Poll::Pending;
        };

        loop {
            match response {
                ResponseState::Waiting(mut future) => match future.as_mut().poll(cx) {
                    Poll::Ready(Ok(assistant)) => {
                        return Poll::Ready(self.submit_assistant_msg(assistant));
                    }
                    // A non-streaming request has not appended an assistant message yet.
                    Poll::Ready(Err(error)) => {
                        return Poll::Ready(StepEvent::Error(error));
                    }
                    Poll::Pending => {
                        self.pending_response = Some(ResponseState::Waiting(future));
                        return Poll::Pending;
                    }
                },
                ResponseState::Streaming {
                    mut events,
                    mut tool_calls,
                } => match events.as_mut().poll_next(cx) {
                    Poll::Ready(Some(Ok(event))) => {
                        // TODO: `updated` can get removed in the future once we also make tool calls show one at a time rather
                        // than batch.
                        let mut chat_was_updated = false;
                        if let Some(usage) = event.usage
                            && self.usage != usage
                        {
                            // The last delta is always empty with usage metadata.
                            self.usage = usage;
                            chat_was_updated = true;
                        }

                        let MessageEntry {
                            message: Message::Assistant(message),
                            timestamp,
                        } = self.uncommitted.assistant.as_mut().unwrap()
                        else {
                            unreachable!()
                        };
                        let reasoning = event.reasoning.as_ref().map_or(0, String::len);
                        let content = event.content.as_ref().map_or(0, String::len);

                        // append any assistant (reasoning) content
                        if let Some(delta) = event.reasoning.filter(|delta| !delta.is_empty()) {
                            assert!(
                                message.content.as_deref().is_none_or(str::is_empty),
                                "reasoning cannot be appended after content"
                            );
                            message
                                .reasoning
                                .get_or_insert_with(String::new)
                                .push_str(&delta);
                            chat_was_updated = true;
                        }
                        if let Some(delta) = event.content.filter(|delta| !delta.is_empty()) {
                            message
                                .content
                                .get_or_insert_with(String::new)
                                .push_str(&delta);
                            chat_was_updated = true;
                        }

                        // append any tool calls
                        for delta in event.tool_calls {
                            if delta.index == tool_calls.len() {
                                tool_calls.push(ToolCall {
                                    id: String::new(),
                                    kind: ToolCallKind::Function,
                                    function_name: String::new(),
                                    arguments: String::new(),
                                });
                            }
                            let call = &mut tool_calls[delta.index];

                            if let Some(kind) = delta.kind {
                                // FIXME: a buggy provider could override call.kind in a subsequent delta,
                                // and this needs some fixing.
                                call.kind = kind;
                            }
                            if let Some(id) = delta.id {
                                call.id.push_str(&id);
                            }
                            if let Some(function_name) = delta.function_name {
                                call.function_name.push_str(&function_name);
                            }
                            if let Some(arguments) = delta.arguments {
                                call.arguments.push_str(&arguments);
                            }
                        }

                        // The stream is over.
                        if event.finished {
                            message.tool_calls = tool_calls;
                            *timestamp = Some(Timestamp::now());
                            self.version += 1;

                            let finished = message.tool_calls.is_empty();
                            if finished {
                                self.history.commit(self.uncommitted.take_completed());
                            }
                            return Poll::Ready(if finished {
                                StepEvent::Finished
                            } else {
                                StepEvent::ToolCalls
                            });
                        }

                        if chat_was_updated {
                            self.version += 1;
                            self.pending_response =
                                Some(ResponseState::Streaming { events, tool_calls });
                            return Poll::Ready(StepEvent::Updated {
                                n_added_reasoning: reasoning,
                                n_added_content: content,
                            });
                        }
                        response = ResponseState::Streaming { events, tool_calls };
                    }
                    Poll::Ready(option) => {
                        let error = match option {
                            Some(Err(error)) => error,
                            Some(Ok(_)) => unreachable!("handled by the arm above"),
                            None => BackendError::Protocol {
                                message: "backend stream ended before it finished".into(),
                            },
                        };
                        // The streamed assistant message is transient and is discarded: an error may have
                        // interrupted it mid-thought, and truncated tool-call arguments must never
                        // enter the history as a real call.
                        self.uncommitted.clear();
                        self.version += 1;
                        return Poll::Ready(StepEvent::Error(error));
                    }
                    Poll::Pending => {
                        self.pending_response =
                            Some(ResponseState::Streaming { events, tool_calls });
                        return Poll::Pending;
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::task::Context;

    use super::*;
    use crate::{AssistantMessage, BackendEvent, TokenUsage, ToolCallDelta, ToolResult};
    use futures_util::{Stream, StreamExt, stream};

    const FOO_TOOL_NAME: &str = "foo";
    const BAR_TOOL_NAME: &str = "bar";

    /// Clones the agent's visible messages so assertion failures print a diff.
    fn visible(agent: &Agent<impl Backend>) -> Vec<Message> {
        agent.messages().iter().cloned().collect()
    }

    /// Responds to requests with predefined responses.
    #[derive(Default)]
    struct DummyBackend {
        // A VecDeque is used only to make the code more readable.
        responses: VecDeque<AssistantMessage>,
    }

    impl DummyBackend {
        fn new(responses: impl IntoIterator<Item = AssistantMessage>) -> Self {
            Self {
                responses: responses.into_iter().collect(),
            }
        }
    }

    impl Backend for DummyBackend {
        fn request(&mut self, _request: BackendRequest<'_>) -> BackendFuture {
            let response = self.responses.pop_front().unwrap();
            Box::pin(async move { Ok(response) })
        }
    }

    struct FailingBackend;

    impl Backend for FailingBackend {
        fn request(&mut self, _request: BackendRequest<'_>) -> BackendFuture {
            Box::pin(async {
                Err(BackendError::Http {
                    status: 401,
                    message: "bad key".into(),
                })
            })
        }
    }

    struct StreamingBackend {
        events: Option<BackendStream>,
    }

    impl StreamingBackend {
        fn new(events: impl Stream<Item = BackendEvent> + Send + 'static) -> Self {
            Self {
                events: Some(Box::pin(events.map(Ok))),
            }
        }
    }

    impl Backend for StreamingBackend {
        fn request(&mut self, _request: BackendRequest<'_>) -> BackendFuture {
            unreachable!()
        }

        fn request_stream(&mut self, _request: BackendRequest<'_>) -> BackendStream {
            self.events.take().unwrap()
        }
    }

    #[tokio::test]
    async fn basic_tool_call() {
        let call = ToolCall {
            id: "".into(),
            kind: ToolCallKind::Function,
            function_name: "".into(),
            arguments: "".into(),
        };
        let first_response = AssistantMessage {
            reasoning: Some("I should use the function.".into()),
            content: None,
            tool_calls: vec![call.clone()],
        };
        let final_response = AssistantMessage {
            reasoning: None,
            content: Some("foo".into()),
            tool_calls: Vec::new(),
        };

        let backend = DummyBackend::new([first_response.clone(), final_response.clone()]);
        let mut agent = Agent::new(backend);

        // Do a tool call chain
        agent.start_turn("bar".into());
        agent.start_step(&[], false);

        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::ToolCalls
        );

        let result = ToolResult {
            tool_call_id: call.id,
            content: "baz".into(),
        };
        agent.append_tool_result(result.clone());
        agent.step(&[]).await;

        assert_eq!(
            visible(&agent),
            vec![
                Message::User("bar".into()),
                Message::Assistant(first_response),
                Message::Tool(vec![result]),
                Message::Assistant(final_response),
            ]
        );
        assert_eq!(agent.history.current_path().len(), 4);
    }

    #[test]
    fn partial_tool_results_remain_uncommitted() {
        let mut agent = Agent::new(DummyBackend::default());
        agent.start_turn("prompt".into());
        agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: None,
            tool_calls: ["first", "second"]
                .into_iter()
                .map(|id| ToolCall {
                    id: id.into(),
                    kind: ToolCallKind::Function,
                    function_name: FOO_TOOL_NAME.into(),
                    arguments: "{}".into(),
                })
                .collect(),
        });
        let assistant_timestamp = agent.messages().timestamp(1).unwrap();
        agent.append_tool_result(ToolResult {
            tool_call_id: "first".into(),
            content: "done".into(),
        });
        assert_eq!(agent.history.current_path().len(), 1);
        assert_eq!(agent.messages().timestamp(1), Some(assistant_timestamp));
        assert_eq!(agent.messages().timestamp(2), None);

        agent.append_tool_result(ToolResult {
            tool_call_id: "second".into(),
            content: "done".into(),
        });
        assert_eq!(agent.messages().timestamp(1), Some(assistant_timestamp));
        assert!(agent.messages().timestamp(2).is_some());
    }

    #[test]
    fn undo_then_new_turn_creates_a_sibling() {
        let mut agent =
            Agent::with_messages(DummyBackend::default(), [Message::System("root".into())]);
        agent.start_turn("first".into());
        agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: Some("old".into()),
            tool_calls: Vec::new(),
        });
        let version = agent.version();
        assert_eq!(agent.undo(1).unwrap(), "first");
        assert_eq!(visible(&agent), vec![Message::System("root".into())]);
        agent.start_turn("second".into());
        agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: Some("new".into()),
            tool_calls: Vec::new(),
        });
        assert_eq!(agent.version(), version + 3);
        assert_eq!(agent.undo(1).unwrap(), "second");
        assert!(agent.redo().unwrap());
        assert_eq!(
            agent.messages()[2],
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some("new".into()),
                tool_calls: Vec::new(),
            })
        );
    }

    #[test]
    fn undo_returns_the_prompt_from_the_selected_turn() {
        let mut agent = Agent::with_messages(
            DummyBackend::default(),
            [
                Message::User("first".into()),
                Message::User("second".into()),
                Message::User("third".into()),
            ],
        );

        assert_eq!(agent.undo(2).unwrap(), "second");
        assert_eq!(visible(&agent), vec![Message::User("first".into())]);
        assert_eq!(agent.undo(usize::MAX).unwrap(), "first");
        assert!(visible(&agent).is_empty());
    }

    #[test]
    fn committing_and_navigating_keep_the_message_payload() {
        let mut agent = Agent::new(DummyBackend::new([]));
        agent.start_turn("unchanged payload".into());
        let Some(Message::User(content)) = agent.messages().get(0) else {
            unreachable!()
        };
        let payload = content.as_ptr();

        let undone = agent.undo(1).unwrap();
        assert_eq!(undone, "unchanged payload");
        assert_eq!(undone.as_ptr(), payload);
        assert!(agent.redo().unwrap());
        let Some(Message::User(content)) = agent.messages().get(0) else {
            unreachable!()
        };
        assert_eq!(content.as_ptr(), payload);
    }

    #[tokio::test]
    async fn terminal_chunk_publishes_all_fields_once_and_preserves_other_tool_calls() {
        let events = stream::iter([
            BackendEvent {
                reasoning: Some("reason\ntail".into()),
                content: Some("answer\ntail".into()),
                tool_calls: vec![
                    ToolCallDelta {
                        index: 0,
                        id: Some("first".into()),
                        kind: Some(ToolCallKind::Function),
                        function_name: Some(BAR_TOOL_NAME.into()),
                        arguments: Some("{".into()),
                    },
                    ToolCallDelta {
                        index: 1,
                        id: Some("second".into()),
                        kind: Some(ToolCallKind::Function),
                        function_name: Some(FOO_TOOL_NAME.into()),
                        arguments: Some("{".into()),
                    },
                ],
                ..Default::default()
            },
            BackendEvent {
                tool_calls: vec![
                    ToolCallDelta {
                        index: 0,
                        arguments: Some("}".into()),
                        ..Default::default()
                    },
                    ToolCallDelta {
                        index: 1,
                        arguments: Some("}".into()),
                        ..Default::default()
                    },
                ],
                finished: true,
                ..Default::default()
            },
        ]);
        let mut agent = Agent::new(StreamingBackend::new(events));
        agent.start_turn("hi".into());
        agent.start_step(&[], true);

        let calls = vec![
            ToolCall {
                id: "first".into(),
                kind: ToolCallKind::Function,
                function_name: BAR_TOOL_NAME.into(),
                arguments: "{}".into(),
            },
            ToolCall {
                id: "second".into(),
                kind: ToolCallKind::Function,
                function_name: FOO_TOOL_NAME.into(),
                arguments: "{}".into(),
            },
        ];
        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::Updated {
                n_added_reasoning: 11,
                n_added_content: 11
            }
        );
        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::ToolCalls
        );
        assert_eq!(agent.version(), 4);
        assert_eq!(
            visible(&agent),
            vec![
                Message::User("hi".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: Some("reason\ntail".into()),
                    content: Some("answer\ntail".into()),
                    tool_calls: calls,
                })
            ]
        );
    }

    #[test]
    fn streamed_updates_publish_content_and_usage() {
        let events = stream::iter([
            BackendEvent {
                usage: Some(TokenUsage {
                    input_tokens: 12,
                    output_tokens: 3,
                    total_tokens: 15,
                }),
                ..Default::default()
            },
            BackendEvent {
                content: Some("partial".into()),
                ..Default::default()
            },
        ])
        .chain(stream::pending::<BackendEvent>());
        let mut agent = Agent::new(StreamingBackend::new(events));
        agent.start_turn("hi".into());
        agent.start_step(&[], true);

        assert_eq!(agent.version(), 2);

        let waker = futures_util::task::noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 0
            })
        );
        assert_eq!(
            agent.usage(),
            TokenUsage {
                input_tokens: 12,
                output_tokens: 3,
                total_tokens: 15,
            }
        );
        assert_eq!(
            visible(&agent),
            vec![
                Message::User("hi".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: Vec::new(),
                })
            ]
        );
        assert_eq!(agent.version(), 3);
        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 7
            })
        );
        assert_eq!(
            visible(&agent),
            vec![
                Message::User("hi".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("partial".into()),
                    tool_calls: Vec::new(),
                })
            ]
        );
    }

    #[test]
    fn streamed_updates_report_appended_bytes() {
        let mut agent = Agent::new(StreamingBackend::new(stream::iter([
            BackendEvent {
                reasoning: Some("考え".into()),
                content: Some("é".into()),
                ..Default::default()
            },
            BackendEvent {
                usage: Some(TokenUsage {
                    output_tokens: 1,
                    ..Default::default()
                }),
                ..Default::default()
            },
            BackendEvent {
                content: Some("!".into()),
                finished: true,
                ..Default::default()
            },
        ])));
        agent.start_turn("hi".into());
        agent.start_step(&[], true);
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(StepEvent::Updated {
                n_added_reasoning: 6,
                n_added_content: 2
            })
        );
        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 0
            })
        );
        assert_eq!(agent.poll_step(&mut cx), Poll::Ready(StepEvent::Finished));
        assert_eq!(
            agent.messages().last(),
            Some(&Message::Assistant(AssistantMessage {
                reasoning: Some("考え".into()),
                content: Some("é!".into()),
                tool_calls: Vec::new(),
            }))
        );
        assert_eq!(agent.poll_step(&mut cx), Poll::Pending);
    }

    #[test]
    fn cancel_response_discards_the_streamed_response() {
        let events = stream::iter([BackendEvent {
            content: Some("partial".into()),
            ..Default::default()
        }])
        .chain(stream::pending::<BackendEvent>());
        let mut agent = Agent::new(StreamingBackend::new(events));
        agent.start_turn("hi".into());
        agent.start_step(&[], true);

        let waker = futures_util::task::noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 7
            })
        );
        let version = agent.version();

        assert!(agent.cancel_response());

        assert_eq!(visible(&agent), vec![Message::User("hi".into())]);
        assert_eq!(agent.version(), version + 1);
        assert_eq!(agent.history.current_path().len(), 1);
    }

    #[tokio::test]
    async fn streamed_lines_update_one_message_and_assemble_tool_calls() {
        let events = stream::iter([
            BackendEvent {
                content: Some(String::new()),
                ..Default::default()
            },
            BackendEvent {
                reasoning: Some("thinking".into()),
                ..Default::default()
            },
            BackendEvent {
                content: Some("answer one\n".into()),
                ..Default::default()
            },
            BackendEvent {
                content: Some("answer two\r".into()),
                ..Default::default()
            },
            BackendEvent {
                content: Some("\n".into()),
                ..Default::default()
            },
            BackendEvent {
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: Some("call-1".into()),
                    kind: Some(ToolCallKind::Function),
                    function_name: Some(FOO_TOOL_NAME.into()),
                    arguments: Some(r#"{"pa"#.into()),
                }],
                ..Default::default()
            },
            BackendEvent {
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: None,
                    kind: None,
                    function_name: None,
                    arguments: Some(r#"th":"notes.txt"}"#.into()),
                }],
                ..Default::default()
            },
            BackendEvent {
                usage: Some(TokenUsage {
                    input_tokens: 12,
                    output_tokens: 3,
                    total_tokens: 15,
                }),
                finished: true,
                ..Default::default()
            },
        ]);
        let mut agent = Agent::new(StreamingBackend::new(events));
        agent.start_turn("hi".into());
        agent.start_step(&[], true);

        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::Updated {
                n_added_reasoning: 8,
                n_added_content: 0
            }
        );
        assert_eq!(agent.messages().timestamp(1), None);
        assert_eq!(
            visible(&agent),
            vec![
                Message::User("hi".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: Some("thinking".into()),
                    content: None,
                    tool_calls: Vec::new(),
                })
            ]
        );

        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 11
            }
        );
        assert_eq!(
            visible(&agent),
            vec![
                Message::User("hi".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: Some("thinking".into()),
                    content: Some("answer one\n".into()),
                    tool_calls: Vec::new(),
                })
            ]
        );

        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 11
            }
        );
        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 1
            }
        );

        let call = ToolCall {
            id: "call-1".into(),
            kind: ToolCallKind::Function,
            function_name: FOO_TOOL_NAME.into(),
            arguments: r#"{"path":"notes.txt"}"#.into(),
        };
        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::ToolCalls
        );
        assert!(agent.messages().timestamp(1).is_some());
        assert_eq!(
            visible(&agent),
            vec![
                Message::User("hi".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: Some("thinking".into()),
                    content: Some("answer one\nanswer two\r\n".into()),
                    tool_calls: vec![call],
                })
            ]
        );
        assert_eq!(
            agent.usage(),
            TokenUsage {
                input_tokens: 12,
                output_tokens: 3,
                total_tokens: 15,
            }
        );
    }

    #[tokio::test]
    async fn incomplete_streams_discard_partial_messages() {
        let cases = [
            (
                vec![
                    Ok(BackendEvent {
                        content: Some("par".into()),
                        ..Default::default()
                    }),
                    Err(BackendError::Transport {
                        message: "connection reset".into(),
                    }),
                ],
                BackendError::Transport {
                    message: "connection reset".into(),
                },
            ),
            (
                vec![Ok(BackendEvent {
                    content: Some("partial".into()),
                    ..Default::default()
                })],
                BackendError::Protocol {
                    message: "backend stream ended before it finished".into(),
                },
            ),
        ];

        for (events, expected) in cases {
            let mut agent = Agent::new(StreamingBackend {
                events: Some(Box::pin(stream::iter(events))),
            });
            agent.start_turn("hi".into());
            agent.start_step(&[], true);

            assert!(matches!(
                poll_fn(|cx| agent.poll_step(cx)).await,
                StepEvent::Updated { .. }
            ));
            assert_eq!(
                poll_fn(|cx| agent.poll_step(cx)).await,
                StepEvent::Error(expected)
            );
            // The transient streamed assistant message is discarded; the prompt it answered remains.
            assert_eq!(visible(&agent), vec![Message::User("hi".into())]);
            assert!(!agent.has_active_request());
        }
    }

    #[tokio::test]
    async fn a_non_streaming_error_leaves_history_untouched() {
        let mut agent = Agent::new(FailingBackend);
        agent.start_turn("hi".into());
        agent.start_step(&[], false);

        assert_eq!(
            poll_fn(|cx| agent.poll_step(cx)).await,
            StepEvent::Error(BackendError::Http {
                status: 401,
                message: "bad key".into()
            })
        );
        // No assistant placeholder exists in this mode, so the history is exactly as it was.
        assert_eq!(visible(&agent), vec![Message::User("hi".into())]);
        assert!(!agent.has_active_request());
    }
}
