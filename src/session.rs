use std::collections::VecDeque;
use std::time::Duration;

use kyun_agent::{Agent, Message, ToolCall, ToolDefinition, ToolResult, openai::OpenAiBackend};
use tokio::time::Instant;

use crate::compositor::Compositor;
use crate::timer::{Output, TimerRegistry};
use crate::tools::ToolFuture;
use crate::widgets::{Approval, get_tool_arguments};

/// An application tool future awaiting a concrete result.
pub struct PendingToolCall {
    pub future: ToolFuture,
    /// Whether the future has been polled yet: an interrupt before the first poll means the call never
    /// started executing and must be reported as skipped, not interrupted.
    pub started: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UIMode {
    Command,
    Insert,
    #[default]
    Normal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ApprovalState {
    #[default]
    Inactive,
    WaitingForIdle,
    AwaitingDecision,
    Approved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Interrupt,
    StepAgent,
    EditPrompt,
    ViewTranscript,
    NewSession,
    ResolveToolApproval(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Info,
    Error,
}

const STATUS_TIMER_ID: &str = "status";
const STATUS_DURATION: Duration = Duration::from_secs(3);
/// Errors outlive notices by an order of magnitude: they carry the reason a turn died and exist nowhere else.
const ERROR_STATUS_DURATION: Duration = Duration::from_secs(30);

/// State for the currently open session. A subagent can be opened by switching which session the
/// components receive.
pub struct Session {
    pub mode: UIMode,
    // TODO: how do I represent orchestration? a btree? just a flat vec?
    pub agent: Agent<OpenAiBackend>,
    pub should_exit: bool,
    pub pending_action: Option<Action>,
    pub step_count: usize,
    pub pending_tool_calls: VecDeque<PendingToolCall>,
    pub approval: ApprovalState,
    pub timers: TimerRegistry,
    status: Option<(String, Severity)>,
}

impl Session {
    pub fn new(agent: Agent<OpenAiBackend>) -> Self {
        Self {
            mode: UIMode::default(),
            agent,
            should_exit: false,
            pending_action: None,
            step_count: 0,
            pending_tool_calls: VecDeque::new(),
            approval: ApprovalState::Inactive,
            timers: TimerRegistry::default(),
            status: None,
        }
    }

    pub fn status(&self) -> Option<(&str, Severity)> {
        self.status
            .as_ref()
            .map(|(message, severity)| (message.as_str(), *severity))
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.set_status_for(status.into(), Severity::Info, STATUS_DURATION);
    }

    pub fn set_error(&mut self, error: impl Into<String>) {
        self.set_status_for(error.into(), Severity::Error, ERROR_STATUS_DURATION);
    }

    pub fn clear_status(&mut self) {
        self.status = None;
        self.timers.cancel(STATUS_TIMER_ID);
    }

    fn set_status_for(&mut self, message: String, severity: Severity, duration: Duration) {
        self.status = Some((message, severity));
        self.timers
            .schedule(STATUS_TIMER_ID, Instant::now() + duration, |_, session| {
                let redraw = session.status.take().is_some();
                (redraw, Output::Stop)
            });
    }

    pub fn is_busy(&self) -> bool {
        self.step_count > 0
            || self.agent.has_active_request()
            || !self.pending_tool_calls.is_empty()
    }

    pub fn fire_timer(&mut self, compositor: &mut Compositor) -> bool {
        let Some(mut timer) = self.timers.pop_due() else {
            return false;
        };
        let (redraw, output) = timer.run(compositor, self);
        self.timers.finish(timer, output);
        redraw
    }

    /// Prompts the user to approve the next pending tool call.
    pub fn show_tool_approval(&mut self, compositor: &mut Compositor) {
        let call = {
            let tool_calls = self.get_last_tool_calls();
            let completed = tool_calls.len() - self.pending_tool_calls.len();
            &tool_calls[completed]
        };

        // TODO: get_tool_arguments should be pulled out if I'm using it here instead of in a widget!
        compositor.push(Box::new(Approval::new(
            call.function_name.clone(),
            get_tool_arguments(call),
            |_, session| session.pending_action = Some(Action::ResolveToolApproval(true)),
            |_, session| session.pending_action = Some(Action::ResolveToolApproval(false)),
        )));
        self.mode = UIMode::Normal;
        self.approval = ApprovalState::AwaitingDecision;
    }

    pub fn interrupt(&mut self) -> bool {
        if !self.is_busy() {
            return false;
        }

        // A streamed assistant message is transient and is discarded. A tool batch is not: a result that already
        // completed, or a call whose execution already started, may have had external effects.
        let interrupting_a_tool_batch = !self.pending_tool_calls.is_empty();

        let discard_streamed_response = self.agent.cancel_response();
        self.step_count = 0;
        self.approval = ApprovalState::Inactive;
        self.timers.cancel(crate::timer::APPROVAL_TIMER_ID);
        self.timers.cancel(crate::timer::STATUSLINE_TIMER_ID);

        if interrupting_a_tool_batch {
            while !self.pending_tool_calls.is_empty() {
                let tool_call_id = {
                    let tool_calls = self.get_last_tool_calls();
                    let completed = tool_calls.len() - self.pending_tool_calls.len();
                    tool_calls[completed].id.clone()
                };

                // Only the front call could have started executing (one-at-a-time execution).
                let content = if self.pending_tool_calls.front().unwrap().started {
                    crate::TOOL_CALL_INTERRUPTED_MESSAGE
                } else {
                    crate::TOOL_CALL_SKIPPED_MESSAGE
                };
                self.finish_tool_call(ToolResult {
                    tool_call_id,
                    content: content.into(),
                });
            }
        }

        discard_streamed_response
    }

    pub fn finish_tool_call(&mut self, result: ToolResult) {
        self.pending_tool_calls.pop_front().unwrap();
        self.approval = ApprovalState::Inactive;
        self.agent.append_tool_result(result);
    }

    pub fn start_agent_step(
        &mut self,
        max_steps: usize,
        tools: &[ToolDefinition],
        stream: bool,
    ) -> bool {
        if self.step_count >= max_steps {
            self.step_count = 0;
            return false;
        }

        self.step_count += 1;
        self.agent.start_step(tools, stream);
        true
    }

    /// The tool calls of the batch currently awaiting results.
    ///
    /// The assistant message that made the calls is last until the first result lands, after which a tool message
    /// follows it.
    pub fn get_last_tool_calls(&self) -> &[ToolCall] {
        let messages = self.agent.messages();
        let message_idx = match messages.last() {
            Some(Message::Tool(_)) => messages.len() - 2,
            _ => messages.len() - 1,
        };
        let Some(Message::Assistant(message)) = messages.get(message_idx) else {
            unreachable!()
        };
        &message.tool_calls
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use kyun_agent::{
        Agent, AssistantMessage, Message, ToolCall, ToolCallKind, ToolResult,
        openai::{Config, OpenAiBackend},
    };

    use super::{PendingToolCall, Session, Severity};
    use crate::compositor::Compositor;
    use crate::tools::write;

    fn session() -> Session {
        Session::new(Agent::new(OpenAiBackend::new(Config {
            model: String::new(),
            base_url: String::new(),
            api_key: None,
            reasoning_effort: None,
            extra_body: None,
            ..kyun_agent::openai::Config::default()
        })))
    }

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            kind: ToolCallKind::Function,
            function_name: write::NAME.into(),
            arguments: "{}".into(),
        }
    }

    fn pending() -> PendingToolCall {
        PendingToolCall {
            future: Box::pin(std::future::pending::<ToolResult>()),
            started: false,
        }
    }

    #[test]
    fn status_expiration_does_not_clear_a_replacement() {
        let mut compositor = Compositor::new();
        let mut session = session();

        session.set_status_for("old".into(), Severity::Info, Duration::ZERO);
        session.set_status_for("new".into(), Severity::Info, Duration::from_secs(60));
        assert!(!session.fire_timer(&mut compositor));
        assert_eq!(session.status(), Some(("new", Severity::Info)));

        session.set_status_for("done".into(), Severity::Info, Duration::ZERO);
        assert!(session.fire_timer(&mut compositor));
        assert_eq!(session.status(), None);
    }

    fn start_two_call_batch(session: &mut Session) {
        session.step_count = 1;
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("write two files".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![call("call-1"), call("call-2")],
                }),
            ],
        );
        session.pending_tool_calls.extend([pending(), pending()]);
    }

    #[test]
    fn interrupt_preserves_completed_results_and_skips_the_rest() {
        let mut session = session();
        start_two_call_batch(&mut session);
        // call-1 executed before the interrupt.
        session.finish_tool_call(ToolResult {
            tool_call_id: "call-1".into(),
            content: "Created 13 bytes.".into(),
        });

        session.interrupt();

        let Message::Tool(results) = session.agent.messages().last().unwrap() else {
            unreachable!()
        };
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool_call_id, "call-1");
        assert_eq!(results[0].content, "Created 13 bytes.");
        assert_eq!(results[1].tool_call_id, "call-2");
        assert_eq!(results[1].content, crate::TOOL_CALL_SKIPPED_MESSAGE);
        assert!(session.pending_tool_calls.is_empty());
        assert_eq!(session.agent.messages().len(), 3);
        // The assistant message that requested the calls is kept.
        assert!(matches!(session.agent.messages()[1], Message::Assistant(_)));
    }

    #[test]
    fn interrupt_marks_an_executing_call_as_interrupted() {
        let mut session = session();
        start_two_call_batch(&mut session);
        session.approval = super::ApprovalState::Approved;
        session.pending_tool_calls[0].started = true;

        session.interrupt();

        let Message::Tool(results) = session.agent.messages().last().unwrap() else {
            unreachable!()
        };
        assert_eq!(results[0].tool_call_id, "call-1");
        assert_eq!(results[0].content, crate::TOOL_CALL_INTERRUPTED_MESSAGE);
        assert_eq!(results[1].tool_call_id, "call-2");
        assert_eq!(results[1].content, crate::TOOL_CALL_SKIPPED_MESSAGE);
    }
}

// TODO: must handle (de)serialization
