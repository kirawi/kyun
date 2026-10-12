use kyun_agent::{
    Agent, Backend, BackendEvent, BackendFuture, BackendRequest, BackendStream, Message,
    openai::OpenAiBackend,
};

/// A backend that returns a fixed sequence of streaming events.
pub struct StreamingBackend {
    pub events: Vec<BackendEvent>,
}

impl Backend for StreamingBackend {
    fn request(&mut self, _request: BackendRequest<'_>) -> BackendFuture {
        unreachable!("streaming fixture only supports streaming requests")
    }

    fn request_stream(&mut self, _request: BackendRequest<'_>) -> BackendStream {
        Box::pin(futures_util::stream::iter(
            std::mem::take(&mut self.events).into_iter().map(Ok),
        ))
    }
}

/// A backend whose requests are never polled; it exists only so test fixtures can build idle agents.
#[derive(Default)]
pub struct InertBackend;

impl Backend for InertBackend {
    fn request(&mut self, _request: BackendRequest<'_>) -> BackendFuture {
        unreachable!("idle fixtures never run steps")
    }
}

pub fn agent_with(messages: impl IntoIterator<Item = Message>) -> Agent<InertBackend> {
    Agent::with_messages(InertBackend, messages)
}

pub fn visible_messages(agent: &Agent<OpenAiBackend>) -> Vec<Message> {
    agent.messages().iter().cloned().collect()
}

/// Builds test conversations through the same transitions used by the application.
pub fn append_messages(
    agent: &mut Agent<OpenAiBackend>,
    messages: impl IntoIterator<Item = Message>,
) {
    for message in messages {
        match message {
            Message::User(content) => {
                agent.start_turn(content);
            }
            Message::Assistant(assistant) => {
                if agent.messages().is_empty() {
                    // Some transcript fixtures begin with a synthetic empty user message.
                    agent.start_turn(String::new());
                }
                agent.submit_assistant_msg(assistant);
            }
            Message::Tool(results) => {
                assert!(!results.is_empty());
                for result in results {
                    agent.append_tool_result(result);
                }
            }
            Message::System(_) => {
                panic!("construct test agents with Agent::with_messages for initial messages")
            }
        }
    }
}
