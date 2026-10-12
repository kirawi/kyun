use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::{anyhow, bail};
use kyun_agent::Message;

use crate::compositor::Compositor;
use crate::session::{Action, Session};
use crate::widgets::{Prompt, Transcript};

pub const UNCOMMITTED_PROMPT_ERROR: &str = "cannot undo while the prompt has uncommitted text";

#[derive(Clone)]
pub struct TypedCommand {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub doc: &'static str,
    pub fun: fn(&mut Context<'_>, &str) -> anyhow::Result<()>,
}

pub struct Context<'a> {
    pub session: &'a mut Session,
    pub compositor: &'a mut Compositor,
}

pub const TYPED_COMMAND_LIST: &[TypedCommand] = &[
    TypedCommand {
        name: "new",
        aliases: &[],
        doc: "Discard the current session and start a new, empty one.",
        fun: new,
    },
    TypedCommand {
        name: "undo",
        aliases: &[],
        doc: "Undo recent turns and restore the oldest removed prompt.",
        fun: undo,
    },
];

pub static TYPED_COMMAND_MAP: LazyLock<HashMap<&'static str, &'static TypedCommand>> =
    LazyLock::new(|| {
        debug_assert!(
            TYPED_COMMAND_LIST
                .iter()
                .all(|command| !command.doc.is_empty())
        );
        TYPED_COMMAND_LIST
            .iter()
            .flat_map(|command| {
                std::iter::once((command.name, command))
                    .chain(command.aliases.iter().map(move |&alias| (alias, command)))
            })
            .collect()
    });

impl Session {
    pub fn execute_command(
        &mut self,
        compositor: &mut Compositor,
        input: &str,
    ) -> anyhow::Result<()> {
        execute_command_line(
            &mut Context {
                session: self,
                compositor,
            },
            input,
        )
    }
}

fn execute_command_line(cx: &mut Context<'_>, input: &str) -> anyhow::Result<()> {
    let input = input.trim_start();
    if input.is_empty() {
        return Ok(());
    }
    let (name, arguments) = input.split_once(char::is_whitespace).unwrap_or((input, ""));
    let command = TYPED_COMMAND_MAP
        .get(name)
        .ok_or_else(|| anyhow!("no such command: '{name}'"))?;
    execute_command(cx, command, arguments)
}

fn execute_command(
    cx: &mut Context<'_>,
    command: &TypedCommand,
    arguments: &str,
) -> anyhow::Result<()> {
    (command.fun)(cx, arguments).map_err(|error| anyhow!("'{}': {error}", command.name))
}

fn new(cx: &mut Context<'_>, arguments: &str) -> anyhow::Result<()> {
    if cx.session.is_busy() {
        bail!("cannot start a new session while the agent is running");
    }
    if !arguments.trim().is_empty() {
        bail!("usage: new");
    }
    cx.session.pending_action = Some(Action::NewSession);
    Ok(())
}

fn undo(cx: &mut Context<'_>, arguments: &str) -> anyhow::Result<()> {
    if cx
        .compositor
        .find::<Prompt>()
        .is_some_and(|prompt| !prompt.is_empty())
    {
        bail!(UNCOMMITTED_PROMPT_ERROR);
    }
    if cx.session.is_busy() {
        bail!("cannot undo while the agent is running");
    }

    let arguments = arguments.trim();
    let count = if arguments.is_empty() {
        1
    } else {
        arguments
            .parse::<usize>()
            .map_err(|_| anyhow!("usage: undo [positive count]"))?
    };
    if count == 0 {
        bail!("usage: undo [positive count]");
    }

    let mut seen_turns = 0;
    let mut had_tool_calls = false;
    for message in cx.session.agent.messages().iter().rev() {
        match message {
            Message::User(_) => {
                seen_turns += 1;
                if seen_turns == count {
                    break;
                }
            }
            Message::Assistant(assistant) if !assistant.tool_calls.is_empty() => {
                had_tool_calls = true;
            }
            _ => {}
        }
    }
    if seen_turns == 0 {
        bail!("there are no turns to undo");
    }

    let last_user_content = cx.session.agent.undo(count).unwrap();
    if let Some(prompt) = cx.compositor.find::<Prompt>() {
        prompt.replace_content(last_user_content);
    }
    cx.compositor
        .find::<Transcript>()
        .unwrap()
        .reset_after_history_change(cx.session.agent.messages());

    let noun = if seen_turns == 1 { "turn" } else { "turns" };
    let mut message = format!("Undid {seen_turns} {noun}");
    if had_tool_calls {
        message.push_str("; tool effects were not reverted");
    }
    cx.session.set_status(message);
    Ok(())
}

#[cfg(test)]
mod tests {
    use kyun_agent::{
        Agent, AssistantMessage, Message, ToolCall, ToolCallKind, ToolResult,
        openai::{Config, OpenAiBackend},
    };

    use crate::compositor::Compositor;
    use crate::session::{Action, Session};
    use crate::test_support::visible_messages;
    use crate::tools::write;
    use crate::widgets::Transcript;

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

    fn assistant(content: &str) -> Message {
        Message::Assistant(AssistantMessage {
            reasoning: None,
            content: Some(content.into()),
            tool_calls: Vec::new(),
        })
    }

    fn apply(command: &str, session: &mut Session) -> String {
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        session.execute_command(&mut compositor, command).unwrap();
        session.status().unwrap().0.to_owned()
    }

    #[test]
    fn undo_removes_complete_turns_and_reports_tool_effects() {
        let call = ToolCall {
            id: "call".into(),
            kind: ToolCallKind::Function,
            function_name: write::NAME.into(),
            arguments: "{}".into(),
        };
        let result = ToolResult {
            tool_call_id: "call".into(),
            content: "done".into(),
        };
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("first".into()),
                assistant("first answer"),
                Message::User("second".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![call],
                }),
                Message::Tool(vec![result]),
                assistant("second answer"),
                Message::User("third".into()),
                assistant("third answer"),
            ],
        );
        let message = apply("undo 2", &mut session);

        assert_eq!(
            visible_messages(&session.agent),
            vec![Message::User("first".into()), assistant("first answer")]
        );
        assert_eq!(message, "Undid 2 turns; tool effects were not reverted");
    }

    #[test]
    fn undo_cmd_no_args() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("first".into()),
                assistant("first answer"),
                Message::User("second".into()),
                assistant("second answer"),
            ],
        );
        let _ = apply("undo", &mut session);

        assert_eq!(
            visible_messages(&session.agent),
            vec![Message::User("first".into()), assistant("first answer")]
        );
    }

    #[test]
    fn undo_count_above_available_removes_all_turns() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("first".into()),
                assistant("first answer"),
                Message::User("second".into()),
                assistant("second answer"),
            ],
        );

        assert_eq!(apply("undo 3", &mut session), "Undid 2 turns");
        assert!(visible_messages(&session.agent).is_empty());
    }

    #[test]
    fn new_defers_to_the_application_and_refuses_while_busy() {
        let mut session = session();
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));

        // The command only records the request; building the session belongs to the application.
        session.execute_command(&mut compositor, "new").unwrap();
        assert_eq!(session.pending_action, Some(Action::NewSession));

        session.pending_action = None;
        assert_eq!(
            session
                .execute_command(&mut compositor, "new name")
                .unwrap_err()
                .to_string(),
            "'new': usage: new"
        );
        assert!(session.pending_action.is_none());

        session.step_count = 1;
        assert_eq!(
            session
                .execute_command(&mut compositor, "new")
                .unwrap_err()
                .to_string(),
            "'new': cannot start a new session while the agent is running"
        );
        assert!(session.pending_action.is_none());
    }

    #[test]
    fn dispatch_and_undo_validation_leave_history_unchanged_on_error() {
        let mut session = session();
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));

        assert_eq!(
            session
                .execute_command(&mut compositor, "unknown")
                .unwrap_err()
                .to_string(),
            "no such command: 'unknown'"
        );
        assert_eq!(
            session
                .execute_command(&mut compositor, "undo")
                .unwrap_err()
                .to_string(),
            "'undo': there are no turns to undo"
        );

        crate::test_support::append_messages(&mut session.agent, [Message::User("prompt".into())]);
        for input in ["undo 0", "undo invalid"] {
            assert_eq!(
                session
                    .execute_command(&mut compositor, input)
                    .unwrap_err()
                    .to_string(),
                "'undo': usage: undo [positive count]"
            );
        }
        session.step_count = 1;
        assert_eq!(
            session
                .execute_command(&mut compositor, "undo")
                .unwrap_err()
                .to_string(),
            "'undo': cannot undo while the agent is running"
        );
        assert_eq!(
            visible_messages(&session.agent),
            vec![Message::User("prompt".into())]
        );
    }
}
