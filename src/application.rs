use std::collections::VecDeque;
use std::env;
use std::future::poll_fn;
use std::io::stdout;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    EventStream, KeyEventKind, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use futures_util::StreamExt;
use kyun_agent::{StepEvent, ToolDefinition};
use ratatui::DefaultTerminal;
use tokio::time::Instant;

use crate::compositor::Compositor;
use crate::config::Config;
use crate::event::Event;
use crate::session::{Action, ApprovalState, PendingToolCall, Session};
use crate::timer::{Output, STATUSLINE_TIMER_ID};
use crate::widgets::statusline::SPINNER_INTERVAL;
use crate::widgets::transcript::{TOOL_LIMITS, ToolLimits};
use crate::widgets::{CommandLine, Prompt, StatusLine, Transcript};

fn enable_terminal_modes(keyboard_enhancement_enabled: bool) {
    execute!(stdout(), EnableMouseCapture, EnableBracketedPaste).unwrap();
    if keyboard_enhancement_enabled {
        execute!(
            stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
        .unwrap();
    }
}

fn disable_terminal_modes(keyboard_enhancement_enabled: bool) -> std::io::Result<()> {
    if keyboard_enhancement_enabled {
        execute!(stdout(), PopKeyboardEnhancementFlags)?;
    }
    execute!(stdout(), DisableBracketedPaste, DisableMouseCapture)?;
    Ok(())
}

pub struct Application {
    config: Arc<Config>,
    terminal: DefaultTerminal,
    compositor: Compositor,
    // TODO: ensure there is always at least one agent
    session: Session,
    tool_definitions: Vec<ToolDefinition>,
}

const APPROVAL_IDLE_DELAY: Duration = Duration::from_secs(1);

impl Application {
    pub fn new(config: Config) -> Self {
        let terminal = ratatui::init();
        let tool_definitions = config.tool_definitions();
        let mut compositor = Compositor::new();
        let session = Session::new(config.make_agent());
        let config = Arc::new(config);

        let limits = ToolLimits {
            output_line_limit: config.tool_output_line_limit,
            diff_line_limit: config.tool_diff_line_limit,
        };
        let mut runtime_limits = TOOL_LIMITS.write().unwrap();
        runtime_limits.current = limits;
        runtime_limits.default = limits;

        // Needed to avoid a deadlock on TOOL_LIMITS which also gets used by Transcript.
        drop(runtime_limits);

        // TODO: Handle edge case terminal heights later. And widths (e.g. not all elements fit on status line)
        // [Content]      = height - 8
        // [Prompt]       = 6
        // [status line]  = 1
        // [command line] = 1
        compositor.push(Box::new(Transcript::new(config.render_markdown)));
        compositor.push(Box::new(Prompt::default()));
        compositor.push(Box::new(StatusLine::new(config.max_tokens)));
        compositor.push(Box::new(CommandLine::default()));

        Self {
            config,
            terminal,
            compositor,
            session,
            tool_definitions,
        }
    }

    // The main loop.
    pub async fn run(&mut self) {
        self.render().unwrap();

        let keyboard_enhancement_enabled =
            crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
        let panic_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            let _ = disable_terminal_modes(keyboard_enhancement_enabled);
            panic_hook(panic_info);
        }));
        enable_terminal_modes(keyboard_enhancement_enabled);

        let mut input_stream = EventStream::new();
        while !self.session.should_exit {
            let should_redraw = tokio::select! {
                biased;

                event = input_stream.next() => {
                    match event {
                        Some(Ok(event)) => {
                            let mut should_redraw = self.handle_input(event.into(), keyboard_enhancement_enabled);

                            // To not starve other work
                            // TODO: make conditional on whether bracketed paste is detected; otherwise human input too slow to be a problem
                            // TODO: wait, consider other types of events that might happen a lot?
                            const MAX_DRAINED: usize = 64;
                            let mut drained = 0;
                            while drained < MAX_DRAINED && let Ok(Some(Ok(event))) = tokio::time::timeout(Duration::ZERO, input_stream.next()).await {
                                if self.handle_input(event.into(), keyboard_enhancement_enabled) {
                                    should_redraw = true;
                                }
                                drained += 1;
                            }
                            should_redraw
                        }
                        // A dead tty (EIO) errors forever; input - including the quit key - would never
                        // arrive again while the status timer keeps the loop alive, so exit instead.
                        Some(Err(_)) | None => {
                            self.session.should_exit = true;
                            false
                        }
                    }
                }
                Some(()) = self.session.timers.wait_next_deadline() => self.session.fire_timer(&mut self.compositor),
                event = poll_fn(|cx| self.session.agent.poll_step(cx)) => {
                    self.handle_agent_event(event).await;
                    true
                },
                result = poll_fn(|cx| {
                    let Some(execution) = self.session.pending_tool_calls.front_mut() else {
                        return Poll::Pending;
                    };

                    if self.session.approval != ApprovalState::Approved {
                        return Poll::Pending;
                    }
                    execution.started = true;
                    execution.future.as_mut().poll(cx)
                }) => {
                    self.session.finish_tool_call(result);
                    self.start_next_tool_call();
                    true
                }
            };

            if should_redraw {
                self.render().unwrap();
            }
        }

        // TODO: maybe move out to exit() function?
        disable_terminal_modes(keyboard_enhancement_enabled).unwrap();
        ratatui::restore();
    }

    /// Returns `true` if the input event was consumed (i.e. the application acted upon it).
    fn handle_input(&mut self, event: Event, keyboard_enhancement_enabled: bool) -> bool {
        // Debounce approval prompt when the user is typing anything.
        if (matches!(&event, Event::Key(key) if key.kind == KeyEventKind::Press)
            || matches!(&event, Event::Paste(_)))
            && self.session.approval == ApprovalState::WaitingForIdle
        {
            self.schedule_pending_approval(Instant::now() + APPROVAL_IDLE_DELAY);
        }

        let frame = self.terminal.get_frame();
        let area = frame.area();

        let consumed = self
            .compositor
            .handle_event(&event, area, &mut self.session);

        // Certain widgets can produce actions which we need to resolve first.
        match self.session.pending_action.take() {
            Some(Action::Interrupt) => {
                self.interrupt();
                true
            }
            Some(Action::ResolveToolApproval(approved)) => {
                self.resolve_tool_approval(approved);
                true
            }
            Some(Action::StepAgent) => {
                self.start_step_or_notice();
                true
            }
            Some(Action::NewSession) => {
                // The session is replaced before the widgets are reset so they observe the new, empty history.
                self.session = Session::new(self.config.make_agent());
                self.compositor
                    .find::<Prompt>()
                    .unwrap()
                    .replace_content("");
                self.compositor
                    .find::<Transcript>()
                    .unwrap()
                    .reset_after_history_change(self.session.agent.messages());
                true
            }
            Some(action @ Action::EditPrompt) | Some(action @ Action::ViewTranscript) => {
                disable_terminal_modes(keyboard_enhancement_enabled).unwrap();
                ratatui::restore();

                let editor = env::var("EDITOR").unwrap_or_else(|_| "vi".into());
                let outcome = match action {
                    Action::EditPrompt => self.compositor.find::<Prompt>().unwrap().edit(&editor),
                    Action::ViewTranscript => self
                        .compositor
                        .find::<Transcript>()
                        .unwrap()
                        .view(self.session.agent.messages(), &editor),
                    _ => unreachable!(),
                };
                if let Err(error) = outcome {
                    // The terminal is already re-enabled by this point, so the message is visible.
                    self.session.set_error(format!("{error:#}"));
                }

                enable_raw_mode().unwrap();
                execute!(stdout(), EnterAlternateScreen).unwrap();
                enable_terminal_modes(keyboard_enhancement_enabled);
                self.terminal.clear().unwrap();
                true
            }
            None => consumed,
        }
    }

    fn interrupt(&mut self) {
        if self.session.interrupt() {
            self.compositor
                .find::<Transcript>()
                .unwrap()
                .reset_after_history_change(self.session.agent.messages());
        }
    }

    async fn handle_agent_event(&mut self, event: StepEvent) {
        match event {
            StepEvent::Updated {
                n_added_reasoning: reasoning,
                n_added_content: content,
            } => {
                self.compositor.handle_event(
                    &Event::AgentUpdated { reasoning, content },
                    self.terminal.get_frame().area(),
                    &mut self.session,
                );
            }
            StepEvent::Finished => {
                self.session.step_count = 0;
                self.session.timers.cancel(STATUSLINE_TIMER_ID);
            }
            StepEvent::Error(error) => {
                // The turn is over: nothing auto-starts afterwards, and the user must be able to submit again.
                self.session.step_count = 0;
                self.session.timers.cancel(STATUSLINE_TIMER_ID);
                self.compositor
                    .find::<Transcript>()
                    .unwrap()
                    .reset_after_history_change(self.session.agent.messages());
                self.session.set_error(error.to_string());
            }
            StepEvent::ToolCalls => {
                let mut mutated_paths = std::collections::HashSet::new();
                let mut context = crate::tools::Context {
                    compositor: &mut self.compositor,
                    mutated_paths: &mut mutated_paths,
                };

                // Prepare every call before prompting so each mutation snapshot is captured before execution.
                let calls = self.session.get_last_tool_calls();
                let mut pending = VecDeque::with_capacity(calls.len());
                for call in calls {
                    let future = crate::tools::prepare_execution(
                        call,
                        Arc::clone(&self.config),
                        &mut context,
                    )
                    .await;
                    pending.push_back(PendingToolCall {
                        future,
                        started: false,
                    });
                }
                self.session.pending_tool_calls = pending;
                self.session.approval = ApprovalState::Inactive;
                self.start_next_tool_call();
            }
        }
    }

    /// Starts the next agent step, surfacing the step limit instead of silently ending the turn.
    fn start_step_or_notice(&mut self) {
        let started = self.session.start_agent_step(
            self.config.max_steps,
            &self.tool_definitions,
            self.config.stream_assistant_messages,
        );
        if started {
            self.session.timers.schedule(
                STATUSLINE_TIMER_ID,
                Instant::now() + SPINNER_INTERVAL,
                |compositor, session| {
                    if !session.is_busy() {
                        return (false, Output::Stop);
                    }
                    compositor.find::<StatusLine>().unwrap().advance_spinner();
                    (true, Output::Reschedule(Instant::now() + SPINNER_INTERVAL))
                },
            );
        } else {
            self.session.timers.cancel(STATUSLINE_TIMER_ID);
            self.session.set_status(format!(
                "turn stopped: reached max_steps ({})",
                self.config.max_steps
            ));
        }
    }

    fn start_next_tool_call(&mut self) {
        if self.session.pending_tool_calls.is_empty() {
            self.start_step_or_notice();
        } else if self.config.skip_approval {
            self.resolve_tool_approval(true);
        } else {
            // Schedule approval prompt to appear (debounced on user input).
            self.session.approval = ApprovalState::WaitingForIdle;
            self.schedule_pending_approval(Instant::now() + APPROVAL_IDLE_DELAY);
        }
    }

    fn schedule_pending_approval(&mut self, deadline: Instant) {
        self.session.timers.schedule(
            crate::timer::APPROVAL_TIMER_ID,
            deadline,
            |compositor, session| {
                session.show_tool_approval(compositor);
                (true, Output::Stop)
            },
        );
    }

    fn resolve_tool_approval(&mut self, approved: bool) {
        if approved {
            self.session.approval = ApprovalState::Approved;
        } else {
            self.session.approval = ApprovalState::Inactive;

            // Tool calls still need results; so use the denial message for remaining tool calls.
            // TODO: make tool call rejection granular so you deny one at a time, maybe? with a way to deny all if desired?
            // could make agent confused though if user allows that.
            self.session.pending_tool_calls.clear();
            self.session
                .agent
                .resolve_remaining_tool_calls(crate::TOOL_CALL_DENIED_MESSAGE);
            self.session.step_count = 0;
            self.session.timers.cancel(STATUSLINE_TIMER_ID);
        }
    }

    // TODO: figure out if we can avoid rendering for ocluded widgets, i.e. widgets whose area is occupied by a later
    // widget
    fn render(&mut self) -> anyhow::Result<()> {
        self.terminal.draw(|frame| {
            let area = frame.area();
            self.compositor
                .handle_event(&Event::Redraw, area, &mut self.session);
            self.compositor.render(frame, area, &mut self.session);
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use kyun_agent::{
        Agent, AssistantMessage, Message, StepEvent, ToolCall, ToolCallKind, ToolResult,
        openai::OpenAiBackend,
    };
    use ratatui::{Terminal, TerminalOptions, backend::CrosstermBackend, layout::Rect};

    use super::Application;
    use crate::component::Component;
    use crate::compositor::Compositor;
    use crate::config::Config;
    use crate::event::Event;
    use crate::session::{Action, ApprovalState, PendingToolCall, Session, Severity, UIMode};
    use crate::test_support::visible_messages;
    use crate::tools::{edit, read, write};
    use crate::widgets::{Approval, CommandLine, Prompt, Transcript};

    fn pending() -> PendingToolCall {
        PendingToolCall {
            future: Box::pin(std::future::pending::<ToolResult>()),
            started: false,
        }
    }

    #[test]
    fn typing_delays_approval_without_stealing_prompt_input() {
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        session.mode = UIMode::Insert;
        session.pending_tool_calls.push_back(pending());
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    kind: ToolCallKind::Function,
                    function_name: read::NAME.into(),
                    arguments: "{}".into(),
                }],
            })],
        );

        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Prompt::default()));
        let mut application = Application {
            config: std::sync::Arc::new(Config::default()),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };

        application.start_next_tool_call();
        assert_eq!(application.session.approval, ApprovalState::WaitingForIdle);
        assert!(application.compositor.find::<Approval>().is_none());
        let original_deadline = application.session.timers.next_deadline().unwrap();

        assert!(application.handle_input(
            Event::Key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE)),
            false,
        ));
        assert_eq!(
            application.compositor.find::<Prompt>().unwrap().content(),
            "y"
        );
        assert!(application.session.timers.next_deadline().unwrap() > original_deadline);
        assert_eq!(application.session.approval, ApprovalState::WaitingForIdle);

        application.schedule_pending_approval(tokio::time::Instant::now());
        assert!(application.session.fire_timer(&mut application.compositor));
        assert_eq!(
            application.session.approval,
            ApprovalState::AwaitingDecision
        );
        assert!(application.compositor.find::<Approval>().is_some());
    }

    #[test]
    fn rejecting_a_tool_batch_denies_every_call_and_stops_the_agent() {
        let calls = vec![
            ToolCall {
                id: "call-1".into(),
                kind: ToolCallKind::Function,
                function_name: read::NAME.into(),
                arguments: "{}".into(),
            },
            ToolCall {
                id: "call-2".into(),
                kind: ToolCallKind::Function,
                function_name: write::NAME.into(),
                arguments: "{}".into(),
            },
        ];
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        session.step_count = 1;
        session
            .pending_tool_calls
            .extend(calls.iter().map(|_| pending()));
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: calls.clone(),
            })],
        );

        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let mut config = Config::default();
        config.skip_approval = false;
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        let mut application = Application {
            config: std::sync::Arc::new(config),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };
        application
            .session
            .show_tool_approval(&mut application.compositor);

        assert!(application.handle_input(
            Event::Key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE)),
            false,
        ));

        // Denying stops the whole batch: every pending call is denied, no further approvals are prompted, and the
        // session is idle instead of starting another agent step.
        assert_eq!(application.session.step_count, 0);
        assert!(application.session.pending_tool_calls.is_empty());
        assert!(!application.session.is_busy());
        assert!(application.compositor.find::<Approval>().is_none());

        let Message::Tool(results) = &application.session.agent.messages()[2] else {
            unreachable!()
        };
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool_call_id, "call-1");
        assert_eq!(results[1].tool_call_id, "call-2");
        assert!(
            results
                .iter()
                .all(|result| result.content == crate::TOOL_CALL_DENIED_MESSAGE)
        );
    }

    #[test]
    fn skip_approval_keeps_the_call_pending_during_execution() {
        let call = ToolCall {
            id: "call-1".into(),
            kind: ToolCallKind::Function,
            function_name: read::NAME.into(),
            arguments: "{}".into(),
        };
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        session.pending_tool_calls.push_back(pending());
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![call],
            })],
        );

        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let config = Config {
            skip_approval: true,
            ..Config::default()
        };
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        let mut application = Application {
            config: std::sync::Arc::new(config),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };

        application.start_next_tool_call();

        assert_eq!(application.session.pending_tool_calls.len(), 1);
        assert_eq!(application.session.approval, ApprovalState::Approved);
        assert!(application.compositor.find::<Approval>().is_none());

        // The main loop marked the call as executing (its future was polled).
        application.session.pending_tool_calls[0].started = true;

        application.interrupt();

        assert!(application.session.pending_tool_calls.is_empty());
        // The call had started executing, so it cannot be reported as never having run.
        let Message::Tool(results) = application.session.agent.messages().last().unwrap() else {
            unreachable!()
        };
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool_call_id, "call-1");
        assert_eq!(results[0].content, crate::TOOL_CALL_INTERRUPTED_MESSAGE);
    }

    #[tokio::test]
    async fn reaching_max_steps_surfaces_a_notice_and_stops_the_turn() {
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        session.agent.start_turn("hi".into());
        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let mut config = Config::default();
        config.max_steps = 0;
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(CommandLine::default()));
        let mut application = Application {
            config: std::sync::Arc::new(config),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };

        // A submitted turn that can never run says why instead of silently doing nothing.
        application.start_step_or_notice();

        assert_eq!(
            application.session.status(),
            Some(("turn stopped: reached max_steps (0)", Severity::Info))
        );
        assert!(!application.session.is_busy());
        assert_eq!(application.session.agent.messages().len(), 1);
    }

    #[tokio::test]
    async fn same_file_calls_in_one_batch_chain_instead_of_failing() {
        // The write tool works relative to the process working directory. The filename is unique per run.
        let path = std::env::current_dir().unwrap().join(format!(
            "target/same_file_calls_in_one_batch_chain_{}.txt",
            std::process::id()
        ));
        tokio::fs::write(&path, "one\ntwo\n").await.unwrap();
        let calls = vec![
            ToolCall {
                id: "call-1".into(),
                kind: ToolCallKind::Function,
                function_name: write::NAME.into(),
                arguments: serde_json::json!({
                    "path": path.display().to_string(),
                    "content": "one\ntwo\nthree\n",
                })
                .to_string(),
            },
            // The edit's context only exists after call 1 ran; preparing it against the pre-batch state
            // fails, and executing it against the stale snapshot would have failed before this fix.
            ToolCall {
                id: "call-2".into(),
                kind: ToolCallKind::Function,
                function_name: edit::NAME.into(),
                arguments: serde_json::json!({
                    "path": path.display().to_string(),
                    "edits": [{ "oldText": "two\nthree", "newText": "two\nTHREE" }],
                })
                .to_string(),
            },
        ];
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: calls.clone(),
            })],
        );

        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let config = Config {
            skip_approval: true,
            ..Config::default()
        };
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        let mut application = Application {
            config: std::sync::Arc::new(config),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };

        application.compositor.handle_event(
            &Event::Redraw,
            Rect::new(0, 0, 80, 24),
            &mut application.session,
        );
        application.handle_agent_event(StepEvent::ToolCalls).await;
        assert_eq!(application.session.pending_tool_calls.len(), 2);

        let mut results = Vec::new();
        while !application.session.pending_tool_calls.is_empty() {
            application.start_next_tool_call();
            let result = std::future::poll_fn(|cx| {
                application
                    .session
                    .pending_tool_calls
                    .front_mut()
                    .unwrap()
                    .future
                    .as_mut()
                    .poll(cx)
            })
            .await;
            results.push(result.content.clone());
            application.session.finish_tool_call(result);
            application.start_next_tool_call();
        }

        // Both calls executed against the batch's own chain of results.
        assert_eq!(results.len(), 2);
        assert!(!results[0].starts_with("Error:"), "{}", results[0]);
        assert!(!results[1].starts_with("Error:"), "{}", results[1]);
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "one\ntwo\nTHREE\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn a_backend_error_resets_the_turn_and_leaves_the_session_submittable() {
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        // A turn was in progress when the provider failed.
        session.step_count = 3;
        crate::test_support::append_messages(&mut session.agent, [Message::User("hi".into())]);

        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(CommandLine::default()));
        let mut application = Application {
            config: std::sync::Arc::new(Config::default()),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };

        application
            .handle_agent_event(StepEvent::Error(kyun_agent::BackendError::Http {
                status: 429,
                message: "rate limited".into(),
            }))
            .await;

        // The turn is over and a new one can start immediately; the history keeps the user's prompt.
        assert_eq!(application.session.step_count, 0);
        assert!(!application.session.is_busy());
        assert_eq!(
            application.session.status(),
            Some(("provider returned HTTP 429: rate limited", Severity::Error))
        );
        application.session.agent.start_turn("next".into());
        assert!(application.session.start_agent_step(5, &[], false));
        assert_eq!(
            visible_messages(&application.session.agent),
            vec![Message::User("hi".into()), Message::User("next".into())]
        );
    }

    #[test]
    fn a_new_session_action_replaces_the_session_and_resets_the_views() {
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::User("sentinel turn".into())],
        );
        session.mode = UIMode::Insert;
        session.set_status("stale status");

        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(Prompt::default()));
        let mut application = Application {
            config: std::sync::Arc::new(Config::default()),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };
        application
            .compositor
            .find::<Prompt>()
            .unwrap()
            .replace_content("draft");

        application.session.pending_action = Some(Action::NewSession);
        application.handle_input(
            Event::Key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE)),
            false,
        );

        let agent = &application.session.agent;
        assert_eq!(agent.messages().len(), 1);
        assert!(matches!(agent.messages()[0], Message::System(_)));
        assert_eq!(application.session.mode, UIMode::Normal);
        assert_eq!(application.session.status(), None);
        assert!(application.session.pending_action.is_none());
        assert_eq!(
            application.compositor.find::<Prompt>().unwrap().content(),
            ""
        );

        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(120, 16)).unwrap();
        application.compositor.handle_event(
            &Event::Redraw,
            terminal.get_frame().area(),
            &mut application.session,
        );
        terminal
            .draw(|frame| {
                application.compositor.find::<Transcript>().unwrap().render(
                    frame,
                    frame.area(),
                    &mut application.session,
                )
            })
            .unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(!rendered.contains("sentinel"));
    }

    #[tokio::test]
    async fn approved_mutation_renders_from_the_context_cached_diff() {
        // The write tool works relative to the process working directory. The filename is unique per run so
        // the mutation is never a no-op against a leftover file.
        let path = std::env::current_dir().unwrap().join(format!(
            "target/approved_mutation_renders_from_the_context_cached_diff_{}.txt",
            std::process::id()
        ));
        let call = ToolCall {
            id: "write".into(),
            kind: ToolCallKind::Function,
            function_name: write::NAME.into(),
            arguments: serde_json::json!({
                "path": path.display().to_string(),
                "content": "first\nsecond\n",
            })
            .to_string(),
        };
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![call.clone()],
            })],
        );

        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        let mut config = Config::default();
        config.skip_approval = false;
        let mut application = Application {
            config: std::sync::Arc::new(config),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };
        application.compositor.handle_event(
            &Event::Redraw,
            Rect::new(0, 0, 80, 24),
            &mut application.session,
        );
        let mut mutated_paths = std::collections::HashSet::new();
        let mut context = crate::tools::Context {
            compositor: &mut application.compositor,
            mutated_paths: &mut mutated_paths,
        };
        let future = crate::tools::prepare_execution(
            &call,
            std::sync::Arc::clone(&application.config),
            &mut context,
        )
        .await;
        application
            .session
            .pending_tool_calls
            .push_back(PendingToolCall {
                future,
                started: false,
            });
        application
            .session
            .show_tool_approval(&mut application.compositor);

        assert!(application.handle_input(
            Event::Key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE)),
            false,
        ));
        let result = std::future::poll_fn(|cx| {
            application.session.pending_tool_calls[0]
                .future
                .as_mut()
                .poll(cx)
        })
        .await;
        let executed: crate::tools::FileMutationOutput =
            serde_json::from_str(&result.content).unwrap();
        assert!(executed.diff.contains("+first\n+second"));
        application.session.finish_tool_call(result);
        application.start_next_tool_call();
        let Message::Tool(results) = &application.session.agent.messages()[2] else {
            unreachable!()
        };
        let output: crate::tools::FileMutationOutput =
            serde_json::from_str(&results[0].content).unwrap();
        assert_eq!(output.summary, "Created 13 bytes.");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first\nsecond\n");

        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(120, 16)).unwrap();
        application.compositor.handle_event(
            &Event::Redraw,
            terminal.get_frame().area(),
            &mut application.session,
        );
        terminal
            .draw(|frame| {
                application.compositor.find::<Transcript>().unwrap().render(
                    frame,
                    frame.area(),
                    &mut application.session,
                )
            })
            .unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        let pathname = path.file_name().unwrap().to_string_lossy();
        assert!(
            rendered.contains(&format!("{pathname} (+2 -0)")),
            "{rendered:?}"
        );
        assert!(rendered.contains("1 + first"), "{rendered:?}");
        assert!(rendered.contains("2 + second"), "{rendered:?}");
        assert!(!rendered.contains("@@"), "{rendered:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn denied_mutation_still_renders_the_context_cached_diff() {
        // The write tool works relative to the process working directory. The filename is unique per run.
        let path = std::env::current_dir().unwrap().join(format!(
            "target/denied_mutation_still_renders_the_context_cached_diff_{}.txt",
            std::process::id()
        ));
        let call = ToolCall {
            id: "write".into(),
            kind: ToolCallKind::Function,
            function_name: write::NAME.into(),
            arguments: serde_json::json!({
                "path": path.display().to_string(),
                "content": "first\nsecond\n",
            })
            .to_string(),
        };
        let mut session =
            Session::new(Agent::new(OpenAiBackend::new(kyun_agent::openai::Config {
                model: String::new(),
                base_url: String::new(),
                api_key: None,
                reasoning_effort: None,
                extra_body: None,
                ..kyun_agent::openai::Config::default()
            })));
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![call.clone()],
            })],
        );

        let terminal = Terminal::with_options(
            CrosstermBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: ratatui::Viewport::Fixed(Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap();
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        let mut config = Config::default();
        config.skip_approval = false;
        let mut application = Application {
            config: std::sync::Arc::new(config),
            terminal,
            compositor,
            session,
            tool_definitions: Vec::new(),
        };
        application.compositor.handle_event(
            &Event::Redraw,
            Rect::new(0, 0, 80, 24),
            &mut application.session,
        );
        let mut mutated_paths = std::collections::HashSet::new();
        let mut context = crate::tools::Context {
            compositor: &mut application.compositor,
            mutated_paths: &mut mutated_paths,
        };
        let future = crate::tools::prepare_execution(
            &call,
            std::sync::Arc::clone(&application.config),
            &mut context,
        )
        .await;
        application
            .session
            .pending_tool_calls
            .push_back(PendingToolCall {
                future,
                started: false,
            });
        application
            .session
            .show_tool_approval(&mut application.compositor);

        assert!(application.handle_input(
            Event::Key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE)),
            false,
        ));
        let Message::Tool(results) = &application.session.agent.messages()[2] else {
            unreachable!()
        };
        assert_eq!(results[0].content, crate::TOOL_CALL_DENIED_MESSAGE);
        assert!(std::fs::read_to_string(&path).is_err());

        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(120, 16)).unwrap();
        application.compositor.handle_event(
            &Event::Redraw,
            terminal.get_frame().area(),
            &mut application.session,
        );
        terminal
            .draw(|frame| {
                application.compositor.find::<Transcript>().unwrap().render(
                    frame,
                    frame.area(),
                    &mut application.session,
                )
            })
            .unwrap();
        let buffer = &terminal.backend().buffer();
        let rendered = buffer
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        let pathname = path.file_name().unwrap().to_string_lossy();
        assert!(
            rendered.contains(&format!("{pathname} (+2 -0)")),
            "{rendered:?}"
        );
        assert!(rendered.contains("+ first"), "{rendered:?}");
        assert!(buffer.content.iter().any(|cell| {
            cell.symbol() == "+" && cell.modifier.contains(ratatui::style::Modifier::DIM)
        }));
    }
}
