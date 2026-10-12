use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

use crate::component::{Component, EventResult};
use crate::event::Event;
use crate::session::{Session, Severity, UIMode};
use crate::widgets::{
    base_areas, display_width, expand_tabs, next_word_boundary, previous_word_boundary, real_lines,
};

#[derive(Default)]
pub struct CommandLine {
    line: String,
    cursor: usize,
    anchor: usize,
}

impl CommandLine {
    fn reset(&mut self, session: &mut Session) {
        self.line.clear();
        self.cursor = 0;
        self.anchor = 0;
        session.clear_status();
    }
}

impl Component for CommandLine {
    fn render(&mut self, frame: &mut Frame, area: Rect, cx: &mut Session) {
        let area = base_areas(area).command_line;
        if area.width == 0 || area.height == 0 {
            return;
        }

        if cx.mode != UIMode::Command {
            // This state is possible when, for instance, an API error message arises while typing a command.
            if let Some((message, severity)) = cx.status() {
                frame.render_widget(
                    Line::styled(
                        message,
                        if severity == Severity::Error {
                            Style::new().fg(Color::Red)
                        } else {
                            Style::default()
                        },
                    ),
                    area,
                );
            }
            return;
        }

        let available = (area.width - 1) as usize;
        if self.cursor < self.anchor {
            self.anchor = self.cursor;
        }

        while available > 0 && display_width(&self.line[self.anchor..self.cursor], 0) >= available {
            let grapheme = self.line[self.anchor..].graphemes(true).next().unwrap();
            self.anchor += grapheme.len();
        }
        frame.render_widget(
            Line::from(vec![
                Span::raw(":"),
                Span::raw(expand_tabs(&self.line[self.anchor..], 0)),
            ]),
            area,
        );

        let cursor_x = if available == 0 {
            area.x
        } else {
            area.x + 1 + display_width(&self.line[self.anchor..self.cursor], 0) as u16
        };
        frame.set_cursor_position((cursor_x, area.y));
    }

    // TODO: need some generic abstraction... the commandline is basically just a specialized prompt
    fn handle_event(&mut self, event: &Event, _area: Rect, cx: &mut Session) -> EventResult {
        if let Event::Paste(content) = event {
            if cx.mode != UIMode::Command {
                return EventResult::Ignored(None);
            }
            let content = real_lines(content).collect::<Vec<_>>().join(" ");
            self.line.insert_str(self.cursor, &content);
            self.cursor += content.len();
            return EventResult::Consumed(None);
        }

        let Event::Key(key) = event else {
            return if cx.mode == UIMode::Command {
                EventResult::Consumed(None)
            } else {
                EventResult::Ignored(None)
            };
        };
        if key.kind != KeyEventKind::Press {
            return EventResult::Ignored(None);
        }

        if cx.mode != UIMode::Command {
            if cx.mode == UIMode::Normal && key.code == KeyCode::Char(':') {
                self.reset(cx);
                cx.mode = UIMode::Command;
                return EventResult::Consumed(None);
            }
            return EventResult::Ignored(None);
        }

        match key.code {
            KeyCode::Esc => {
                self.reset(cx);
                cx.mode = UIMode::Normal;
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.reset(cx);
                cx.mode = UIMode::Normal;
            }
            KeyCode::Enter => {
                let input = std::mem::take(&mut self.line);
                self.cursor = 0;
                self.anchor = 0;
                cx.mode = UIMode::Normal;
                if input.trim().is_empty() {
                    // An unread error must survive an accidental Enter; it is the only record of it.
                    if !cx
                        .status()
                        .is_some_and(|(_, severity)| severity == Severity::Error)
                    {
                        cx.clear_status();
                    }
                    return EventResult::Consumed(None);
                }
                return EventResult::Consumed(Some(Box::new(move |compositor, session| {
                    if let Err(error) = session.execute_command(compositor, &input) {
                        session.set_error(error.to_string());
                    }
                })));
            }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => self.cursor = 0,
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor = self.line.len();
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some((idx, _)) = self.line[..self.cursor].grapheme_indices(true).next_back()
                {
                    self.cursor = idx;
                }
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(grapheme) = self.line[self.cursor..].graphemes(true).next() {
                    self.cursor += grapheme.len();
                }
            }
            KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let boundary = previous_word_boundary(&self.line, self.cursor);
                self.line.replace_range(boundary..self.cursor, "");
                self.cursor = boundary;
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.line.replace_range(..self.cursor, "");
                self.cursor = 0;
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.line.truncate(self.cursor);
            }
            KeyCode::Left
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    || key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.cursor = previous_word_boundary(&self.line, self.cursor);
            }
            KeyCode::Right
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    || key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.cursor = next_word_boundary(&self.line, self.cursor);
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.cursor = previous_word_boundary(&self.line, self.cursor);
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.cursor = next_word_boundary(&self.line, self.cursor);
            }
            KeyCode::Backspace if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let boundary = previous_word_boundary(&self.line, self.cursor);
                self.line.replace_range(boundary..self.cursor, "");
                self.cursor = boundary;
            }
            KeyCode::Delete if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let boundary = next_word_boundary(&self.line, self.cursor);
                self.line.replace_range(self.cursor..boundary, "");
            }
            KeyCode::Char('h') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some((idx, _)) = self.line[..self.cursor].grapheme_indices(true).next_back()
                {
                    self.line.replace_range(idx..self.cursor, "");
                    self.cursor = idx;
                }
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(grapheme) = self.line[self.cursor..].graphemes(true).next() {
                    self.line
                        .replace_range(self.cursor..self.cursor + grapheme.len(), "");
                }
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.line.insert(self.cursor, c);
                self.cursor += c.len_utf8();
            }
            KeyCode::Backspace => {
                if let Some((idx, _)) = self.line[..self.cursor].grapheme_indices(true).next_back()
                {
                    self.line.replace_range(idx..self.cursor, "");
                    self.cursor = idx;
                }
            }
            KeyCode::Delete => {
                if let Some(grapheme) = self.line[self.cursor..].graphemes(true).next() {
                    self.line
                        .replace_range(self.cursor..self.cursor + grapheme.len(), "");
                }
            }
            KeyCode::Left => {
                if let Some((idx, _)) = self.line[..self.cursor].grapheme_indices(true).next_back()
                {
                    self.cursor = idx;
                }
            }
            KeyCode::Right => {
                if let Some(grapheme) = self.line[self.cursor..].graphemes(true).next() {
                    self.cursor += grapheme.len();
                }
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.line.len(),
            _ => {}
        }
        EventResult::Consumed(None)
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use kyun_agent::{
        Agent, AssistantMessage, Message,
        openai::{Config, OpenAiBackend},
    };

    use ratatui::layout::Rect;

    use super::CommandLine;
    use crate::commands::UNCOMMITTED_PROMPT_ERROR;
    use crate::component::{Component, EventResult};
    use crate::compositor::Compositor;
    use crate::event::Event;
    use crate::session::{Session, Severity, UIMode};
    use crate::test_support::visible_messages;
    use crate::tools::read;
    use crate::widgets::{Approval, Prompt, Transcript};

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

    fn key(code: KeyCode, modifiers: KeyModifiers) -> Event {
        Event::Key(KeyEvent::new(code, modifiers))
    }

    fn command_line() -> CommandLine {
        CommandLine::default()
    }

    fn type_text(compositor: &mut Compositor, session: &mut Session, text: &str) {
        for character in text.chars() {
            assert!(compositor.handle_event(
                &key(KeyCode::Char(character), KeyModifiers::NONE),
                Default::default(),
                session,
            ));
        }
    }

    #[test]
    fn colon_activates_only_in_normal_mode() {
        let mut command_line = command_line();
        let mut session = session();
        session.mode = UIMode::Insert;

        assert!(matches!(
            command_line.handle_event(
                &key(KeyCode::Char(':'), KeyModifiers::NONE),
                Default::default(),
                &mut session,
            ),
            EventResult::Ignored(None)
        ));
        assert_eq!(session.mode, UIMode::Insert);

        session.mode = UIMode::Normal;
        assert!(matches!(
            command_line.handle_event(
                &key(KeyCode::Char(':'), KeyModifiers::NONE),
                Default::default(),
                &mut session,
            ),
            EventResult::Consumed(None)
        ));
        assert_eq!(session.mode, UIMode::Command);
    }

    #[test]
    fn escape_discards_the_command() {
        let mut session = session();
        crate::test_support::append_messages(&mut session.agent, [Message::User("first".into())]);
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(command_line()));

        let _ = compositor.handle_event(
            &key(KeyCode::Char(':'), KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        type_text(&mut compositor, &mut session, "unknown");
        let _ = compositor.handle_event(
            &key(KeyCode::Esc, KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        assert_eq!(session.mode, UIMode::Normal);

        let _ = compositor.handle_event(
            &key(KeyCode::Char(':'), KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        type_text(&mut compositor, &mut session, "undo");
        let _ = compositor.handle_event(
            &key(KeyCode::Enter, KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        assert!(session.agent.messages().is_empty());
    }

    #[test]
    fn undo_fills_the_prompt_with_the_last_user_turn() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("first".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("first answer".into()),
                    tool_calls: Vec::new(),
                }),
                Message::User("second\nturn".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("second answer".into()),
                    tool_calls: Vec::new(),
                }),
            ],
        );
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(Prompt::default()));
        compositor.push(Box::new(command_line()));

        let _ = compositor.handle_event(
            &key(KeyCode::Char(':'), KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        type_text(&mut compositor, &mut session, "undo");
        let _ = compositor.handle_event(
            &key(KeyCode::Enter, KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );

        assert_eq!(
            visible_messages(&session.agent),
            vec![
                Message::User("first".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("first answer".into()),
                    tool_calls: Vec::new(),
                }),
            ]
        );
        let prompt = compositor.find::<Prompt>().unwrap();
        assert_eq!(prompt.content(), "second\nturn");
        assert_eq!(session.status(), Some(("Undid 1 turn", Severity::Info)));
    }

    #[test]
    fn undo_is_rejected_when_the_prompt_has_uncommitted_text() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("last turn".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("answer".into()),
                    tool_calls: Vec::new(),
                }),
            ],
        );
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(Prompt::default()));
        compositor.push(Box::new(command_line()));
        compositor
            .find::<Prompt>()
            .unwrap()
            .replace_content("uncommitted");

        let _ = compositor.handle_event(
            &key(KeyCode::Char(':'), KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        type_text(&mut compositor, &mut session, "undo");
        let _ = compositor.handle_event(
            &key(KeyCode::Enter, KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );

        assert_eq!(
            visible_messages(&session.agent),
            vec![
                Message::User("last turn".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("answer".into()),
                    tool_calls: Vec::new(),
                }),
            ]
        );
        assert_eq!(
            compositor.find::<Prompt>().unwrap().content(),
            "uncommitted"
        );
        assert_eq!(
            session.status(),
            Some((
                format!("'undo': {UNCOMMITTED_PROMPT_ERROR}").as_str(),
                Severity::Error
            ))
        );
    }

    #[test]
    fn backspace_removes_one_unicode_grapheme() {
        let mut session = session();
        crate::test_support::append_messages(&mut session.agent, [Message::User("first".into())]);
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(command_line()));

        let _ = compositor.handle_event(
            &key(KeyCode::Char(':'), KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        type_text(&mut compositor, &mut session, "undoe\u{301}");
        let _ = compositor.handle_event(
            &key(KeyCode::Backspace, KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        let _ = compositor.handle_event(
            &key(KeyCode::Enter, KeyModifiers::NONE),
            Default::default(),
            &mut session,
        );
        assert!(session.agent.messages().is_empty());
    }

    #[test]
    fn approval_takes_precedence_over_command_activation() {
        let mut session = session();
        let mut compositor = Compositor::new();
        compositor.push(Box::new(command_line()));
        compositor.push(Box::new(Approval::new(
            read::NAME,
            "{}",
            |_, _| {},
            |_, _| {},
        )));

        assert!(compositor.handle_event(
            &key(KeyCode::Char(':'), KeyModifiers::NONE),
            Default::default(),
            &mut session,
        ));
        assert_eq!(session.mode, UIMode::Normal);
    }

    #[test]
    fn empty_enter_keeps_an_unread_error_and_clears_a_notice() {
        let mut compositor = Compositor::new();
        compositor.push(Box::new(CommandLine::default()));
        let mut session = session();
        session.mode = UIMode::Command;
        session.set_error("provider failed");

        let component = compositor.find::<CommandLine>().unwrap();
        component.handle_event(
            &Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )),
            Rect::new(0, 0, 80, 24),
            &mut session,
        );

        assert_eq!(session.status(), Some(("provider failed", Severity::Error)));

        // Re-enter command mode (Enter left it), then verify a plain notice is still cleared.
        let component = compositor.find::<CommandLine>().unwrap();
        component.handle_event(
            &Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char(':'),
                KeyModifiers::NONE,
            )),
            Rect::new(0, 0, 80, 24),
            &mut session,
        );
        session.set_status("done");
        let component = compositor.find::<CommandLine>().unwrap();
        component.handle_event(
            &Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )),
            Rect::new(0, 0, 80, 24),
            &mut session,
        );

        assert_eq!(session.status(), None);
    }
}
