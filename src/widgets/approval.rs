use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear};

use crate::component::{Callback, Component, EventResult};
use crate::event::Event;
use crate::session::{Action, Session};
use crate::widgets::{PENDING, TERM_BG, TERM_BORDER, TOOL, VIRTUAL, expand_tabs};

pub struct Approval {
    tool_name: String,
    arguments: String,
    on_accept: Option<Callback>,
    on_deny: Option<Callback>,
}

impl Approval {
    pub fn new(
        tool_name: impl Into<String>,
        arguments: impl Into<String>,
        on_accept: impl FnOnce(&mut crate::compositor::Compositor, &mut Session) + 'static,
        on_deny: impl FnOnce(&mut crate::compositor::Compositor, &mut Session) + 'static,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments: arguments.into(),
            on_accept: Some(Box::new(on_accept)),
            on_deny: Some(Box::new(on_deny)),
        }
    }
}

impl Component for Approval {
    fn render(&mut self, frame: &mut Frame, area: Rect, _cx: &mut Session) {
        let area = Layout::vertical([
            Constraint::Min(0),
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area)[1];
        let block = Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(TERM_BORDER))
            .style(Style::default().bg(TERM_BG));
        let inner = area.inner(Margin {
            vertical: 1,
            horizontal: 1,
        });

        frame.render_widget(Clear, area);
        frame.render_widget(block, area);
        frame.render_widget(
            Line::from(vec![
                Span::styled(
                    "Approval Required:",
                    Style::default().fg(PENDING).add_modifier(Modifier::BOLD),
                ),
                Span::raw("  Allow "),
                Span::styled(
                    self.tool_name.clone(),
                    Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(
                    expand_tabs(&self.arguments, 0),
                    Style::default().fg(VIRTUAL),
                ),
                Span::raw("?   "),
                Span::styled("[", Style::default().fg(VIRTUAL)),
                Span::styled(
                    "Y",
                    Style::default().fg(PENDING).add_modifier(Modifier::BOLD),
                ),
                Span::styled("es   [", Style::default().fg(VIRTUAL)),
                Span::styled(
                    "N",
                    Style::default().fg(PENDING).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "]o   <C-v> Transcript   <C-c> Cancel",
                    Style::default().fg(VIRTUAL),
                ),
            ]),
            inner,
        );
    }

    fn handle_event(&mut self, event: &Event, _area: Rect, cx: &mut Session) -> EventResult {
        let Event::Key(key) = event else {
            return EventResult::Ignored(None);
        };

        // The transcript stays scrollable while the approval is displayed: its motion keys pass through so the
        // layer beneath handles them. Esc is excluded because the transcript treats it as an interrupt request.
        if key.kind == KeyEventKind::Press {
            let navigates = match key.code {
                KeyCode::Down
                | KeyCode::Up
                | KeyCode::PageDown
                | KeyCode::PageUp
                | KeyCode::Home
                | KeyCode::End
                | KeyCode::Char('j')
                | KeyCode::Char('k')
                | KeyCode::Char(' ')
                | KeyCode::Char('g')
                | KeyCode::Char('G') => true,
                KeyCode::Char('d') | KeyCode::Char('u') => key.modifiers == KeyModifiers::CONTROL,
                _ => false,
            };
            if navigates {
                return EventResult::Ignored(None);
            }
        }

        let callback = match (key.kind, key.modifiers, key.code) {
            // Approve and deny accept only bare keys: Ctrl+Y approving a mutation nobody meant to approve
            // is worse than ignoring the chord.
            (KeyEventKind::Press, KeyModifiers::NONE, KeyCode::Char('y')) => self.on_accept.take(),
            (KeyEventKind::Press, KeyModifiers::NONE, KeyCode::Char('n')) => self.on_deny.take(),
            (KeyEventKind::Press, KeyModifiers::CONTROL, KeyCode::Char('v')) => {
                cx.pending_action = Some(Action::ViewTranscript);
                return EventResult::Consumed(None);
            }
            (KeyEventKind::Press, KeyModifiers::CONTROL, KeyCode::Char('t')) => {
                // To be consumed by Transcript
                return EventResult::Ignored(None);
            }
            (KeyEventKind::Press, KeyModifiers::CONTROL, KeyCode::Char('c')) => self.on_deny.take(),
            _ => return EventResult::Consumed(None),
        };

        EventResult::Consumed(Some(Box::new(move |compositor, session| {
            compositor.pop();
            callback.unwrap()(compositor, session);
        })))
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use kyun_agent::{
        Agent, Message,
        openai::{Config, OpenAiBackend},
    };
    use ratatui::{Terminal, backend::TestBackend, layout::Rect};

    use super::Approval;
    use crate::component::{Component, EventResult};
    use crate::compositor::Compositor;
    use crate::event::Event;
    use crate::session::{Action, Session};
    use crate::tools::read;
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

    #[test]
    fn control_v_requests_the_transcript_without_resolving_the_approval() {
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Approval::new(
            read::NAME,
            "{}",
            |_, session| session.pending_action = Some(Action::ResolveToolApproval(true)),
            |_, session| session.pending_action = Some(Action::ResolveToolApproval(false)),
        )));
        let mut session = Session::new(Agent::new(OpenAiBackend::new(Config {
            model: String::new(),
            base_url: String::new(),
            api_key: None,
            reasoning_effort: None,
            extra_body: None,
            ..kyun_agent::openai::Config::default()
        })));

        compositor.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL)),
            Default::default(),
            &mut session,
        );
        assert_eq!(session.pending_action, Some(Action::ViewTranscript));
        assert!(compositor.find::<Approval>().is_some());

        session.pending_action = None;
        compositor.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE)),
            Default::default(),
            &mut session,
        );
        assert_eq!(
            session.pending_action,
            Some(Action::ResolveToolApproval(true))
        );
        assert!(compositor.find::<Approval>().is_none());
    }

    #[test]
    fn transcript_motion_keys_pass_through() {
        let mut session = session();
        let mut approval = Approval::new(read::NAME, "{}", |_, _| {}, |_, _| {});

        for code in [
            KeyCode::Down,
            KeyCode::Up,
            KeyCode::PageDown,
            KeyCode::PageUp,
            KeyCode::Home,
            KeyCode::End,
            KeyCode::Char('j'),
            KeyCode::Char('k'),
            KeyCode::Char(' '),
            KeyCode::Char('g'),
            KeyCode::Char('G'),
        ] {
            let result = approval.handle_event(
                &Event::Key(KeyEvent::new(code, KeyModifiers::NONE)),
                Default::default(),
                &mut session,
            );
            assert!(matches!(result, EventResult::Ignored(None)));
        }
        // The transcript scrolls by a page for these two only with control held.
        for code in [KeyCode::Char('d'), KeyCode::Char('u')] {
            let result = approval.handle_event(
                &Event::Key(KeyEvent::new(code, KeyModifiers::CONTROL)),
                Default::default(),
                &mut session,
            );
            assert!(matches!(result, EventResult::Ignored(None)));
        }

        // Esc is not a motion key either way: keeping it swallowed prevents an accidental interrupt.
        let result = approval.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            Default::default(),
            &mut session,
        );
        assert!(matches!(result, EventResult::Consumed(None)));
        assert!(session.pending_action.is_none());
    }

    #[test]
    fn a_scroll_key_reaches_the_transcript_beneath_an_approval() {
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(Approval::new(
            read::NAME,
            "{}",
            |_, _| {},
            |_, _| {},
        )));
        let mut session = session();
        // The message overflows the viewport, so a one-row scroll must change what is rendered.
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::User(
                (0..8)
                    .map(|line| format!("line {line}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )],
        );

        let area = Rect::new(0, 0, 80, 12);
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        compositor.handle_event(&Event::Redraw, area, &mut session);
        terminal
            .draw(|frame| compositor.render(frame, frame.area(), &mut session))
            .unwrap();
        let before = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(compositor.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)),
            area,
            &mut session,
        ));

        terminal
            .draw(|frame| compositor.render(frame, frame.area(), &mut session))
            .unwrap();
        let after = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(compositor.find::<Approval>().is_some());
        assert_ne!(before, after);
    }
}
