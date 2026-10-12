use std::time::Duration;

use ratatui::prelude::*;
use ratatui::widgets::Block;

use crate::component::{Component, EventResult};
use crate::event::Event;
use crate::session::{ApprovalState, Session, UIMode};
use crate::widgets::{PENDING, STATUS_ACTIVE_BG, STATUS_BG, STATUS_PENDING_BG, base_areas};

const SPINNER_FRAMES: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
pub const SPINNER_INTERVAL: Duration = Duration::from_millis(100);

fn format_token_count(tokens: u64) -> String {
    const THOUSAND: u64 = 1_000;
    const MILLION: u64 = 1_000_000;

    if tokens >= MILLION {
        let whole = tokens / MILLION;
        let tenth = tokens % MILLION / 100_000;
        if tenth == 0 {
            format!("{whole}M")
        } else {
            format!("{whole}.{tenth}M")
        }
    } else if tokens >= THOUSAND {
        format!("{}k", tokens / THOUSAND)
    } else {
        tokens.to_string()
    }
}

pub struct StatusLine {
    layout: Layout,
    context_limit: usize,
    current_row: usize,
    bottom_row: usize,
    spinner_frame: usize,
}

impl StatusLine {
    pub fn new(context_limit: usize) -> Self {
        let layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(10),
                Constraint::Min(0),
                Constraint::Length(5),
            ]);
        Self {
            layout,
            context_limit,
            current_row: 0,
            bottom_row: 0,
            spinner_frame: 0,
        }
    }

    pub fn set_scroll(&mut self, current_row: usize, bottom_row: usize) {
        self.current_row = current_row;
        self.bottom_row = bottom_row;
    }

    pub fn advance_spinner(&mut self) {
        self.spinner_frame = (self.spinner_frame + 1) % SPINNER_FRAMES.len();
    }
}

impl Component for StatusLine {
    fn render(&mut self, frame: &mut Frame, area: Rect, cx: &mut Session) {
        let area = base_areas(area).status_line;
        let layout = self.layout.split(area);
        let background = if cx.approval == ApprovalState::WaitingForIdle {
            STATUS_PENDING_BG
        } else if cx.is_busy() {
            STATUS_ACTIVE_BG
        } else {
            STATUS_BG
        };
        frame.render_widget(Block::new().style(Style::new().bg(background)), area);

        let label = match cx.mode {
            UIMode::Command => "COMMAND",
            UIMode::Insert => "INSERT",
            UIMode::Normal => "NORMAL",
        };
        frame.render_widget(
            Line::from(vec![
                Span::raw(" "),
                Span::raw(label),
                Span::raw(" "),
                Span::raw(if cx.is_busy() {
                    SPINNER_FRAMES[self.spinner_frame]
                } else {
                    " "
                }),
            ]),
            layout[0],
        );

        // Assuming that if it's 0 (default value), it means the user didn't configure it
        if self.context_limit != 0 {
            let usage = format!(
                "{}/{} ",
                format_token_count(cx.agent.usage().total_tokens),
                format_token_count(self.context_limit as u64),
            );
            frame.render_widget(Line::from(usage.as_str()).right_aligned(), layout[1]);
            if cx.approval == ApprovalState::WaitingForIdle {
                frame.render_widget(
                    Line::styled("Approval waiting", Style::new().fg(PENDING)),
                    Rect {
                        width: layout[1].width.saturating_sub(usage.len() as u16),
                        ..layout[1]
                    },
                );
            }
        } else if cx.approval == ApprovalState::WaitingForIdle {
            frame.render_widget(
                Line::styled("Approval waiting", Style::new().fg(PENDING)),
                layout[1],
            );
        }
        frame.render_widget(
            Line::from(vec![
                Span::raw(if self.bottom_row == 0 {
                    "All".into()
                } else if self.current_row == 0 {
                    "Top".into()
                } else if self.current_row == self.bottom_row {
                    "Bot".into()
                } else {
                    format!(
                        "{}%",
                        (self.current_row.saturating_mul(100) / self.bottom_row).clamp(1, 99)
                    )
                }),
                Span::raw(" "),
            ])
            .right_aligned(),
            layout[2],
        );
    }

    fn handle_event(&mut self, _event: &Event, _area: Rect, _cx: &mut Session) -> EventResult {
        EventResult::Ignored(None)
    }
}

#[cfg(test)]
mod tests {
    use super::format_token_count;

    #[test]
    fn formats_token_counts_compactly() {
        assert_eq!(format_token_count(999), "999");
        assert_eq!(format_token_count(185_000), "185k");
        assert_eq!(format_token_count(1_000_000), "1M");
        assert_eq!(format_token_count(1_850_000), "1.8M");
    }
}
