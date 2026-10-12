use std::io::Write;
use std::process::Command;

use anyhow::Context as _;
use ratatui::prelude::*;
use ropey::{Rope, RopeSlice};

use crate::component::{Component, EventResult};
use crate::event::Event;
use crate::session::{Action, Session, UIMode};
use crate::widgets::{
    MUTED, SOFT_WRAP_INDICATOR, TextFormatter, base_areas, display_width, expand_tabs,
    next_word_boundary, previous_word_boundary, real_lines, soft_wrap_prefix_width,
};
// TODO: Maybe enum? Maybe there can be different types of prompts that will render and process differently?
pub struct Prompt {
    content: Rope,
    /// Cursor as `(row, col)`.
    cursor: (usize, usize),
    scroll: usize,
}

struct VisualLine<'a> {
    text: &'a str,
    is_continuation: bool,
}

impl Default for Prompt {
    fn default() -> Self {
        Self {
            content: Rope::new(),
            cursor: (0, 0),
            scroll: 0,
        }
    }
}

impl Prompt {
    fn line(&self, row: usize) -> RopeSlice<'_> {
        let line = self.content.line(row);
        if line.len_chars() > 0 && line.char(line.len_chars() - 1) == '\n' {
            line.slice(..line.len_chars() - 1)
        } else {
            line
        }
    }

    fn line_char(&self, row: usize, byte: usize) -> usize {
        self.content.line_to_char(row) + self.line(row).byte_to_char(byte)
    }

    fn remove_bytes(&mut self, row: usize, range: std::ops::Range<usize>) {
        let start = self.line_char(row, range.start);
        let end = self.line_char(row, range.end);
        self.content.remove(start..end);
    }

    fn insert_text(&mut self, row: usize, byte: usize, text: &str) {
        self.content.insert(self.line_char(row, byte), text);
    }

    fn join_lines(&mut self, row: usize) {
        let newline = self.content.line_to_char(row + 1) - 1;
        self.content.remove(newline..newline + 1);
    }

    pub fn is_empty(&self) -> bool {
        self.content.len_bytes() == 0
    }

    #[cfg(test)]
    pub fn content(&self) -> String {
        self.content.to_string()
    }

    pub fn replace_content(&mut self, content: &str) {
        let lines = real_lines(content).collect::<Vec<_>>();
        self.content = Rope::from(lines.join("\n"));
        let last_row = lines.len() - 1;
        self.cursor = (last_row, lines[last_row].len());
        self.scroll = 0;
    }

    pub fn edit(&mut self, editor: &str) -> anyhow::Result<()> {
        let mut file = tempfile::Builder::new()
            .prefix("kyun-prompt-")
            .suffix(".md")
            .tempfile()
            .unwrap();
        for chunk in self.content.chunks() {
            file.write_all(chunk.as_bytes()).unwrap();
        }
        file.flush().unwrap();

        let status = Command::new(editor)
            .arg(file.path())
            .status()
            .with_context(|| format!("could not start editor {editor:?}"))?;
        if !status.success() {
            return Ok(());
        }

        let content = std::fs::read_to_string(file.path()).unwrap();
        self.replace_content(&content);
        Ok(())
    }
}

impl Component for Prompt {
    fn render(&mut self, frame: &mut Frame, area: Rect, cx: &mut Session) {
        const PADDING: usize = 2;
        const PROMPT_GLYPH: Style = Style::new().fg(Color::Green).add_modifier(Modifier::BOLD);

        let prompt_area = base_areas(area).prompt;

        let height = prompt_area.height as usize;
        let max_prompt_w = prompt_area.width as usize;
        let max_content_w = max_prompt_w.saturating_sub(PADDING * 2); // +4 total padding on left-right
        if height == 0 || max_prompt_w == 0 {
            return;
        }

        let lines = (0..self.content.len_lines())
            .map(|row| self.line(row).to_string())
            .collect::<Vec<_>>();
        let (cursor_line_idx, cursor_byte_idx) = self.cursor;
        let mut visual_lines = Vec::new();
        // The column a continuation line's text starts at, matching the prefix width the wrapping TextFormatter
        // reserves for the soft-wrap indicator.
        let continuation_prefix = soft_wrap_prefix_width(max_content_w);

        // The visual line (i.e. hard or soft) the cursor is currently on.
        let mut cursor_visual_line = None;

        // The column the cursor is currently on; initialized to the start of the visual line (PADDING).
        let mut cursor_column = PADDING;

        for (real_line_idx, full_line) in lines.iter().enumerate() {
            let mut visual_line_byte_start = 0;
            for (text, is_continuation) in TextFormatter::new(full_line, max_content_w, 0) {
                let visual_line_byte_end = visual_line_byte_start + text.len();

                // Check if the current row to be rendered also contains the cursor, which is to say
                // that it is the first line for which cursor_byte_idx does not exceed its span.
                if real_line_idx == cursor_line_idx
                    && cursor_visual_line.is_none()
                    && (max_content_w == 0
                        || cursor_byte_idx < visual_line_byte_end
                        || visual_line_byte_end == full_line.len())
                {
                    cursor_visual_line = Some(visual_lines.len());
                    if max_content_w > 0 {
                        cursor_column = display_width(
                            &full_line[visual_line_byte_start..cursor_byte_idx],
                            if is_continuation {
                                continuation_prefix
                            } else {
                                0
                            },
                        ) + if is_continuation {
                            continuation_prefix
                        } else {
                            PADDING
                        };
                    }
                }
                visual_line_byte_start = visual_line_byte_end;
                visual_lines.push(VisualLine {
                    text,
                    is_continuation,
                });
            }
        }

        let cursor_visual_line = cursor_visual_line.unwrap();

        // If the content updates, then self.scroll may get outdated (e.g. if line is deleted) and hence
        // need to update.
        if cursor_visual_line < self.scroll {
            self.scroll = cursor_visual_line;
        } else if cursor_visual_line >= self.scroll + height {
            self.scroll = cursor_visual_line + 1 - height;
        }
        self.scroll = self.scroll.min(visual_lines.len().saturating_sub(height));

        // Render the prompt text.
        for ((visual_line_idx, visual_line), row_area) in visual_lines
            .iter()
            .enumerate()
            .skip(self.scroll)
            .zip(prompt_area.rows())
        {
            let mut spans: Vec<Span> = if visual_line_idx == 0 {
                vec![Span::styled(">", PROMPT_GLYPH), Span::raw(" ")]
            } else if visual_line.is_continuation && continuation_prefix > 0 {
                vec![Span::styled(SOFT_WRAP_INDICATOR, Style::new().fg(MUTED))]
            } else {
                vec![Span::raw(" ".repeat(PADDING))]
            };
            spans.push(Span::raw(expand_tabs(
                visual_line.text,
                if visual_line.is_continuation {
                    continuation_prefix
                } else {
                    0
                },
            )));
            frame.render_widget(Line::from(spans), row_area);
        }

        if cx.mode == UIMode::Insert {
            let cursor_x = prompt_area.x + cursor_column.min(max_prompt_w - 1) as u16;
            let cursor_y = prompt_area.y + (cursor_visual_line - self.scroll) as u16;
            frame.set_cursor_position((cursor_x, cursor_y));
        }
    }

    fn handle_event(&mut self, event: &Event, _area: Rect, cx: &mut Session) -> EventResult {
        use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

        if let Event::Paste(content) = event {
            if cx.mode != UIMode::Insert {
                return EventResult::Ignored(None);
            }

            let (row, col) = self.cursor;
            let pasted_lines = real_lines(content).collect::<Vec<_>>();
            self.insert_text(row, col, &pasted_lines.join("\n"));

            // Jump cursor to the end of the pasted block.
            let last_row = row + pasted_lines.len() - 1;
            let last_col = if pasted_lines.len() == 1 {
                col + pasted_lines[0].len()
            } else {
                pasted_lines.last().unwrap().len()
            };
            self.cursor = (last_row, last_col);

            return EventResult::Consumed(None);
        }

        let Event::Key(key) = event else {
            return EventResult::Ignored(None);
        };

        if key.kind != KeyEventKind::Press {
            return EventResult::Ignored(None);
        }

        if cx.mode != UIMode::Insert {
            if cx.mode != UIMode::Normal {
                return EventResult::Ignored(None);
            }
            match key.code {
                KeyCode::Char('i') => {
                    cx.mode = UIMode::Insert;
                    return EventResult::Consumed(None);
                }
                KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    cx.pending_action = Some(Action::ViewTranscript);
                    return EventResult::Consumed(None);
                }
                KeyCode::Char('q') => {
                    cx.should_exit = true;
                    return EventResult::Consumed(None);
                }
                _ => return EventResult::Ignored(None),
            }
        }

        let (cur_y, cur_x) = self.cursor;

        // TODO: replace char len with grapheme iteration
        match key.code {
            KeyCode::Esc => {
                // TODO: Add Vim-like keybinds for e.g. undo
                cx.mode = UIMode::Normal;
            }
            KeyCode::Enter => {
                if !cx.is_busy() && !self.is_empty() {
                    cx.agent.start_turn(self.content.to_string());
                    cx.pending_action = Some(Action::StepAgent);
                    self.content = Rope::new();
                    self.cursor = (0, 0);
                    self.scroll = 0;
                    cx.mode = UIMode::Normal;
                }
            }
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert_text(cur_y, cur_x, "\n");
                self.cursor = (cur_y + 1, 0);
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.content = Rope::new();
                self.cursor = (0, 0);
                self.scroll = 0;
            }
            KeyCode::Char('g') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                cx.pending_action = Some(Action::EditPrompt);
            }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor = (cur_y, 0);
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor = (cur_y, self.line(cur_y).len_bytes());
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let line = self.line(cur_y);
                let idx = line.byte_to_char(cur_x);
                if idx > 0 {
                    self.cursor = (cur_y, line.char_to_byte(idx - 1));
                } else if cur_y > 0 {
                    self.cursor = (cur_y - 1, self.line(cur_y - 1).len_bytes());
                }
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let line = self.line(cur_y);
                let idx = line.byte_to_char(cur_x);
                if idx < line.len_chars() {
                    self.cursor = (cur_y, line.char_to_byte(idx + 1));
                } else if cur_y + 1 < self.content.len_lines() {
                    self.cursor = (cur_y + 1, 0);
                }
            }
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if cur_y > 0 {
                    let col = self.line(cur_y).byte_to_char(cur_x);
                    let prev = self.line(cur_y - 1);
                    self.cursor = (cur_y - 1, prev.char_to_byte(col.min(prev.len_chars())));
                }
            }
            KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if cur_y + 1 < self.content.len_lines() {
                    let col = self.line(cur_y).byte_to_char(cur_x);
                    let next = self.line(cur_y + 1);
                    self.cursor = (cur_y + 1, next.char_to_byte(col.min(next.len_chars())));
                }
            }
            KeyCode::Char('h') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let line = self.line(cur_y);
                let idx = line.byte_to_char(cur_x);
                if idx > 0 {
                    let start = line.char_to_byte(idx - 1);
                    self.remove_bytes(cur_y, start..cur_x);
                    self.cursor = (cur_y, start);
                } else if cur_y > 0 {
                    let prev_len = self.line(cur_y - 1).len_bytes();
                    self.join_lines(cur_y - 1);
                    self.cursor = (cur_y - 1, prev_len);
                }
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let line = self.line(cur_y);
                let idx = line.byte_to_char(cur_x);
                if idx < line.len_chars() {
                    let end = line.char_to_byte(idx + 1);
                    self.remove_bytes(cur_y, cur_x..end);
                } else if cur_y + 1 < self.content.len_lines() {
                    self.join_lines(cur_y);
                }
            }
            KeyCode::Char('w') | KeyCode::Backspace
                if key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                if cur_x > 0 {
                    let line = self.line(cur_y).to_string();
                    let boundary = previous_word_boundary(&line, cur_x);
                    self.remove_bytes(cur_y, boundary..cur_x);
                    self.cursor = (cur_y, boundary);
                } else if cur_y > 0 {
                    let prev_len = self.line(cur_y - 1).len_bytes();
                    self.join_lines(cur_y - 1);
                    self.cursor = (cur_y - 1, prev_len);
                }
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.remove_bytes(cur_y, 0..cur_x);
                self.cursor = (cur_y, 0);
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let end = self.line(cur_y).len_bytes();
                self.remove_bytes(cur_y, cur_x..end);
            }
            KeyCode::Left
                if key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                if cur_x > 0 {
                    let line = self.line(cur_y).to_string();
                    self.cursor = (cur_y, previous_word_boundary(&line, cur_x));
                } else if cur_y > 0 {
                    let line = self.line(cur_y - 1).to_string();
                    self.cursor = (cur_y - 1, previous_word_boundary(&line, line.len()));
                }
            }
            KeyCode::Right
                if key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                if cur_x < self.line(cur_y).len_bytes() {
                    let line = self.line(cur_y).to_string();
                    self.cursor = (cur_y, next_word_boundary(&line, cur_x));
                } else if cur_y + 1 < self.content.len_lines() {
                    let next = self.line(cur_y + 1).to_string();
                    let byte = next
                        .char_indices()
                        .find(|(_, ch)| ch.is_alphanumeric() || *ch == '_')
                        .map(|(idx, _)| idx)
                        .unwrap_or(next.len());
                    self.cursor = (cur_y + 1, byte);
                }
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::ALT) => {
                if cur_x > 0 {
                    let line = self.line(cur_y).to_string();
                    self.cursor = (cur_y, previous_word_boundary(&line, cur_x));
                } else if cur_y > 0 {
                    let line = self.line(cur_y - 1).to_string();
                    self.cursor = (cur_y - 1, previous_word_boundary(&line, line.len()));
                }
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::ALT) => {
                if cur_x < self.line(cur_y).len_bytes() {
                    let line = self.line(cur_y).to_string();
                    self.cursor = (cur_y, next_word_boundary(&line, cur_x));
                } else if cur_y + 1 < self.content.len_lines() {
                    let next = self.line(cur_y + 1).to_string();
                    let byte = next
                        .char_indices()
                        .find(|(_, ch)| ch.is_alphanumeric() || *ch == '_')
                        .map(|(idx, _)| idx)
                        .unwrap_or(next.len());
                    self.cursor = (cur_y + 1, byte);
                }
            }
            KeyCode::Delete if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if cur_x < self.line(cur_y).len_bytes() {
                    let line = self.line(cur_y).to_string();
                    let boundary = next_word_boundary(&line, cur_x);
                    self.remove_bytes(cur_y, cur_x..boundary);
                } else if cur_y + 1 < self.content.len_lines() {
                    self.join_lines(cur_y);
                }
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert_text(cur_y, cur_x, &c.to_string());
                self.cursor = (cur_y, cur_x + c.len_utf8());
            }
            KeyCode::Backspace => {
                let line = self.line(cur_y);
                let idx = line.byte_to_char(cur_x);
                if idx > 0 {
                    let start = line.char_to_byte(idx - 1);
                    self.remove_bytes(cur_y, start..cur_x);
                    self.cursor = (cur_y, start);
                } else if cur_y > 0 {
                    let prev_len = self.line(cur_y - 1).len_bytes();
                    self.join_lines(cur_y - 1);
                    self.cursor = (cur_y - 1, prev_len);
                }
            }
            KeyCode::Delete => {
                let line = self.line(cur_y);
                let idx = line.byte_to_char(cur_x);
                if idx < line.len_chars() {
                    let end = line.char_to_byte(idx + 1);
                    self.remove_bytes(cur_y, cur_x..end);
                } else if cur_y + 1 < self.content.len_lines() {
                    self.join_lines(cur_y);
                }
            }
            KeyCode::Left => {
                let line = self.line(cur_y);
                let idx = line.byte_to_char(cur_x);
                if idx > 0 {
                    self.cursor = (cur_y, line.char_to_byte(idx - 1));
                } else if cur_y > 0 {
                    self.cursor = (cur_y - 1, self.line(cur_y - 1).len_bytes());
                }
            }
            KeyCode::Right => {
                let line = self.line(cur_y);
                let idx = line.byte_to_char(cur_x);
                if idx < line.len_chars() {
                    self.cursor = (cur_y, line.char_to_byte(idx + 1));
                } else if cur_y + 1 < self.content.len_lines() {
                    self.cursor = (cur_y + 1, 0);
                }
            }
            KeyCode::Up => {
                if cur_y > 0 {
                    let col = self.line(cur_y).byte_to_char(cur_x);
                    let prev = self.line(cur_y - 1);
                    self.cursor = (cur_y - 1, prev.char_to_byte(col.min(prev.len_chars())));
                }
            }
            KeyCode::Down => {
                if cur_y + 1 < self.content.len_lines() {
                    let col = self.line(cur_y).byte_to_char(cur_x);
                    let next = self.line(cur_y + 1);
                    self.cursor = (cur_y + 1, next.char_to_byte(col.min(next.len_chars())));
                }
            }
            KeyCode::Home => self.cursor = (cur_y, 0),
            KeyCode::End => self.cursor = (cur_y, self.line(cur_y).len_bytes()),
            _ => return EventResult::Ignored(None),
        }
        EventResult::Consumed(None)
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use kyun_agent::{
        Agent, AssistantMessage, ToolCall, ToolCallKind, ToolResult,
        openai::{Config, OpenAiBackend},
    };
    use ratatui::{
        Terminal,
        backend::TestBackend,
        layout::{Position, Rect},
    };

    use super::{Prompt, Rope};
    use crate::{
        component::Component,
        event::Event,
        session::{Action, PendingToolCall, Session, UIMode},
        test_support::visible_messages,
        tools::read,
    };

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
    fn control_enter_does_not_submit_while_tool_calls_are_pending() {
        let mut prompt = Prompt::default();
        prompt.replace_content("follow-up");
        let mut session = session();
        session.mode = UIMode::Insert;
        session.agent.start_turn("first prompt".into());
        session.agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: None,
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                kind: ToolCallKind::Function,
                function_name: read::NAME.into(),
                arguments: "{}".into(),
            }],
        });
        session.pending_tool_calls.push_back(PendingToolCall {
            future: Box::pin(std::future::pending::<ToolResult>()),
            started: false,
        });
        let messages_before = visible_messages(&session.agent);

        prompt.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
            Rect::default(),
            &mut session,
        );

        assert_eq!(visible_messages(&session.agent), messages_before);
        assert_eq!(session.pending_action, None);
        assert_eq!(prompt.content(), "follow-up");
    }

    #[test]
    fn word_navigation_and_deletion_handle_unicode_and_separators() {
        let mut prompt = Prompt {
            content: Rope::from("one, two κόσμε"),
            cursor: (0, "one, two κόσμε".len()),
            scroll: 0,
        };
        let mut session = session();
        session.mode = UIMode::Insert;

        let _ = prompt.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::CONTROL)),
            Rect::default(),
            &mut session,
        );
        assert_eq!(&prompt.content()[prompt.cursor.1..], "κόσμε");

        let _ = prompt.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL)),
            Rect::default(),
            &mut session,
        );
        assert_eq!(prompt.content(), "one, κόσμε");
        assert_eq!(&prompt.content()[prompt.cursor.1..], "κόσμε");

        let _ = prompt.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Delete, KeyModifiers::CONTROL)),
            Rect::default(),
            &mut session,
        );
        assert_eq!(prompt.content(), "one, ");
        assert_eq!(prompt.cursor, (0, "one, ".len()));
    }

    #[test]
    fn bracketed_paste_inserts_multiline_text_and_control_c_resets_the_prompt() {
        let mut prompt = Prompt {
            content: Rope::from("before after\nfollowing"),
            cursor: (0, "before ".len()),
            scroll: 3,
        };
        let mut session = session();
        session.mode = UIMode::Insert;

        let _ = prompt.handle_event(
            &Event::Paste("α\r\nbeta\n".into()),
            Rect::default(),
            &mut session,
        );
        assert_eq!(prompt.content(), "before α\nbeta\nafter\nfollowing");
        assert_eq!(prompt.cursor, (2, 0));

        let _ = prompt.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Rect::default(),
            &mut session,
        );
        assert_eq!(prompt.content(), "");
        assert_eq!(prompt.cursor, (0, 0));
        assert_eq!(prompt.scroll, 0);
    }

    #[cfg(unix)]
    #[test]
    fn control_g_requests_the_editor_and_edited_text_replaces_the_prompt() {
        use std::os::unix::fs::PermissionsExt;

        let mut prompt = Prompt {
            content: Rope::from("old prompt"),
            cursor: (0, 3),
            scroll: 2,
        };
        let mut session = session();
        session.mode = UIMode::Insert;

        let _ = prompt.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL)),
            Rect::default(),
            &mut session,
        );
        assert_eq!(session.pending_action, Some(Action::EditPrompt));

        let directory = tempfile::tempdir().unwrap();
        let editor = directory.path().join("editor");
        std::fs::write(&editor, "#!/bin/sh\nprintf 'edited\\n第二行' > \"$1\"\n").unwrap();
        let mut permissions = std::fs::metadata(&editor).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&editor, permissions).unwrap();

        prompt.edit(editor.to_str().unwrap()).unwrap();
        assert_eq!(prompt.content(), "edited\n第二行");
        assert_eq!(prompt.cursor, (1, "第二行".len()));
        assert_eq!(prompt.scroll, 0);
    }

    #[test]
    fn an_editor_that_cannot_spawn_reports_an_error() {
        let mut prompt = Prompt::default();
        prompt.replace_content("draft");

        let error = prompt
            .edit("kyun-definitely-not-an-editor")
            .unwrap_err()
            .to_string();

        assert!(error.contains("could not start editor"), "{error}");
        // The prompt keeps its content when the editor never ran.
        assert_eq!(prompt.content(), "draft");
    }

    #[test]
    fn pasted_text_places_the_rendered_cursor_on_its_final_wrapped_row() {
        let mut session = session();
        session.mode = UIMode::Insert;

        for (content, width, position, scroll) in [
            ("", 12, (2, 0), 0),
            ("hello, world", 12, (7, 1), 0),
            ("abcdefgh", 8, (4, 2), 0),
            ("日本語", 9, (4, 1), 0),
            ("abcdefghijklmnopqrstuvwxyz012345", 12, (8, 4), 0),
            ("head\nhello, world", 12, (7, 2), 0),
            ("hello, world\n", 12, (2, 2), 0),
        ] {
            let mut prompt = Prompt::default();
            let mut terminal = Terminal::new(TestBackend::new(width, 8)).unwrap();
            let _ =
                prompt.handle_event(&Event::Paste(content.into()), Rect::default(), &mut session);
            terminal
                .draw(|frame| prompt.render(frame, frame.area(), &mut session))
                .unwrap();
            assert_eq!(
                terminal.get_cursor_position().unwrap(),
                Position::from(position),
                "{content:?}"
            );
            assert_eq!(prompt.scroll, scroll, "{content:?}");
        }
    }

    #[test]
    fn pasted_text_remains_visible_when_moving_resizing_and_clearing() {
        let mut session = session();
        session.mode = UIMode::Insert;
        let mut prompt = Prompt::default();
        let mut terminal = Terminal::new(TestBackend::new(12, 8)).unwrap();
        let _ = prompt.handle_event(
            &Event::Paste("abcdefghijklmnopqrstuvwxyz012345".into()),
            Rect::default(),
            &mut session,
        );
        terminal
            .draw(|frame| prompt.render(frame, frame.area(), &mut session))
            .unwrap();
        assert_eq!(
            terminal
                .backend()
                .buffer()
                .content
                .chunks(12)
                .take(5)
                .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
                .collect::<Vec<_>>(),
            [
                "> abcdefgh  ",
                "↪ ijklmn    ",
                "↪ opqrst    ",
                "↪ uvwxyz    ",
                "↪ 012345    "
            ],
        );

        for (key, position, scroll) in [(KeyCode::Home, (2, 0), 0), (KeyCode::End, (8, 4), 0)] {
            let _ = prompt.handle_event(
                &Event::Key(KeyEvent::new(key, KeyModifiers::NONE)),
                Rect::default(),
                &mut session,
            );
            terminal
                .draw(|frame| prompt.render(frame, frame.area(), &mut session))
                .unwrap();
            assert_eq!(
                terminal.get_cursor_position().unwrap(),
                Position::from(position)
            );
            assert_eq!(prompt.scroll, scroll);
        }

        terminal.backend_mut().resize(20, 8);
        terminal
            .draw(|frame| prompt.render(frame, frame.area(), &mut session))
            .unwrap();
        assert_eq!(terminal.get_cursor_position().unwrap(), Position::new(4, 2));
        assert_eq!(prompt.scroll, 0);

        let _ = prompt.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL)),
            Rect::default(),
            &mut session,
        );
        terminal
            .draw(|frame| prompt.render(frame, frame.area(), &mut session))
            .unwrap();
        assert_eq!(terminal.get_cursor_position().unwrap(), Position::new(2, 0));
        assert_eq!(prompt.scroll, 0);

        let _ = prompt.handle_event(
            &Event::Paste("hello, world".into()),
            Rect::default(),
            &mut session,
        );
        terminal.backend_mut().resize(12, 8);
        for _ in 0..5 {
            let _ = prompt.handle_event(
                &Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)),
                Rect::default(),
                &mut session,
            );
        }
        terminal
            .draw(|frame| prompt.render(frame, frame.area(), &mut session))
            .unwrap();
        assert_eq!(terminal.get_cursor_position().unwrap(), Position::new(2, 1));
    }
}
