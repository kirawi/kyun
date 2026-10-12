use std::collections::HashMap;
use std::ops::Range;

use kyun_agent::{Message, MessageView};
use ratatui::prelude::Style;

use crate::tools::ToolKind;
use crate::widgets::{TEXT, real_lines};

use super::highlight::{HighlightedLine, TOOL_OUTPUT_LINE_NUMBER_WIDTH};

pub const TOOL_OUTPUT_LINE_LIMIT: usize = 10;

const OUTPUT_RESERVED_WIDTH: usize = 2 + TOOL_OUTPUT_LINE_NUMBER_WIDTH + 1;

#[derive(Clone)]
pub struct OutputLayout {
    spans: Vec<Range<usize>>,
    rows: Vec<usize>,
    pub lines: Vec<Range<usize>>,
    pub width: usize,
    pub truncated: bool,
    line_limit: usize,
}

impl OutputLayout {
    fn with_line_limit(content: &str, width: usize, line_limit: usize) -> Self {
        let mut spans = Vec::new();
        let mut rows = vec![0];
        let mut line_ranges = Vec::new();
        let display_content = content
            .strip_suffix("\r\n")
            .or_else(|| content.strip_suffix('\n'))
            .or_else(|| content.strip_suffix('\r'))
            .unwrap_or(content);
        let mut lines = real_lines(display_content);
        let mut start = 0;
        for line in lines.by_ref().take(line_limit) {
            line_ranges.push(start..start + line.len());
            spans.extend(
                HighlightedLine::plain(line, Style::new().fg(TEXT))
                    .wrap(start, width.saturating_sub(OUTPUT_RESERVED_WIDTH)),
            );
            rows.push(spans.len());
            start += line.len();
            if content.as_bytes().get(start) == Some(&b'\r') {
                start += 1;
            }
            if content.as_bytes().get(start) == Some(&b'\n') {
                start += 1;
            }
        }
        Self {
            spans,
            rows,
            lines: line_ranges,
            width,
            truncated: lines.next().is_some(),
            line_limit,
        }
    }

    pub fn content_rows(&self) -> usize {
        self.spans.len()
    }

    pub fn last_row(&self) -> usize {
        // Row zero is the top border. The content and optional truncation marker follow it, and the final
        // row is the bottom border.
        self.content_rows() + usize::from(self.truncated) + 1
    }

    pub fn row_position(&self, row: usize) -> Option<(usize, usize)> {
        if row >= self.content_rows() {
            return None;
        }
        let line = self.rows.partition_point(|&start| start <= row) - 1;
        Some((line, row - self.rows[line]))
    }

    pub fn line_row(&self, line: usize, segment: usize) -> usize {
        self.rows[line] + segment
    }

    pub fn segment_count(&self, line: usize) -> usize {
        self.rows[line + 1] - self.rows[line]
    }

    pub fn segment_range(&self, row: usize, line: usize) -> Range<usize> {
        let start = self.lines[line].start;
        self.spans[row].start - start..self.spans[row].end - start
    }
}

#[derive(Default)]
pub struct OutputCache {
    layouts: HashMap<(usize, usize), OutputLayout>,
}

impl OutputCache {
    pub fn truncate(&mut self, message_count: usize) {
        self.layouts
            .retain(|&(message_idx, _), _| message_idx < message_count);
    }

    pub fn layout(
        &mut self,
        messages: MessageView<'_>,
        message_idx: usize,
        call_idx: usize,
        width: usize,
    ) -> Option<&OutputLayout> {
        let (call, result) = messages.get_tool_call(message_idx, call_idx);
        let result = result?;
        if !matches!(
            ToolKind::from_function_name(&call.function_name),
            ToolKind::Shell
        ) || result.content.is_empty()
            || result.content == crate::TOOL_CALL_DENIED_MESSAGE
            || result.content == crate::TOOL_CALL_INTERRUPTED_MESSAGE
        {
            return None;
        }
        let key = (
            if matches!(messages[message_idx], Message::Tool(_)) {
                message_idx
            } else {
                message_idx + 1
            },
            call_idx,
        );
        let line_limit = super::tool_limits().output_line_limit;
        let layout = self
            .layouts
            .entry(key)
            .or_insert_with(|| OutputLayout::with_line_limit(&result.content, width, line_limit));
        if layout.width != width || layout.line_limit != line_limit {
            *layout = OutputLayout::with_line_limit(&result.content, width, line_limit);
        }
        Some(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::{OutputLayout, TOOL_OUTPUT_LINE_LIMIT};

    #[test]
    fn output_limit_counts_source_lines_after_wrapping() {
        let content = (0..TOOL_OUTPUT_LINE_LIMIT)
            .map(|line| format!("line {line} long words\n"))
            .collect::<String>();
        let layout = OutputLayout::with_line_limit(&content, 14, TOOL_OUTPUT_LINE_LIMIT);
        assert!(!layout.truncated);
        assert!(layout.content_rows() > TOOL_OUTPUT_LINE_LIMIT);
        assert_eq!(
            layout.row_position(layout.content_rows() - 1).unwrap().0,
            TOOL_OUTPUT_LINE_LIMIT - 1
        );

        let extended = format!("{content}one more\n");
        let layout = OutputLayout::with_line_limit(&extended, 14, TOOL_OUTPUT_LINE_LIMIT);
        assert!(layout.truncated);
        assert_eq!(layout.last_row(), layout.content_rows() + 2);
    }

    #[test]
    fn output_segments_preserve_text_and_source_line_positions() {
        let content = "aaaa bbbb cccc\r\nnext\n";
        let layout = OutputLayout::with_line_limit(content, 14, TOOL_OUTPUT_LINE_LIMIT);
        let first_line = (0..layout.content_rows())
            .filter_map(|row| {
                (layout.row_position(row).unwrap().0 == 0)
                    .then_some(&content[layout.spans[row].clone()])
            })
            .collect::<String>();
        assert_eq!(first_line, "aaaa bbbb cccc");
        let row = layout.content_rows() - 1;
        assert_eq!(
            (
                &content[layout.spans[row].clone()],
                layout.row_position(row).unwrap()
            ),
            ("next", (1, 0))
        );
    }
}
