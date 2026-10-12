use std::num::NonZeroU32;
use std::ops::{Index, Range};

use kyun_agent::{Message, MessageView, ToolCall, ToolResult};
use ratatui::prelude::Style;

use crate::tools::{FileMutationOutput, ToolKind};
use crate::widgets::{DIFF_ADD, DIFF_TEXT, JUMP, TEXT};

use super::highlight::{HighlightedLine, TOOL_OUTPUT_LINE_NUMBER_WIDTH};

pub use crate::tools::write::DiffLineKind;
use crate::tools::write::FileDiff;

// TODO: make this configurable at runtime as an Option where None = unlimited
pub const TOOL_DIFF_LINE_LIMIT: usize = 100;

/// Width reserved by the diff box: 1-cell left and right borders plus the two 5-cell line-number gutters.
const TOOL_DIFF_RESERVED_WIDTH: usize = 2 + (TOOL_OUTPUT_LINE_NUMBER_WIDTH + 1) * 2;

/// The width available to soft-wrapped diff line text for a given transcript content width (the walker's
/// `width`).
pub fn diff_text_width(width: usize) -> usize {
    width.saturating_sub(TOOL_DIFF_RESERVED_WIDTH)
}

pub struct DiffDisplayLine<'a> {
    pub content: &'a str,
    pub old_line: Option<NonZeroU32>,
    pub new_line: Option<NonZeroU32>,
    pub kind: DiffLineKind,
    pub additions: u32,
    pub removals: u32,
}

pub fn is_file_mutation(call: &ToolCall) -> bool {
    matches!(
        ToolKind::from_function_name(&call.function_name),
        ToolKind::FileMutation(_)
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct DiffFileTotals {
    line: u32,
    additions: u32,
    removals: u32,
}

/// Flat soft-wrap ranges with cumulative row offsets for each source line. Indexing by a source line returns its
/// contiguous range of soft-wrap segments, preserving the old `spans[line]` access pattern without per-line Vecs.
pub struct DiffSpans {
    ranges: Vec<Range<usize>>,
    rows: Box<[usize]>,
}

impl Default for DiffSpans {
    fn default() -> Self {
        Self {
            ranges: Vec::new(),
            rows: Box::new([0]),
        }
    }
}

impl Index<usize> for DiffSpans {
    type Output = [Range<usize>];

    fn index(&self, line: usize) -> &Self::Output {
        &self.ranges[self.rows[line]..self.rows[line + 1]]
    }
}

pub struct DiffLayout {
    source: FileDiff,
    file_totals: Box<[DiffFileTotals]>,
    /// Soft-wrapped segments as flat byte ranges into the source text. File lines are never wrapped, so they hold
    /// exactly one span; every other line is segmented by `soft_spans`. Indexing by source line uses the cumulative
    /// row offsets stored alongside the ranges.
    pub spans: DiffSpans,
    /// The transcript content width this layout was computed for.
    pub width: usize,
    pub files: usize,
    pub additions: usize,
    pub removals: usize,
    line_limit: usize,
}

/// Diff layouts in the order their file-mutation calls arrived. Once displayed, a diff remains cached when the
/// mutation is denied or fails so it can still be rendered.
#[derive(Default)]
pub struct DiffCache {
    entries: Vec<Option<DiffLayout>>,
}

impl DiffCache {
    /// Records a file-mutation call, with `None` for a mutation that could not produce a text diff. Every
    /// mutation call occupies a slot so repeated calls with identical arguments remain distinct.
    pub fn insert(&mut self, diff: Option<FileDiff>, width: usize) {
        let line_limit = super::tool_limits().diff_line_limit;
        self.entries.push(
            diff.filter(|diff| !diff.text.is_empty())
                .map(|diff| DiffLayout::with_line_limit(diff, width, line_limit)),
        );
    }

    /// Removes previews belonging to tool calls discarded from the end of the transcript.
    pub fn truncate(&mut self, len: usize) {
        self.entries.truncate(len);
    }

    fn entry_index(
        messages: MessageView<'_>,
        message_idx: usize,
        call_idx: usize,
    ) -> Option<usize> {
        let assistant_idx = match messages[message_idx] {
            Message::Assistant(_) => message_idx,
            Message::Tool(_) => message_idx - 1,
            Message::System(_) | Message::User(_) => return None,
        };
        let mut entry_idx = 0;
        for (message_index, message) in messages.iter().take(assistant_idx + 1).enumerate() {
            let Message::Assistant(message) = message else {
                continue;
            };
            for (index, call) in message.tool_calls.iter().enumerate() {
                if !is_file_mutation(call) {
                    continue;
                }
                if message_index == assistant_idx && index == call_idx {
                    return Some(entry_idx);
                }
                entry_idx += 1;
            }
        }
        None
    }

    pub fn source(
        &self,
        messages: MessageView<'_>,
        message_idx: usize,
        call_idx: usize,
    ) -> Option<&str> {
        let entry_idx = Self::entry_index(messages, message_idx, call_idx)?;
        Some(&self.entries.get(entry_idx)?.as_ref()?.source.text)
    }

    /// Returns the cached layout for a file-mutation tool call, reflowing it when the transcript width changes.
    pub fn layout(
        &mut self,
        messages: MessageView<'_>,
        message_idx: usize,
        call_idx: usize,
        width: usize,
    ) -> Option<&DiffLayout> {
        let entry_idx = Self::entry_index(messages, message_idx, call_idx)?;
        let diff = self.entries.get_mut(entry_idx)?.as_mut()?;
        let line_limit = super::tool_limits().diff_line_limit;
        if diff.width != width || diff.line_limit != line_limit {
            diff.set_layout(width, line_limit);
        }
        Some(diff)
    }
}

impl DiffLayout {
    fn with_line_limit(source: FileDiff, width: usize, line_limit: usize) -> Self {
        let mut files = 0;
        let mut additions = 0;
        let mut removals = 0;
        for line in &source.lines {
            match line.kind {
                DiffLineKind::File => files += 1,
                DiffLineKind::Addition => additions += 1,
                DiffLineKind::Removal => removals += 1,
                DiffLineKind::Context | DiffLineKind::Note => {}
            }
        }

        let mut layout = Self {
            source,
            file_totals: Box::new([]),
            spans: DiffSpans::default(),
            width,
            files: files.max(1),
            additions,
            removals,
            line_limit,
        };
        layout.set_layout(width, line_limit);
        layout
    }

    fn set_layout(&mut self, width: usize, line_limit: usize) {
        self.line_limit = line_limit;
        let retained_lines = self.source.lines.len().min(line_limit);
        let mut file_totals = Vec::new();
        let mut current_file: Option<usize> = None;
        for (index, line) in self.source.lines.iter().take(retained_lines).enumerate() {
            if line.kind == DiffLineKind::File {
                file_totals.push(DiffFileTotals {
                    line: u32::try_from(index).unwrap(),
                    ..DiffFileTotals::default()
                });
                current_file = Some(file_totals.len() - 1);
            }
            if let Some(file) = current_file {
                match line.kind {
                    DiffLineKind::Addition => file_totals[file].additions += 1,
                    DiffLineKind::Removal => file_totals[file].removals += 1,
                    _ => {}
                }
            }
        }
        self.file_totals = file_totals.into_boxed_slice();
        self.relayout(width);
    }

    /// Reflows only the width-dependent layout (soft-wrap spans and cumulative rows) over the retained
    /// lines, leaving the diff text, line structure, totals, and truncation status untouched.
    pub fn relayout(&mut self, width: usize) {
        let text_width = diff_text_width(width);
        let retained_lines = self.source.lines.len().min(self.line_limit);
        let mut ranges = Vec::new();
        let mut rows = Vec::with_capacity(retained_lines + 1);
        rows.push(0);

        for (index, line) in self.source.lines.iter().take(retained_lines).enumerate() {
            if line.kind == DiffLineKind::File {
                ranges.push(line.content.clone());
            } else {
                ranges.extend(
                    self.highlighted_line(index)
                        .wrap(line.content.start, text_width),
                );
            }
            rows.push(ranges.len());
        }
        self.spans = DiffSpans {
            ranges,
            rows: rows.into_boxed_slice(),
        };
        self.width = width;
    }

    pub fn line(&self, index: usize) -> Option<DiffDisplayLine<'_>> {
        // TODO: should this not be an assert to enforce that this condition must never be reached?
        // In practice, this function is always unwrapped except one test
        // Furthermore: it only has one use in production
        if index >= self.spans.rows.len() - 1 {
            return None;
        }

        let line = &self.source.lines[index];
        let totals = self
            .file_totals
            .binary_search_by_key(&u32::try_from(index).unwrap(), |totals| totals.line)
            .ok()
            .map(|index| self.file_totals[index])
            .unwrap_or_default();
        Some(DiffDisplayLine {
            content: &self.source.text[line.content.clone()],
            old_line: line.old_line,
            new_line: line.new_line,
            kind: line.kind,
            additions: totals.additions,
            removals: totals.removals,
        })
    }

    // TODO: prob should remove...
    #[cfg(test)]
    fn segment(&self, line: usize, segment: usize) -> &str {
        &self.source.text[self.spans.ranges[self.spans.rows[line] + segment].clone()]
    }

    pub fn segment_range(&self, line: usize, segment: usize) -> Range<usize> {
        let range = &self.spans.ranges[self.spans.rows[line] + segment];
        let start = self.source.lines[line].content.start;
        range.start - start..range.end - start
    }

    pub fn highlighted_line(&self, index: usize) -> HighlightedLine<'_> {
        let line = &self.source.lines[index];
        let text = &self.source.text[line.content.clone()];
        let marker = match line.kind {
            DiffLineKind::Addition => Some(DIFF_ADD),
            DiffLineKind::Removal => Some(JUMP),
            DiffLineKind::Context => Some(TEXT),
            DiffLineKind::Note => None,
            DiffLineKind::File => unreachable!(),
        };
        if let Some(marker) = marker.filter(|_| !text.is_empty()) {
            HighlightedLine {
                text,
                highlights: Box::new([
                    (Style::new().fg(marker), 0..1),
                    (Style::new().fg(DIFF_TEXT), 1..text.len()),
                ]),
                marker_space: true,
            }
        } else {
            HighlightedLine::plain(text, Style::new().fg(DIFF_TEXT))
        }
    }

    /// The number of virtual rows occupied by diff line content.
    pub fn content_rows(&self) -> usize {
        *self.spans.rows.last().unwrap()
    }

    pub fn is_truncated(&self) -> bool {
        self.source.lines.len() > self.line_limit
    }

    /// The index of the final virtual row: the bottom-border row of the diff box. When the diff is truncated,
    /// the `...` marker occupies `content_rows()` and the border row follows it.
    pub fn last_row(&self) -> usize {
        self.content_rows() + usize::from(self.is_truncated())
    }

    /// The line and soft-wrap segment rendered at a content `row`, if it is one.
    pub fn row_position(&self, row: usize) -> Option<(usize, usize)> {
        if row >= self.content_rows() {
            return None;
        }
        let line = self.spans.rows.partition_point(|&start| start <= row) - 1;
        Some((line, row - self.spans.rows[line]))
    }

    /// The flat content row at which `line` and `segment` are rendered.
    pub fn line_row(&self, line: usize, segment: usize) -> usize {
        self.spans.rows[line] + segment
    }
}

pub fn get_tool_diff(call: &ToolCall, result: Option<&ToolResult>) -> Option<String> {
    if !is_file_mutation(call) {
        return None;
    }

    let output: FileMutationOutput = serde_json::from_str(&result?.content).ok()?;
    (!output.diff.is_empty()).then(|| output.diff.as_ref().to_owned())
}

#[cfg(test)]
mod tests {
    use kyun_agent::{AssistantMessage, Message, ToolCall, ToolCallKind, ToolResult};

    use super::{DiffCache, DiffLayout, DiffLineKind, diff_text_width};
    use crate::widgets::{display_width, soft_wrap_prefix_width};
    use crate::{test_support::agent_with, tools::write};

    /// Builds a layout from a structured file diff.
    fn layout(diff: write::FileDiff, width: usize) -> DiffLayout {
        DiffLayout::with_line_limit(diff, width, super::TOOL_DIFF_LINE_LIMIT)
    }

    #[test]
    fn display_diff_collapses_file_headers_and_tracks_line_numbers() {
        let before = "one\ntwo\nthree\nfour\nsame\nold\nseven\n";
        let after = "one\ntwo\nthree\nfour\nsame\nnew\nseven\n";
        let diff = layout(
            write::file_diff(Some("src/main.rs"), Some("src/main.rs"), before, after),
            80,
        );
        let lines = (0..)
            .map_while(|index| diff.line(index))
            .collect::<Vec<_>>();

        assert_eq!(lines[0].kind, DiffLineKind::File);
        assert_eq!(lines[0].content, "src/main.rs");
        assert_eq!((lines[0].additions, lines[0].removals), (1, 1));
        // The hunk's context, removal, and addition carry their old/new line numbers.
        let context = lines
            .iter()
            .find(|line| line.kind == DiffLineKind::Context)
            .unwrap();
        assert!(context.old_line.is_some() && context.new_line.is_some());
        let removal = lines
            .iter()
            .find(|line| line.kind == DiffLineKind::Removal)
            .unwrap();
        assert_eq!(
            (
                removal.old_line.map(|line| line.get()),
                removal.new_line.map(|line| line.get())
            ),
            (Some(6), None)
        );
        let addition = lines
            .iter()
            .find(|line| line.kind == DiffLineKind::Addition)
            .unwrap();
        assert_eq!(
            (
                addition.old_line.map(|line| line.get()),
                addition.new_line.map(|line| line.get())
            ),
            (None, Some(6))
        );
        assert_eq!((diff.files, diff.additions, diff.removals), (1, 1, 1));
    }

    #[test]
    fn multi_file_stats_are_assigned_to_each_file_header() {
        // Merge two per-file diffs the way apply_patch composes them.
        let mut text = String::new();
        let mut lines = Vec::new();
        for file_diff in [
            write::file_diff(None, Some("first"), "", "new\n"),
            write::file_diff(Some("second"), Some("second"), "old\n", "new\n"),
        ] {
            let mut offset = text.len();
            if !text.is_empty() {
                text.push('\n');
                offset += 1;
            }
            text.push_str(&file_diff.text);
            for mut line in file_diff.lines {
                line.content.start += offset;
                line.content.end += offset;
                lines.push(line);
            }
        }
        let diff = layout(
            write::FileDiff {
                text: text.into(),
                lines: lines.into_boxed_slice(),
            },
            80,
        );

        assert_eq!((diff.files, diff.additions, diff.removals), (2, 2, 1));
        let first = diff.line(0).unwrap();
        assert_eq!(first.content, "first");
        assert_eq!((first.additions, first.removals), (1, 0));
        // The creation diff contributes 2 lines (File + Addition), so the second file's header is at index 2.
        let second = diff.line(2).unwrap();
        assert_eq!(second.content, "second");
        assert_eq!((second.additions, second.removals), (1, 1));
    }

    #[test]
    fn diff_lines_soft_wrap_into_virtual_rows_at_the_given_width() {
        // text width = 22 - 12 reserved = 10 cells; wrapping verified against the text formatter.
        let before = "aaaa bbbb cccc dddd eeee ffff\naaaa bbbb cccc\n";
        let after = "aaaa bbbb cccc dddd eeee ffff\naaaa bbbb cccc dddd\n";
        let diff = layout(
            write::file_diff(Some("src/main.rs"), Some("src/main.rs"), before, after),
            22,
        );

        // File header (never wrapped), context, removal, addition.
        assert_eq!(diff.content_rows(), 1 + 6 + 3 + 4);

        assert_eq!(diff.row_position(0).unwrap(), (0, 0));
        assert_eq!(diff.row_position(1).unwrap(), (1, 0));
        assert_eq!(diff.row_position(2).unwrap(), (1, 1));
        assert_eq!(diff.row_position(6).unwrap(), (1, 5));
        assert_eq!(diff.row_position(7).unwrap(), (2, 0));
        assert_eq!(diff.row_position(10).unwrap(), (3, 0));

        let segment_text = |row: usize| -> &str {
            let (line, segment) = diff.row_position(row).unwrap();
            diff.segment(line, segment)
        };
        assert_eq!(segment_text(0), "src/main.rs");
        assert_eq!(segment_text(1), " aaaa ");
        assert_eq!(segment_text(2), "bbbb ");
    }

    #[test]
    fn tabbed_body_first_segments_stay_within_the_wrap_budget() {
        // A tab after a body prefix of width 2 (mod 4) is the worst case: expanding from column 2 advances it
        // four columns, which the old wrap budget (one cell reserved, tabs counted from column 0) underestimated
        // by up to three columns (FINDINGS M-g), clipping full-width lines.
        let body = format!("ab\t{}\n", "c".repeat(30));
        let diff = layout(write::file_diff(None, Some("file.txt"), "", &body), 40);
        let text_width = diff_text_width(40);

        let mut wrapped = false;
        for row in 0..diff.content_rows() {
            let (line, segment) = diff.row_position(row).unwrap();
            let kind = diff.line(line).unwrap().kind;
            let text = diff.segment(line, segment);
            let width = if matches!(
                kind,
                DiffLineKind::Addition | DiffLineKind::Removal | DiffLineKind::Context
            ) && segment == 0
            {
                wrapped = true;
                2 + display_width(&text[1..], 2)
            } else {
                let prefix_width = soft_wrap_prefix_width(text_width);
                prefix_width + display_width(text, prefix_width)
            };
            assert!(
                width <= text_width,
                "row {row} renders {width} > {text_width}: {text:?}"
            );
        }
        // The tabbed line actually wrapped (file line plus two body segments), so the bound is not vacuous.
        assert!(wrapped && diff.content_rows() >= 3);
    }

    #[test]
    fn truncation_marker_occupies_the_row_after_wrapped_content() {
        // The file header plus the additions exceed the retained limit boundary; the lines exceed the text width
        // of 12 cells (24 - 12 reserved), so each occupies two rows.
        let after = (0..super::TOOL_DIFF_LINE_LIMIT)
            .map(|line| format!("line {line} aaaa bbbb\n"))
            .collect::<String>();
        let diff = layout(write::file_diff(None, Some("file.rs"), "", &after), 24);
        assert!(diff.is_truncated());
        // Unwrapped, the layout would hold 101 rows (the file row plus 100 lines); more rows mean the lines
        // actually occupied multiple wrapped rows.
        assert!(diff.content_rows() > 101);
        assert_eq!(diff.last_row(), diff.content_rows() + 1);
        assert!(diff.row_position(diff.content_rows()).is_none());
    }

    #[test]
    fn cached_layouts_stay_with_their_tool_calls() {
        let mut cache = DiffCache::default();
        // The first write failed during preparation. Its empty slot keeps the following two previews aligned,
        // even though all three calls have the same empty ID.
        cache.insert(None, 80);
        cache.insert(
            Some(write::file_diff(None, Some("first.txt"), "", "one\n")),
            80,
        );
        cache.insert(
            Some(write::file_diff(None, Some("second.txt"), "", "two\n")),
            80,
        );
        let messages = [
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![
                    ToolCall {
                        id: String::new(),
                        kind: ToolCallKind::Function,
                        function_name: write::NAME.into(),
                        arguments: String::new(),
                    },
                    ToolCall {
                        id: String::new(),
                        kind: ToolCallKind::Function,
                        function_name: write::NAME.into(),
                        arguments: String::new(),
                    },
                    ToolCall {
                        id: String::new(),
                        kind: ToolCallKind::Function,
                        function_name: write::NAME.into(),
                        arguments: String::new(),
                    },
                ],
            }),
            Message::Tool(vec![
                ToolResult {
                    tool_call_id: String::new(),
                    content: String::new(),
                },
                ToolResult {
                    tool_call_id: String::new(),
                    content: String::new(),
                },
                ToolResult {
                    tool_call_id: String::new(),
                    content: String::new(),
                },
            ]),
        ];
        let agent = agent_with(messages);

        assert!(cache.layout(agent.messages(), 1, 0, 80).is_none());
        assert_eq!(
            cache
                .layout(agent.messages(), 1, 1, 80)
                .unwrap()
                .line(0)
                .unwrap()
                .content,
            "first.txt"
        );
        assert_eq!(
            cache
                .layout(agent.messages(), 1, 2, 80)
                .unwrap()
                .line(0)
                .unwrap()
                .content,
            "second.txt"
        );
    }
}
