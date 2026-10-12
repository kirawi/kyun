use std::ops::Range;

use crate::widgets::{
    formatter::{PlainText, State},
    real_lines,
};
use kyun_agent::{Message, MessageView};

use super::diff::DiffCache;
use super::markdown::{MarkdownCache, MarkdownPosition, MarkdownReflow};
use super::output::OutputCache;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Position {
    pub msg_idx: usize,
    pub row: Row,
}

impl Position {
    pub fn start() -> Self {
        Self {
            msg_idx: 0,
            row: Row::Start,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Row {
    Start,
    RoleHeader,
    Text {
        is_reasoning: bool,
        span: Span,
    },
    // Step4: The position uses tree block/cell ownership and cached wrapped boundaries.
    // Pending text uses a separate plain cursor at sentinel ID zero; it does not inherit Markdown ancestors.
    Markdown {
        position: MarkdownPosition,
    },
    ToolHeader {
        call_idx: usize,
    },
    /// A body, ellipsis, or bottom-border row in a diff. `row == last_row` is the bottom border.
    ToolDiff {
        call_idx: usize,
        row: usize,
        last_row: usize,
    },
    /// A top-border, body, ellipsis, or bottom-border row in a shell result. `row == last_row` is the bottom
    /// border.
    ToolOutput {
        call_idx: usize,
        row: usize,
        last_row: usize,
    },
    Padding,
}

impl Row {
    fn with_body_row(self, row: usize, last_row: usize) -> Self {
        match self {
            Self::ToolDiff { call_idx, .. } => Self::ToolDiff {
                call_idx,
                row,
                last_row,
            },
            Self::ToolOutput { call_idx, .. } => Self::ToolOutput {
                call_idx,
                row,
                last_row,
            },
            _ => unreachable!(),
        }
    }
}

pub type Span = Range<usize>;

pub fn get_msg_text(message: &Message, is_reasoning: bool) -> &str {
    match message {
        Message::Assistant(message) if is_reasoning => message.reasoning.as_deref().unwrap(),
        Message::Assistant(message) => message.content.as_deref().unwrap(),
        Message::User(content) => content,
        Message::System(_) | Message::Tool(_) => unreachable!(),
    }
}

pub fn find_line_start(s: &str, byte: usize) -> usize {
    s.as_bytes()[..byte]
        .iter()
        .rposition(|byte| matches!(byte, b'\r' | b'\n'))
        .map_or(0, |offset| offset + 1)
}

/// Cached wrapping for the plain-text line containing the cursor.
// TODO: Resetting the cache instead of replacing it could reuse the spans allocation.
// TP1: Markdown and plain text have separate positions; this cache serves only plain text.
// Step4: Markdown retains wrapped boundaries by block/cell node ID and invalidates affected blocks or tables.
// TODO: Apply this walker-local scope to Markdown wrapping, memoizing leaf/byte boundaries only when needed.
pub struct SoftCache {
    line: Span,
    spans: Vec<Span>,
    row: usize,
}

impl SoftCache {
    fn new(line: &str, start: usize, width: usize) -> Self {
        let mut spans = Vec::new();
        if line.is_empty() || width == 0 {
            spans.push(start..start);
        } else {
            let mut state = State::default();
            while let Some(row) = state.next_row(&PlainText(line), width) {
                spans.push(start + row.span.start..start + row.span.end);
            }
        }
        Self {
            line: start..start + line.len(),
            spans,
            row: 0,
        }
    }
}

pub struct Walker<'a> {
    messages: MessageView<'a>,
    pub cur: Position,
    width: usize,
    pub soft_cache: Option<SoftCache>,
    /// Parsed layouts shared with the transcript, so hot-path stepping does not re-parse tool output.
    diffs: &'a mut DiffCache,
    markdown: &'a mut MarkdownCache,
    outputs: &'a mut OutputCache,
}

impl<'a> Walker<'a> {
    /// Constructs a walker positioned after the transcript's final row.
    pub fn from_end(
        messages: MessageView<'a>,
        width: usize,
        diffs: &'a mut DiffCache,
        markdown: &'a mut MarkdownCache,
        outputs: &'a mut OutputCache,
    ) -> Self {
        let mut walker = Self::new(
            messages,
            Position::start(),
            width,
            None,
            diffs,
            markdown,
            outputs,
        );
        let Some(idx) = messages
            .iter()
            .rposition(|message| !matches!(message, Message::System(_)))
        else {
            return walker;
        };

        let data = match &messages[idx] {
            Message::User(_) if idx + 1 == messages.len() => walker.tail_row(idx),
            Message::User(_) => Row::Padding,
            Message::Assistant(message) if !message.tool_calls.is_empty() => {
                let call_idx = walker.visible_tool_call_count(idx).checked_sub(1).unwrap();
                walker.tail_tool_call(idx, call_idx)
            }
            Message::Assistant(_) if idx + 1 == messages.len() => walker.tail_row(idx),
            Message::Assistant(_) if walker.is_tool_call_only(idx) => Row::RoleHeader,
            Message::Assistant(_) if idx + 1 != messages.len() => Row::Padding,
            Message::Assistant(_) | Message::Tool(_) => walker.tail_row(idx),
            Message::System(_) => unreachable!(),
        };
        walker.cur = Position {
            msg_idx: idx,
            row: data,
        };
        walker
    }

    /// Any supplied cache must accompany its original cursor and width. Use `update_text` to change width.
    pub fn new(
        messages: MessageView<'a>,
        cur: Position,
        width: usize,
        cache: Option<SoftCache>,
        diffs: &'a mut DiffCache,
        markdown: &'a mut MarkdownCache,
        outputs: &'a mut OutputCache,
    ) -> Self {
        // A pending tool row is initially owned by the Assistant message. Once its ordered Tool result arrives,
        // the same row is owned by that adjacent Tool message; preserve the cursor's row and call index.
        let cur = if matches!(
            &cur.row,
            Row::ToolHeader { .. } | Row::ToolDiff { .. } | Row::ToolOutput { .. }
        ) && matches!(messages.get(cur.msg_idx), Some(Message::Assistant(_)))
            && matches!(messages.get(cur.msg_idx + 1), Some(Message::Tool(_)))
        {
            Position {
                msg_idx: cur.msg_idx + 1,
                row: cur.row,
            }
        } else {
            cur
        };
        Self {
            messages,
            cur,
            width,
            soft_cache: cache,
            diffs,
            markdown,
            outputs,
        }
    }

    fn markdown_first_row(&mut self, message_idx: usize) -> Option<Row> {
        let messages = self.messages;
        let source = get_msg_text(&messages[message_idx], false);
        let width = self.width;
        let document = self.markdown.msg(messages, message_idx, width)?;
        document
            .first_position(source, width)
            .map(|position| Row::Markdown { position })
    }

    fn markdown_tail_row(&mut self, message_idx: usize) -> Option<Row> {
        let messages = self.messages;
        let source = get_msg_text(&messages[message_idx], false);
        let width = self.width;
        let document = self.markdown.msg(messages, message_idx, width)?;
        document
            .last_position(source, width)
            .map(|position| Row::Markdown { position })
    }

    pub fn before_markdown_node(&mut self, message_idx: usize, node: usize) -> Position {
        // Step4: Start before the changed block/table ancestor. Zero selects the plain pending suffix boundary.
        let messages = self.messages;
        let source = get_msg_text(&messages[message_idx], false);
        let width = self.width;
        let position = self
            .markdown
            .msg(messages, message_idx, width)
            .and_then(|document| document.position_before_node(source, width, node));
        let data = if let Some(position) = position {
            Row::Markdown { position }
        } else if let Message::Assistant(message) = &messages[message_idx]
            && let Some(reasoning) = message
                .reasoning
                .as_deref()
                .filter(|reasoning| !reasoning.is_empty())
        {
            self.tail_text_row(reasoning, true)
        } else {
            Row::RoleHeader
        };
        Position {
            msg_idx: message_idx,
            row: data,
        }
    }

    // QUESTION: Why does it add a 1 for an unresolved result? Is it to represent the pending
    // tool result? If so, I wonder if this permits a data race between the callback that
    // triggers the pending approval and an event on the transcript. This depends on how callbacks
    // and renders are executed/ordered, and particularly whether apprpoval makes use of callbacks
    // at all.
    fn visible_tool_call_count(&self, message_idx: usize) -> usize {
        let (calls, results) = self.messages.get_tool_calls(message_idx);
        results.len() + usize::from(results.len() < calls.len())
    }

    fn first_tool_call_after_assistant(&mut self, message_idx: &mut usize) -> Row {
        if matches!(self.messages.get(*message_idx + 1), Some(Message::Tool(_))) {
            *message_idx += 1;
        }
        self.get_first_tool_call_row(*message_idx, 0)
    }

    fn is_tool_call_only(&mut self, message_idx: usize) -> bool {
        let Message::Assistant(message) = &self.messages[message_idx] else {
            return false;
        };
        if message.tool_calls.is_empty() {
            return false;
        }
        let has_reasoning = message
            .reasoning
            .as_deref()
            .is_some_and(|reasoning| !reasoning.is_empty());
        // Parsed content can be nonempty without producing a visible row.
        let has_visible_content = message
            .content
            .as_deref()
            .is_some_and(|content| !content.is_empty())
            && (!self.markdown.is_enabled()
                || self
                    .markdown
                    .msg(self.messages, message_idx, self.width)
                    .is_some());
        !has_reasoning && !has_visible_content
    }

    fn row_before_pending_tool(&mut self, message_idx: usize) -> Row {
        if self.is_tool_call_only(message_idx) {
            Row::RoleHeader
        } else {
            Row::Padding
        }
    }

    fn tool_body_row(&mut self, message_idx: usize, call_idx: usize, at_end: bool) -> Option<Row> {
        if let Some(diff) = self
            .diffs
            .layout(self.messages, message_idx, call_idx, self.width)
        {
            return Some(Row::ToolDiff {
                call_idx,
                row: if at_end { diff.last_row() } else { 0 },
                last_row: diff.last_row(),
            });
        }
        self.outputs
            .layout(self.messages, message_idx, call_idx, self.width)
            .map(|output| Row::ToolOutput {
                call_idx,
                row: if at_end { output.last_row() } else { 0 },
                last_row: output.last_row(),
            })
    }

    fn get_first_tool_call_row(&mut self, message_idx: usize, call_idx: usize) -> Row {
        if let Some(diff) = self
            .diffs
            .layout(self.messages, message_idx, call_idx, self.width)
            && diff.files == 1
        {
            return Row::ToolDiff {
                call_idx,
                row: 0,
                last_row: diff.last_row(),
            };
        }
        Row::ToolHeader { call_idx }
    }

    fn tail_tool_call(&mut self, message_idx: usize, call_idx: usize) -> Row {
        self.tool_body_row(message_idx, call_idx, true)
            .unwrap_or(Row::ToolHeader { call_idx })
    }

    /// Constructs a [`Row`] for the last soft-wrapped line of `s`.
    fn tail_text_row(&mut self, s: &str, is_reasoning: bool) -> Row {
        let real_start = find_line_start(s, s.len());
        let mut cache = SoftCache::new(&s[real_start..], real_start, self.width);
        cache.row = cache.spans.len() - 1; // 1-index -> 0-index; spans can never be empty

        let span = cache.spans[cache.row].clone();
        self.soft_cache = Some(cache);
        Row::Text { is_reasoning, span }
    }

    /// Constructs the final non-padding row of a message.
    fn tail_row(&mut self, idx: usize) -> Row {
        match self.messages.get(idx).unwrap() {
            Message::User(content) => self.tail_text_row(content, false),
            Message::Assistant(message) => {
                if let Some(content) = message
                    .content
                    .as_deref()
                    .filter(|content| !content.is_empty())
                {
                    if self.markdown.is_enabled() {
                        if let Some(row) = self.markdown_tail_row(idx) {
                            return row;
                        }
                        return message
                            .reasoning
                            .as_deref()
                            .filter(|reasoning| !reasoning.is_empty())
                            .map_or(Row::RoleHeader, |reasoning| {
                                self.tail_text_row(reasoning, true)
                            });
                    }
                    self.tail_text_row(content, false)
                } else if let Some(reasoning) = message
                    .reasoning
                    .as_deref()
                    .filter(|reasoning| !reasoning.is_empty())
                {
                    self.tail_text_row(reasoning, true)
                } else {
                    Row::RoleHeader
                }
            }
            Message::Tool(_) => {
                let call_idx = self.visible_tool_call_count(idx).checked_sub(1).unwrap();
                self.tail_tool_call(idx, call_idx)
            }
            Message::System(_) => unreachable!(),
        }
    }

    /// Returns the next text [`Row`] which may be soft-wrapped.
    fn walk_text(&self, is_reasoning: bool, start: usize) -> Row {
        let s = get_msg_text(&self.messages[self.cur.msg_idx], is_reasoning);
        if self.width == 0 {
            return Row::Text {
                is_reasoning,
                span: start..start,
            };
        }

        let line_start = find_line_start(s, start);
        let line = real_lines(&s[line_start..]).next().unwrap();
        // TP1: PlainText supplies its continuation width to the shared formatter.
        let span = State {
            byte: start - line_start,
            soft: start > line_start,
            finished: false,
        }
        .next_row(&PlainText(line), self.width)
        .map_or(start..start, |row| {
            line_start + row.span.start..line_start + row.span.end
        });
        Row::Text { is_reasoning, span }
    }

    fn row_after_text(&self, message_idx: usize) -> Option<Row> {
        if matches!(
            &self.messages[message_idx],
            Message::Assistant(message) if !message.tool_calls.is_empty()
        ) || message_idx + 1 < self.messages.len()
        {
            Some(Row::Padding)
        } else {
            None
        }
    }

    pub fn previous(&mut self) -> bool {
        if let Some(cache) = &mut self.soft_cache
            && cache.row > 0
            && let Row::Text { span, .. } = &mut self.cur.row
        {
            cache.row -= 1;
            *span = cache.spans[cache.row].clone();
            return true;
        } // cache does not correspond to the current line, thus invalidated
        self.soft_cache = None;

        let mut idx = self.cur.msg_idx;
        let row = match self.cur.row.clone() {
            Row::Start => return false,

            Row::ToolHeader { call_idx: 0 }
                if matches!(self.messages[idx], Message::Assistant(_)) =>
            {
                Some(self.row_before_pending_tool(idx))
            }

            // Transition from current message to the previous message (if there is one) or clamp to Start state.
            Row::RoleHeader | Row::ToolHeader { call_idx: 0 } => {
                // Skip system messages.
                let Some(previous_idx) = self
                    .messages
                    .iter()
                    .take(idx)
                    .rposition(|message| !matches!(message, Message::System(_)))
                else {
                    self.cur = Position::start();
                    return true;
                };
                idx = previous_idx;

                // A tool result needs padding before a new user turn, but not before an assistant response.
                if self.is_tool_call_only(idx)
                    || matches!(self.messages[idx], Message::Tool(_))
                        && !matches!(self.messages[self.cur.msg_idx], Message::User(_))
                {
                    None
                } else {
                    Some(Row::Padding)
                }
            }
            Row::ToolHeader { call_idx } => Some(self.tail_tool_call(idx, call_idx - 1)),
            Row::ToolDiff {
                call_idx: 0,
                row: 0,
                ..
            } if matches!(self.messages[idx], Message::Assistant(_))
                && self
                    .diffs
                    .layout(self.messages, idx, 0, self.width)
                    .is_some_and(|diff| diff.files == 1) =>
            {
                Some(self.row_before_pending_tool(idx))
            }
            Row::ToolDiff {
                call_idx: 0,
                row: 0,
                ..
            } if self
                .diffs
                .layout(self.messages, idx, 0, self.width)
                .is_some_and(|diff| diff.files == 1) =>
            {
                // A single-file diff starts directly at its first diff row, without a summary row.
                let Some(previous_idx) = self
                    .messages
                    .iter()
                    .take(idx)
                    .rposition(|message| !matches!(message, Message::System(_)))
                else {
                    self.cur = Position::start();
                    return true;
                };
                idx = previous_idx;

                if matches!(self.messages[idx], Message::Tool(_)) || self.is_tool_call_only(idx) {
                    None
                } else {
                    Some(Row::Padding)
                }
            }
            Row::ToolDiff {
                call_idx,
                row,
                last_row,
            } => {
                if row == 0 {
                    if self
                        .diffs
                        .layout(self.messages, idx, call_idx, self.width)
                        .is_some_and(|diff| diff.files == 1)
                    {
                        Some(self.tail_tool_call(idx, call_idx - 1))
                    } else {
                        Some(Row::ToolHeader { call_idx })
                    }
                } else {
                    Some(Row::ToolDiff {
                        call_idx,
                        row: row - 1,
                        last_row,
                    })
                }
            }
            Row::ToolOutput {
                call_idx,
                row,
                last_row,
            } => {
                if row == 0 {
                    Some(Row::ToolHeader { call_idx })
                } else {
                    Some(Row::ToolOutput {
                        call_idx,
                        row: row - 1,
                        last_row,
                    })
                }
            }
            Row::Markdown { position } => {
                let messages = self.messages;
                let source = get_msg_text(&messages[idx], false);
                let width = self.width;
                let previous_row = self
                    .markdown
                    .msg(messages, idx, width)
                    .and_then(|document| document.previous(source, width, &position))
                    .map(|position| Row::Markdown { position });
                if let Some(row) = previous_row {
                    Some(row)
                } else {
                    let messages = self.messages;
                    if let Message::Assistant(message) = &messages[idx]
                        && let Some(reasoning) = message
                            .reasoning
                            .as_deref()
                            .filter(|reasoning| !reasoning.is_empty())
                    {
                        Some(self.tail_text_row(reasoning, true))
                    } else {
                        Some(Row::RoleHeader)
                    }
                }
            }
            Row::Padding => {
                // At this point, we are in the correct message but just need to render the above text row if any.
                None
            }
            Row::Text { is_reasoning, span } => {
                let messages = self.messages;
                let source = get_msg_text(&messages[self.cur.msg_idx], is_reasoning);
                let line_start = find_line_start(source, span.start);

                // NOTE: in the case of width == 0, span.start is always equal to line_start. Thus that case
                // is always implicitly handled.
                if span.start != line_start {
                    // This is a soft-wrapped line of text, and thus there exists a prior soft-wrapped line to go to.
                    let line = real_lines(&source[line_start..]).next().unwrap();
                    let mut cache = SoftCache::new(line, line_start, self.width);

                    // Fetch the line above the current line.
                    cache.row = cache
                        .spans
                        .partition_point(|previous| previous.start < span.start)
                        - 1;

                    let previous = cache.spans[cache.row].clone();
                    self.soft_cache = Some(cache);
                    Some(Row::Text {
                        is_reasoning,
                        span: previous,
                    })
                } else if span.start > 0 {
                    // The line is finished.
                    // Go to the previous real line
                    let end = if source[..span.start].ends_with("\r\n") {
                        span.start - 2
                    } else {
                        span.start - 1
                    };
                    Some(self.tail_text_row(&source[..end], is_reasoning))
                } else if !is_reasoning
                    && let Message::Assistant(message) = &messages[self.cur.msg_idx]
                    && let Some(reasoning) = message
                        .reasoning
                        .as_deref()
                        .filter(|reasoning| !reasoning.is_empty())
                {
                    // The *content* of the message has been fully read if this is an assistant message,
                    // therefore proceed to rendering reasoning.
                    Some(self.tail_text_row(reasoning, true))
                } else {
                    Some(Row::RoleHeader)
                }
            }
        };

        // Jump to the previous message and render its last row if necessary.
        let row = match row {
            Some(row) => row,
            None => self.tail_row(idx),
        };
        self.cur = Position { msg_idx: idx, row };
        true
    }

    /// Reflows the cursor's text line at the new width, preserving the boundary before the first displayed text row,
    /// if any. Tool body rows are re-anchored to the source line shown at the top of the viewport.
    pub fn update_text(&mut self, width: usize) {
        let old_width = std::mem::replace(&mut self.width, width);
        if old_width == width {
            return;
        }

        // The cache is invalidated by width changes, so it must be invalidated regardless of the current row
        // for correctness.
        let old_cache = self.soft_cache.take();

        match self.cur.row.clone() {
            Row::ToolDiff { .. } | Row::ToolOutput { .. } => {
                let (anchor, old_last_row) = self.tool_row_anchor(old_width).unwrap();
                self.restore_tool_row(anchor, old_last_row);
            }
            Row::Text { is_reasoning, span } => {
                // Get the actual full line being inspected (since only it matters for adjusting the row).
                let source = get_msg_text(&self.messages[self.cur.msg_idx], is_reasoning);

                let line_span = old_cache.map_or_else(
                    || {
                        let start = find_line_start(source, span.start);
                        start..start + real_lines(&source[start..]).next().unwrap().len()
                    },
                    |cache| cache.line,
                );
                let line = &source[line_span.clone()];

                // Compute the soft-wrapped version of the current line and save it to cache for future lookups
                // when scrolling up.
                let mut cache = SoftCache::new(line, line_span.start, width);

                // Find the reflowed soft row containing the current row's end.
                let current_row = if old_width == 0 || span.end == line_span.end {
                    // The current row ends at the end of its physical line.
                    // With a width of zero, traversal always consumes the entire real line.
                    cache.spans.len()
                } else {
                    cache.spans.partition_point(|soft_span| {
                        !soft_span.is_empty() && soft_span.end <= span.end
                    })
                }
                .checked_sub(1);

                if let Some(current_row) = current_row {
                    cache.row = current_row;
                    let span = cache.spans[cache.row].clone();
                    self.soft_cache = Some(cache);
                    self.cur.row = Row::Text { is_reasoning, span };
                } else {
                    // The original row no longer maps to the state prior to the first rendered row (i.e. it is now
                    // rendered). Therefore, we must look at the prior row to set the new state to be equivalent to
                    // the pre-reflow state such that the first row of the viewport contains the starting byte idx of
                    // the originally rendered row.
                    self.cur.row = Row::Text {
                        is_reasoning,
                        span: line_span.start..line_span.start,
                    };
                    self.previous();
                }
            }
            Row::Markdown { position } => {
                let messages = self.messages;
                let source = get_msg_text(&messages[self.cur.msg_idx], false);
                if let Some(document) = self.markdown.msg(messages, self.cur.msg_idx, width) {
                    match document.reflow(source, old_width, width, &position) {
                        Some(MarkdownReflow::At(position)) => {
                            self.cur.row = Row::Markdown { position };
                        }
                        Some(MarkdownReflow::Before(position)) => {
                            self.cur.row = Row::Markdown { position };
                            self.previous();
                        }
                        None => {}
                    }
                }
            }
            _ => {}
        }
    }

    /// Returns the source line and segment for the next displayed tool row at `width`.
    pub fn tool_row_anchor(&mut self, width: usize) -> Option<(Option<(usize, usize)>, usize)> {
        match self.cur.row {
            Row::ToolDiff { call_idx, row, .. } => self
                .diffs
                .layout(self.messages, self.cur.msg_idx, call_idx, width)
                .map(|layout| (layout.row_position(row + 1), layout.last_row())),
            Row::ToolOutput { call_idx, row, .. } => self
                .outputs
                .layout(self.messages, self.cur.msg_idx, call_idx, width)
                .map(|layout| (layout.row_position(row), layout.last_row())),
            _ => None,
        }
    }

    /// Restores a tool row after its layout changes, keeping its source line when it remains visible.
    pub fn restore_tool_row(&mut self, anchor: Option<(usize, usize)>, old_last_row: usize) {
        let (call_idx, row) = match self.cur.row {
            Row::ToolDiff { call_idx, row, .. } | Row::ToolOutput { call_idx, row, .. } => {
                (call_idx, row)
            }
            _ => unreachable!(),
        };
        let (last_row, new_row) = match self.cur.row {
            Row::ToolDiff { .. } => self
                .diffs
                .layout(self.messages, self.cur.msg_idx, call_idx, self.width)
                .map(|layout| {
                    let row = anchor.map(|(line, segment)| {
                        if layout.line(line).is_none() {
                            layout.content_rows()
                        } else {
                            layout.line_row(line, segment.min(layout.spans[line].len() - 1))
                        }
                    });
                    (layout.last_row(), row)
                })
                .unwrap(),
            Row::ToolOutput { .. } => self
                .outputs
                .layout(self.messages, self.cur.msg_idx, call_idx, self.width)
                .map(|layout| {
                    let row = anchor.map(|(line, segment)| {
                        if line >= layout.lines.len() {
                            layout.content_rows()
                        } else {
                            layout.line_row(line, segment.min(layout.segment_count(line) - 1))
                        }
                    });
                    (layout.last_row(), row)
                })
                .unwrap(),
            _ => unreachable!(),
        };
        if let Some(new_row) = new_row {
            if matches!(self.cur.row, Row::ToolOutput { .. }) {
                self.cur.row = self.cur.row.clone().with_body_row(new_row, last_row);
                return;
            }
            if new_row > 0 {
                self.cur.row = self.cur.row.clone().with_body_row(new_row - 1, last_row);
            } else {
                self.cur.row = self.cur.row.clone().with_body_row(0, last_row);
                self.previous();
            }
        } else {
            self.cur.row = self.cur.row.clone().with_body_row(
                last_row.saturating_sub(old_last_row.saturating_sub(row)),
                last_row,
            );
        }
    }

    /// Push the cursor forward by `rows`, i.e. the top of the viewport.
    pub fn advance(&mut self, rows: usize) {
        for _ in 0..rows {
            if self.next().is_none() {
                break;
            }
        }
    }

    /// Push the cursor back by `rows`, i.e. the top of the viewport.
    pub fn retreat(&mut self, rows: usize) {
        for _ in 0..rows {
            if !self.previous() {
                break;
            }
        }
    }
}

impl Iterator for Walker<'_> {
    type Item = Row;

    fn next(&mut self) -> Option<Self::Item> {
        // TP1: Short-circuit when width = 0 since it's a nonsense condition.
        if let Some(cache) = &mut self.soft_cache
            && cache.row + 1 < cache.spans.len()
            && let Row::Text { span, .. } = &mut self.cur.row
        {
            cache.row += 1;
            *span = cache.spans[cache.row].clone();
            return Some(self.cur.row.clone());
        }
        self.soft_cache = None; // cache is invalid per above `if let`

        let mut idx = self.cur.msg_idx;

        // Cannot hold an immutable reference to the message; therefore, save index.
        let message_idx = idx;
        self.messages.get(message_idx)?;

        // Get the next message that has a valid row to emit.
        // TP1: This should be turned into a helper function. It also erroneously conflates two responsibilities:
        // getting the message, and getting its first row.
        let get_next_message = |walker: &mut Walker<'_>, idx: &mut usize| {
            let next_idx = walker
                .messages
                .iter()
                .enumerate()
                .skip(*idx)
                .find(|(_, message)| !matches!(message, Message::System(_)))?
                .0;
            *idx = next_idx;

            // Return the initial row for the new message.
            if matches!(walker.messages[*idx], Message::Tool(_)) {
                Some(walker.get_first_tool_call_row(*idx, 0))
            } else {
                Some(Row::RoleHeader)
            }
        };

        // TP1: This should also be turned into a helper method for the same reason as above.
        let get_next_tool_call = |walker: &mut Walker<'_>, idx: &mut usize, call_idx: usize| {
            if call_idx + 1 < walker.visible_tool_call_count(*idx) {
                Some(walker.get_first_tool_call_row(*idx, call_idx + 1))
            } else if matches!(walker.messages[*idx], Message::Tool(_))
                && walker
                    .messages
                    .iter()
                    .skip(*idx + 1)
                    .find(|message| !matches!(message, Message::System(_)))
                    .is_some_and(|message| matches!(message, Message::User(_)))
            {
                Some(Row::Padding)
            } else {
                *idx += 1;
                get_next_message(walker, idx)
            }
        };

        // Results form an ordered prefix of the calls. The next call remains visible while it is pending.
        let row = match self.cur.row.clone() {
            // Transition from sentinel.
            Row::Start => get_next_message(self, &mut idx),

            // Transition from a role header to its first row of content.
            Row::RoleHeader => match &self.messages[message_idx] {
                Message::User(_) => Some(self.walk_text(false, 0)),
                Message::Assistant(message) => {
                    let has_reasoning = message
                        .reasoning
                        .as_deref()
                        .is_some_and(|reasoning| !reasoning.is_empty());
                    let has_content = message
                        .content
                        .as_deref()
                        .is_some_and(|content| !content.is_empty());
                    let has_tool_calls = !message.tool_calls.is_empty();
                    if has_reasoning {
                        Some(self.walk_text(true, 0))
                    } else if has_content {
                        if let Some(row) = self.markdown_first_row(message_idx) {
                            Some(row)
                        } else if self.markdown.is_enabled() {
                            if has_tool_calls {
                                Some(self.first_tool_call_after_assistant(&mut idx))
                            } else if idx + 1 == self.messages.len() {
                                None
                            } else {
                                Some(Row::Padding)
                            }
                        } else {
                            Some(self.walk_text(false, 0))
                        }
                    } else if has_tool_calls {
                        Some(self.first_tool_call_after_assistant(&mut idx))
                    } else if idx + 1 == self.messages.len() {
                        None
                    } else {
                        Some(Row::Padding)
                    }
                }
                Message::System(_) | Message::Tool(_) => unreachable!(),
            },
            Row::Text { is_reasoning, span } => {
                let s = get_msg_text(&self.messages[message_idx], is_reasoning);
                let mut start = span.end;

                // The transcript area itself may be width=0.
                // TODO: eliminate this check by removing the gutter altogether on transcript in
                // such a case.
                if self.width == 0 {
                    start += real_lines(&s[start..]).next().unwrap().len();
                }

                // Go to the next real line if there is one
                if start < s.len() {
                    if s.as_bytes().get(start) == Some(&b'\r') {
                        start += 1;
                    }
                    if s.as_bytes().get(start) == Some(&b'\n') {
                        start += 1;
                    }
                    Some(self.walk_text(is_reasoning, start))
                } else if is_reasoning
                    && let Message::Assistant(message) = &self.messages[message_idx]
                    && message
                        .content
                        .as_deref()
                        .is_some_and(|content| !content.is_empty())
                {
                    if let Some(row) = self.markdown_first_row(message_idx) {
                        Some(row)
                    } else if self.markdown.is_enabled() {
                        self.row_after_text(message_idx)
                    } else {
                        Some(self.walk_text(false, 0))
                    }
                } else {
                    self.row_after_text(message_idx)
                }
            }
            Row::Markdown { position } => {
                let messages = self.messages;
                let source = get_msg_text(&messages[message_idx], false);
                let width = self.width;
                let next_row = self
                    .markdown
                    .msg(messages, message_idx, width)
                    .and_then(|document| document.next(source, width, &position))
                    .map(|position| Row::Markdown { position });
                if let Some(row) = next_row {
                    Some(row)
                } else {
                    self.row_after_text(message_idx)
                }
            }
            Row::ToolHeader { call_idx } => self
                .tool_body_row(idx, call_idx, false)
                .or_else(|| get_next_tool_call(self, &mut idx, call_idx)),
            Row::Padding => {
                if matches!(
                    &self.messages[message_idx],
                    Message::Assistant(message) if !message.tool_calls.is_empty()
                ) {
                    Some(self.first_tool_call_after_assistant(&mut idx))
                } else {
                    idx += 1;
                    get_next_message(self, &mut idx)
                }
            }
            body @ (Row::ToolDiff {
                call_idx,
                row,
                last_row,
            }
            | Row::ToolOutput {
                call_idx,
                row,
                last_row,
            }) => {
                if row < last_row {
                    Some(body.with_body_row(row + 1, last_row))
                } else {
                    get_next_tool_call(self, &mut idx, call_idx)
                }
            }
        }?;

        self.cur = Position {
            msg_idx: idx,
            row: row.clone(),
        };
        Some(row)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        ops::Range,
        task::{Context, Poll},
    };

    use kyun_agent::{
        Agent, AssistantMessage, BackendEvent, Message, MessageView, StepEvent, ToolCall,
        ToolCallKind, ToolResult,
    };

    use super::{DiffCache, MarkdownCache, OutputCache, Position, Row, Walker};
    use crate::test_support::{InertBackend, StreamingBackend, agent_with};
    use crate::tools::{apply_patch, edit, read, shell, write};

    fn assistant(content: String) -> Message {
        Message::Assistant(AssistantMessage {
            reasoning: None,
            content: Some(content),
            tool_calls: Vec::new(),
        })
    }

    /// An agent holding an assistant message with unresolved tool calls.
    fn pending_agent(assistant: AssistantMessage) -> Agent<InertBackend> {
        let mut agent = agent_with([]);
        agent.submit_assistant_msg(assistant);
        agent
    }

    fn cursors_with(
        messages: MessageView<'_>,
        width: usize,
        diffs: &mut DiffCache,
    ) -> Vec<Position> {
        let mut markdown = MarkdownCache::default();
        let mut outputs = OutputCache::default();
        let mut walker = Walker::new(
            messages,
            Position::start(),
            width,
            None,
            diffs,
            &mut markdown,
            &mut outputs,
        );
        let mut cursors = vec![walker.cur.clone()];
        while walker.next().is_some() {
            cursors.push(walker.cur.clone());
        }
        cursors
    }

    fn cursors(messages: MessageView<'_>, width: usize) -> Vec<Position> {
        cursors_with(messages, width, &mut DiffCache::default())
    }

    /// Seeds a write diff into the test transcript.
    fn seed_write_diff(
        diffs: &mut DiffCache,
        before_path: Option<&str>,
        after_path: Option<&str>,
        before: &str,
        after: &str,
        width: usize,
    ) {
        diffs.insert(
            Some(crate::tools::write::file_diff(
                before_path,
                after_path,
                before,
                after,
            )),
            width,
        );
    }

    fn text_spans(cursors: &[Position]) -> Vec<Range<usize>> {
        cursors
            .iter()
            .filter_map(|cursor| match &cursor.row {
                Row::Text { span, .. } => Some(span.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn soft_wrapped_text_can_be_walked_in_both_directions() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        for (content, spans) in [
            ("a".repeat(30), vec![0..10, 10..18, 18..26, 26..30]),
            ("hello, world again".into(), vec![0..7, 7..13, 13..18]),
        ] {
            let agent = agent_with([assistant(content)]);
            let cursors = cursors(agent.messages(), 10);
            assert_eq!(text_spans(&cursors), spans);
            let mut markdown = MarkdownCache::default();
            let mut walker = Walker::new(
                agent.messages(),
                cursors.last().unwrap().clone(),
                10,
                None,
                &mut diffs,
                &mut markdown,
                &mut outputs,
            );
            for expected in cursors.iter().rev().skip(1) {
                assert!(walker.previous());
                assert_eq!(&walker.cur, expected);
            }
            assert!(!walker.previous());
        }
    }

    #[test]
    fn backward_traversal_does_not_reuse_soft_cache_across_messages() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        // Both texts have a soft-wrapping hard line starting at offset 0; stepping back across the message
        // boundary must not reuse the other message's cached spans.
        let agent = agent_with([
            Message::User("aaaa bbbb cccc dddd eeee ffff".into()),
            assistant("a".repeat(30)),
        ]);
        let cursors = cursors(agent.messages(), 10);
        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            cursors.last().unwrap().clone(),
            10,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in cursors.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }
        assert!(!walker.previous());
    }

    #[test]
    fn interrupted_tool_turn_has_padding_before_next_user_across_system_message() {
        // The system message is committed between turns, a state the agent's transitions never produce; the
        // walker's system-skip paths need it to exist mid-list.
        let agent = agent_with([
            Message::User("first".into()),
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call".into(),
                    kind: ToolCallKind::Function,
                    function_name: read::NAME.into(),
                    arguments: "{}".into(),
                }],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: "call".into(),
                content: crate::TOOL_CALL_INTERRUPTED_MESSAGE.into(),
            }]),
            Message::System("instructions".into()),
            Message::User("next".into()),
        ]);
        let cursors = cursors(agent.messages(), 80);
        assert_eq!(
            cursors,
            [
                Position::start(),
                Position {
                    msg_idx: 0,
                    row: Row::RoleHeader,
                },
                Position {
                    msg_idx: 0,
                    row: Row::Text {
                        is_reasoning: false,
                        span: 0..5,
                    },
                },
                Position {
                    msg_idx: 0,
                    row: Row::Padding,
                },
                Position {
                    msg_idx: 1,
                    row: Row::RoleHeader,
                },
                Position {
                    msg_idx: 2,
                    row: Row::ToolHeader { call_idx: 0 },
                },
                Position {
                    msg_idx: 2,
                    row: Row::Padding,
                },
                Position {
                    msg_idx: 4,
                    row: Row::RoleHeader,
                },
                Position {
                    msg_idx: 4,
                    row: Row::Text {
                        is_reasoning: false,
                        span: 0..4,
                    },
                },
            ]
        );

        let mut diffs = DiffCache::default();
        let mut markdown = MarkdownCache::default();
        let mut outputs = OutputCache::default();
        let mut walker = Walker::from_end(
            agent.messages(),
            80,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in cursors.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }
        assert!(!walker.previous());
    }

    #[test]
    fn transferred_soft_cache_remains_consistent_during_bidirectional_traversal() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        let agent = agent_with([Message::Assistant(AssistantMessage {
            reasoning: Some("reasoning words ".repeat(10)),
            content: Some(format!(
                "{}\r\n\n{}\rlast\n",
                "a".repeat(100),
                "界 e\u{301} words ".repeat(10)
            )),
            tool_calls: Vec::new(),
        })]);
        for width in [0, 1, 2, 10, 20] {
            let cursors = cursors(agent.messages(), width);
            let mut cursor = cursors.last().unwrap().clone();
            let mut cache = None;
            for expected in cursors.iter().rev().skip(1) {
                let mut markdown = MarkdownCache::default();
                let mut walker = Walker::new(
                    agent.messages(),
                    cursor,
                    width,
                    cache,
                    &mut diffs,
                    &mut markdown,
                    &mut outputs,
                );
                let previous_row = walker.soft_cache.as_ref().map(|cached| cached.row);
                assert!(walker.previous());
                assert_eq!(&walker.cur, expected);
                if let Some(row) = previous_row
                    && row > 0
                {
                    let cached = walker.soft_cache.as_ref().unwrap();
                    assert_eq!(cached.row, row - 1);
                    assert!(walker.next().is_some());
                    assert_eq!(walker.soft_cache.as_ref().unwrap().row, row);
                    assert!(walker.previous());
                    assert_eq!(&walker.cur, expected);
                }
                if let Some(cached) = &walker.soft_cache {
                    let Row::Text { span, .. } = &walker.cur.row else {
                        panic!()
                    };
                    assert_eq!(span, &cached.spans[cached.row]);
                }
                cursor = walker.cur;
                cache = walker.soft_cache;
            }
        }
    }

    #[test]
    fn transferred_soft_cache_remains_valid_after_appending_messages() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        let mut agent = agent_with([assistant(format!("head\r{}", "aaaa bbbb ".repeat(10)))]);
        let (cursor, cache) = {
            let mut markdown = MarkdownCache::default();
            let mut walker = Walker::new(
                agent.messages(),
                Position {
                    msg_idx: 0,
                    row: Row::Padding,
                },
                10,
                None,
                &mut diffs,
                &mut markdown,
                &mut outputs,
            );
            walker.retreat(4);
            (walker.cur, walker.soft_cache.take().unwrap())
        };
        agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: Some("cccc dddd ".repeat(10)),
            tool_calls: Vec::new(),
        });

        let cursors = cursors(agent.messages(), 10);
        let position = cursors
            .iter()
            .position(|candidate| candidate == &cursor)
            .unwrap();
        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            cursor,
            10,
            Some(cache),
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in &cursors[position + 1..] {
            assert!(walker.next().is_some());
            assert_eq!(&walker.cur, expected);
            if let Some(cached) = &walker.soft_cache {
                let Row::Text { span, .. } = &walker.cur.row else {
                    panic!()
                };
                assert_eq!(span, &cached.spans[cached.row]);
            }
        }
        for expected in cursors.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }
    }

    #[test]
    fn retained_source_is_part_of_the_markdown_document_walk() {
        for (content, reasoning) in [
            ("intro paragraph\n\npending paragraph", None),
            ("open paragraph only", None),
            ("open paragraph only", Some("thinking about the answer")),
        ] {
            let mut agent = agent_with([Message::User("question".into())]);
            agent.submit_assistant_msg(AssistantMessage {
                reasoning: reasoning.map(str::to_string),
                content: Some(content.to_string()),
                tool_calls: Vec::new(),
            });
            let mut diffs = DiffCache::default();
            let mut outputs = OutputCache::default();
            let mut markdown = MarkdownCache::new(true);
            markdown.sync(agent.messages(), true, 20);

            let mut walker = Walker::new(
                agent.messages(),
                Position::start(),
                20,
                None,
                &mut diffs,
                &mut markdown,
                &mut outputs,
            );
            let mut rows = vec![walker.cur.clone()];
            while walker.next().is_some() {
                rows.push(walker.cur.clone());
            }

            assert!(rows.iter().any(|cursor| {
                cursor.msg_idx == 1 && matches!(cursor.row, Row::Markdown { .. })
            }));
            assert!(!rows.iter().any(|cursor| {
                cursor.msg_idx == 1
                    && matches!(
                        cursor.row,
                        Row::Text {
                            is_reasoning: false,
                            ..
                        }
                    )
            }));

            let mut walker = Walker::from_end(
                agent.messages(),
                20,
                &mut diffs,
                &mut markdown,
                &mut outputs,
            );
            for expected in rows.iter().rev().skip(1) {
                assert!(walker.previous());
                assert_eq!(&walker.cur, expected);
            }
        }
    }

    #[test]
    fn streaming_rows_are_an_append_only_prefix_and_keep_the_soft_cache_valid() {
        let mut agent = Agent::new(StreamingBackend {
            events: vec![
                BackendEvent {
                    content: Some("first line words".into()),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("\nsecond".into()),
                    ..Default::default()
                },
                BackendEvent {
                    finished: true,
                    ..Default::default()
                },
            ],
        });
        agent.start_turn("question".into());
        agent.start_step(&[], true);
        let waker = futures_util::task::noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 16
            })
        );
        let first = cursors(agent.messages(), 80);
        assert!(matches!(first.last().unwrap().row, Row::Text { .. }));

        let (cursor, cache) = {
            let mut markdown = MarkdownCache::default();
            let mut walker =
                Walker::from_end(agent.messages(), 8, &mut diffs, &mut markdown, &mut outputs);
            assert!(walker.previous());
            (walker.cur, walker.soft_cache.take().unwrap())
        };

        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 7
            })
        );
        let second = cursors(agent.messages(), 80);
        assert_eq!(first, second[..first.len()]);
        assert!(matches!(second.last().unwrap().row, Row::Text { .. }));

        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            cursor,
            8,
            Some(cache),
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        assert!(walker.next().is_some());
        assert!(walker.next().is_some());
        assert_eq!(walker.cur.msg_idx, 1);
        assert_eq!(
            walker.cur.row,
            Row::Text {
                is_reasoning: false,
                span: 17..23,
            }
        );

        assert_eq!(agent.poll_step(&mut cx), Poll::Ready(StepEvent::Finished));
        assert_eq!(second, cursors(agent.messages(), 80));
    }

    #[test]
    fn mixed_line_endings_have_matching_forward_and_backward_rows() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        let agent = agent_with([assistant("one\r\ntwo\nthree\rfour".into())]);
        let cursors = cursors(agent.messages(), 10);
        assert_eq!(text_spans(&cursors), [0..3, 5..8, 9..14, 15..19]);

        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            cursors.last().unwrap().clone(),
            10,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in cursors.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }
    }

    #[test]
    fn zero_width_advances_between_real_lines() {
        let cursors = cursors(agent_with([assistant("one\ntwo".into())]).messages(), 0);
        assert_eq!(text_spans(&cursors), [0..0, 4..4]);
    }

    #[test]
    fn end_walker_matches_forward_traversal() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        let call = ToolCall {
            id: "call".into(),
            kind: ToolCallKind::Function,
            function_name: "tool".into(),
            arguments: "{}".into(),
        };
        // `Agent::with_messages` validates tool exchanges, so pending calls use transitions while the
        // remaining fixtures use ordinary initial messages.
        let histories = [
            agent_with([Message::System("system".into())]),
            agent_with([Message::User("user words ".repeat(10))]),
            agent_with([assistant("assistant words ".repeat(10))]),
            pending_agent(AssistantMessage {
                reasoning: Some("reasoning words ".repeat(10)),
                content: Some("content words ".repeat(10)),
                tool_calls: vec![call.clone()],
            }),
            pending_agent(AssistantMessage {
                reasoning: Some("reasoning words ".repeat(10)),
                content: None,
                tool_calls: vec![call.clone()],
            }),
            pending_agent(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![call.clone()],
            }),
            agent_with([
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![call.clone()],
                }),
                Message::Tool(vec![ToolResult {
                    tool_call_id: "call".into(),
                    content: "result".into(),
                }]),
                Message::System("system".into()),
            ]),
        ];

        for agent in histories {
            for width in [0, 1, 10, 80] {
                let expected = cursors(agent.messages(), width).pop().unwrap();
                assert_eq!(
                    Walker::from_end(
                        agent.messages(),
                        width,
                        &mut diffs,
                        &mut MarkdownCache::default(),
                        &mut outputs
                    )
                    .cur,
                    expected,
                    "width {width}, messages {:?}",
                    agent.messages(),
                );
            }
        }
    }

    #[test]
    fn resize_preserves_the_visible_anchor_across_widths_and_row_boundaries() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        // The mid-list system messages are states the agent's transitions never produce; the resize paths
        // need them to exist mid-list.
        let agent = agent_with([
            Message::System("system".into()),
            Message::User(format!("{}\r\n\n界 e\u{301} words\rlast\n", "a".repeat(50))),
            Message::Assistant(AssistantMessage {
                reasoning: Some("reasoning words ".repeat(3)),
                content: Some("answer\n\nmore words here".into()),
                tool_calls: Vec::new(),
            }),
            assistant(String::new()),
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some("calling a tool".into()),
                tool_calls: vec![ToolCall {
                    id: "call".into(),
                    kind: ToolCallKind::Function,
                    function_name: "tool".into(),
                    arguments: "{}".into(),
                }],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: "call".into(),
                content: "result".into(),
            }]),
            Message::System("system".into()),
        ]);

        for old_width in [0, 1, 2, 10, 20, 80] {
            let old_cursors = cursors(agent.messages(), old_width);
            for width in [0, 1, 2, 10, 20, 80] {
                let new_cursors = cursors(agent.messages(), width);
                for (position, cursor) in old_cursors.iter().enumerate() {
                    let visible = old_cursors.get(position + 1);
                    let expected = match visible {
                        Some(Position { msg_idx: idx, row: Row::Text { is_reasoning, span } }) => new_cursors
                            .iter()
                            .rposition(|cursor| cursor.msg_idx == *idx && matches!(
                                &cursor.row,
                                Row::Text { is_reasoning: reasoning, span: candidate }
                                    if reasoning == is_reasoning && candidate.start <= span.start
                            ))
                            .unwrap(),
                        Some(cursor) => new_cursors.iter().position(|candidate| candidate == cursor).unwrap(),
                        None => new_cursors.len(),
                    };

                    let mut markdown = MarkdownCache::default();
                    let mut walker = Walker::new(
                        agent.messages(),
                        cursor.clone(),
                        old_width,
                        None,
                        &mut diffs,
                        &mut markdown,
                        &mut outputs,
                    );
                    if matches!(cursor.row, Row::Text { .. }) && walker.next().is_some() {
                        assert!(walker.previous());
                        assert_eq!(&walker.cur, cursor);
                    }
                    walker.update_text(width);
                    assert_eq!(
                        walker.cur,
                        new_cursors[expected - 1],
                        "width {old_width} -> {width}, cursor {cursor:?}, visible {visible:?}",
                    );
                    assert_eq!(
                        walker.next(),
                        new_cursors.get(expected).map(|cursor| cursor.row.clone()),
                        "width {old_width} -> {width}, cursor {cursor:?}, visible {visible:?}",
                    );
                }
            }
        }

        let agent = agent_with([assistant(format!("first\n{}", "a".repeat(50)))]);
        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            Position {
                msg_idx: 0,
                row: Row::Text {
                    is_reasoning: false,
                    span: 6..26,
                },
            },
            20,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        walker.update_text(10);
        assert_eq!(
            walker.next(),
            Some(Row::Text {
                is_reasoning: false,
                span: 24..32,
            })
        );
    }

    #[test]
    fn reasoning_and_content_keep_their_text_source() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        let agent = agent_with([Message::Assistant(AssistantMessage {
            reasoning: Some("think".into()),
            content: Some("answer".into()),
            tool_calls: Vec::new(),
        })]);
        let initial_cursors = cursors(agent.messages(), 10);
        assert_eq!(
            initial_cursors
                .iter()
                .map(|cursor| cursor.row.clone())
                .collect::<Vec<_>>(),
            [
                Row::Start,
                Row::RoleHeader,
                Row::Text {
                    is_reasoning: true,
                    span: 0..5,
                },
                Row::Text {
                    is_reasoning: false,
                    span: 0..6,
                },
            ]
        );

        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            initial_cursors.last().unwrap().clone(),
            10,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in initial_cursors.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }

        let agent = agent_with([Message::Assistant(AssistantMessage {
            reasoning: Some("think".into()),
            content: None,
            tool_calls: Vec::new(),
        })]);
        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            Position {
                msg_idx: 0,
                row: Row::Padding,
            },
            10,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        assert!(walker.previous());
        assert_eq!(
            walker.cur.row,
            Row::Text {
                is_reasoning: true,
                span: 0..5,
            }
        );
    }

    #[test]
    fn assistant_parts_skip_empty_strings_but_preserve_empty_lines() {
        let agent = agent_with([
            Message::Assistant(AssistantMessage {
                reasoning: Some(String::new()),
                content: Some("answer".into()),
                tool_calls: Vec::new(),
            }),
            Message::Assistant(AssistantMessage {
                reasoning: Some("thinking".into()),
                content: Some(String::new()),
                tool_calls: Vec::new(),
            }),
            Message::Assistant(AssistantMessage {
                reasoning: Some(String::new()),
                content: Some(String::new()),
                tool_calls: Vec::new(),
            }),
        ]);
        let initial_cursors = cursors(agent.messages(), 10);
        assert_eq!(
            initial_cursors
                .iter()
                .filter_map(|cursor| match &cursor.row {
                    Row::Text { is_reasoning, span } => Some((*is_reasoning, span.clone())),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            [(false, 0..6), (true, 0..8)]
        );

        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();
        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            initial_cursors.last().unwrap().clone(),
            10,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in initial_cursors.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }
        assert!(!walker.previous());

        let agent = agent_with([Message::Assistant(AssistantMessage {
            reasoning: Some(String::new()),
            content: Some("\n".into()),
            tool_calls: Vec::new(),
        })]);
        assert_eq!(
            cursors(agent.messages(), 10)
                .iter()
                .filter_map(|cursor| match &cursor.row {
                    Row::Text { is_reasoning, span } => Some((*is_reasoning, span.clone())),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            [(false, 0..0), (false, 1..1)]
        );
    }

    #[test]
    fn tool_batches_reveal_only_the_next_pending_call() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        let mut agent = agent_with([]);
        agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: None,
            tool_calls: vec![
                ToolCall {
                    id: "first".into(),
                    kind: ToolCallKind::Function,
                    function_name: "first_tool".into(),
                    arguments: "{}".into(),
                },
                ToolCall {
                    id: "second".into(),
                    kind: ToolCallKind::Function,
                    function_name: write::NAME.into(),
                    arguments: r#"{"path":"output.txt","content":"pending"}"#.into(),
                },
                ToolCall {
                    id: "third".into(),
                    kind: ToolCallKind::Function,
                    function_name: "third_tool".into(),
                    arguments: "{}".into(),
                },
            ],
        });
        agent.append_tool_result(ToolResult {
            tool_call_id: "first".into(),
            content: "one".into(),
        });

        let cursors_before = cursors(agent.messages(), 10);
        assert_eq!(
            cursors_before
                .iter()
                .filter_map(|cursor| match cursor {
                    Position {
                        msg_idx: 1,
                        row: Row::ToolHeader { call_idx },
                    } => Some(*call_idx),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            [0, 1],
        );
        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            cursors_before.last().unwrap().clone(),
            10,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in cursors_before.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }

        agent.append_tool_result(ToolResult {
            tool_call_id: "second".into(),
            content: "two".into(),
        });
        assert_eq!(
            cursors(agent.messages(), 10)
                .iter()
                .filter_map(|cursor| match cursor {
                    Position {
                        msg_idx: 1,
                        row: Row::ToolHeader { call_idx },
                    } => Some(*call_idx),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            [0, 1, 2],
        );
    }

    #[test]
    fn completed_write_diff_rows_include_truncation_and_the_bottom_border() {
        for (diff_line_count, expected_last_row) in [(3, 4), (99, 100), (100, 101)] {
            let mut diffs = DiffCache::default();
            let mut outputs = OutputCache::default();
            let after: String = (0..diff_line_count)
                .map(|line| format!("line {line} aaaa bbbb\n"))
                .collect();
            let file_diff = crate::tools::write::file_diff(None, Some("output.txt"), "", &after);
            let agent = agent_with([
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![
                        ToolCall {
                            id: "write".into(),
                            kind: ToolCallKind::Function,
                            function_name: write::NAME.into(),
                            arguments: r#"{"path":"output.txt","content":"first\nsecond\nthird"}"#
                                .into(),
                        },
                        ToolCall {
                            id: "next".into(),
                            kind: ToolCallKind::Function,
                            function_name: "next".into(),
                            arguments: "{}".into(),
                        },
                    ],
                }),
                Message::Tool(vec![
                    ToolResult {
                        tool_call_id: "write".into(),
                        content: serde_json::to_string(&crate::tools::FileMutationOutput {
                            summary: "changed".into(),
                            diff: file_diff.text.clone(),
                        })
                        .unwrap(),
                    },
                    ToolResult {
                        tool_call_id: "next".into(),
                        content: String::new(),
                    },
                ]),
            ]);
            diffs.insert(Some(file_diff), 80);

            let completed = cursors_with(agent.messages(), 80, &mut diffs);
            let mut expected = vec![
                Position::start(),
                Position {
                    msg_idx: 0,
                    row: Row::RoleHeader,
                },
            ];
            expected.extend((0..=expected_last_row).map(|row| Position {
                msg_idx: 1,
                row: Row::ToolDiff {
                    call_idx: 0,
                    row,
                    last_row: expected_last_row,
                },
            }));
            expected.push(Position {
                msg_idx: 1,
                row: Row::ToolHeader { call_idx: 1 },
            });
            assert_eq!(completed, expected);
            assert_eq!(
                Walker::from_end(
                    agent.messages(),
                    80,
                    &mut diffs,
                    &mut MarkdownCache::default(),
                    &mut outputs
                )
                .cur,
                *completed.last().unwrap()
            );

            let mut markdown = MarkdownCache::default();
            let mut walker = Walker::from_end(
                agent.messages(),
                80,
                &mut diffs,
                &mut markdown,
                &mut outputs,
            );
            for expected in completed.iter().rev().skip(1) {
                assert!(walker.previous());
                assert_eq!(&walker.cur, expected);
            }
            assert!(!walker.previous());
        }
    }

    #[test]
    fn shell_output_appears_after_result_and_walks_in_both_directions() {
        let mut outputs = OutputCache::default();
        let shell_request = AssistantMessage {
            reasoning: None,
            content: None,
            tool_calls: vec![
                ToolCall {
                    id: "shell".into(),
                    kind: ToolCallKind::Function,
                    function_name: shell::NAME.into(),
                    arguments: r#"{"command":"echo test"}"#.into(),
                },
                ToolCall {
                    id: "next".into(),
                    kind: ToolCallKind::Function,
                    function_name: read::NAME.into(),
                    arguments: "{}".into(),
                },
            ],
        };
        // The batch with no results yet.
        let mut agent = agent_with([]);
        agent.submit_assistant_msg(shell_request.clone());
        assert_eq!(
            cursors(agent.messages(), 14).last().unwrap().row,
            Row::ToolHeader { call_idx: 0 }
        );
        agent.append_tool_result(ToolResult {
            tool_call_id: "shell".into(),
            content: "aaaa bbbb cccc\nnext".into(),
        });
        let completed = cursors(agent.messages(), 14);
        assert!(matches!(completed[2].row, Row::ToolHeader { call_idx: 0 }));
        assert!(matches!(
            completed[3].row,
            Row::ToolOutput {
                call_idx: 0,
                row: 0,
                ..
            }
        ));
        assert!(matches!(
            completed.last().unwrap().row,
            Row::ToolHeader { call_idx: 1 }
        ));
        assert!(
            completed
                .iter()
                .filter(|cursor| matches!(cursor.row, Row::ToolOutput { .. }))
                .count()
                > 3
        );

        let mut diffs = DiffCache::default();
        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::from_end(
            agent.messages(),
            14,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in completed.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }
        assert!(!walker.previous());

        walker.cur = completed[3].clone();
        let old_line = walker
            .outputs
            .layout(agent.messages(), 1, 0, 14)
            .unwrap()
            .row_position(0)
            .unwrap()
            .0;
        walker.update_text(18);
        let Row::ToolOutput { row, .. } = walker.cur.row else {
            panic!("expected shell output row after resize");
        };
        assert_eq!(
            walker
                .outputs
                .layout(agent.messages(), 1, 0, 18)
                .unwrap()
                .row_position(row)
                .unwrap()
                .0,
            old_line
        );

        // Each shell result that is empty, denied, or interrupted suppresses its output box.
        for content in [
            "",
            crate::TOOL_CALL_DENIED_MESSAGE,
            crate::TOOL_CALL_INTERRUPTED_MESSAGE,
        ] {
            let mut agent = agent_with([]);
            agent.submit_assistant_msg(shell_request.clone());
            agent.append_tool_result(ToolResult {
                tool_call_id: "shell".into(),
                content: content.into(),
            });
            assert!(
                cursors(agent.messages(), 14)
                    .iter()
                    .all(|cursor| !matches!(cursor.row, Row::ToolOutput { .. }))
            );
        }
    }

    fn write_call_messages(diff: &str) -> Vec<Message> {
        vec![
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: "write".into(),
                    kind: ToolCallKind::Function,
                    function_name: write::NAME.into(),
                    arguments: "{}".into(),
                }],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: "write".into(),
                content: serde_json::to_string(&crate::tools::FileMutationOutput {
                    summary: "changed".into(),
                    diff: diff.into(),
                })
                .unwrap(),
            }]),
        ]
    }

    #[test]
    fn soft_wrapped_diff_rows_round_trip_through_forward_and_backward_walks() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();
        let agent = agent_with(write_call_messages(concat!(
            "--- src/main.rs\n",
            "+++ src/main.rs\n",
            "@@ -1,2 +1,2 @@\n",
            " aaaa bbbb cccc dddd eeee ffff\n",
            "-aaaa bbbb cccc\n",
            "+aaaa bbbb cccc dddd\n",
        )));
        seed_write_diff(
            &mut diffs,
            Some("src/main.rs"),
            Some("src/main.rs"),
            "aaaa bbbb cccc dddd eeee ffff\naaaa bbbb cccc\n",
            "aaaa bbbb cccc dddd eeee ffff\naaaa bbbb cccc dddd\n",
            22,
        );

        // At width 22 the text width is 10: the context line wraps into 6 segments, the removal into 3, and the
        // addition into 4 (verified against the text formatter), plus the unwrapped file header and border row.
        // The cursor sequence includes the assistant header before its results.
        let last_row = 14;
        let cursors = cursors_with(agent.messages(), 22, &mut diffs);
        let mut expected = vec![
            Position::start(),
            Position {
                msg_idx: 0,
                row: Row::RoleHeader,
            },
        ];
        expected.extend((0..=last_row).map(|row| Position {
            msg_idx: 1,
            row: Row::ToolDiff {
                call_idx: 0,
                row,
                last_row,
            },
        }));
        assert_eq!(cursors, expected);
        let mut markdown = MarkdownCache::default();
        assert_eq!(
            Walker::from_end(
                agent.messages(),
                22,
                &mut diffs,
                &mut markdown,
                &mut outputs
            )
            .cur,
            *cursors.last().unwrap()
        );

        let mut walker = Walker::from_end(
            agent.messages(),
            22,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for expected in cursors.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }
        assert!(!walker.previous());
    }

    #[test]
    fn resize_reanchors_diff_rows_to_the_first_visible_diff_line() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();
        let agent = agent_with(write_call_messages(concat!(
            "--- src/main.rs\n",
            "+++ src/main.rs\n",
            "@@ -1,2 +1,2 @@\n",
            " aaaa bbbb cccc dddd eeee ffff\n",
            "-aaaa bbbb cccc\n",
            "+aaaa bbbb cccc dddd\n",
        )));
        seed_write_diff(
            &mut diffs,
            Some("src/main.rs"),
            Some("src/main.rs"),
            "aaaa bbbb cccc dddd eeee ffff\naaaa bbbb cccc\n",
            "aaaa bbbb cccc dddd eeee ffff\naaaa bbbb cccc dddd\n",
            22,
        );

        // Content rows at width 22: [file 1, context 6, removal 3, addition 4]; last_row 14.
        // At width 40 the context wraps once and the other lines fit: [1, 2, 1, 1]; last_row 5.
        // The first visible row keeps its diff line, with its segment clamped to the new segment count.
        for (row, width, expected_row) in [
            (0, 40, 0),  // the context line's first segment
            (1, 40, 1),  // the context line's second segment, which still exists
            (2, 40, 1),  // the context line's third segment, clamped to its last new segment
            (5, 40, 1),  // the final context segment, clamped to the new final context segment
            (10, 40, 3), // the addition line, after the removal line
            (11, 40, 3), // another segment of the addition line
            (12, 40, 3), // another segment of the addition line
            (13, 40, 4), // the final addition segment, anchored before the border row
            (14, 40, 5), // the border row
        ] {
            let mut markdown = MarkdownCache::default();
            let mut walker = Walker::new(
                agent.messages(),
                Position {
                    msg_idx: 1,
                    row: Row::ToolDiff {
                        call_idx: 0,
                        row,
                        last_row: 14,
                    },
                },
                22,
                None,
                &mut diffs,
                &mut markdown,
                &mut outputs,
            );
            walker.update_text(width);
            assert_eq!(
                walker.cur.row,
                Row::ToolDiff {
                    call_idx: 0,
                    row: expected_row,
                    last_row: 5,
                },
                "cursor row {row} resized to width {width}",
            );
        }

        // When the anchored line no longer wraps at all and is the diff's first row, the cursor becomes the row
        // preceding the diff itself (here: the tool call's message header), so that the first
        // visible row is still the start of that line.
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();
        let agent = agent_with(write_call_messages(
            " aaaa bbbb cccc dddd eeee ffff\n-old\n+new\n",
        ));
        let mut file_diff = crate::tools::write::file_diff(
            Some("p"),
            Some("p"),
            "aaaa bbbb cccc dddd eeee ffff\nold\n",
            "aaaa bbbb cccc dddd eeee ffff\nnew\n",
        );
        // This test case deliberately omits the file header, preserving its original row positions.
        file_diff.lines = file_diff.lines.into_vec().into_iter().skip(1).collect();
        diffs.insert(Some(file_diff), 22);
        let mut markdown = MarkdownCache::default();
        let mut walker = Walker::new(
            agent.messages(),
            Position {
                msg_idx: 1,
                row: Row::ToolDiff {
                    call_idx: 0,
                    row: 1,
                    last_row: 7,
                },
            },
            22,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        walker.update_text(60);
        assert_eq!(walker.cur.msg_idx, 0);
        assert_eq!(walker.cur.row, Row::RoleHeader);
        assert_eq!(
            walker.next(),
            Some(Row::ToolDiff {
                call_idx: 0,
                row: 0,
                last_row: 3,
            })
        );
    }

    #[test]
    fn completed_edits_and_patches_append_diff_rows() {
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();

        for function_name in [edit::NAME, apply_patch::NAME] {
            let file_diff =
                crate::tools::write::file_diff(Some("file"), Some("file"), "old\n", "new\n");
            let agent = agent_with([
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "change".into(),
                        kind: ToolCallKind::Function,
                        function_name: function_name.into(),
                        arguments: "{}".into(),
                    }],
                }),
                Message::Tool(vec![ToolResult {
                    tool_call_id: "change".into(),
                    content: serde_json::to_string(&crate::tools::FileMutationOutput {
                        summary: "changed".into(),
                        diff: file_diff.text.clone(),
                    })
                    .unwrap(),
                }]),
            ]);
            diffs.insert(Some(file_diff), 80);
            let cursors = cursors_with(agent.messages(), 80, &mut diffs);
            let tool_rows = cursors
                .iter()
                .filter(|cursor| cursor.msg_idx == 1)
                .map(|cursor| cursor.row.clone())
                .collect::<Vec<_>>();

            assert_eq!(
                &tool_rows[..],
                &(0..=3)
                    .map(|row| Row::ToolDiff {
                        call_idx: 0,
                        row,
                        last_row: 3,
                    })
                    .collect::<Vec<_>>()
            );

            let mut markdown = MarkdownCache::default();
            let mut walker = Walker::from_end(
                agent.messages(),
                80,
                &mut diffs,
                &mut markdown,
                &mut outputs,
            );
            for expected in cursors.iter().rev().skip(1) {
                assert!(walker.previous());
                assert_eq!(&walker.cur, expected);
            }
            assert!(!walker.previous());
        }
    }

    #[test]
    fn hidden_markdown_content_with_tool_calls_walks_identically_both_ways() {
        // Content that renders no Markdown lines behaves exactly like absent content: no padding
        // row between the header and the tool summary, in both walk directions.
        let mut agent = agent_with([Message::User("q".into())]);
        agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: Some(" \n".into()),
            tool_calls: vec![ToolCall {
                id: "call".into(),
                kind: ToolCallKind::Function,
                function_name: read::NAME.into(),
                arguments: "{}".into(),
            }],
        });
        let mut diffs = DiffCache::default();
        let mut outputs = OutputCache::default();
        let mut markdown = MarkdownCache::new(true);
        markdown.initialize_for_test(agent.messages());
        let mut walker = Walker::new(
            agent.messages(),
            Position::start(),
            80,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        let mut cursors = vec![walker.cur.clone()];
        while walker.next().is_some() {
            cursors.push(walker.cur.clone());
        }
        // The assistant header goes straight to the tool summary: no padding row and no
        // content rows for the whitespace content.
        assert_eq!(cursors[4].msg_idx, 1);
        assert_eq!(cursors[4].row, Row::RoleHeader, "{cursors:?}");
        assert_eq!(
            cursors[5].row,
            Row::ToolHeader { call_idx: 0 },
            "{cursors:?}"
        );
        assert!(
            !cursors.iter().any(|cursor| cursor.msg_idx == 1
                && matches!(cursor.row, Row::Padding | Row::Markdown { .. })),
            "{cursors:?}"
        );
        for expected in cursors.iter().rev().skip(1) {
            assert!(walker.previous());
            assert_eq!(&walker.cur, expected);
        }
        assert!(!walker.previous());
    }
}
