mod diff;
mod highlight;
mod markdown;
mod output;
mod walker;

use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::io::Write;
use std::num::NonZeroU16;
use std::process::{Command, Stdio};
use std::sync::{LazyLock, RwLock};

use anyhow::Context as _;

use crate::component::{Callback, Component, EventResult};
use crate::event::Event;
use crate::session::{Action, Session, UIMode};
use crate::tools::{FileMutationOutput, ToolKind, apply_patch, file_mutation};
use crate::widgets::{
    BOX_BORDER, CODE_BG, DIFF_ADD, DIFF_TRUNCATED, DIFF_TRUNCATED_BG, JUMP, MUTED, PENDING,
    SOFT_WRAP_INDICATOR, StatusLine, TEXT, expand_tabs,
    formatter::{PlainText, State},
    get_tool_arguments, real_lines, soft_wrap_prefix_width,
};

use diff::{
    DiffCache, DiffDisplayLine, DiffLineKind, diff_text_width, get_tool_diff, is_file_mutation,
};
use highlight::TOOL_OUTPUT_LINE_NUMBER_WIDTH;
use markdown::MarkdownCache;
use output::OutputCache;
use walker::{Position, Row, SoftCache, Walker, find_line_start, get_msg_text};

use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};
use kyun_agent::{Message, MessageView};
use ratatui::buffer::{CellDiffOption, CellWidth};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolLimits {
    pub output_line_limit: usize,
    pub diff_line_limit: usize,
}

impl Default for ToolLimits {
    fn default() -> Self {
        Self {
            output_line_limit: output::TOOL_OUTPUT_LINE_LIMIT,
            diff_line_limit: diff::TOOL_DIFF_LINE_LIMIT,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeToolLimits {
    pub current: ToolLimits,
    pub default: ToolLimits,
}

impl Default for RuntimeToolLimits {
    fn default() -> Self {
        let default = ToolLimits::default();
        Self {
            current: default,
            default,
        }
    }
}

pub static TOOL_LIMITS: LazyLock<RwLock<RuntimeToolLimits>> =
    LazyLock::new(|| RwLock::new(RuntimeToolLimits::default()));

fn tool_limits() -> ToolLimits {
    TOOL_LIMITS.read().unwrap().current
}

fn push_section(transcript: &mut String, heading: &str, content: &str) {
    if !transcript.is_empty() {
        transcript.push('\n');
    }
    transcript.push_str("# ");
    transcript.push_str(heading);
    transcript.push_str("\n\n");
    transcript.push_str(content);
    if !content.ends_with('\n') {
        transcript.push('\n');
    }
}

fn format_transcript(messages: MessageView<'_>, diff_cache: &DiffCache) -> String {
    let mut transcript = String::new();
    let mut turn = 0;
    for (message_idx, message) in messages.iter().enumerate() {
        match message {
            Message::System(content) => push_section(&mut transcript, "System", content),
            Message::User(content) => {
                turn += 1;
                push_section(&mut transcript, &format!("Turn {turn}: User"), content);
            }
            Message::Assistant(message) => {
                if let Some(reasoning) = &message.reasoning {
                    push_section(
                        &mut transcript,
                        &format!("Turn {turn}: Assistant reasoning"),
                        reasoning,
                    );
                }
                if let Some(content) = &message.content {
                    push_section(&mut transcript, &format!("Turn {turn}: Assistant"), content);
                }
                for (call_idx, call) in message.tool_calls.iter().enumerate() {
                    let result = messages.get(message_idx + 1).and_then(|message| {
                        let Message::Tool(results) = message else {
                            return None;
                        };
                        results.get(call_idx)
                    });
                    let arguments = if let Some(diff) = get_tool_diff(call, result) {
                        format!("```diff\n{}\n```", diff.trim_end())
                    } else if result.is_none()
                        && let Some(diff) = diff_cache.source(messages, message_idx, call_idx)
                    {
                        format!("```diff\n{}\n```", diff.trim_end())
                    } else if result.is_none()
                        && matches!(
                            ToolKind::from_function_name(&call.function_name),
                            ToolKind::FileMutation(file_mutation::Kind::ApplyPatch)
                        )
                    {
                        match apply_patch::patch_text(&call.arguments) {
                            Some(patch) => format!("```diff\n{}\n```", patch.trim()),
                            None => call.arguments.clone(),
                        }
                    } else {
                        call.arguments.clone()
                    };
                    push_section(
                        &mut transcript,
                        &format!(
                            "Turn {turn}: Tool call: {} ({})",
                            call.function_name, call.id
                        ),
                        &arguments,
                    );
                }
            }
            Message::Tool(results) => {
                let Message::Assistant(assistant) = &messages[message_idx - 1] else {
                    unreachable!()
                };
                for (call, result) in assistant.tool_calls.iter().zip(results) {
                    let content = if matches!(
                        ToolKind::from_function_name(&call.function_name),
                        ToolKind::FileMutation(_)
                    ) && let Ok(output) =
                        serde_json::from_str::<FileMutationOutput>(&result.content)
                    {
                        output.summary
                    } else {
                        result.content.clone()
                    };

                    push_section(
                        &mut transcript,
                        &format!(
                            "Turn {turn}: Tool result: {} ({})",
                            call.function_name, call.id
                        ),
                        &content,
                    );
                }
            }
        }
    }
    transcript
}

fn gutter_width(messages: MessageView<'_>) -> usize {
    let user_turn_count = messages
        .iter()
        .filter(|message| matches!(message, Message::User(_)))
        .count();
    user_turn_count.max(1).ilog10() as usize + 3 // +1 digit, +2 padding
}

/// Renders the file header row, including its path and change totals.
fn styled_diff_file(diff_line: DiffDisplayLine<'_>, content_width: usize, dim: bool) -> Line<'_> {
    let additions = format!(" (+{}", diff_line.additions);
    let removals = format!(" -{})", diff_line.removals);
    let reserved_width = 3 + additions.width() + removals.width();
    let max_width = content_width.saturating_sub(reserved_width);
    let path = if diff_line.content.is_empty() {
        Cow::Borrowed(diff_line.content)
    } else {
        let ellipsis = "…";
        match max_width.checked_sub(ellipsis.width()) {
            None => Cow::Borrowed(""),
            Some(content_width) => {
                let mut end = 0;
                let mut width = 0;
                let mut truncated = None;
                for (offset, grapheme) in diff_line.content.grapheme_indices(true) {
                    width += grapheme.width();
                    if width <= content_width {
                        end = offset + grapheme.len();
                    }
                    if width > max_width {
                        truncated = Some(format!("{}{ellipsis}", &diff_line.content[..end]));
                        break;
                    }
                }
                truncated.map_or(Cow::Borrowed(diff_line.content), Cow::Owned)
            }
        }
    };
    let mut style = Style::new();

    if dim {
        style = style.add_modifier(Modifier::DIM);
    }
    Line::from(vec![
        Span::styled(" @ ", Style::new().fg(TEXT)),
        Span::styled(path, Style::new().fg(TEXT).add_modifier(Modifier::BOLD)),
        Span::styled(additions, Style::new().fg(DIFF_ADD)),
        Span::styled(removals, Style::new().fg(JUMP)),
        Span::raw(" "),
    ])
    .style(style)
}

/// Numbers appear only on the first visual row of a source line; continuations keep their gutters blank.
fn output_gutter(
    numbers: &[Option<u32>],
    continuation: bool,
    text_width: usize,
    style: Style,
) -> Vec<Span<'static>> {
    let mut spans = numbers
        .iter()
        .map(|number| {
            Span::styled(
                format!(
                    "{:>width$} ",
                    if continuation {
                        String::new()
                    } else {
                        number.map(|value| value.to_string()).unwrap_or_default()
                    },
                    width = TOOL_OUTPUT_LINE_NUMBER_WIDTH
                ),
                style,
            )
        })
        .collect::<Vec<_>>();
    if continuation && soft_wrap_prefix_width(text_width) > 0 {
        spans.push(Span::styled(SOFT_WRAP_INDICATOR, Style::new().fg(MUTED)));
    }
    spans
}

// NOTE: The existence of this seems extremely suspicious!
fn text_row_count(text: &str, width: usize) -> usize {
    real_lines(text)
        .map(|line| {
            if line.is_empty() || width == 0 {
                return 1;
            }
            let mut rows = 0;
            let mut state = State::default();
            while state.next_row(&PlainText(line), width).is_some() {
                rows += 1;
            }
            rows
        })
        .sum()
}

/// Identifies the `cursor` that renders a bottom-aligned viewport for a [`Transcript`] version. Whenever the version,
/// terminal size, or tool limits change, `cursor` must be updated.
struct BottomCache {
    cursor: Position,
    /// The count of the top-most row when the viewport is at the bottom of the transcript.
    row: usize,
    version: u64,
    limits: ToolLimits,
}

struct Reflow {
    is_reasoning: bool,
    start: usize,
    row: usize,
    old_rows: usize,
    bottom_row: usize,
    markdown_first_node: Option<usize>,
}

/// Renders the full conversation history as a scrollable list of turns.
pub struct Transcript {
    /// The cursor preceding the first displayed row at the top of the viewport.
    cursor: Position,
    // TODO: `Cursor` should be renamed to `Position`, and the cursor concept used here should be its own type with
    // position and Row. BottomCache would also become BottomCursor.
    /// The count of the top-most row of the current viewport with respect to the overall transcript.
    row: usize,
    bottom: BottomCache,
    /// (width, height): used to determine if there was a resize and `cursor` needs to be adjusted.
    // TODO: `size` is always set since the application always calls render() first, therefore should not be an Opt
    size: Option<(usize, usize)>,
    /// The cached plain-text line at `cursor` as an optimization for scrolling up.
    /// Read [`Walker`] for context if needed.
    soft_cache: Option<SoftCache>,
    /// The cursors for up to one viewport of rows displayed after `cursor`, in screen order.
    // NOTE: row_cache needs to be benchmarked to see if it's justified.
    // Suspicion: it is not. It only saves on plaintext/markdown row traversal. Can instead maintain
    // dedicated cache for that; potentially with timers and max height (like browser or PDF reader does).
    // Markdown now caches boundaries by node; a shared cache would need to preserve those owners and plain offsets.
    row_cache: VecDeque<Position>,
    diff_cache: DiffCache,
    // TODO: unify with DiffCache
    output_cache: OutputCache,
    // Step4: Markdown keeps its own node-indexed rows; SoftCache remains for plain-text lines.
    markdown: MarkdownCache,
}

enum Motion {
    GoDown(usize),
    GoUp(usize),
    Top,
    Bottom,
}

impl Default for Transcript {
    fn default() -> Self {
        let limits = tool_limits();
        Self {
            cursor: Position::start(),
            row: 0,
            bottom: BottomCache {
                cursor: Position::start(),
                row: 0,
                version: 0,
                limits,
            },
            size: None,
            soft_cache: None,
            row_cache: VecDeque::new(),
            diff_cache: DiffCache::default(),
            output_cache: OutputCache::default(),
            markdown: MarkdownCache::default(),
        }
    }
}

impl Transcript {
    pub fn new(enable_markdown_rendering: bool) -> Self {
        Self {
            markdown: MarkdownCache::new(enable_markdown_rendering),
            ..Self::default()
        }
    }

    /// Caches a file mutation's structured diff at the current transcript width.
    pub fn cache_diff(&mut self, diff: Option<crate::tools::write::FileDiff>) {
        // Width zero until the first render; the layout reflows itself when one arrives.
        self.diff_cache
            .insert(diff, self.size.map_or(0, |size| size.0));
    }

    /// Resets viewport state after history is shortened while keeping Markdown and previews for retained messages.
    pub fn reset_after_history_change(&mut self, messages: MessageView<'_>) {
        self.cursor = Position::start();
        self.row = 0;
        self.bottom = BottomCache {
            cursor: Position::start(),
            row: 0,
            version: 0,
            limits: tool_limits(),
        };
        self.size = None;
        self.soft_cache = None;
        self.row_cache.clear();
        self.markdown.truncate(messages);
        self.diff_cache.truncate(
            messages
                .iter()
                .filter_map(|message| match message {
                    Message::Assistant(message) => Some(&message.tool_calls),
                    _ => None,
                })
                .flatten()
                .filter(|call| is_file_mutation(call))
                .count(),
        );
        self.output_cache.truncate(messages.len());
    }

    pub fn view(&self, messages: MessageView<'_>, editor: &str) -> anyhow::Result<()> {
        let mut child = Command::new(editor)
            .arg("-")
            .stdin(Stdio::piped())
            .spawn()
            .with_context(|| format!("could not start editor {editor:?}"))?;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(format_transcript(messages, &self.diff_cache).as_bytes())
            .with_context(|| format!("could not write the transcript to {editor:?}"))?;
        child
            .wait()
            .with_context(|| format!("could not wait for {editor:?}"))?;
        Ok(())
    }

    fn fill_row_cache(&mut self, messages: MessageView<'_>, width: usize, height: usize) {
        self.row_cache.truncate(height);
        if self.row_cache.len() == height {
            return;
        }

        let cursor = self.cursor.clone();
        let mut walker = Walker::new(
            messages,
            self.row_cache.back().unwrap_or(&cursor).clone(),
            width,
            None,
            &mut self.diff_cache,
            &mut self.markdown,
            &mut self.output_cache,
        );

        while self.row_cache.len() < height {
            let Some(data) = walker.next() else {
                break;
            };
            self.row_cache.push_back(Position {
                msg_idx: walker.cur.msg_idx,
                row: data,
            });
        }
    }

    /// Returns the logical row offsets of the current and bottom-aligned viewports at the given size.
    fn scroll_rows(
        &mut self,
        messages: MessageView<'_>,
        width: usize,
        height: usize,
    ) -> (usize, usize) {
        let cursor = self.cursor.clone();
        let mut current_row = (cursor == Position::start()).then_some(0);
        let mut total_rows = 0;
        let mut walker = Walker::new(
            messages,
            Position::start(),
            width,
            None,
            &mut self.diff_cache,
            &mut self.markdown,
            &mut self.output_cache,
        );
        while walker.next().is_some() {
            total_rows += 1;
            if walker.cur == cursor {
                current_row = Some(total_rows);
            }
        }
        (current_row.unwrap(), total_rows.saturating_sub(height))
    }

    fn updated_bottom_row(
        &mut self,
        messages: MessageView<'_>,
        width: usize,
        version: u64,
        new_cursor: &Position,
    ) -> usize {
        // A single agent update changes only the history suffix. Reuse the old row count only if the cursor is
        // before the last message and no updates were missed. History shortening resets Transcript.
        let unchanged_prefix =
            version == self.bottom.version + 1 && self.bottom.cursor.msg_idx + 1 < messages.len();
        if unchanged_prefix && &self.bottom.cursor == new_cursor {
            return self.bottom.row;
        }

        // An unchanged-width bottom cursor normally moves by only the rows added or removed at the tail.
        let mut walker = Walker::new(
            messages,
            new_cursor.clone(),
            width,
            None,
            &mut self.diff_cache,
            &mut self.markdown,
            &mut self.output_cache,
        );
        let mut distance = 0;
        while walker.previous() {
            distance += 1;
            if unchanged_prefix && walker.cur == self.bottom.cursor {
                return self.bottom.row + distance;
            }
        }
        // Without a reusable old cursor, reaching the start gives the new cursor's absolute row.
        distance
    }

    fn update_viewport(
        &mut self,
        messages: MessageView<'_>,
        width: usize,
        height: usize,
        version: u64,
        has_active_request: bool,
        // (n_new_reasoning_tokens, n_new_content_tokens) as per `StepEvent::Updated`
        delta: Option<(usize, usize)>,
    ) {
        let version_changed = self.bottom.version != version;
        if let Some((old_width, old_height)) = self.size
            && old_width != width
            && version_changed
        {
            // Repair positions at the old width before resize reflow uses them.
            // The first call cannot repeat this branch.
            self.update_viewport(
                messages,
                old_width,
                old_height,
                version,
                has_active_request,
                delta,
            );
        }
        let size = (width, height);
        let size_changed = self.size != Some(size);
        let is_bottom_view = self.cursor == self.bottom.cursor;
        let tool_render_limits_changed = self.bottom.limits != tool_limits();
        let content_changed = version_changed || tool_render_limits_changed;

        // Step5: Use Step3's distinction between parsed-tree and pending-text changes here. A pending-only append can
        // update plain suffix rows while leaving parsed Markdown wrapping state alone. Changes to parsed nodes start
        // at the affected block/table ancestor, identified by markdown_first_node.
        // Transcript still needs the changed row extent to adjust the viewport. Tree growth, pending growth, and equal
        // old/new row counts are different facts, so an unchanged height must not suppress drawing changed text.
        // Parser and cursor updates must not depend on whether the transcript has visible rows.
        let markdown_update = self.markdown.sync(messages, has_active_request, width);

        // This fast path handles an append whose changed suffix can be measured without walking unchanged history.
        let reflow = if !size_changed
            && !tool_render_limits_changed
            && version == self.bottom.version + 1
            && let Some((reasoning, content)) = delta
            && (!self.markdown.is_enabled() || markdown_update.is_some())
        {
            let markdown_update = markdown_update.unwrap_or_default();
            if reasoning == 0 && content == 0 && !markdown_update.changed {
                // Update the version to match against subsequent updates
                self.bottom.version = version;
                return;
            }

            let has_reasoning = reasoning > 0;
            let last = messages.last().unwrap();
            let (start, old_rows, new_rows) = if !has_reasoning && self.markdown.is_enabled() {
                (
                    0,
                    markdown_update.removed_rows,
                    markdown_update.inserted_rows,
                )
            } else {
                let source = get_msg_text(last, has_reasoning);
                let old_len = source.len() - if has_reasoning { reasoning } else { content };

                // A trailing CR may become CRLF, so include the line before that unfinished line ending.
                let start = walker::find_line_start(
                    source,
                    old_len - usize::from(source[..old_len].ends_with('\r')),
                );

                // An append can change wrapping in the final hard line.
                let old_rows = if old_len == 0 {
                    0
                } else {
                    text_row_count(&source[start..old_len], width)
                };
                let mut new_rows = text_row_count(&source[start..], width);
                if has_reasoning && content > 0 {
                    // A boundary update can finish reasoning and add the first content rows.
                    if self.markdown.is_enabled() {
                        new_rows += markdown_update.inserted_rows;
                    } else {
                        let text = get_msg_text(last, false);
                        debug_assert_eq!(text.len(), content);
                        new_rows += text_row_count(text, width);
                    }
                }
                (start, old_rows, new_rows)
            };

            if old_rows == 0 && new_rows == 0 {
                // Record the update so the next content delta remains consecutive.
                self.bottom.version = version;
                return;
            }

            // Every viewport is full unless the entire transcript is shorter than its height.
            let row = self.bottom.row + self.row_cache.len() - old_rows;
            let bottom_row = (row + new_rows).saturating_sub(height);
            Some(Reflow {
                is_reasoning: has_reasoning,
                start,
                row,
                old_rows,
                bottom_row,
                markdown_first_node: (!has_reasoning && self.markdown.is_enabled())
                    .then_some(markdown_update.first_node),
            })
        } else {
            None
        };

        let mut replaced_view = None;
        if size_changed || content_changed {
            let (new_bottom_cursor, bottom_soft_cache) = {
                let mut bottom_walker = Walker::from_end(
                    messages,
                    width,
                    &mut self.diff_cache,
                    &mut self.markdown,
                    &mut self.output_cache,
                );
                bottom_walker.retreat(height);
                (bottom_walker.cur.clone(), bottom_walker.soft_cache.take())
            };
            let bottom_row = if size_changed || tool_render_limits_changed {
                0
            } else if let Some(reflow) = &reflow {
                reflow.bottom_row
            } else {
                self.updated_bottom_row(messages, width, version, &new_bottom_cursor)
            };

            // The viewport's top row lands on the final message and the view follows content growth there.
            let is_tail_view =
                self.cursor.msg_idx + 1 == messages.len() && !matches!(self.cursor.row, Row::Start);

            if (content_changed
                || size_changed
                    && (height == 0 || self.size.is_some_and(|(_, old_height)| old_height == 0)))
                && (is_bottom_view
                    || (!size_changed
                        && !tool_render_limits_changed
                        && bottom_row < self.bottom.row
                        && self.row >= bottom_row))
            {
                if let Some(reflow) = &reflow {
                    // Rows through the changed suffix's boundary keep their positions, even when the viewport moves.
                    self.row_cache.truncate(reflow.row.saturating_sub(self.row));
                    if bottom_row >= self.row {
                        self.row_cache
                            .drain(..(bottom_row - self.row).min(self.row_cache.len()));
                    } else if !self.row_cache.is_empty() {
                        // Keep the overlap and prepend only the unchanged rows exposed above the old viewport.
                        let mut walker = Walker::new(
                            messages,
                            self.cursor.clone(),
                            width,
                            None,
                            &mut self.diff_cache,
                            &mut self.markdown,
                            &mut self.output_cache,
                        );
                        for _ in bottom_row..self.row {
                            self.row_cache.push_front(walker.cur.clone());
                            walker.previous();
                        }
                    }
                } else {
                    self.row_cache.clear();
                }
                self.row = bottom_row;
                replaced_view = Some((new_bottom_cursor.clone(), bottom_soft_cache));
            } else if let Some(reflow) = &reflow {
                if self.row > reflow.row {
                    let msg_idx = messages.len() - 1;
                    let mut walker = if let Some(node) = reflow.markdown_first_node {
                        let mut walker = Walker::new(
                            messages,
                            Position::start(),
                            width,
                            None,
                            &mut self.diff_cache,
                            &mut self.markdown,
                            &mut self.output_cache,
                        );
                        walker.cur = walker.before_markdown_node(msg_idx, node);
                        walker
                    } else {
                        let mut walker = Walker::new(
                            messages,
                            Position {
                                msg_idx,
                                row: Row::Text {
                                    is_reasoning: reflow.is_reasoning,
                                    span: reflow.start..reflow.start,
                                },
                            },
                            width,
                            None,
                            &mut self.diff_cache,
                            &mut self.markdown,
                            &mut self.output_cache,
                        );
                        walker.previous();
                        walker
                    };
                    walker.advance(self.row - reflow.row);
                    replaced_view = Some((walker.cur.clone(), walker.soft_cache.take()));
                    self.row_cache.clear();
                } else if reflow.old_rows > 0 {
                    self.row_cache.truncate(reflow.row - self.row);
                }
            } else if tool_render_limits_changed {
                // A limits change keeps the viewport's content position. Recalculate its numeric row after rebuilding.
                self.row_cache.clear();
            } else if !size_changed
                && (version_changed
                    && (is_tail_view
                        || version != self.bottom.version + 1
                        || (self.markdown.is_enabled() && markdown_update.is_none())))
            {
                let mut walker = Walker::new(
                    messages,
                    Position::start(),
                    width,
                    None,
                    &mut self.diff_cache,
                    &mut self.markdown,
                    &mut self.output_cache,
                );
                walker.advance(self.row);
                replaced_view = Some((walker.cur.clone(), walker.soft_cache.take()));
                self.row_cache.clear();
            } else if version_changed {
                self.row_cache.clear();
            }
            self.bottom = BottomCache {
                cursor: new_bottom_cursor,
                row: bottom_row,
                version,
                limits: tool_limits(),
            };
        }

        let mut walker = match replaced_view {
            Some((cursor, soft_cache)) => Walker::new(
                messages,
                cursor,
                width,
                soft_cache,
                &mut self.diff_cache,
                &mut self.markdown,
                &mut self.output_cache,
            ),
            None => {
                let mut walker = Walker::new(
                    messages,
                    self.cursor.clone(),
                    self.size.map_or(width, |(old_width, _)| old_width),
                    self.soft_cache.take(),
                    &mut self.diff_cache,
                    &mut self.markdown,
                    &mut self.output_cache,
                );

                // Update the cursor if the terminal has been resized, ensuring text is reflowed correctly.
                if size_changed {
                    walker.update_text(width);
                }
                walker
            }
        };

        // After a resize, clamp the viewport to not exceed the bottom (if possible).
        // NOTE: This can do repeated work if the walker ends up equivalent to the bottom walker after clamping; but
        // optimizing it is likely not worth it given how rare the case is (resize).
        if !is_bottom_view && (size_changed || tool_render_limits_changed) && self.size.is_some() {
            walker.advance(height);
            walker.retreat(height);
        }

        // Viewport is invalidated upon a resize because text gets soft-wrapped at the new width.
        // It's possible to selectively invalidate for when only height changes, but not worth the hassle at this point.
        if size_changed || tool_render_limits_changed {
            self.row_cache.clear();
        }

        self.size = Some(size);
        self.cursor = walker.cur.clone();
        self.soft_cache = walker.soft_cache.take();
        self.fill_row_cache(messages, width, height);
        if size_changed || tool_render_limits_changed {
            (self.row, self.bottom.row) = self.scroll_rows(messages, width, height);
        } else if is_bottom_view {
            self.row = self.bottom.row;
        }
    }

    /// Creates a callback to update the scroll percentage on the statusline.
    fn update_scroll_pct(&self) -> Callback {
        let current_row = self.row;
        let bottom_row = self.bottom.row;
        Box::new(move |compositor, _| {
            if let Some(status_line) = compositor.find::<StatusLine>() {
                status_line.set_scroll(current_row, bottom_row);
            }
        })
    }
}

impl Component for Transcript {
    // Layout: ` [gutter] [box]`
    fn render(&mut self, frame: &mut Frame, area: Rect, cx: &mut Session) {
        const MSG_NUM_STYLE: Style = Style::new().fg(Color::Yellow);
        const ASSISTANT_ROLE: Style = Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD);
        const ASSISTANT_BORDER: Style = Style::new().fg(Color::Magenta);
        const USER_ROLE: Style = Style::new().add_modifier(Modifier::BOLD);

        let area = crate::widgets::base_areas(area).transcript;
        if area.height == 0 {
            // The terminal is too short to render anything.
            // TODO: 0 width should be directly handled
            return;
        }

        let messages = cx.agent.messages();

        // TP1: cache this when width doesn't change
        let gutter_w = gutter_width(messages);
        let height = (area.height as usize).saturating_sub(1);

        // -1 for any message's left-flanking border and -2 for its left-right padding.
        let content_w = (area.width as usize).saturating_sub(gutter_w + 3);

        // Costs basically nothing to compute (i.e. no reason to cache).
        let mut turn_number = cx
            .agent
            .messages()
            .iter()
            .take(self.cursor.msg_idx + 1)
            .filter(|message| matches!(message, Message::User(_)))
            .count();

        // Render the rows to be drawn to the viewport.
        //
        // Record the message_idx corresponding to top-most rendered row of current view. Because the message
        // may be rendered partially (e.g. the last line), we want to have the turn number be rendered for the
        // top-most row. Therefore, it is special-cased as otherwise turn numbers are created only for the first
        // rows of each turn. Recall: turns are counted by user messages, not by messages in general.
        let mut last_idx = self.cursor.msg_idx;
        let mut first_visible_row = true; // to always render the number for the top-most turn
        for (row_area, position) in area.rows().take(height).zip(&self.row_cache) {
            let message = &messages[position.msg_idx];

            // Fast path (this also excludes turn numbers for top-most rows that are padding)
            if matches!(position.row, Row::Padding) {
                // Ensure that padding between assistant messages and their tool calls gets rendered with the
                // turn's left-hand border.
                if matches!(
                    &messages[position.msg_idx],
                    Message::Assistant(message) if !message.tool_calls.is_empty()
                ) {
                    frame.render_widget(
                        Line::from(vec![
                            Span::raw(" ".repeat(gutter_w)),
                            Span::styled("│ ", ASSISTANT_BORDER),
                        ]),
                        row_area,
                    );
                }
                continue;
            }

            let is_user = matches!(message, Message::User(_));
            let is_new_turn = is_user && position.msg_idx != last_idx;
            if is_new_turn {
                turn_number += 1;
            }

            // Render the turn number
            let gutter = if turn_number > 0 && (first_visible_row || is_new_turn) {
                first_visible_row = false;
                format!("{:>gw$} ", turn_number, gw = gutter_w - 1)
            } else {
                " ".repeat(gutter_w)
            };
            last_idx = position.msg_idx;

            let mut content_prefix = Line::from(vec![
                Span::styled(gutter, MSG_NUM_STYLE),
                Span::styled(
                    "│ ",
                    if is_user {
                        Style::default()
                    } else {
                        ASSISTANT_BORDER
                    },
                ),
            ]);
            let prefix_width = gutter_w as u16 + 2; // +2 for padding and left-flanking message border

            // TODO: add fast path for zero-width content area?
            // is this even tested?
            if content_w > 0 {
                // The area after the gutter
                let content_area = Rect {
                    x: row_area.x + prefix_width,
                    width: content_w as u16,
                    ..row_area
                };
                match &position.row {
                    Row::Start | Row::Padding => {
                        unreachable!(
                            "padding should already have been separately handled; and row_cache being iterated over excludes sentinel row"
                        )
                    }
                    Row::ToolHeader { call_idx } => {
                        let (call, result) = messages.get_tool_call(position.msg_idx, *call_idx);
                        let was_denied = result.is_some_and(|result| {
                            result.content == crate::TOOL_CALL_DENIED_MESSAGE
                        });

                        let tool_name_style = if was_denied {
                            Style::new()
                                .fg(Color::Red)
                                .add_modifier(Modifier::CROSSED_OUT)
                        } else {
                            Style::new().fg(PENDING)
                        };

                        if let Some(diff) =
                            self.diff_cache
                                .layout(messages, position.msg_idx, *call_idx, content_w)
                        {
                            // Render aggregate info about multi-file batch edits.
                            let (files, additions, removals) =
                                (diff.files, diff.additions, diff.removals);
                            let noun = if files == 1 { "file" } else { "files" };
                            let title = Line::from(vec![
                                Span::styled(
                                    format!("Edited {files} {noun} "),
                                    tool_name_style.add_modifier(Modifier::BOLD),
                                ),
                                Span::styled(format!("(+{additions}"), Style::new().fg(DIFF_ADD)),
                                Span::styled(format!(" -{removals})"), Style::new().fg(JUMP)),
                            ]);
                            frame.render_widget(
                                Block::new()
                                    .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
                                    .border_style(Style::new().fg(BOX_BORDER))
                                    .title_top(title),
                                content_area,
                            );
                        } else {
                            // Render other tool call headers.
                            let arguments = get_tool_arguments(call);
                            let args_style = if was_denied {
                                Style::new().fg(MUTED).add_modifier(Modifier::CROSSED_OUT)
                            } else {
                                Style::new().fg(MUTED)
                            };
                            let content = Line::from(vec![
                                Span::styled(call.function_name.as_str(), tool_name_style),
                                Span::raw(" "),
                                Span::styled(expand_tabs(&arguments, 0), args_style),
                            ]);
                            frame.render_widget(content, content_area);
                        }
                    }
                    Row::ToolDiff {
                        call_idx,
                        row,
                        last_row,
                    }
                    | Row::ToolOutput {
                        call_idx,
                        row,
                        last_row,
                    } => {
                        let (_, result) = messages.get_tool_call(position.msg_idx, *call_idx);
                        let mut borders = Borders::LEFT | Borders::RIGHT;
                        if *row == 0 {
                            borders |= Borders::TOP;
                        }
                        if row == last_row {
                            borders |= Borders::BOTTOM;
                        }
                        frame.render_widget(
                            Block::new()
                                .borders(borders)
                                .border_style(Style::new().fg(BOX_BORDER)),
                            content_area,
                        );

                        if (matches!(position.row, Row::ToolDiff { .. }) || *row > 0)
                            && row < last_row
                        {
                            let box_interior = Rect::new(
                                content_area.x + 1,
                                content_area.y,
                                content_area.width.saturating_sub(2),
                                1,
                            );
                            let content = if matches!(position.row, Row::ToolDiff { .. }) {
                                let diff = self
                                    .diff_cache
                                    .layout(messages, position.msg_idx, *call_idx, content_w)
                                    .unwrap();
                                let was_denied = result.is_some_and(|result| {
                                    result.content == crate::TOOL_CALL_DENIED_MESSAGE
                                });

                                if diff.is_truncated() && *row == diff.content_rows() {
                                    // Signal to the user that the diff has been truncated to row limits.
                                    frame.render_widget(
                                        Block::new().style(Style::new().bg(DIFF_TRUNCATED_BG)),
                                        box_interior,
                                    );

                                    let mut style = Style::new()
                                        .fg(DIFF_TRUNCATED)
                                        .add_modifier(Modifier::ITALIC);
                                    if was_denied {
                                        style = style.add_modifier(Modifier::DIM);
                                    }
                                    Line::from(Span::styled(" ...", style))
                                } else {
                                    let (line, segment) = diff.row_position(*row).unwrap();
                                    let diff_line = diff.line(line).unwrap();
                                    if diff_line.kind == DiffLineKind::File {
                                        styled_diff_file(
                                            diff_line,
                                            box_interior.width as usize,
                                            was_denied,
                                        )
                                    } else {
                                        let background = match diff_line.kind {
                                            DiffLineKind::Addition => Color::Rgb(0x21, 0x3A, 0x2B),
                                            DiffLineKind::Removal => Color::Rgb(0x4A, 0x22, 0x1D),
                                            DiffLineKind::Context => CODE_BG,
                                            DiffLineKind::Note => Color::Reset,
                                            DiffLineKind::File => unreachable!(),
                                        };
                                        frame.render_widget(
                                            Block::new().style(Style::new().bg(background)),
                                            box_interior,
                                        );

                                        let text_width = diff_text_width(diff.width);
                                        let mut spans = output_gutter(
                                            &[
                                                diff_line.old_line.map(|number| number.get()),
                                                diff_line.new_line.map(|number| number.get()),
                                            ],
                                            segment > 0,
                                            text_width,
                                            Style::new().fg(TEXT),
                                        );
                                        spans.extend(diff.highlighted_line(line).render(
                                            diff.segment_range(line, segment),
                                            if segment > 0 {
                                                soft_wrap_prefix_width(text_width)
                                            } else {
                                                0
                                            },
                                        ));
                                        let mut style = Style::new().bg(background);
                                        if was_denied {
                                            style = style.add_modifier(Modifier::DIM);
                                        }
                                        Line::from(spans).style(style)
                                    }
                                }
                            } else {
                                let output = self
                                    .output_cache
                                    .layout(messages, position.msg_idx, *call_idx, content_w)
                                    .unwrap();
                                if output.truncated && *row == output.content_rows() + 1 {
                                    Line::from(Span::styled(
                                        "     ...",
                                        Style::new()
                                            .fg(DIFF_TRUNCATED)
                                            .add_modifier(Modifier::ITALIC),
                                    ))
                                } else {
                                    let (line, segment) = output.row_position(*row - 1).unwrap();
                                    let source =
                                        &result.unwrap().content[output.lines[line].clone()];
                                    let text_width = (box_interior.width as usize)
                                        .saturating_sub(TOOL_OUTPUT_LINE_NUMBER_WIDTH + 1);
                                    let mut spans = output_gutter(
                                        &[Some(u32::try_from(line + 1).unwrap())],
                                        segment > 0,
                                        text_width,
                                        Style::new().fg(MUTED),
                                    );
                                    spans.extend(
                                        highlight::HighlightedLine::plain(
                                            source,
                                            Style::new().fg(TEXT),
                                        )
                                        .render(
                                            output.segment_range(*row - 1, line),
                                            if segment > 0 {
                                                soft_wrap_prefix_width(text_width)
                                            } else {
                                                0
                                            },
                                        ),
                                    );
                                    Line::from(spans)
                                }
                            };
                            frame.render_widget(content, box_interior);
                        }
                    }
                    Row::RoleHeader => {
                        // Headers of assistant turns aren't rendered again if it's already following an
                        // existing assistant turn (i.e. tool call loop).
                        if is_user
                            || !matches!(
                                position
                                    .msg_idx
                                    .checked_sub(1)
                                    .and_then(|idx| messages.get(idx)),
                                Some(Message::Assistant(_) | Message::Tool(_))
                            )
                        {
                            // Use a different symbol for the left-flanking border to signify it's the start of
                            // either a user or assistant turn.
                            content_prefix.spans[1].content = "╿ ".into();
                            frame.render_widget(
                                Line::from(if is_user {
                                    Span::styled("User", USER_ROLE)
                                } else {
                                    Span::styled("Assistant", ASSISTANT_ROLE)
                                }),
                                content_area,
                            );
                        }
                    }
                    Row::Text { is_reasoning, span } => {
                        let source = get_msg_text(message, *is_reasoning);
                        let line_start = find_line_start(source, span.start);
                        let continuation =
                            span.start > line_start && soft_wrap_prefix_width(content_w) > 0;
                        let mut spans = Vec::new();
                        if continuation {
                            spans.push(Span::styled(SOFT_WRAP_INDICATOR, Style::new().fg(MUTED)));
                        }
                        spans.push(Span::styled(
                            super::expand_tabs(
                                &source[span.clone()],
                                if continuation {
                                    SOFT_WRAP_INDICATOR.width()
                                } else {
                                    0
                                },
                            ),
                            if *is_reasoning {
                                Style::new().fg(MUTED)
                            } else {
                                Style::new()
                            },
                        ));
                        let content = Line::from(spans);
                        frame.render_widget(content, content_area);
                    }
                    Row::Markdown {
                        position: md_position,
                    } => {
                        // Step5: Resolve the tree row's leaf spans and traversal-derived prefixes/styles here, then
                        // draw them in content_area. The row data can also specify the full-row code background, so
                        // Markdown no longer needs its own Widget implementation to fill or draw this area.
                        let row = self.markdown.render(
                            cx.agent.messages(),
                            position.msg_idx,
                            get_msg_text(message, false),
                            content_w,
                            md_position,
                        );
                        if let Some(background) = row.background {
                            frame.buffer_mut().set_style(content_area, background);
                        }
                        frame.render_widget(&row.line, content_area);
                        let mut column = usize::from(content_area.x);
                        let mut links = row.links.iter().peekable();
                        for (span_idx, span) in row.line.spans.iter().enumerate() {
                            if let Some(link) = links.peek()
                                && link.span == span_idx
                            {
                                let mut url = String::new();
                                for byte in link.url.bytes() {
                                    if (0x21..=0x7e).contains(&byte) {
                                        url.push(char::from(byte));
                                    } else {
                                        write!(url, "%{byte:02X}").unwrap();
                                    }
                                }
                                let mut x = column;
                                for grapheme in span.styled_graphemes(Style::default()) {
                                    let width = grapheme.symbol.cell_width();
                                    if width == 0 {
                                        continue;
                                    }
                                    if x + usize::from(width) > usize::from(content_area.right()) {
                                        break;
                                    }
                                    let cell = &mut frame.buffer_mut()[(x as u16, content_area.y)];
                                    // Close each cell's link so partial redraws cannot link unrelated text.
                                    cell.set_symbol(&format!(
                                        "\x1b]8;id=kyun-{}-{};{url}\x1b\\{}\x1b]8;;\x1b\\",
                                        position.msg_idx,
                                        link.node,
                                        cell.symbol(),
                                    ));
                                    cell.set_diff_option(CellDiffOption::ForcedWidth(
                                        NonZeroU16::new(width).unwrap(),
                                    ));
                                    x += usize::from(width);
                                }
                                links.next();
                            }
                            column += span.width();
                        }
                    }
                }
            }

            // Draw the gutter
            frame.render_widget(
                content_prefix,
                Rect {
                    width: prefix_width.min(row_area.width),
                    ..row_area
                },
            );
        }

        let border_area = Rect {
            y: area.y + area.height - 1,
            height: 1,
            ..area
        };
        frame.render_widget(
            Block::default().borders(Borders::BOTTOM).border_style(
                if self.cursor == self.bottom.cursor {
                    Style::new().add_modifier(Modifier::DIM)
                } else {
                    Style::new()
                },
            ),
            border_area,
        );
    }

    fn handle_event(&mut self, event: &Event, area: Rect, cx: &mut Session) -> EventResult {
        let lifecycle_area = match event {
            Event::Resize(width, height) => Some(Rect::new(0, 0, *width, *height)),
            Event::Redraw | Event::AgentUpdated { .. } => Some(area),
            _ => None,
        };

        // TODO: simplify per size comment and also that area is recreated the same way (duplicate code)
        // Also potentially risky here to duplicate layout here and also in render() -- it's possible they
        // could get out of sync rather easily
        let (width, height) = match lifecycle_area {
            Some(area) => {
                let area = crate::widgets::base_areas(area).transcript;
                (
                    (area.width as usize).saturating_sub(gutter_width(cx.agent.messages()) + 3),
                    (area.height as usize).saturating_sub(1),
                )
            }
            None => self.size.unwrap_or_else(|| {
                let area = crate::widgets::base_areas(area).transcript;
                (
                    (area.width as usize).saturating_sub(gutter_width(cx.agent.messages()) + 3),
                    (area.height as usize).saturating_sub(1),
                )
            }),
        };

        if lifecycle_area.is_some() {
            if matches!(event, Event::Redraw)
                && self.size == Some((width, height))
                && self.bottom.version == cx.agent.version()
                && self.bottom.limits == tool_limits()
            {
                // Was just a redraw
                // TODO: I don't believe that bottom version, size, nor limits really matter?
                // I believe the transcript already gets updated in such a case.
                // Should probably not even consume? Should ignore.
                return EventResult::Consumed(None);
            }

            let delta = match event {
                Event::AgentUpdated { reasoning, content } => Some((*reasoning, *content)),
                _ => None,
            };
            self.update_viewport(
                cx.agent.messages(),
                width,
                height,
                cx.agent.version(),
                cx.agent.has_active_request(),
                delta,
            );
            return EventResult::Consumed(Some(self.update_scroll_pct()));
        }

        if height == 0 {
            // No clearly meaningful way to resolve this case except to do nothing.
            // Scroll pct doesn't have to be updated since it would've already been done by some such (re)draw event.
            return EventResult::Consumed(None);
        }
        self.size = Some((width, height));

        // Toggles whether all rows for tool outputs are shown rather than truncated according to configured limits.
        if let Event::Key(key) = event
            && cx.mode == UIMode::Normal
            && key.kind == KeyEventKind::Press
            && key.code == KeyCode::Char('t')
            && key.modifiers == KeyModifiers::CONTROL
        {
            let anchor = if self.cursor == self.bottom.cursor {
                None
            } else {
                let mut walker = Walker::new(
                    cx.agent.messages(),
                    self.cursor.clone(),
                    width,
                    None,
                    &mut self.diff_cache,
                    &mut self.markdown,
                    &mut self.output_cache,
                );
                walker.tool_row_anchor(width)
            };
            let mut runtime_limits = TOOL_LIMITS.write().unwrap();

            // TODO: this should be a constant def on impl ToolLimits itself
            let unlimited = ToolLimits {
                output_line_limit: usize::MAX,
                diff_line_limit: usize::MAX,
            };

            // TODO: could also be turned into a method for encapsulation of purpose
            runtime_limits.current = if runtime_limits.current == unlimited {
                runtime_limits.default
            } else {
                unlimited
            };
            drop(runtime_limits); // to avoid holding lock since transcript needs access to update viewport

            if let Some((anchor, old_last_row)) = anchor {
                let mut walker = Walker::new(
                    cx.agent.messages(),
                    self.cursor.clone(),
                    width,
                    None,
                    &mut self.diff_cache,
                    &mut self.markdown,
                    &mut self.output_cache,
                );
                walker.restore_tool_row(anchor, old_last_row);
                self.cursor = walker.cur;
                self.soft_cache = walker.soft_cache;
            }
            self.update_viewport(
                cx.agent.messages(),
                width,
                height,
                cx.agent.version(),
                cx.agent.has_active_request(),
                None,
            );
            return EventResult::Consumed(Some(self.update_scroll_pct()));
        }

        let motion = match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Esc if cx.mode == UIMode::Normal => {
                    cx.pending_action = Some(Action::Interrupt);
                    return EventResult::Consumed(None);
                }
                KeyCode::Down | KeyCode::Char('j') => Motion::GoDown(1),
                KeyCode::Up | KeyCode::Char('k') => Motion::GoUp(1),
                KeyCode::PageDown | KeyCode::Char(' ') => Motion::GoDown(height),
                KeyCode::Char('d') if key.modifiers == KeyModifiers::CONTROL => {
                    Motion::GoDown(height)
                }
                KeyCode::PageUp => Motion::GoUp(height),
                KeyCode::Char('u') if key.modifiers == KeyModifiers::CONTROL => {
                    Motion::GoUp(height)
                }
                KeyCode::Home | KeyCode::Char('g') => Motion::Top,
                KeyCode::End | KeyCode::Char('G') => Motion::Bottom,
                _ => return EventResult::Ignored(None),
            },
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollDown => Motion::GoDown(3),
                MouseEventKind::ScrollUp => Motion::GoUp(3),
                _ => return EventResult::Ignored(None),
            },
            _ => return EventResult::Ignored(None),
        };

        let mut walker = Walker::new(
            cx.agent.messages(),
            self.cursor.clone(),
            width,
            self.soft_cache.take(),
            &mut self.diff_cache,
            &mut self.markdown,
            &mut self.output_cache,
        );
        match motion {
            Motion::GoDown(count) => {
                for _ in 0..count {
                    if walker.cur == self.bottom.cursor {
                        break;
                    }

                    if let Some(cursor) = self.row_cache.pop_front() {
                        walker.cur = cursor;
                        walker.soft_cache = None;
                    } else if walker.next().is_none() {
                        break;
                    }
                    self.row += 1;
                }
            }
            Motion::GoUp(count) => {
                for _ in 0..count {
                    let old_cursor = walker.cur.clone();
                    if !walker.previous() {
                        break;
                    }
                    self.row_cache.push_front(old_cursor);
                    self.row_cache.truncate(height);
                    self.row -= 1;
                }
            }
            Motion::Top => {
                walker.cur = Position::start();
                walker.soft_cache = None;
                self.row_cache.clear();
                self.row = 0;
            }
            Motion::Bottom => {
                walker.cur = self.bottom.cursor.clone();
                walker.soft_cache = None;
                self.row_cache.clear();
                self.row = self.bottom.row;
            }
        }
        self.cursor = walker.cur;
        self.soft_cache = walker.soft_cache;
        self.fill_row_cache(cx.agent.messages(), width, height);
        EventResult::Consumed(Some(self.update_scroll_pct()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::task::{Context, Poll};

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
    use kyun_agent::{
        Agent, AssistantMessage, BackendEvent, Message, MessageView, StepEvent, TokenUsage,
        ToolCall, ToolCallKind, ToolResult,
        openai::{Config, OpenAiBackend},
    };
    use ratatui::{Terminal, backend::TestBackend, layout::Rect, style::Color};

    use super::markdown;
    use super::{DiffCache, OutputCache, Position, Transcript, format_transcript};
    use crate::widgets::{CODE_BG, MUTED};
    use crate::{
        component::Component,
        compositor::Compositor,
        event::Event,
        session::{Session, UIMode},
        test_support::{StreamingBackend, agent_with},
        tools::{FileMutationOutput, apply_patch, edit, read, shell, write},
        widgets::{CommandLine, Prompt, StatusLine},
    };

    use super::walker;
    use crate::widgets::SOFT_WRAP_INDICATOR;

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

    /// Applies one streamed update through the production viewport path.
    fn stream_update(
        agent: &mut Agent<StreamingBackend>,
        transcript: &mut Transcript,
        cx: &mut Context<'_>,
        width: usize,
        height: usize,
    ) {
        let Poll::Ready(StepEvent::Updated {
            n_added_reasoning: reasoning,
            n_added_content: content,
        }) = agent.poll_step(cx)
        else {
            panic!("expected an updated event");
        };
        transcript.update_viewport(
            agent.messages(),
            width,
            height,
            agent.version(),
            true,
            Some((reasoning, content)),
        );
    }

    fn terminal_text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    fn draw_transcript(
        transcript: &mut Transcript,
        terminal: &mut Terminal<TestBackend>,
        session: &mut Session,
    ) {
        let area = terminal.get_frame().area();
        redraw_transcript(transcript, area, session);
        terminal
            .draw(|frame| transcript.render(frame, frame.area(), session))
            .unwrap();
    }

    fn redraw_transcript(transcript: &mut Transcript, area: Rect, session: &mut Session) {
        transcript.handle_event(&Event::Redraw, area, session);
    }

    #[test]
    fn history_reset_keeps_diffs_for_retained_tool_calls() {
        let agent = agent_with([
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: String::new(),
                    kind: ToolCallKind::Function,
                    function_name: write::NAME.into(),
                    arguments: String::new(),
                }],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: String::new(),
                content: String::new(),
            }]),
        ]);
        let mut session = session();
        let mut transcript = Transcript::default();
        redraw_transcript(&mut transcript, Rect::new(0, 0, 80, 16), &mut session);
        transcript.cache_diff(Some(crate::tools::write::file_diff(
            None,
            Some("retained.txt"),
            "",
            "one\n",
        )));
        transcript.cache_diff(Some(crate::tools::write::file_diff(
            None,
            Some("discarded.txt"),
            "",
            "two\n",
        )));

        transcript.reset_after_history_change(agent.messages());

        let diff = transcript
            .diff_cache
            .layout(agent.messages(), 1, 0, 80)
            .unwrap();
        assert_eq!(diff.line(0).unwrap().content, "retained.txt");

        transcript.reset_after_history_change(agent_with([]).messages());
        redraw_transcript(&mut transcript, Rect::new(0, 0, 80, 16), &mut session);
        transcript.cache_diff(Some(crate::tools::write::file_diff(
            None,
            Some("new.txt"),
            "",
            "three\n",
        )));
        let mut pending = agent_with([]);
        pending.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: None,
            tool_calls: vec![ToolCall {
                id: String::new(),
                kind: ToolCallKind::Function,
                function_name: write::NAME.into(),
                arguments: String::new(),
            }],
        });
        let diff = transcript
            .diff_cache
            .layout(pending.messages(), 0, 0, 80)
            .unwrap();
        assert_eq!(diff.line(0).unwrap().content, "new.txt");
    }

    #[test]
    fn formatted_transcript_preserves_messages_tool_arguments_and_full_results() {
        let agent = agent_with([
            Message::System("instructions".into()),
            Message::User("question".into()),
            Message::Assistant(AssistantMessage {
                reasoning: Some("thinking".into()),
                content: Some("I will read it.".into()),
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    kind: ToolCallKind::Function,
                    function_name: read::NAME.into(),
                    arguments: r#"{"path":"notes.txt"}"#.into(),
                }],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: "call-1".into(),
                content: "first line\nsecond line\nthird line".into(),
            }]),
            Message::User("follow-up".into()),
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some("second answer".into()),
                tool_calls: Vec::new(),
            }),
        ]);

        assert_eq!(
            format_transcript(agent.messages(), &DiffCache::default()),
            format!(
                concat!(
                    "# System\n\ninstructions\n\n",
                    "# Turn 1: User\n\nquestion\n\n",
                    "# Turn 1: Assistant reasoning\n\nthinking\n\n",
                    "# Turn 1: Assistant\n\nI will read it.\n\n",
                    "# Turn 1: Tool call: {} (call-1)\n\n{{\"path\":\"notes.txt\"}}\n\n",
                    "# Turn 1: Tool result: {} (call-1)\n\nfirst line\nsecond line\nthird line\n\n",
                    "# Turn 2: User\n\nfollow-up\n\n",
                    "# Turn 2: Assistant\n\nsecond answer\n",
                ),
                read::NAME,
                read::NAME,
            )
        );
    }

    #[test]
    fn formatted_transcript_puts_apply_patch_calls_in_diff_codeblocks() {
        let mut agent = agent_with([]);
        agent.start_turn("change it".into());
        agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: None,
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                kind: ToolCallKind::Custom,
                function_name: apply_patch::NAME.into(),
                arguments: r#"{"patch":"*** Begin Patch\n*** Update File: src/main.rs\n@@\n-old\n+new\n*** End Patch"}"#.into(),
            }],
        });

        assert_eq!(
            format_transcript(agent.messages(), &DiffCache::default()),
            format!(
                concat!(
                    "# Turn 1: User\n\nchange it\n\n",
                    "# Turn 1: Tool call: {} (call-1)\n\n",
                    "```diff\n",
                    "*** Begin Patch\n",
                    "*** Update File: src/main.rs\n",
                    "@@\n",
                    "-old\n",
                    "+new\n",
                    "*** End Patch\n",
                    "```\n",
                ),
                apply_patch::NAME,
            )
        );
    }

    #[test]
    fn formatted_transcript_uses_the_cached_pending_write_diff() {
        let mut agent = agent_with([]);
        agent.start_turn("change it".into());
        agent.submit_assistant_msg(AssistantMessage {
            reasoning: None,
            content: None,
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                kind: ToolCallKind::Function,
                function_name: write::NAME.into(),
                arguments: r#"{"path":"notes.txt","content":"new\n"}"#.into(),
            }],
        });
        let mut diffs = DiffCache::default();
        diffs.insert(
            Some(write::file_diff(None, Some("notes.txt"), "", "new\n")),
            80,
        );

        let transcript = format_transcript(agent.messages(), &diffs);

        assert!(transcript.contains("```diff\n--- /dev/null\n+++ notes.txt"));
        assert!(!transcript.contains(r#"{"path":"notes.txt","content":"new\n"}"#));
    }

    #[test]
    fn formatted_transcript_puts_a_completed_edit_diff_on_the_tool_call() {
        let agent = agent_with([
            Message::User("change it".into()),
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    kind: ToolCallKind::Function,
                    function_name: edit::NAME.into(),
                    arguments:
                        r#"{"path":"src/main.rs","edits":[{"oldText":"old","newText":"new"}]}"#
                            .into(),
                }],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: "call-1".into(),
                content: serde_json::to_string(&crate::tools::FileMutationOutput {
                    summary: "Success. Updated the following files:\nM src/main.rs\n".into(),
                    diff: "--- src/main.rs\n+++ src/main.rs\n@@ -1,1 +1,1 @@\n-old\n+new\n".into(),
                })
                .unwrap(),
            }]),
        ]);

        assert_eq!(
            format_transcript(agent.messages(), &DiffCache::default()),
            format!(
                concat!(
                    "# Turn 1: User\n\nchange it\n\n",
                    "# Turn 1: Tool call: {} (call-1)\n\n",
                    "```diff\n",
                    "--- src/main.rs\n",
                    "+++ src/main.rs\n",
                    "@@ -1,1 +1,1 @@\n",
                    "-old\n",
                    "+new\n",
                    "```\n\n",
                    "# Turn 1: Tool result: {} (call-1)\n\n",
                    "Success. Updated the following files:\n",
                    "M src/main.rs\n",
                ),
                edit::NAME,
                edit::NAME,
            )
        );
    }

    #[test]
    fn tabs_render_as_expanded_spaces_in_text_rows() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("a\tbc".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("answer".into()),
                    tool_calls: Vec::new(),
                }),
            ],
        );

        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).unwrap();
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        // Raw tabs do not render as tab stops, so the spaces prove the tab went through `expand_tabs`.
        assert!(rendered.contains("a   bc"), "{rendered:?}");
    }

    #[test]
    fn diff_lines_have_consistent_marker_padding() {
        let mut session = session();
        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).unwrap();
        redraw_transcript(&mut transcript, Rect::new(0, 0, 80, 16), &mut session);
        crate::test_support::append_messages(&mut session.agent, [
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "edit".into(),
                        kind: ToolCallKind::Function,
                        function_name: edit::NAME.into(),
                        arguments: r#"{"path":"src/main.rs","edits":1}"#.into(),
                    }],
                }),
                Message::Tool(vec![ToolResult {
                    tool_call_id: "edit".into(),
                    content: serde_json::to_string(&crate::tools::FileMutationOutput {
                        summary: "changed".into(),
                        diff: "--- src/main.rs\n+++ src/main.rs\n@@ -1,2 +1,2 @@\n keep\n-old\n+\tindented\n".into(),
                    })
                    .unwrap(),
                }]),
            ]);

        transcript.cache_diff(Some(crate::tools::write::file_diff(
            Some("src/main.rs"),
            Some("src/main.rs"),
            "keep\nold\n",
            "keep\n\tindented\n",
        )));
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        // Both the context and added lines have a separator after their diff marker.
        assert!(rendered.contains("  keep"), "{rendered:?}");
        assert!(rendered.contains("+   indented"), "{rendered:?}");
    }

    #[test]
    fn headers_render_role_labels_in_content_area() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("question".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("answer".into()),
                    tool_calls: Vec::new(),
                }),
            ],
        );

        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).unwrap();
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(rendered.contains("question"), "{rendered:?}");
        assert!(rendered.contains("answer"), "{rendered:?}");
        assert!(rendered.contains("User"), "{rendered:?}");
        assert!(rendered.contains("Assistant"), "{rendered:?}");
    }

    #[test]
    fn completed_edit_renders_the_diff_as_the_tool_call_body() {
        let mut session = session();
        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).unwrap();
        redraw_transcript(&mut transcript, Rect::new(0, 0, 80, 16), &mut session);
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "edit".into(),
                        kind: ToolCallKind::Function,
                        function_name: edit::NAME.into(),
                        arguments: r#"{"path":"src/main.rs","edits":1}"#.into(),
                    }],
                }),
                Message::Tool(vec![ToolResult {
                    tool_call_id: "edit".into(),
                    content: serde_json::to_string(&crate::tools::FileMutationOutput {
                        summary: "changed".into(),
                        diff: "--- src/main.rs\n+++ src/main.rs\n@@ -1,1 +1,1 @@\n-old\n+new\n"
                            .into(),
                    })
                    .unwrap(),
                }]),
            ],
        );

        transcript.cache_diff(Some(crate::tools::write::file_diff(
            Some("src/main.rs"),
            Some("src/main.rs"),
            "old\n",
            "new\n",
        )));
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let rows = terminal
            .backend()
            .buffer()
            .content
            .chunks(80)
            .map(|cells| cells.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>();
        let file_row = rows
            .iter()
            .position(|row| row.contains("src/main.rs (+1 -1)"))
            .unwrap_or_else(|| panic!("{rows:#?}"));

        assert!(!rows.iter().any(|row| row.contains("Edited 1 file (+1 -1)")));
        assert!(!rows.iter().any(|row| row.contains("edits")));
        assert!(!rows.iter().any(|row| row.contains("diff")));
        assert!(!rows.iter().any(|row| row.contains("@@")));
        assert!(rows[file_row].contains("src/main.rs (+1 -1)"));
        assert!(
            rows[file_row - 1].starts_with("   ╿ Assistant"),
            "{rows:#?}"
        );
        assert!(rows[file_row + 1].contains("   1      - old"));
        assert!(rows[file_row + 2].contains("       1 + new"));
        let buffer = terminal.backend().buffer();
        assert!(buffer.content.iter().any(|cell| {
            cell.symbol() == "-"
                && cell.fg == crate::widgets::JUMP
                && cell.bg == ratatui::style::Color::Rgb(0x4A, 0x22, 0x1D)
        }));
        assert!(buffer.content.iter().any(|cell| {
            cell.symbol() == "o"
                && cell.fg == crate::widgets::DIFF_TEXT
                && cell.bg == ratatui::style::Color::Rgb(0x4A, 0x22, 0x1D)
        }));
        assert!(buffer.content.iter().any(|cell| {
            cell.symbol() == "+"
                && cell.fg == crate::widgets::DIFF_ADD
                && cell.bg == ratatui::style::Color::Rgb(0x21, 0x3A, 0x2B)
        }));
        assert!(buffer.content.iter().any(|cell| {
            cell.symbol() == "n"
                && cell.fg == crate::widgets::DIFF_TEXT
                && cell.bg == ratatui::style::Color::Rgb(0x21, 0x3A, 0x2B)
        }));
    }

    #[test]
    fn pending_and_failed_mutations_render_their_argument_summaries() {
        for (function_name, arguments, result, expected) in [
            (
                edit::NAME,
                r#"{"path":"src/main.rs","edits":[{"oldText":"old","newText":"new"}]}"#,
                None,
                format!(r#"{} {{ "path": "src/main.rs", "edits": 1 }}"#, edit::NAME),
            ),
            (
                apply_patch::NAME,
                "*** Begin Patch\n*** Update File: src/main.rs\n@@\n-old\n+new\n*** End Patch",
                Some("Error: change did not apply"),
                format!(r#"{} {{ "patch": "src/main.rs" }}"#, apply_patch::NAME),
            ),
        ] {
            let mut session = session();
            session.agent.start_turn(String::new());
            session.agent.submit_assistant_msg(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: "change".into(),
                    kind: ToolCallKind::Function,
                    function_name: function_name.into(),
                    arguments: arguments.into(),
                }],
            });
            if let Some(content) = result {
                session.agent.append_tool_result(ToolResult {
                    tool_call_id: "change".into(),
                    content: content.into(),
                });
            }

            let mut transcript = Transcript::default();
            let mut terminal = Terminal::new(TestBackend::new(80, 16)).unwrap();
            draw_transcript(&mut transcript, &mut terminal, &mut session);
            let rendered = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();

            assert!(rendered.contains(expected.as_str()), "{rendered:?}");
        }
    }

    #[test]
    fn mouse_scroll_down_matches_three_keyboard_down_steps_in_insert_mode() {
        let mut compositor = Compositor::new();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(Prompt::default()));
        let mut session = session();
        session.mode = UIMode::Insert;
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
        let transcript = compositor.find::<Transcript>().unwrap();
        transcript.cursor = walker::Position::start();
        transcript.soft_cache = None;
        transcript.row_cache.clear();

        assert!(compositor.handle_event(
            &Event::Mouse(MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 10,
                row: 5,
                modifiers: KeyModifiers::NONE,
            }),
            area,
            &mut session,
        ));
        let mouse_cursor = compositor.find::<Transcript>().unwrap().cursor.clone();

        let transcript = compositor.find::<Transcript>().unwrap();
        transcript.cursor = walker::Position::start();
        transcript.soft_cache = None;
        transcript.row_cache.clear();
        for _ in 0..3 {
            let _ = transcript.handle_event(
                &Event::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)),
                area,
                &mut session,
            );
        }
        assert_eq!(mouse_cursor, transcript.cursor);
    }

    #[test]
    fn scroll_events_update_the_status_line() {
        let mut compositor = Compositor::new();
        let mut session = session();
        compositor.push(Box::new(Transcript::default()));
        compositor.push(Box::new(StatusLine::new(0)));
        compositor.push(Box::new(CommandLine::default()));
        session.mode = UIMode::Command;
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::User(
                (0..12)
                    .map(|line| format!("line {line}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )],
        );

        let area = Rect::new(0, 0, 80, 12);
        let mut terminal = Terminal::new(TestBackend::new(area.width, area.height)).unwrap();
        let rendered = |terminal: &Terminal<TestBackend>| {
            terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>()
        };

        compositor.handle_event(&Event::Redraw, area, &mut session);
        terminal
            .draw(|frame| compositor.render(frame, frame.area(), &mut session))
            .unwrap();
        assert!(rendered(&terminal).contains("Bot"));
        assert_eq!(compositor.find::<Transcript>().unwrap().bottom.row, 10);

        session.mode = UIMode::Normal;
        compositor.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE)),
            area,
            &mut session,
        );
        terminal
            .draw(|frame| compositor.render(frame, frame.area(), &mut session))
            .unwrap();
        assert!(rendered(&terminal).contains("Top"));

        compositor.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)),
            area,
            &mut session,
        );
        terminal
            .draw(|frame| compositor.render(frame, frame.area(), &mut session))
            .unwrap();
        assert!(rendered(&terminal).contains("10%"));

        session.mode = UIMode::Command;
        terminal.backend_mut().resize(80, 30);
        compositor.handle_event(&Event::Resize(80, 30), area, &mut session);
        terminal
            .draw(|frame| compositor.render(frame, frame.area(), &mut session))
            .unwrap();
        assert!(rendered(&terminal).contains("All"));
    }

    #[test]
    fn appended_messages_refresh_the_bottom_cache() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::User(
                (0..12)
                    .map(|line| format!("user line {line}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )],
        );

        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        assert_eq!(transcript.cursor, transcript.bottom.cursor);

        transcript.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)),
            Rect::new(0, 0, 80, 12),
            &mut session,
        );
        let scrolled_cursor = transcript.cursor.clone();
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some("first answer".into()),
                tool_calls: Vec::new(),
            })],
        );
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        assert_eq!(transcript.cursor, scrolled_cursor);
        assert_ne!(transcript.cursor, transcript.bottom.cursor);

        transcript.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE)),
            Rect::new(0, 0, 80, 12),
            &mut session,
        );
        let previous_bottom = transcript.bottom.cursor.clone();
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some("second answer".into()),
                tool_calls: Vec::new(),
            })],
        );
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        assert_ne!(transcript.bottom.cursor, previous_bottom);
        assert_eq!(transcript.cursor, transcript.bottom.cursor);

        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: ["first", "second"]
                    .into_iter()
                    .map(|id| ToolCall {
                        id: id.into(),
                        kind: ToolCallKind::Function,
                        function_name: format!("{id}_tool"),
                        arguments: "{}".into(),
                    })
                    .collect(),
            })],
        );

        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let previous_bottom = transcript.bottom.cursor.clone();
        assert_eq!(transcript.cursor, previous_bottom);

        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Tool(vec![
                ToolResult {
                    tool_call_id: "first".into(),
                    content: "done".into(),
                },
                ToolResult {
                    tool_call_id: "second".into(),
                    content: String::new(),
                },
            ])],
        );
        draw_transcript(&mut transcript, &mut terminal, &mut session);

        assert_ne!(transcript.bottom.cursor, previous_bottom);
        assert_eq!(transcript.cursor, transcript.bottom.cursor);
    }

    #[test]
    fn denied_pending_mutation_keeps_its_cached_diff_dimmed() {
        let mut session = session();
        let area = Rect::new(0, 0, 80, 14);
        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 14)).unwrap();
        redraw_transcript(&mut transcript, area, &mut session);
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("one\ntwo\nthree\nfour".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "write".into(),
                        kind: ToolCallKind::Function,
                        function_name: write::NAME.into(),
                        arguments: r#"{"path":"hello.txt","content":"..."}"#.into(),
                    }],
                }),
            ],
        );

        transcript.cache_diff(Some(crate::tools::write::file_diff(
            None,
            Some("hello.txt"),
            "",
            "one\ntwo\nthree\n",
        )));
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        // Put the viewport on the pending preview itself so denial must move this cursor to the Tool message.
        let (cursor, row) = {
            let (width, _) = transcript.size.unwrap();
            let mut walker = walker::Walker::new(
                session.agent.messages(),
                walker::Position::start(),
                width,
                None,
                &mut transcript.diff_cache,
                &mut transcript.markdown,
                &mut transcript.output_cache,
            );
            let mut row = 0;
            loop {
                let cursor = walker.cur.clone();
                walker.next().unwrap();
                if matches!(cursor.row, walker::Row::ToolDiff { .. }) {
                    break (cursor, row);
                }
                row += 1;
            }
        };
        transcript.cursor = cursor;
        transcript.row = row;
        transcript.row_cache.clear();
        let (width, height) = transcript.size.unwrap();
        transcript.fill_row_cache(session.agent.messages(), width, height);
        assert_ne!(transcript.cursor, transcript.bottom.cursor);

        // The call is denied after its pending preview has been rendered.
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Tool(vec![ToolResult {
                tool_call_id: "write".into(),
                content: crate::TOOL_CALL_DENIED_MESSAGE.into(),
            }])],
        );
        draw_transcript(&mut transcript, &mut terminal, &mut session);

        let content = &terminal.backend().buffer().content;
        let rendered = content.iter().map(|cell| cell.symbol()).collect::<String>();
        assert_eq!(
            transcript.cursor.msg_idx,
            session.agent.messages().len() - 1
        );
        assert_eq!(rendered.matches("+ three").count(), 1);
        assert!(rendered.contains("+ three"));
        assert!(content.iter().any(|cell| {
            cell.symbol() == "+" && cell.modifier.contains(ratatui::style::Modifier::DIM)
        }));
    }

    #[test]
    fn toggling_unlimited_diffs_keeps_the_viewport_within_the_bottom_clamp() {
        let mut session = session();
        let area = Rect::new(0, 0, 80, 16);
        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).unwrap();
        redraw_transcript(&mut transcript, area, &mut session);
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("make it long".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "write".into(),
                        kind: ToolCallKind::Function,
                        function_name: write::NAME.into(),
                        arguments: r#"{"path":"output.txt","content":"..."}"#.into(),
                    }],
                }),
            ],
        );
        transcript.cache_diff(Some(crate::tools::write::file_diff(
            None,
            Some("output.txt"),
            "",
            &(0..150)
                .map(|line| format!("line {line} aaaa bbbb\n"))
                .collect::<String>(),
        )));
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        assert_eq!(transcript.cursor, transcript.bottom.cursor);
        assert!(transcript.bottom.row > 0);
        let truncated_bottom_row = transcript.bottom.row;

        // Ctrl-t expands the truncated diff; the bottom-aligned view follows the growth.
        transcript.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL)),
            area,
            &mut session,
        );
        assert!(transcript.bottom.row > truncated_bottom_row);
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        assert_eq!(transcript.cursor, transcript.bottom.cursor);
        let expanded_bottom_row = transcript.bottom.row;

        // Scroll up once so the viewport's bottom row sits inside the expanded diff, past the line limit.
        transcript.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)),
            area,
            &mut session,
        );
        assert_ne!(transcript.cursor, transcript.bottom.cursor);

        // Toggling back shrinks the transcript; the viewport must not remain past the bottom.
        transcript.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL)),
            area,
            &mut session,
        );
        assert!(transcript.bottom.row < expanded_bottom_row);
        assert!(transcript.row <= transcript.bottom.row);
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        assert!(
            transcript.row <= transcript.bottom.row,
            "viewport top row {} is past the bottom clamp {}",
            transcript.row,
            transcript.bottom.row,
        );

        // Down-scrolling hits the clamp again instead of walking past the end of the transcript.
        transcript.handle_event(
            &Event::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)),
            area,
            &mut session,
        );
        assert!(transcript.row <= transcript.bottom.row);
        assert!(!transcript.row_cache.is_empty());
    }

    #[test]
    fn long_diff_lines_soft_wrap_across_rows_instead_of_truncating() {
        let mut session = session();
        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(40, 16)).unwrap();
        redraw_transcript(&mut transcript, Rect::new(0, 0, 40, 16), &mut session);
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "write".into(),
                        kind: ToolCallKind::Function,
                        function_name: write::NAME.into(),
                        arguments: r#"{"path":"output.txt","content":"..."}"#.into(),
                    }],
                }),
                Message::Tool(vec![ToolResult {
                    tool_call_id: "write".into(),
                    content: serde_json::to_string(&FileMutationOutput {
                        summary: "changed".into(),
                        diff: concat!(
                            "--- output.txt\n",
                            "+++ output.txt\n",
                            "@@ -1,1 +1,1 @@\n",
                            "-old short\n",
                            "+aaaa bbbb cccc dddd eeee ffff\n",
                        )
                        .into(),
                    })
                    .unwrap(),
                }]),
            ],
        );

        transcript.cache_diff(Some(crate::tools::write::file_diff(
            Some("output.txt"),
            Some("output.txt"),
            "old short\n",
            "aaaa bbbb cccc dddd eeee ffff\n",
        )));
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let rows = terminal
            .backend()
            .buffer()
            .content
            .chunks(40)
            .map(|cells| cells.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>();

        let removal_row = rows
            .iter()
            .position(|row| row.contains("- old short"))
            .unwrap_or_else(|| panic!("{rows:#?}"));
        // Content width is 34, so the text width is 22: the addition wraps after "dddd".
        assert!(rows[removal_row + 1].contains("+ aaaa bbbb cccc dddd"));
        assert!(rows[removal_row + 2].contains(&format!("{SOFT_WRAP_INDICATOR}eeee ffff")));
        // Continuations render a blank number gutter instead of repeating line numbers.
        assert!(!rows[removal_row + 2].contains("2 "));
        assert!(!rows.iter().any(|row| row.contains('…')));
    }

    #[test]
    fn shell_result_wraps_in_the_transcript_after_it_arrives() {
        let mut session = session();
        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(30, 24)).unwrap();
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: "shell".into(),
                    kind: ToolCallKind::Function,
                    function_name: shell::NAME.into(),
                    arguments: r#"{"command":"echo output"}"#.into(),
                }],
            })],
        );
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        assert!(
            transcript
                .row_cache
                .iter()
                .all(|cursor| !matches!(cursor.row, walker::Row::ToolOutput { .. }))
        );

        session.agent.append_tool_result(ToolResult {
            tool_call_id: "shell".into(),
            content: "first\tsecond third fourth\nnext".into(),
        });
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let rows = terminal
            .backend()
            .buffer()
            .content
            .chunks(30)
            .map(|cells| cells.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>();
        let first_output_row = rows
            .iter()
            .position(|row| row.contains("1 first"))
            .unwrap_or_else(|| panic!("{rows:#?}"));
        assert!(rows[first_output_row - 1].contains('─'), "{rows:#?}");
        assert!(rows.iter().any(|row| row.contains("1 first")));
        assert!(rows.iter().any(|row| row.contains("2 next")));
        assert!(rows.iter().any(|row| row.contains(SOFT_WRAP_INDICATOR)));
        assert!(rows.iter().any(|row| row.contains("second")));
        assert!(rows.iter().any(|row| row.contains("fourth")));
        assert!(rows.iter().all(|row| !row.contains('\t')));
        assert!(
            transcript
                .row_cache
                .iter()
                .any(|cursor| matches!(cursor.row, walker::Row::ToolOutput { .. }))
        );
    }

    #[test]
    fn turn_number_renders_after_top_padding() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("first".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some("answer".into()),
                    tool_calls: vec![ToolCall {
                        id: "shell".into(),
                        kind: ToolCallKind::Function,
                        function_name: shell::NAME.into(),
                        arguments: "{}".into(),
                    }],
                }),
                Message::Tool(vec![ToolResult {
                    tool_call_id: "shell".into(),
                    content: "output".into(),
                }]),
            ],
        );

        let mut transcript = Transcript::default();
        transcript.cursor = walker::Position {
            msg_idx: 1,
            row: walker::Row::Text {
                is_reasoning: false,
                span: 0..6,
            },
        };
        transcript.row_cache = VecDeque::from([
            walker::Position {
                msg_idx: 1,
                row: walker::Row::Padding,
            },
            walker::Position {
                msg_idx: 2,
                row: walker::Row::ToolOutput {
                    call_idx: 0,
                    row: 0,
                    last_row: 2,
                },
            },
        ]);
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|frame| transcript.render(frame, frame.area(), &mut session))
            .unwrap();
        let rows = terminal
            .backend()
            .buffer()
            .content
            .chunks(80)
            .map(|cells| cells.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>();
        assert!(rows.iter().any(|row| row.starts_with(" 1 │")), "{rows:#?}");
    }

    #[test]
    fn pending_and_completed_write_render_the_same_diff_rows() {
        let mut session = session();
        let mut transcript = Transcript::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        redraw_transcript(&mut transcript, Rect::new(0, 0, 80, 12), &mut session);
        let diff = concat!(
            "--- /dev/null\n",
            "+++ hello.txt\n",
            "@@ -0,0 +1,1 @@\n",
            "+Hello, world!\n",
            "\\ No newline at end of file\n",
        );
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: "write".into(),
                    kind: ToolCallKind::Function,
                    function_name: write::NAME.into(),
                    arguments: r#"{"path":"hello.txt","content":"Hello, world!"}"#.into(),
                }],
            })],
        );

        transcript.cache_diff(Some(crate::tools::write::file_diff(
            None,
            Some("hello.txt"),
            "",
            "Hello, world!",
        )));
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let pending = terminal
            .backend()
            .buffer()
            .content
            .chunks(80)
            .map(|cells| cells.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>();
        assert!(terminal.backend().buffer().content.iter().any(|cell| {
            cell.symbol() == "+" && !cell.modifier.contains(ratatui::style::Modifier::DIM)
        }));

        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Tool(vec![ToolResult {
                tool_call_id: "write".into(),
                content: serde_json::to_string(&FileMutationOutput {
                    summary: "Created 13 bytes.".into(),
                    diff: diff.into(),
                })
                .unwrap(),
            }])],
        );
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let completed = terminal
            .backend()
            .buffer()
            .content
            .chunks(80)
            .map(|cells| cells.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>();

        assert_eq!(pending, completed);
        assert!(pending.iter().any(|row| row.contains("hello.txt (+1 -0)")));
        assert!(pending.iter().any(|row| row.contains("+ Hello, world!")));
        assert!(!pending.iter().any(|row| row.contains("@@")));
        assert!(!pending.iter().any(|row| row.contains("No newline")));
        assert!(terminal.backend().buffer().content.iter().any(|cell| {
            cell.symbol() == "+" && !cell.modifier.contains(ratatui::style::Modifier::DIM)
        }));
    }

    #[test]
    fn cached_and_uncached_scrolling_visit_the_same_rows_across_resizes_and_bounds() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("question words ".repeat(30)),
                Message::Assistant(AssistantMessage {
                    reasoning: Some("reasoning words ".repeat(30)),
                    content: Some(format!(
                        "{}\r\n{}",
                        "answer words ".repeat(30),
                        "last words ".repeat(30)
                    )),
                    tool_calls: Vec::new(),
                }),
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
                    content: serde_json::to_string(&FileMutationOutput {
                        summary: "changed".into(),
                        diff: format!(
                            "--- output.txt\n+++ output.txt\n@@ -1,1 +1,1 @@\n-old\n+{}\n",
                            "wrapped diff words ".repeat(6)
                        )
                        .into(),
                    })
                    .unwrap(),
                }]),
            ],
        );
        let mut cached = Transcript::default();
        let mut uncached = Transcript::default();
        let mut used_soft_cache = false;
        for width in [30, 20, 40] {
            let mut terminal = Terminal::new(TestBackend::new(width, 13)).unwrap();
            let area = Rect::new(0, 0, width, 13);
            cached.handle_event(&Event::Redraw, area, &mut session);
            uncached.handle_event(&Event::Redraw, area, &mut session);
            for key in [
                KeyCode::End,
                KeyCode::Up,
                KeyCode::Up,
                KeyCode::Down,
                KeyCode::PageUp,
                KeyCode::PageDown,
                KeyCode::Home,
                KeyCode::Down,
                KeyCode::End,
                KeyCode::Up,
            ] {
                let event = Event::Key(KeyEvent::new(key, KeyModifiers::NONE));
                uncached.soft_cache = None;
                uncached.row_cache.clear();
                cached.handle_event(&event, area, &mut session);
                uncached.handle_event(&event, area, &mut session);
                assert_eq!(cached.cursor, uncached.cursor);
                used_soft_cache |= cached.soft_cache.is_some();
                uncached.soft_cache = None;
                uncached.row_cache.clear();
                uncached.size = None;
                draw_transcript(&mut cached, &mut terminal, &mut session);
                draw_transcript(&mut uncached, &mut terminal, &mut session);
                assert_eq!(cached.cursor, uncached.cursor);
                assert_eq!(cached.row_cache, uncached.row_cache);
            }
        }
        assert!(used_soft_cache);
    }

    #[test]
    fn streamed_real_line_updates_match_a_fresh_walk_at_every_scroll_position() {
        let threshold = format!("{} {}", "a".repeat(40), "b".repeat(40));
        let joining_grapheme = format!("zzzzzz abcdef{}", "👩\u{200d}".repeat(6));
        for chunks in [
            // Reflow within a word, followed by a CRLF split across updates.
            vec![
                ("", "first line\nsecond line\nhello wor hello wor hello wor"),
                ("", "ld"),
                ("", "\n"),
                ("", "next\r"),
                ("", "\n with words"),
            ],
            // An append that changes the wrap count.
            vec![("", threshold.as_str()), ("", "b"), ("", " extra")],
            // A grapheme cluster split across updates.
            vec![("", joining_grapheme.as_str()), ("", "〰"), ("", "\tmore")],
            // The boundary update finishes reasoning and starts content.
            vec![
                ("thinking wor", ""),
                ("ds", ""),
                (" later", "answer wor"),
                ("", "ds"),
                ("", " and more"),
                ("", "\nlast"),
            ],
        ] {
            for width in [0, 1, 2, 10, 80] {
                for height in [1, 3, 12] {
                    let mut agent = Agent::new(StreamingBackend {
                        events: chunks
                            .iter()
                            .map(|(reasoning, content)| BackendEvent {
                                reasoning: Some((*reasoning).into()),
                                content: Some((*content).into()),
                                ..Default::default()
                            })
                            .chain([BackendEvent {
                                content: Some(" final".into()),
                                finished: true,
                                ..Default::default()
                            }])
                            .collect(),
                    });
                    agent.start_turn("question\nwith history".into());
                    agent.start_step(&[], true);
                    let mut views = Vec::new();
                    let mut transcript = Transcript::default();
                    transcript.update_viewport(
                        agent.messages(),
                        width,
                        height,
                        agent.version(),
                        true,
                        None,
                    );
                    views.push(transcript);
                    let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
                    for update in 0..=chunks.len() {
                        let delta = match agent.poll_step(&mut cx) {
                            Poll::Ready(StepEvent::Updated {
                                n_added_reasoning: reasoning,
                                n_added_content: content,
                            }) => {
                                assert_eq!(
                                    (reasoning, content),
                                    (chunks[update].0.len(), chunks[update].1.len())
                                );
                                Some((reasoning, content))
                            }
                            Poll::Ready(StepEvent::Finished) => {
                                assert_eq!(update, chunks.len());
                                None
                            }
                            event => panic!("unexpected step event: {event:?}"),
                        };
                        let mut diffs = DiffCache::default();
                        let mut markdown = markdown::MarkdownCache::default();
                        let mut outputs = OutputCache::default();
                        let mut walker = walker::Walker::new(
                            agent.messages(),
                            walker::Position::start(),
                            width,
                            None,
                            &mut diffs,
                            &mut markdown,
                            &mut outputs,
                        );
                        let mut rows = vec![walker.cur.clone()];
                        while walker.next().is_some() {
                            rows.push(walker.cur.clone());
                        }
                        let bottom = (rows.len() - 1).saturating_sub(height);
                        for transcript in &mut views {
                            let row = if transcript.cursor == transcript.bottom.cursor {
                                bottom
                            } else {
                                transcript.row.min(bottom)
                            };
                            transcript.update_viewport(
                                agent.messages(),
                                width,
                                height,
                                agent.version(),
                                delta.is_some(),
                                delta,
                            );
                            assert_eq!((transcript.row, transcript.bottom.row), (row, bottom));
                            assert_eq!(transcript.cursor, rows[row]);
                            assert_eq!(transcript.bottom.cursor, rows[bottom]);
                            assert_eq!(
                                transcript.row_cache,
                                rows.iter()
                                    .skip(row + 1)
                                    .take(height)
                                    .cloned()
                                    .collect::<VecDeque<_>>()
                            );
                        }
                        if update == 0 {
                            // Keep a view at every scroll position, including cached rows within the changing real line.
                            for position in 0..bottom {
                                let mut transcript = Transcript::default();
                                transcript.update_viewport(
                                    agent.messages(),
                                    width,
                                    height,
                                    agent.version(),
                                    true,
                                    None,
                                );
                                let mut walker = walker::Walker::new(
                                    agent.messages(),
                                    transcript.cursor.clone(),
                                    width,
                                    transcript.soft_cache.take(),
                                    &mut transcript.diff_cache,
                                    &mut transcript.markdown,
                                    &mut transcript.output_cache,
                                );
                                walker.retreat(bottom - position);
                                transcript.cursor = walker.cur;
                                transcript.soft_cache = walker.soft_cache;
                                transcript.row = position;
                                transcript.row_cache.clear();
                                transcript.fill_row_cache(agent.messages(), width, height);
                                views.push(transcript);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn retained_rows_that_shrink_when_parsed_match_a_fresh_walk() {
        // Publishing Markdown can shrink pending text. Include a stable prefix that remains visible as the view moves up.
        for (height, pending) in [
            (1, "### streaming heading wrapping across narrow rows"),
            (4, "### streaming heading wrapping across narrow rows"),
            (
                8,
                "first\n\nsecond\n\n[link](https://example.com/a/long/path/that/wraps)",
            ),
        ] {
            let width = 12;
            let mut agent = Agent::new(StreamingBackend {
                events: [
                    BackendEvent {
                        content: Some(pending.into()),
                        ..Default::default()
                    },
                    BackendEvent {
                        content: Some("\n\nafter\n".into()),
                        ..Default::default()
                    },
                    BackendEvent {
                        content: Some(" final".into()),
                        finished: true,
                        ..Default::default()
                    },
                ]
                .into(),
            });
            agent.start_turn("question".into());
            agent.start_step(&[], true);
            let mut transcript = markdown_transcript();
            let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
            stream_update(&mut agent, &mut transcript, &mut cx, width, height);

            let old_row = transcript.row;
            let old_cached = transcript.row_cache.clone();
            let mut views = {
                let mut diffs = DiffCache::default();
                let mut markdown = markdown::MarkdownCache::new(true);
                let mut outputs = OutputCache::default();
                markdown.sync(agent.messages(), true, width);
                let mut walker = walker::Walker::new(
                    agent.messages(),
                    Position::start(),
                    width,
                    None,
                    &mut diffs,
                    &mut markdown,
                    &mut outputs,
                );
                let mut rows = vec![walker.cur.clone()];
                while walker.next().is_some() {
                    rows.push(walker.cur.clone());
                }
                let bottom = (rows.len() - 1).saturating_sub(height);
                (0..=bottom)
                    .map(|position| {
                        let mut view = markdown_transcript();
                        view.update_viewport(
                            agent.messages(),
                            width,
                            height,
                            agent.version(),
                            true,
                            None,
                        );
                        let mut walker = walker::Walker::new(
                            agent.messages(),
                            view.cursor.clone(),
                            width,
                            view.soft_cache.take(),
                            &mut view.diff_cache,
                            &mut view.markdown,
                            &mut view.output_cache,
                        );
                        walker.retreat(bottom - position);
                        view.cursor = walker.cur;
                        view.soft_cache = walker.soft_cache;
                        view.row = position;
                        view.row_cache.clear();
                        view.fill_row_cache(agent.messages(), width, height);
                        view
                    })
                    .collect::<Vec<_>>()
            };

            let Poll::Ready(StepEvent::Updated {
                n_added_reasoning: reasoning,
                n_added_content: content,
            }) = agent.poll_step(&mut cx)
            else {
                panic!("expected an updated event")
            };
            for view in &mut views {
                view.update_viewport(
                    agent.messages(),
                    width,
                    height,
                    agent.version(),
                    true,
                    Some((reasoning, content)),
                );
            }
            transcript.update_viewport(
                agent.messages(),
                width,
                height,
                agent.version(),
                true,
                Some((reasoning, content)),
            );
            if height == 8 {
                assert!(transcript.row < old_row);
                assert!(
                    old_cached
                        .iter()
                        .any(|position| transcript.row_cache.contains(position))
                );
            }

            let mut diffs = DiffCache::default();
            let mut markdown = markdown::MarkdownCache::new(true);
            let mut outputs = OutputCache::default();
            markdown.sync(agent.messages(), true, width);
            let mut walker = walker::Walker::new(
                agent.messages(),
                Position::start(),
                width,
                None,
                &mut diffs,
                &mut markdown,
                &mut outputs,
            );
            let mut rows = vec![walker.cur.clone()];
            while walker.next().is_some() {
                rows.push(walker.cur.clone());
            }
            let bottom = (rows.len() - 1).saturating_sub(height);
            for view in &mut views {
                let row = if view.cursor == view.bottom.cursor {
                    bottom
                } else {
                    view.row.min(bottom)
                };
                assert_eq!((view.row, view.bottom.row), (row, bottom));
                assert_eq!(view.cursor, rows[row]);
                assert_eq!(
                    view.row_cache,
                    rows.iter()
                        .skip(row + 1)
                        .take(height)
                        .cloned()
                        .collect::<VecDeque<_>>()
                );
            }
        }
    }

    #[test]
    fn usage_only_updates_consume_the_version_without_touching_the_view() {
        let mut agent = Agent::new(StreamingBackend {
            events: [
                BackendEvent {
                    content: Some("```rust\nlet a = 1;\n".into()),
                    ..Default::default()
                },
                // Usage metadata carries no text delta.
                BackendEvent {
                    usage: Some(TokenUsage {
                        output_tokens: 1,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("let b = 2;\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("```\n".into()),
                    finished: true,
                    ..Default::default()
                },
            ]
            .into(),
        });
        agent.start_turn("question".into());
        agent.start_step(&[], true);
        let mut scrolled = markdown_transcript();
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());

        // Scroll above the streamed code lines.
        stream_update(&mut agent, &mut scrolled, &mut cx, 60, 4);
        scrolled.row = 0;
        scrolled.cursor = Position::start();
        scrolled.soft_cache = None;
        scrolled.row_cache.clear();
        scrolled.fill_row_cache(agent.messages(), 60, 4);
        let cached = scrolled.row_cache.clone();
        let cursor = scrolled.cursor.clone();

        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(StepEvent::Updated {
                n_added_reasoning: 0,
                n_added_content: 0
            })
        );
        let version = agent.version();
        scrolled.update_viewport(agent.messages(), 60, 4, version, true, Some((0, 0)));
        assert_eq!(scrolled.cursor, cursor);
        assert_eq!(scrolled.row_cache, cached);
        assert_eq!(scrolled.bottom.version, version);

        // The following append is below the viewport.
        stream_update(&mut agent, &mut scrolled, &mut cx, 60, 4);
        assert_eq!(scrolled.cursor, cursor);
        assert_eq!(scrolled.row_cache, cached);
    }

    #[test]
    fn streamed_markdown_updates_match_a_fresh_walk_at_every_scroll_position() {
        for width in [10, 60] {
            for height in [1, 8] {
                let mut agent = Agent::new(StreamingBackend {
                    events: [
                        BackendEvent {
                            reasoning: Some("thinking about the code\n".into()),
                            ..Default::default()
                        },
                        // An open paragraph renders through the document's retained-source tail.
                        BackendEvent {
                            content: Some("streaming paragraph".into()),
                            ..Default::default()
                        },
                        // A fenced block publishes the paragraph, a separator, and immediate code lines.
                        BackendEvent {
                            content: Some("\n\n```rust\nlet a = 1;\n".into()),
                            ..Default::default()
                        },
                        // A table streams open.
                        BackendEvent {
                            content: Some(
                                "```\n\n| a | longer header |\n|---|---|\n| 1 | 2 |\n".into(),
                            ),
                            ..Default::default()
                        },
                        // The table closes and aligns, rewriting published rows.
                        BackendEvent {
                            content: Some("| 222 | 3 |\n".into()),
                            ..Default::default()
                        },
                        BackendEvent {
                            content: Some(" final".into()),
                            finished: true,
                            ..Default::default()
                        },
                    ]
                    .into(),
                });
                agent.start_turn("question\nwith history".into());
                agent.start_step(&[], true);
                let mut views = Vec::new();
                let mut transcript = markdown_transcript();
                transcript.update_viewport(
                    agent.messages(),
                    width,
                    height,
                    agent.version(),
                    true,
                    None,
                );
                views.push(transcript);
                let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
                for update in 0..=5 {
                    let delta = match agent.poll_step(&mut cx) {
                        Poll::Ready(StepEvent::Updated {
                            n_added_reasoning: reasoning,
                            n_added_content: content,
                        }) => Some((reasoning, content)),
                        Poll::Ready(StepEvent::Finished) => {
                            assert_eq!(update, 5);
                            None
                        }
                        event => panic!("unexpected step event: {event:?}"),
                    };
                    let mut diffs = DiffCache::default();
                    let mut markdown = markdown::MarkdownCache::new(true);
                    let mut outputs = OutputCache::default();
                    markdown.sync(agent.messages(), delta.is_some(), width);
                    let mut walker = walker::Walker::new(
                        agent.messages(),
                        walker::Position::start(),
                        width,
                        None,
                        &mut diffs,
                        &mut markdown,
                        &mut outputs,
                    );
                    let mut rows = vec![walker.cur.clone()];
                    while walker.next().is_some() {
                        rows.push(walker.cur.clone());
                    }
                    let bottom = (rows.len() - 1).saturating_sub(height);
                    for transcript in &mut views {
                        let row = if transcript.cursor == transcript.bottom.cursor {
                            bottom
                        } else {
                            transcript.row.min(bottom)
                        };
                        transcript.update_viewport(
                            agent.messages(),
                            width,
                            height,
                            agent.version(),
                            delta.is_some(),
                            delta,
                        );
                        assert_eq!((transcript.row, transcript.bottom.row), (row, bottom));
                        assert_eq!(transcript.cursor, rows[row]);
                        assert_eq!(transcript.bottom.cursor, rows[bottom]);
                        assert_eq!(
                            transcript.row_cache,
                            rows.iter()
                                .skip(row + 1)
                                .take(height)
                                .cloned()
                                .collect::<VecDeque<_>>()
                        );
                    }
                    if update <= 1 {
                        // Include every scroll position while Markdown content begins streaming.
                        for position in 0..bottom {
                            let mut transcript = markdown_transcript();
                            transcript.update_viewport(
                                agent.messages(),
                                width,
                                height,
                                agent.version(),
                                true,
                                None,
                            );
                            let mut walker = walker::Walker::new(
                                agent.messages(),
                                transcript.cursor.clone(),
                                width,
                                transcript.soft_cache.take(),
                                &mut transcript.diff_cache,
                                &mut transcript.markdown,
                                &mut transcript.output_cache,
                            );
                            walker.retreat(bottom - position);
                            transcript.cursor = walker.cur;
                            transcript.soft_cache = walker.soft_cache;
                            transcript.row = position;
                            transcript.row_cache.clear();
                            transcript.fill_row_cache(agent.messages(), width, height);
                            views.push(transcript);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scrolling_across_reasoning_and_content_visits_the_expected_text_rows() {
        let mut session = session();
        let reasoning = (0..8).map(|i| format!("reasoning {i}")).collect::<Vec<_>>();
        let content = (0..20).map(|i| format!("content {i}")).collect::<Vec<_>>();
        crate::test_support::append_messages(
            &mut session.agent,
            [
                Message::User("question".into()),
                Message::Assistant(AssistantMessage {
                    reasoning: Some(reasoning.join("\n")),
                    content: Some(content.join("\n")),
                    tool_calls: Vec::new(),
                }),
            ],
        );
        let expected = reasoning.iter().chain(&content).collect::<Vec<_>>();
        let mut transcript = Transcript::default();
        transcript.cursor = walker::Position {
            msg_idx: 1,
            row: walker::Row::RoleHeader,
        };
        let mut terminal = Terminal::new(TestBackend::new(80, 14)).unwrap();
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        for (key, offsets) in [
            (KeyCode::Down, (0..=12).collect::<Vec<_>>()),
            (KeyCode::Up, (0..12).rev().collect::<Vec<_>>()),
        ] {
            for offset in offsets {
                if offset != 0 || key == KeyCode::Up {
                    transcript.handle_event(
                        &Event::Key(KeyEvent::new(key, KeyModifiers::NONE)),
                        Rect::new(0, 0, 80, 14),
                        &mut session,
                    );
                }
                assert_eq!(transcript.row_cache.len(), 5);
                for (cursor, expected) in transcript.row_cache.iter().zip(&expected[offset..]) {
                    let walker::Row::Text { is_reasoning, span } = &cursor.row else {
                        panic!("expected a text row")
                    };
                    let source = walker::get_msg_text(
                        session.agent.messages().get(cursor.msg_idx).unwrap(),
                        *is_reasoning,
                    );
                    assert_eq!(&source[span.clone()], expected.as_str());
                }
            }
        }
    }

    fn markdown_transcript() -> Transcript {
        Transcript::new(true)
    }

    #[test]
    fn markdown_links_keep_cell_widths_and_ids_across_scrolling() {
        use ratatui::buffer::{CellDiffOption, CellWidth};

        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some(format!(
                    "[{}](https://example.com) plain\n",
                    "界 e\u{301} **bold** words ".repeat(16)
                )),
                tool_calls: Vec::new(),
            })],
        );
        let mut transcript = markdown_transcript();
        let mut terminal = Terminal::new(TestBackend::new(32, 16)).unwrap();
        let mut id = None;
        let mut linked_rows = std::collections::HashSet::new();
        let mut wide = false;
        let mut combined = false;
        let mut plain = false;
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        for key in [KeyCode::Home, KeyCode::Down, KeyCode::End, KeyCode::Up] {
            transcript.handle_event(
                &Event::Key(KeyEvent::new(key, KeyModifiers::NONE)),
                Rect::new(0, 0, 32, 16),
                &mut session,
            );
            draw_transcript(&mut transcript, &mut terminal, &mut session);
            let buffer = terminal.backend().buffer();
            for y in 0..buffer.area.height {
                let mut x = 0;
                while x < buffer.area.width {
                    let cell = &buffer[(x, y)];
                    if let Some(link) = cell.symbol().strip_prefix("\x1b]8;id=") {
                        let (opening, text) = link.split_once("\x1b\\").unwrap();
                        let (current_id, url) = opening.split_once(';').unwrap();
                        assert_eq!(id.get_or_insert_with(|| current_id.to_owned()), current_id);
                        assert_eq!(url, "https://example.com");
                        let text = text.strip_suffix("\x1b]8;;\x1b\\").unwrap();
                        assert_eq!(cell.cell_width(), text.cell_width());
                        assert!(matches!(cell.diff_option, CellDiffOption::ForcedWidth(_)));
                        wide |= text == "界";
                        combined |= text == "e\u{301}";
                        linked_rows.insert(y);
                    } else {
                        plain |= cell.symbol() == "p";
                        assert!(!cell.symbol().contains("\x1b]8;"));
                    }
                    x += cell.cell_width().max(1);
                }
            }
        }
        assert!(wide && combined && plain);
        assert!(linked_rows.len() > 1);
    }

    // This checks buffer diffs and emitted OSC8 bytes; it does not verify terminal interpretation or clicking.
    // Possible follow-up: interpret consecutive frames in an OSC8-aware terminal/emulator and inspect cell links.
    // Include URL-only changes, linked-to-plain and wide-to-narrow replacements, clipping, and partial redraws.
    // Intended invariant: each visible cell has its current link, and link state cannot leak into unrelated text.
    #[test]
    fn markdown_link_target_changes_emit_osc8_with_the_same_label() {
        use ratatui::{TerminalOptions, Viewport, backend::CrosstermBackend};

        let area = Rect::new(0, 0, 40, 16);
        let mut output = Vec::new();
        let mut terminal = Terminal::with_options(
            CrosstermBackend::new(&mut output),
            TerminalOptions {
                viewport: Viewport::Fixed(area),
            },
        )
        .unwrap();
        let mut previous: Option<ratatui::buffer::Buffer> = None;
        for url in ["https://example.com/old", "https://example.com/界"] {
            let mut session = session();
            crate::test_support::append_messages(
                &mut session.agent,
                [Message::Assistant(AssistantMessage {
                    reasoning: None,
                    content: Some(format!("[界e\u{301}Z]({url}) plain\n")),
                    tool_calls: Vec::new(),
                })],
            );
            let mut transcript = markdown_transcript();
            redraw_transcript(&mut transcript, area, &mut session);
            let current = terminal
                .draw(|frame| transcript.render(frame, area, &mut session))
                .unwrap()
                .buffer
                .clone();
            if let Some(previous) = previous {
                let diff = previous.diff(&current);
                assert_eq!(diff.len(), 3);
                assert!(
                    diff.iter().all(|(_, _, cell)| cell
                        .symbol()
                        .contains("https://example.com/%E7%95%8C"))
                );
            }
            previous = Some(current);
        }
        drop(terminal);
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains(";https://example.com/old\x1b\\界\x1b]8;;\x1b\\"));
        assert!(output.contains(";https://example.com/%E7%95%8C\x1b\\界\x1b]8;;\x1b\\"));
        assert_eq!(output.matches("\x1b]8;id=").count(), 6);
        assert_eq!(output.matches("\x1b]8;;\x1b\\").count(), 6);
    }

    /// Walks every cursor with a Markdown-enabled walker, returning the populated cache for
    /// follow-up line reads.
    fn markdown_cursors(
        messages: MessageView<'_>,
        width: usize,
    ) -> (Vec<walker::Position>, markdown::MarkdownCache) {
        let mut diffs = DiffCache::default();
        let mut markdown = markdown::MarkdownCache::new(true);
        let mut outputs = OutputCache::default();
        markdown.initialize_for_test(messages);
        let mut walker = walker::Walker::new(
            messages,
            walker::Position::start(),
            width,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        let mut cursors = vec![walker.cur.clone()];
        while walker.next().is_some() {
            cursors.push(walker.cur.clone());
        }
        (cursors, markdown)
    }

    /// The visible text of every Markdown row, in order.
    fn markdown_row_texts(messages: MessageView<'_>, width: usize) -> Vec<String> {
        let (cursors, mut markdown) = markdown_cursors(messages, width);
        cursors
            .iter()
            .filter_map(|cursor| match &cursor.row {
                walker::Row::Markdown { position } => Some(
                    markdown
                        .render(
                            messages,
                            cursor.msg_idx,
                            super::get_msg_text(&messages[cursor.msg_idx], false),
                            width,
                            position,
                        )
                        .text(),
                ),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn disabled_markdown_keeps_plain_assistant_rows() {
        let agent = agent_with([Message::Assistant(AssistantMessage {
            reasoning: None,
            content: Some("# Title\n\n*emph* text\n".into()),
            tool_calls: Vec::new(),
        })]);
        // A default transcript's walker emits plain text rows covering the source's real lines.
        let mut diffs = DiffCache::default();
        let mut markdown = markdown::MarkdownCache::default();
        let mut outputs = OutputCache::default();
        let mut walker = walker::Walker::new(
            agent.messages(),
            walker::Position::start(),
            80,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        let mut spans = Vec::new();
        while let Some(row) = walker.next() {
            if let walker::Row::Text { span, .. } = row {
                spans.push(span);
            }
        }
        assert_eq!(spans, [0..7, 8..8, 9..20, 21..21]);
    }

    #[test]
    fn reasoning_renders_plain_while_content_renders_markdown() {
        let agent = agent_with([Message::Assistant(AssistantMessage {
            reasoning: Some("thinking deeply".into()),
            content: Some("# Title\n\nbody *text*\n".into()),
            tool_calls: Vec::new(),
        })]);
        let (cursors, _) = markdown_cursors(agent.messages(), 80);
        let reasoning_spans: Vec<std::ops::Range<usize>> = cursors
            .iter()
            .filter_map(|cursor| match &cursor.row {
                walker::Row::Text {
                    is_reasoning: true,
                    span,
                } => Some(span.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning_spans, [0..15]);
        let texts = markdown_row_texts(agent.messages(), 80);
        assert_eq!(texts, ["# Title", "", "body text"]);
        // Reasoning precedes content in the walk.
        let reasoning_pos = cursors
            .iter()
            .position(|cursor| {
                matches!(
                    &cursor.row,
                    walker::Row::Text {
                        is_reasoning: true,
                        ..
                    }
                )
            })
            .unwrap();
        let markdown_pos = cursors
            .iter()
            .position(|cursor| matches!(&cursor.row, walker::Row::Markdown { .. }))
            .unwrap();
        assert!(reasoning_pos < markdown_pos);
    }

    #[test]
    fn inline_code_foreground_survives_soft_wrapping_without_a_background() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some(
                    "before `ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ` after\n\ntail\n".to_owned(),
                ),
                tool_calls: Vec::new(),
            })],
        );
        let mut transcript = markdown_transcript();
        let mut terminal = Terminal::new(TestBackend::new(40, 16)).unwrap();
        draw_transcript(&mut transcript, &mut terminal, &mut session);

        // The code span wraps across rows; every code cell keeps its foreground without gaining
        // the fenced-code background.
        let buffer = terminal.backend().buffer();
        let mut code_rows = 0;
        for y in 0..buffer.area.height {
            let cells: Vec<(String, Color, Color)> = (0..buffer.area.width)
                .map(|x| {
                    let cell = &buffer[(x, y)];
                    (cell.symbol().to_owned(), cell.fg, cell.bg)
                })
                .collect();
            if cells.iter().any(|(symbol, _, _)| symbol == "Z") {
                code_rows += 1;
                for (symbol, foreground, background) in cells {
                    if symbol == "Z" {
                        assert_eq!(foreground, Color::Rgb(0x82, 0xD2, 0xCE));
                        assert_ne!(background, CODE_BG);
                    }
                }
            }
        }
        assert!(code_rows >= 2, "the code span must wrap across rows");
    }

    #[test]
    fn code_block_background_fills_nonempty_and_empty_content_rows() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some("```\nblock-code\n\n```\n\nordinary\n".to_owned()),
                tool_calls: Vec::new(),
            })],
        );
        let mut transcript = markdown_transcript();
        let width = 40;
        let mut terminal = Terminal::new(TestBackend::new(width, 16)).unwrap();
        draw_transcript(&mut transcript, &mut terminal, &mut session);

        let buffer = terminal.backend().buffer();
        let content_x = super::gutter_width(session.agent.messages()) as u16 + 2;
        let content_width = width.saturating_sub(content_x + 1);
        let row_text = |y| {
            (content_x..content_x + content_width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        };
        let code_row = (0..buffer.area.height)
            .find(|&y| row_text(y).contains("block-code"))
            .unwrap();
        let ordinary_row = (0..buffer.area.height)
            .find(|&y| row_text(y).contains("ordinary"))
            .unwrap();

        for y in [code_row, code_row + 1] {
            for x in content_x..content_x + content_width {
                assert_eq!(
                    buffer[(x, y)].bg,
                    CODE_BG,
                    "missing code background at ({x}, {y})"
                );
            }
        }
        assert!(
            (content_x..content_x + content_width).all(|x| buffer[(x, ordinary_row)].bg != CODE_BG)
        );
    }

    #[test]
    fn reasoning_rows_stay_muted_while_markdown_rows_do_not() {
        let mut session = session();
        crate::test_support::append_messages(
            &mut session.agent,
            [Message::Assistant(AssistantMessage {
                reasoning: Some("plain reasoning".into()),
                content: Some("plain body".into()),
                tool_calls: Vec::new(),
            })],
        );
        let mut transcript = markdown_transcript();
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).unwrap();
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let buffer = terminal.backend().buffer();
        let row_text = |y: u16| {
            (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol().to_owned())
                .collect::<String>()
        };
        let reasoning_row = (0..buffer.area.height)
            .find(|y| row_text(*y).contains("plain reasoning"))
            .unwrap();
        let body_row = (0..buffer.area.height)
            .find(|y| row_text(*y).contains("plain body"))
            .unwrap();
        let has_muted = |y: u16| (0..buffer.area.width).any(|x| buffer[(x, y)].fg == MUTED);
        assert!(has_muted(reasoning_row), "reasoning renders muted");
        assert!(
            !has_muted(body_row),
            "markdown body text keeps the default style"
        );
    }

    #[test]
    fn streamed_markdown_publishes_immediately_and_finishes() {
        let mut agent = Agent::new(StreamingBackend {
            events: [
                // An open paragraph appears through the document's retained-source tail.
                BackendEvent {
                    content: Some("streaming paragraph".into()),
                    ..Default::default()
                },
                // Immediate code lines publish before the closing fence.
                BackendEvent {
                    content: Some("\n\n```rust\nlet a = 1;\nlet b = 2;\n".into()),
                    ..Default::default()
                },
                // The code block closes and a closed paragraph follows.
                BackendEvent {
                    content: Some("```\nvisible now\n\n".into()),
                    ..Default::default()
                },
                // A deferred trailing paragraph, then the stream ends.
                BackendEvent {
                    content: Some("tail".into()),
                    ..Default::default()
                },
                BackendEvent {
                    finished: true,
                    ..Default::default()
                },
            ]
            .into(),
        });
        agent.start_turn("question".into());
        agent.start_step(&[], true);
        let mut transcript = markdown_transcript();
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());

        stream_update(&mut agent, &mut transcript, &mut cx, 60, 8);
        assert_eq!(
            markdown_row_texts(agent.messages(), 60),
            ["streaming paragraph"]
        );
        assert!(
            !transcript.row_cache.iter().any(|cursor| {
                cursor.msg_idx == 1
                    && matches!(
                        cursor.row,
                        walker::Row::Text {
                            is_reasoning: false,
                            ..
                        }
                    )
            }),
            "retained source uses the same Markdown row type as emitted content"
        );

        // The fenced block closes the paragraph, publishing it with a separator before the code.
        stream_update(&mut agent, &mut transcript, &mut cx, 60, 8);
        assert_eq!(
            markdown_row_texts(agent.messages(), 60),
            ["streaming paragraph", "", "let a = 1;", "let b = 2;"]
        );
        assert!(
            !transcript.row_cache.iter().any(|cursor| {
                cursor.msg_idx == 1
                    && matches!(
                        cursor.row,
                        walker::Row::Text {
                            is_reasoning: false,
                            ..
                        }
                    )
            }),
            "immediate code lines render only as published Markdown rows"
        );

        // The closing fence and a closed trailing paragraph.
        stream_update(&mut agent, &mut transcript, &mut cx, 60, 8);
        assert_eq!(
            markdown_row_texts(agent.messages(), 60),
            [
                "streaming paragraph",
                "",
                "let a = 1;",
                "let b = 2;",
                "",
                "visible now"
            ]
        );

        // Finish publishes the open trailing paragraph; the view matches a fresh walk.
        stream_update(&mut agent, &mut transcript, &mut cx, 60, 8);
        assert_eq!(agent.poll_step(&mut cx), Poll::Ready(StepEvent::Finished));
        transcript.update_viewport(agent.messages(), 60, 8, agent.version(), false, None);
        assert!(
            !transcript.row_cache.iter().any(|cursor| matches!(
                cursor.row,
                walker::Row::Text {
                    is_reasoning: false,
                    ..
                }
            )),
            "completion replaces every pending row with rendered ones"
        );
        assert_eq!(
            markdown_row_texts(agent.messages(), 60),
            [
                "streaming paragraph",
                "",
                "let a = 1;",
                "let b = 2;",
                "",
                "visible now",
                "",
                "tail"
            ]
        );
        let mut fresh = markdown_transcript();
        fresh.update_viewport(agent.messages(), 60, 8, agent.version(), false, None);
        assert_eq!(transcript.cursor, fresh.cursor);
        assert_eq!(transcript.row, fresh.row);
        assert_eq!(transcript.row_cache, fresh.row_cache);
    }

    #[test]
    fn markdown_appends_preserve_cached_viewport_rows() {
        let mut agent = Agent::new(StreamingBackend {
            events: [
                BackendEvent {
                    content: Some("first paragraph\n\nsecond paragraph\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("\n\nthird paragraph\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("\n\nfourth paragraph\n".into()),
                    ..Default::default()
                },
            ]
            .into(),
        });
        agent.start_turn(format!("{} ", "context words").repeat(40));
        agent.start_step(&[], true);
        let width = 60;
        let height = 8;
        let mut transcript = markdown_transcript();
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());

        stream_update(&mut agent, &mut transcript, &mut cx, width, height);
        stream_update(&mut agent, &mut transcript, &mut cx, width, height);
        // Compare the bottom view with a fresh walk after the append.
        let mut fresh = markdown_transcript();
        fresh.update_viewport(agent.messages(), width, height, agent.version(), true, None);
        assert_eq!(transcript.cursor, fresh.cursor);
        assert_eq!(transcript.row_cache, fresh.row_cache);

        // Check a scrolled view while retained source becomes parsed output.
        let mut scrolled = markdown_transcript();
        scrolled.update_viewport(agent.messages(), width, height, agent.version(), true, None);
        let mut diffs = DiffCache::default();
        let mut markdown = markdown::MarkdownCache::new(true);
        let mut outputs = OutputCache::default();
        markdown.sync(agent.messages(), true, width);
        let mut walker = walker::Walker::new(
            agent.messages(),
            scrolled.cursor.clone(),
            width,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        for _ in 0..5 {
            assert!(walker.previous());
        }
        scrolled.cursor = walker.cur;
        scrolled.soft_cache = walker.soft_cache.take();
        scrolled.row_cache.clear();
        scrolled.fill_row_cache(agent.messages(), width, height);
        let row = scrolled.row;
        let cursor = scrolled.cursor.clone();
        stream_update(&mut agent, &mut scrolled, &mut cx, width, height);
        assert_eq!(scrolled.cursor, cursor);
        assert_eq!(scrolled.row, row);
        let mut diffs = DiffCache::default();
        let mut markdown = markdown::MarkdownCache::new(true);
        let mut outputs = OutputCache::default();
        markdown.sync(agent.messages(), true, width);
        let mut walker = walker::Walker::new(
            agent.messages(),
            cursor.clone(),
            width,
            None,
            &mut diffs,
            &mut markdown,
            &mut outputs,
        );
        let mut fresh_rows = VecDeque::new();
        for _ in 0..height {
            if walker.next().is_none() {
                break;
            }
            fresh_rows.push_back(walker.cur.clone());
        }
        assert_eq!(scrolled.row_cache, fresh_rows);
    }

    #[test]
    fn markdown_completion_with_tool_calls_places_tools_after_content() {
        let mut agent = Agent::new(StreamingBackend {
            events: [
                BackendEvent {
                    content: Some("explanation\n\n```\ncode\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    tool_calls: vec![kyun_agent::ToolCallDelta {
                        index: 0,
                        id: Some("call".into()),
                        kind: Some(ToolCallKind::Function),
                        function_name: Some(read::NAME.into()),
                        arguments: Some("{\"path\":\"x\"}".into()),
                    }],
                    finished: true,
                    ..Default::default()
                },
            ]
            .into(),
        });
        agent.start_turn("question".into());
        agent.start_step(&[], true);
        let mut transcript = markdown_transcript();
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
        stream_update(&mut agent, &mut transcript, &mut cx, 60, 10);
        assert_eq!(agent.poll_step(&mut cx), Poll::Ready(StepEvent::ToolCalls));
        transcript.update_viewport(agent.messages(), 60, 10, agent.version(), false, None);

        // Content rows precede the padding and the tool summary rows.
        let (cursors, _) = markdown_cursors(agent.messages(), 60);
        let content_positions: Vec<usize> = cursors
            .iter()
            .enumerate()
            .filter_map(|(position, cursor)| {
                matches!(cursor.row, walker::Row::Markdown { .. }).then_some(position)
            })
            .collect();
        let tool_positions: Vec<usize> = cursors
            .iter()
            .enumerate()
            .filter_map(|(position, cursor)| {
                matches!(cursor.row, walker::Row::ToolHeader { .. }).then_some(position)
            })
            .collect();
        assert!(!content_positions.is_empty());
        assert!(!tool_positions.is_empty());
        assert!(content_positions.last().unwrap() < &tool_positions[0]);
        let texts = markdown_row_texts(agent.messages(), 60);
        assert!(texts.contains(&"explanation".to_owned()));
        assert!(texts.contains(&"code".to_owned()));
    }

    #[test]
    fn history_reset_clears_streaming_markdown_state() {
        let mut agent = Agent::new(StreamingBackend {
            events: [BackendEvent {
                content: Some("partial\n\nmore\n".into()),
                ..Default::default()
            }]
            .into(),
        });
        agent.start_turn("question".into());
        agent.start_step(&[], true);
        let mut transcript = markdown_transcript();
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
        stream_update(&mut agent, &mut transcript, &mut cx, 60, 8);
        assert!(
            transcript
                .row_cache
                .iter()
                .any(|cursor| matches!(cursor.row, walker::Row::Markdown { .. }))
        );

        // An interrupt pops the streamed message; the reset drops the streaming parse.
        let emptied = agent_with([]);
        let empty = emptied.messages();
        transcript.reset_after_history_change(empty);
        let mut fresh = markdown_transcript();
        fresh.update_viewport(empty, 60, 8, 1, false, None);
        transcript.update_viewport(empty, 60, 8, 1, false, None);
        assert_eq!(transcript.cursor, fresh.cursor);
        assert_eq!(transcript.row_cache, fresh.row_cache);
    }

    #[test]
    fn markdown_forward_and_backward_walks_visit_identical_rows() {
        let agent = agent_with([
            Message::User("question".into()),
            Message::Assistant(AssistantMessage {
                reasoning: Some("thinking\nmore thinking".into()),
                content: Some(
                    "# Heading\n\nparagraph with `code` and *emphasis*\n\n- item one\n- item two\n\n> quote\n\n```\nfenced\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n".to_owned(),
                ),
                tool_calls: Vec::new(),
            }),
        ]);
        let (cursors, _) = markdown_cursors(agent.messages(), 30);
        let mut diffs = DiffCache::default();
        let mut markdown = markdown::MarkdownCache::new(true);
        let mut outputs = OutputCache::default();
        markdown.initialize_for_test(agent.messages());
        let mut walker = walker::Walker::from_end(
            agent.messages(),
            30,
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
    fn markdown_resize_produces_a_cursor_in_the_new_layout() {
        let agent = agent_with([Message::Assistant(AssistantMessage {
            reasoning: None,
            content: Some(
                "- a very long list item that certainly wraps at narrow widths\n\nshort\n"
                    .to_owned(),
            ),
            tool_calls: Vec::new(),
        })]);
        for old_width in [10usize, 30, 80] {
            let (old_cursors, _) = markdown_cursors(agent.messages(), old_width);
            for width in [10usize, 30, 80] {
                let (new_cursors, _) = markdown_cursors(agent.messages(), width);
                for cursor in &old_cursors {
                    let mut diffs = DiffCache::default();
                    let mut markdown = markdown::MarkdownCache::new(true);
                    let mut outputs = OutputCache::default();
                    markdown.initialize_for_test(agent.messages());
                    let mut walker = walker::Walker::new(
                        agent.messages(),
                        cursor.clone(),
                        old_width,
                        None,
                        &mut diffs,
                        &mut markdown,
                        &mut outputs,
                    );
                    if matches!(cursor.row, walker::Row::Markdown { .. }) && walker.next().is_some()
                    {
                        assert!(walker.previous());
                        assert_eq!(&walker.cur, cursor);
                    }
                    walker.update_text(width);
                    assert!(
                        new_cursors.contains(&walker.cur),
                        "width {old_width} -> {width}, cursor {cursor:?}, result {:?}",
                        walker.cur
                    );
                }
            }
        }
    }

    #[test]
    fn hidden_viewport_keeps_cursor_valid() {
        use super::walker::{Position, Walker};
        for (initial, appended) in [
            (
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
                " fifteen sixteen seventeen",
            ),
            (
                "| name | long notes column |\n|---|---|\n| alpha | beta gamma delta |\n",
                "| epsilon zeta | eta theta iota kappa |\n\nfinal paragraph",
            ),
        ] {
            for follow in [false, true] {
                for restored_width in [10, 15] {
                    let mut agent = Agent::new(StreamingBackend {
                        events: [
                            BackendEvent {
                                content: Some(initial.into()),
                                ..Default::default()
                            },
                            BackendEvent {
                                content: Some(appended.into()),
                                ..Default::default()
                            },
                            BackendEvent {
                                finished: true,
                                ..Default::default()
                            },
                        ]
                        .into(),
                    });
                    agent.start_turn("question".into());
                    agent.start_step(&[], true);
                    let mut transcript = markdown_transcript();
                    let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
                    stream_update(&mut agent, &mut transcript, &mut cx, 10, 2);
                    if !follow {
                        let mut walker = Walker::new(
                            agent.messages(),
                            transcript.cursor.clone(),
                            10,
                            transcript.soft_cache.take(),
                            &mut transcript.diff_cache,
                            &mut transcript.markdown,
                            &mut transcript.output_cache,
                        );
                        assert!(walker.previous());
                        transcript.cursor = walker.cur;
                        transcript.soft_cache = walker.soft_cache;
                        transcript.row -= 1;
                        transcript.row_cache.clear();
                        transcript.fill_row_cache(agent.messages(), 10, 2);
                    }
                    transcript.update_viewport(
                        agent.messages(),
                        10,
                        0,
                        agent.version(),
                        true,
                        None,
                    );
                    assert_eq!(transcript.cursor == transcript.bottom.cursor, follow);
                    stream_update(&mut agent, &mut transcript, &mut cx, 10, 0);
                    assert_eq!(agent.poll_step(&mut cx), Poll::Ready(StepEvent::Finished));
                    transcript.update_viewport(
                        agent.messages(),
                        restored_width,
                        0,
                        agent.version(),
                        false,
                        None,
                    );
                    transcript.update_viewport(
                        agent.messages(),
                        restored_width,
                        2,
                        agent.version(),
                        false,
                        None,
                    );
                    let mut diffs = DiffCache::default();
                    let mut markdown = markdown::MarkdownCache::new(true);
                    let mut outputs = OutputCache::default();
                    markdown.sync(agent.messages(), false, restored_width);
                    let mut walker = Walker::new(
                        agent.messages(),
                        Position::start(),
                        restored_width,
                        None,
                        &mut diffs,
                        &mut markdown,
                        &mut outputs,
                    );
                    let mut positions = vec![walker.cur.clone()];
                    while walker.next().is_some() {
                        positions.push(walker.cur.clone());
                    }
                    assert_eq!(transcript.cursor, positions[transcript.row]);
                    if follow {
                        assert_eq!(transcript.cursor, transcript.bottom.cursor);
                    }
                }
            }
        }
    }

    #[test]
    fn zero_height_initialization_and_empty_resize_keep_bottom_following() {
        let mut agent = Agent::new(StreamingBackend {
            events: [
                BackendEvent {
                    content: Some("pending paragraph".into()),
                    ..Default::default()
                },
                BackendEvent {
                    finished: true,
                    ..Default::default()
                },
            ]
            .into(),
        });
        agent.start_turn("question".into());
        agent.start_step(&[], true);
        let mut transcript = markdown_transcript();
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
        for height in [0, 2, 0, 2] {
            transcript.update_viewport(agent.messages(), 40, height, agent.version(), true, None);
            assert_eq!(transcript.cursor, transcript.bottom.cursor);
        }
        stream_update(&mut agent, &mut transcript, &mut cx, 40, 0);
        assert_eq!(agent.poll_step(&mut cx), Poll::Ready(StepEvent::Finished));
        transcript.update_viewport(agent.messages(), 40, 0, agent.version(), false, None);
        transcript.update_viewport(agent.messages(), 35, 2, agent.version(), false, None);
        assert_eq!(transcript.cursor, transcript.bottom.cursor);
    }

    #[test]
    fn streamed_tables_keep_the_viewport_consistent() {
        let mut agent = Agent::new(StreamingBackend {
            events: [
                BackendEvent {
                    content: Some("| name | notes |\n|---|---|\n| x | y |\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("after\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    finished: true,
                    ..Default::default()
                },
            ]
            .into(),
        });
        agent.start_turn("question".into());
        agent.start_step(&[], true);
        let mut transcript = markdown_transcript();
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());

        // The table streams open, then closes; each update's view must match a fresh walk.
        for _ in 0..3 {
            stream_update(&mut agent, &mut transcript, &mut cx, 60, 8);
            let mut fresh = markdown_transcript();
            fresh.update_viewport(agent.messages(), 60, 8, agent.version(), true, None);
            assert_eq!(transcript.cursor, fresh.cursor);
            assert_eq!(transcript.row, fresh.row);
            assert_eq!(transcript.row_cache, fresh.row_cache);
        }

        // Completion renders the table followed by the later content.
        assert_eq!(agent.poll_step(&mut cx), Poll::Ready(StepEvent::Finished));
        let mut session = session();
        crate::test_support::append_messages(&mut session.agent, agent.messages().iter().cloned());
        let mut terminal = Terminal::new(TestBackend::new(60, 16)).unwrap();
        draw_transcript(&mut transcript, &mut terminal, &mut session);
        let rendered = terminal_text(&terminal);
        assert!(rendered.contains("name │ notes"), "{rendered:?}");
        assert!(rendered.contains("x    │ y"), "{rendered:?}");
        assert!(rendered.contains("after"), "{rendered:?}");
    }
}
