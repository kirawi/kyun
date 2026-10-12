mod drawing;
mod navigation;
mod table;
mod traversal;
mod tree;
mod utils;

use std::collections::HashMap;

use kyun_agent::{Message, MessageView};
use mdtext::Options;
use ratatui::{
    prelude::{Color, Modifier, Style},
    text::Line,
};

use self::{
    traversal::Block,
    tree::{Element, Tree},
};
use crate::widgets::{CODE_BG, MUTED, TEXT, wrapping::Position};

// TP3: These constants should not be defined here and should be in theme
/// GFM plus immediate fenced-code/HTML lines, before their closing delimiters arrive.
const OPTIONS: Options = Options::from_bits(Options::GFM.bits() | Options::IMMEDIATE_MODE.bits());
const PREFIX_STYLE: Style = Style::new().fg(MUTED);
const HTML_STYLE: Style = Style::new().fg(MUTED);
const CODE_BLOCK_STYLE: Style = Style::new().bg(CODE_BG).fg(TEXT);
const CODE_SPAN_STYLE: Style = Style::new().fg(Color::Rgb(0x82, 0xD2, 0xCE));
const MATH_STYLE: Style = Style::new()
    .fg(Color::Rgb(0x56, 0xB6, 0xC2))
    .add_modifier(Modifier::ITALIC);
const ALT_STYLE: Style = Style::new().fg(MUTED).add_modifier(Modifier::ITALIC);

// These caches store positions and column widths, never paths, markers, styles, or display-row descriptions.
// TODO: Replace RowCache with a walker-local cache following SoftCache's approach. Be lazy about memoization:
// retain wrapped boundaries only when needed for the current physical line or table row and column group.
// Replace the cache when the walker leaves that content; invalidate it when the content or wrapping width changes.
// The tree, source, width, and current position can reconstruct rows. Use only what is needed when it is needed.
// Storing only positions still leaves cache_rows() wrapping every row of a block/cell on first access. SoftCache also
// wraps its whole physical line on creation; reducing cache scope alone does not make boundary computation lazy.
// Possible approach: extend a boundary list as rows are requested, and replay from the physical-line start only when
// backward traversal needs an uncached boundary. Entering at the last row may still require wrapping that whole line.
// Invariant to preserve: cache eviction changes the work performed, never the rows, positions, or update counts.
// Stable node IDs alone do not validate boundaries: the message, content, wrapping width, and prefix/table context
// must still match.
type RowCache = HashMap<usize, Vec<Option<Position>>>;
type ColumnCache = HashMap<usize, Vec<usize>>;

/// The current wrapping location. A table's sibling cells use the same line in their own boundary caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkdownPosition {
    Text {
        node: usize,
        start: Option<Position>,
    },
    Table {
        first_cell: usize,
        line: usize,
    },
    Separator {
        first_cell: usize,
    },
    Gap {
        node: usize,
    },
}

pub enum MarkdownReflow {
    At(MarkdownPosition),
    Before(MarkdownPosition),
}

pub struct MarkdownLink<'a> {
    pub span: usize,
    pub node: usize,
    pub url: &'a str,
}

/// Resolved terminal spans, links, and optional full-row background. Transcript owns drawing them.
// TP1: Rendering is the role of Transcript; Markdown supplies data rather than implementing Widget.
// TP1: Filling the remainder of a code row also belongs in Transcript, alongside the row's spans.
pub struct MarkdownRow<'a> {
    pub line: Line<'a>,
    pub links: Vec<MarkdownLink<'a>>,
    pub background: Option<Style>,
}

#[cfg(test)]
impl MarkdownRow<'_> {
    pub fn text(&self) -> String {
        self.line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }
}

/// Active and completed messages use the same tree. Width-dependent storage is limited to boundary/column caches.
pub struct MdMessage {
    tree: Tree,
    rows: RowCache,
    columns: ColumnCache,
}

impl MdMessage {
    fn new() -> Self {
        Self {
            tree: Tree::new(),
            rows: HashMap::new(),
            columns: HashMap::new(),
        }
    }

    #[cfg(test)]
    fn parse(source: &str) -> Self {
        let mut message = Self::new();
        message.tree.update(source, true);
        message
    }

    fn count_rows_before(
        &mut self,
        source: &str,
        width: usize,
        before_node_idx: Option<usize>,
    ) -> usize {
        // Visits are temporary. Height accounting does not build or retain an owner list or table visual rows.
        // TODO: Separate counting from boundary retention. cache_rows() and table_height() fill boundary caches even
        // when only a count is needed, so measuring message height eagerly fills caches outside the viewport.
        // Possible approaches: count with transient formatter state, or retain scalar counts only where repeated
        // measurement justifies them. Either separates counting from navigation caches; neither alone avoids wrapping
        // the measured text. Reuse of an unchanged prefix needs a separate argument about the affected boundary.
        let mut count = 0;
        for block in self.tree.blocks(source) {
            if before_node_idx
                .is_some_and(|node| block.node_idx == 0 || node != 0 && block.node_idx >= node)
            {
                break;
            }
            count += usize::from(block.needs_gap);

            if block.has_content {
                count += match self.tree.nodes[block.node_idx].data {
                    Element::ThematicBreak | Element::ListItem => 1,
                    Element::Table(_) => self.tree.table_height(
                        source,
                        block.node_idx,
                        width,
                        &mut self.rows,
                        &mut self.columns,
                    ),
                    _ => self
                        .tree
                        .cache_rows(source, block.node_idx, width, &mut self.rows)
                        .len(),
                };
            }
        }
        count
    }

    fn total_rows_count(
        &mut self,
        source: &str,
        width: usize,
        height: &mut Option<usize>,
    ) -> usize {
        if let Some(height) = *height {
            return height;
        }

        let count = self.count_rows_before(source, width, None);
        *height = Some(count);
        count
    }

    fn update(
        &mut self,
        source: &str,
        width: usize,
        finalize: bool,
        state: &mut StreamState,
    ) -> MarkdownChange {
        // TODO: Preserve the old affected extent without requiring a message's total before change detection.
        // This implementation subtracts an unchanged prefix from old/new totals after identifying appended content
        // and container closures. This dependency comes from the update algorithm, not from navigation.
        // Possible approaches: identify and measure the mutable suffix before applying events, or keep compact height
        // summaries for mutable owners. Prove that rows before the chosen boundary cannot change. At fixed width,
        // old_height - removed_rows + inserted_rows must equal a fresh walk, including gaps. A new cell can rewrap
        // earlier table cells, and closing-only events can expose rows without adding nodes.
        // A possible fast path would require unchanged source/width, no parser finalization, and any requested
        // total already available. Equal node counts or equal pending lengths alone do not establish those conditions.
        let old_height = self.total_rows_count(&source[..state.fed_len], width, &mut state.height);
        let pending = (
            self.tree.pending_start(),
            state.fed_len - self.tree.pending_start(),
        );

        // Invariant: pending height is text rows plus its optional gap.
        let old_pending_gap = self.tree.pending_needs_gap(source, state.fed_len);
        let nodes = self.tree.nodes.len();
        let mut container_idx = self.tree.open_container;
        self.tree.update(source, finalize);
        state.fed_len = source.len();

        let mut diff_start_idx = (nodes < self.tree.nodes.len()).then_some(nodes);
        let mut closed_containers = Vec::new();

        // Only the old open ancestor chain can gain descendants or close. Parent links remain valid after either.
        while container_idx != 0 {
            if matches!(self.tree.nodes[container_idx].data, Element::ListItem)
                && !self.tree.is_open_container(container_idx)
            {
                // New list item implicitly closes previous sibling list item.
                // `is_open_container` would've been true if it was a parent list item.
                closed_containers.push(container_idx);
            }

            if self.tree.subtree_end(container_idx) > nodes {
                match self.tree.nodes[container_idx].data {
                    Element::Table(_) => {
                        // Table rows are complete when mdtext emits them. A table owns widths across all its cells.
                        let end = self.tree.subtree_end(container_idx);
                        self.rows
                            .retain(|&node, _| node < container_idx || node >= end);
                        self.columns.remove(&container_idx);
                        diff_start_idx = Some(
                            diff_start_idx.map_or(container_idx, |old| old.min(container_idx)),
                        );
                    }
                    Element::Paragraph
                    | Element::Heading(_)
                    | Element::CodeBlock(_)
                    | Element::HtmlBlock
                    | Element::DisplayMath => {
                        self.rows.remove(&container_idx);
                        diff_start_idx = Some(
                            diff_start_idx.map_or(container_idx, |old| old.min(container_idx)),
                        );
                    }
                    _ => {}
                }
            }
            container_idx = self.tree.nodes[container_idx].parent;
        }

        let new_pending_gap = self.tree.pending_needs_gap(source, source.len());
        for &idx in &closed_containers {
            // blocks() emits a row for a closed item only when its descendants produced no body rows.
            if !self
                .tree
                .children(idx)
                .any(|child| self.tree.renders(source, child))
            {
                diff_start_idx = Some(diff_start_idx.map_or(idx, |old| old.min(idx)));
            }
        }

        let pending_changed = pending
            != (
                self.tree.pending_start(),
                source.len() - self.tree.pending_start(),
            );

        if nodes == self.tree.nodes.len()
            && diff_start_idx.is_none()
            && !pending_changed
            && old_pending_gap == new_pending_gap
        {
            return MarkdownChange::default();
        }

        let first_node = diff_start_idx.unwrap_or(0);
        let (removed_rows, inserted_rows) = if first_node == 0 {
            // Pending-only changes retain parsed block/cell caches. Gaps are separate from cached text rows.
            let removed =
                self.rows.remove(&0).map_or(0, |rows| rows.len()) + usize::from(old_pending_gap);
            let inserted = if self.tree.pending_start() < source.len() {
                self.tree.cache_rows(source, 0, width, &mut self.rows).len()
            } else {
                0
            } + usize::from(new_pending_gap);
            state.height = Some(old_height - removed + inserted);
            (removed, inserted)
        } else {
            // Completed siblings keep their boundaries. Growing blocks and tables were invalidated above.
            self.rows.remove(&0);
            state.height = None;

            let before = self.count_rows_before(source, width, Some(first_node));
            let height = self.total_rows_count(source, width, &mut state.height);
            (old_height - before, height - before)
        };
        MarkdownChange {
            first_node,
            removed_rows,
            inserted_rows,
            changed: true,
        }
    }
}

// TP1: Why is this needed?
/// Counts describe old and new suffix extents at the supplied width, rather than individual edited rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MarkdownChange {
    pub first_node: usize,
    pub removed_rows: usize,
    pub inserted_rows: usize,
    pub changed: bool,
}

/// Bookkeeping for the message being streamed, consumed by `MdMessage::update`. Completed messages do not retain it.
#[derive(Default)]
struct StreamState {
    /// Length of the source at the last update. Used to decide if the content changed (i.e. new assistant delta).
    fed_len: usize,
    /// Cached row total of the streaming message. Only update() reads it, for its suffix subtraction.
    height: Option<usize>,
}

struct Streaming {
    msg_idx: usize,
    state: StreamState,
}

// The vector contains only assistant messages with nonempty content, in message order.
#[derive(Default)]
pub struct MarkdownCache {
    is_enabled: bool,
    entries: Vec<MdMessage>,
    // The active message selects the same message storage used after parser finalization.
    streaming: Option<Streaming>,
    /// Width that the entries' wrapping caches and the streaming height were last invalidated against.
    width: Option<usize>,
}

impl MarkdownCache {
    pub fn new(is_enabled: bool) -> Self {
        Self {
            is_enabled,
            ..Self::default()
        }
    }

    // TODO: This should no longer exist; MarkdownCache shouldn't be called at all probably per my intent.
    // And it's something controlled by config! Violation of abstraction.
    pub fn is_enabled(&self) -> bool {
        self.is_enabled
    }

    pub fn truncate(&mut self, messages: MessageView<'_>) {
        self.entries
            .truncate(entry_count_up_to(messages, messages.len()));
        self.streaming = None;
    }

    #[cfg(test)]
    pub fn initialize_for_test(&mut self, messages: MessageView<'_>) {
        self.entries = messages
            .iter()
            .filter_map(get_assistant_content)
            .map(MdMessage::parse)
            .collect();
        self.streaming = None;
    }

    fn get_entry(&mut self, messages: MessageView<'_>, msg_idx: usize) -> &mut MdMessage {
        let entry_idx = get_entry_idx(messages, msg_idx);
        &mut self.entries[entry_idx]
    }

    // NOTE: the name is bad! It's really an invalidate cache check...
    // Transcript resizing already walks all messages, so per-message width tracking would preserve nothing.
    fn ensure_width(&mut self, width: usize) {
        if self.width != Some(width) {
            for message in &mut self.entries {
                message.rows.clear();
                message.columns.clear();
            }
            if let Some(streaming) = &mut self.streaming {
                streaming.state.height = None;
            }
            self.width = Some(width);
        }
    }

    /// Feed appended source before Walker requests rows. Completed messages are parsed on synchronization.
    /// Every stream update and completion must be synchronized, including when the transcript has no visible rows.
    /// Finishing retains the same tree and boundary caches.
    pub fn sync(
        &mut self,
        messages: MessageView<'_>,
        is_streaming: bool,
        width: usize,
    ) -> Option<MarkdownChange> {
        if !self.is_enabled {
            return None;
        }
        self.ensure_width(width);

        // NOTE: This may be duplicating code from elsewhere? I recall that there is a method elsewhere that
        // also checks if the current message is truly streaming without tool calls?
        let streamed = is_streaming
            && matches!(messages.last(), Some(Message::Assistant(message)) if message.tool_calls.is_empty())
            && messages.timestamp(messages.len() - 1).is_none();
        match self.streaming.take() {
            None if streamed => {
                let msg_idx = messages.len() - 1;
                let Some(content) = get_assistant_content_for_view(messages, msg_idx) else {
                    // There is not yet any new content to render.
                    // The request to the backend has only just begun.
                    return Some(MarkdownChange::default());
                };

                // Does this assert do anything useful? Or is it vacuously true by this point?
                debug_assert_eq!(self.entries.len(), get_entry_idx(messages, msg_idx));

                let mut message = MdMessage::new();
                let mut state = StreamState::default();
                let update = message.update(content, width, false, &mut state);
                self.entries.push(message);
                self.streaming = Some(Streaming { msg_idx, state });
                Some(update)
            }

            // FIXME: it checks streamed so it knows that it is the last len and that it is an assistant message.
            // get_msg_content is poorly named and should be a method; its Option is unnecessary here.
            Some(Streaming { msg_idx, mut state }) => {
                let finish = messages.timestamp(msg_idx).is_some();
                let content = get_assistant_content_for_view(messages, msg_idx).unwrap();
                let message = self.get_entry(messages, msg_idx);

                let update = message.update(content, width, finish, &mut state);
                if !finish {
                    self.streaming = Some(Streaming { msg_idx, state });
                }
                Some(update)
            }
            None => {
                if let Some(msg_idx) = messages.len().checked_sub(1)
                    && let Some(content) = get_assistant_content_for_view(messages, msg_idx)
                {
                    let entry_idx = get_entry_idx(messages, msg_idx);
                    if entry_idx == self.entries.len() {
                        let mut message = MdMessage::new();
                        let update =
                            message.update(content, width, true, &mut StreamState::default());
                        self.entries.push(message);
                        return Some(update);
                    }
                }
                None
            }
        }
    }

    // TODO: needs proper name
    pub fn msg(
        &mut self,
        messages: MessageView<'_>,
        msg_idx: usize,
        width: usize,
    ) -> Option<&mut MdMessage> {
        if !self.is_enabled {
            return None;
        }
        self.ensure_width(width);

        let content = get_assistant_content_for_view(messages, msg_idx)?;
        let message = self.get_entry(messages, msg_idx);

        let visible = message.tree.blocks(content).next().is_some();
        visible.then_some(message)
    }

    /// Resolve one cached position into display data. Padding is local to drawing, not another retained message.
    // TP1: Padding need not live in a mutable Markdown cache field just to serve a terminal width.
    pub fn render<'a>(
        &'a mut self,
        messages: MessageView<'_>,
        msg_idx: usize,
        source: &'a str,
        width: usize,
        position: &MarkdownPosition,
    ) -> MarkdownRow<'a> {
        self.ensure_width(width);
        let message = self.get_entry(messages, msg_idx);
        message.render(source, width, position)
    }
}

// TP1: this does not belong here; this should be scrutinized and if useful, a method on MessageView itself
fn get_assistant_content_for_view(messages: MessageView<'_>, msg_idx: usize) -> Option<&str> {
    get_assistant_content(messages.get(msg_idx)?)
}

// TODO: this should be a method on Message!
fn get_assistant_content(msg: &Message) -> Option<&str> {
    match msg {
        Message::Assistant(msg) => msg.content.as_deref().filter(|content| !content.is_empty()),
        _ => None,
    }
}

fn entry_count_up_to(messages: MessageView<'_>, limit: usize) -> usize {
    messages
        .iter()
        .take(limit)
        .filter_map(get_assistant_content)
        .count()
}

fn get_entry_idx(messages: MessageView<'_>, msg_idx: usize) -> usize {
    entry_count_up_to(messages, msg_idx + 1) - 1
}

#[cfg(test)]
mod tests;
