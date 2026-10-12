// TODO: this should not be a seprate module; it belong son tree.rs!

use mdtext::ListKind;
use unicode_width::UnicodeWidthStr;

use super::{
    RowCache,
    tree::{Element, Tree},
};
use crate::widgets::wrapping::{Content, Position, Row, State};

#[derive(Debug)]
pub struct Block {
    pub node_idx: usize,
    pub needs_gap: bool,
    pub has_content: bool,
}

impl Tree {
    pub fn path(&self, mut node: usize) -> Vec<usize> {
        // Parser events append nodes in preorder. Parent IDs and ancestry survive append-only feeds.
        // Follow parents without scanning preceding siblings; callers need ancestors before descendants.
        // TODO: Carry the required path with traversal instead of rebuilding it for each request.
        let mut path = Vec::new();
        while node != 0 {
            path.push(node);
            node = self.nodes[node].parent;
        }
        path.reverse();
        path
    }

    pub fn leaf<'a>(&'a self, source: &'a str, id: usize) -> &'a str {
        match &self.nodes[id].data {
            Element::Root { .. } => source,
            Element::Plain(text)
            | Element::Code(text)
            | Element::Html(text)
            | Element::InlineMath(text)
            | Element::Math(text) => text.resolve(source, &self.strings),
            Element::SoftBreak | Element::HardBreak => "\n",
            _ => unreachable!(),
        }
    }

    pub fn content<'a>(&'a self, source: &'a str, parent: usize) -> Content<'a> {
        if parent == 0 {
            return Content::with_offsets([(
                0,
                self.pending_start(),
                &source[self.pending_start()..],
            )]);
        }

        // TODO: Avoid rebuilding the full block/cell content index for each row or visibility check. The formatter
        // needs text across inline leaves so style boundaries do not split words or graphemes. A resumable leaf/byte
        // position still triggers this whole-subtree collection; local positions alone do not make the work local.
        // Possible approaches: adapt the formatter to a leaf cursor, or keep Content for the current physical line.
        // First account for Source::len_bytes(), offset mapping, word lookahead, and grapheme context across chunks.
        // Invariant to preserve: splitting the same visible text across inline styles must not change its wrapping.
        // CRLF and graphemes can cross leaf boundaries. Content limited to one line must retain the original
        // heading/indent context when offsets are rebased.
        let mut parts = Vec::new();
        let mut image_depth = 0;
        for (id, enter) in self.traverse_subtree(parent) {
            if matches!(self.nodes[id].data, Element::Image { .. }) {
                if enter {
                    image_depth += 1;
                } else {
                    image_depth -= 1;
                }
            }
            if !enter {
                continue;
            }
            match &self.nodes[id].data {
                Element::Plain(_) | Element::Code(_) => parts.push((id, self.leaf(source, id))),
                Element::Html(_) | Element::InlineMath(_) | Element::Math(_)
                    if image_depth == 0 =>
                {
                    parts.push((id, self.leaf(source, id)));
                }
                Element::SoftBreak | Element::HardBreak => {
                    parts.push((id, if image_depth == 0 { "\n" } else { " " }))
                }
                _ => {}
            }
        }
        let mut content = Content::new(parts);
        if !matches!(self.nodes[parent].data, Element::TableCell) {
            content.indent = self.prefix(&self.path(parent), &[], None).width();
            if let Element::Heading(level) = self.nodes[parent].data {
                content.heading = level as usize + 1;
            }
        }
        content
    }

    pub fn prefix(&self, path: &[usize], markers: &[usize], heading: Option<bool>) -> String {
        let mut prefix = String::new();
        for (depth, &id) in path.iter().enumerate() {
            match self.nodes[id].data {
                Element::Quote => prefix.push_str("│ "),
                Element::ListItem => {
                    let list = path[depth - 1];
                    let mut marker = match self.nodes[list].data {
                        Element::List(ListKind::Unordered) => "• ".to_owned(),
                        Element::List(ListKind::Ordered(start)) => {
                            format!(
                                "{}. ",
                                start as usize
                                    + self.children(list).position(|child| child == id).unwrap()
                            )
                        }
                        _ => unreachable!(),
                    };
                    if let Some(checked) =
                        self.children(id)
                            .find_map(|child| match self.nodes[child].data {
                                Element::TaskListMarker(checked) => Some(checked),
                                _ => None,
                            })
                    {
                        if marker == "• " {
                            marker.clear();
                        }
                        marker.push_str(if checked { "[x] " } else { "[ ] " });
                    }
                    if markers.contains(&id) {
                        prefix.push_str(&marker);
                    } else {
                        prefix.push_str(&" ".repeat(marker.width()));
                    }
                }
                Element::Heading(level) => {
                    // The first physical heading line has hashes; its soft continuations use equal indentation.
                    if let Some(first) = heading {
                        if first {
                            prefix.push_str(&"#".repeat(level as usize));
                            prefix.push(' ');
                        } else {
                            prefix.push_str(&" ".repeat(level as usize + 1));
                        }
                    }
                }
                _ => {}
            }
        }
        prefix
    }

    pub fn markers(&self, source: &str, owner: usize, path: &[usize]) -> Vec<usize> {
        path.iter()
            .copied()
            .filter(|&id| matches!(self.nodes[id].data, Element::ListItem))
            .filter(|&item| {
                // The first visible descendant uses the item marker. This property is independent of wrap width.
                // Search only that item's subtree; no earlier document rows or saved marker decisions are needed.
                let first = self.traverse_subtree(item).find_map(|(id, enter)| {
                    let body = match self.nodes[id].data {
                        Element::Heading(_) | Element::ThematicBreak => enter,
                        Element::Paragraph
                        | Element::CodeBlock(_)
                        | Element::HtmlBlock
                        | Element::DisplayMath => enter && !self.content(source, id).is_empty(),
                        Element::Table(_) => enter && self.table_rows(id).next().is_some(),
                        Element::ListItem => !enter && !self.is_open_container(id),
                        _ => false,
                    };
                    body.then_some(id)
                });
                first.unwrap_or(item) == owner
            })
            .collect()
    }

    // TODO: this probably doesn't belong in Tree since this concerns how it's rendered? Violation of encapsulation!
    /// Creates an iterator over all row-wise elements/blocks loosely similar to the markdown definition with
    /// data about how they should be rendered.
    ///
    /// Every emitted block contributes content or a spacer.
    pub fn blocks<'a>(&'a self, source: &'a str) -> impl Iterator<Item = Block> + 'a {
        /// Represents the renderer state for a markdown container (root, list item, quote) and its children.
        #[derive(Default)]
        struct Scope {
            /// A subsequent sibling block will need a spacer for rendering.
            /// Exception: nested containers do not get spacers
            pending_gap: bool,
            /// There is content within this subtree that can be rendered.
            has_content: bool,
        }

        let mut elements = self.traverse_subtree(0);
        let mut scopes = vec![Scope::default()];
        let mut need_emit_pending = self.pending_start() < source.len();
        std::iter::from_fn(move || {
            for (idx, enter) in elements.by_ref() {
                if !enter {
                    // "Pop" containers on exit
                    if matches!(self.nodes[idx].data, Element::Quote | Element::ListItem) {
                        let current_scope = scopes.pop().unwrap();

                        // Traversal can exit a node that the parser still has open. Wait for parser closure before
                        // emitting an empty item marker, since content may still arrive.
                        let is_empty_list_item = matches!(self.nodes[idx].data, Element::ListItem)
                            && !current_scope.has_content
                            && !self.is_open_container(idx);
                        if current_scope.has_content || is_empty_list_item {
                            let parent_scope = scopes.last_mut().unwrap();
                            parent_scope.pending_gap = true;
                            parent_scope.has_content = true;
                        }

                        if is_empty_list_item {
                            // Ensure empty list items still get rendered (list markers)!
                            return Some(Block {
                                node_idx: idx,
                                needs_gap: false,
                                has_content: true,
                            });
                        }
                    }
                    continue;
                }

                if matches!(self.nodes[idx].data, Element::ListItem) {
                    // Defer rendering list items until all content for it (if any) has been gathered.
                    scopes.push(Scope::default());
                    continue;
                }

                let is_text_block = matches!(
                    self.nodes[idx].data,
                    Element::Paragraph
                        | Element::Heading(_)
                        | Element::CodeBlock(_)
                        | Element::HtmlBlock
                        | Element::DisplayMath
                );
                let is_table = matches!(self.nodes[idx].data, Element::Table(_));
                let is_break = matches!(self.nodes[idx].data, Element::ThematicBreak);
                let is_container =
                    matches!(self.nodes[idx].data, Element::Quote | Element::List(_));

                // Only Markdown blocks can create scopes, not inline elements.
                if is_text_block || is_table || is_break || is_container {
                    let needs_gap = (!is_container || scopes.len() == 1)
                        && std::mem::take(&mut scopes.last_mut().unwrap().pending_gap);

                    if matches!(self.nodes[idx].data, Element::Quote) {
                        scopes.push(Scope::default());
                    }

                    let has_content = is_break
                        || matches!(self.nodes[idx].data, Element::Heading(_))
                        || is_text_block && !self.content(source, idx).is_empty()
                        || is_table && self.table_rows(idx).next().is_some();
                    if has_content {
                        let current_scope = scopes.last_mut().unwrap();
                        current_scope.pending_gap = true;
                        current_scope.has_content = true;
                    }

                    if needs_gap || has_content {
                        return Some(Block {
                            node_idx: idx,
                            needs_gap,
                            has_content,
                        });
                    }
                }
            }

            if std::mem::take(&mut need_emit_pending) {
                // Behavior rationale: A separator above the plain text prevents viewport shuffling in the common case.
                let needs_gap = self.open_container == 0 && scopes[0].pending_gap;
                Some(Block {
                    node_idx: 0,
                    needs_gap,
                    has_content: true,
                })
            } else {
                None
            }
        })
    }

    /// Whether the subtree at `idx` renders at least one body row; same as `blocks()` but without
    /// needing to replay the entire tree each call in the typical case.
    pub fn renders(&self, source: &str, idx: usize) -> bool {
        if matches!(self.nodes[idx].data, Element::ListItem) && !self.is_open_container(idx) {
            // Empty list items should always be rendered (for the marker).
            return true;
        }

        match self.nodes[idx].data {
            Element::ThematicBreak | Element::Heading(_) => true,
            Element::Paragraph
            | Element::CodeBlock(_)
            | Element::HtmlBlock
            | Element::DisplayMath => !self.content(source, idx).is_empty(),
            Element::Table(_) => self.table_rows(idx).next().is_some(),

            // Containers are renderable only when they have content to render. (exception: empty list item)
            Element::List(_) | Element::Quote | Element::ListItem => {
                self.children(idx).any(|child| self.renders(source, child))
            }
            _ => false,
        }
    }

    /// Whether the pending text (not yet parseable markdown) needs top-margin for rendering.
    pub fn pending_needs_gap(&self, source: &str, pending_end: usize) -> bool {
        self.open_container == 0
            && self.pending_start() < pending_end
            && self
                .children(0)
                .last()
                .is_some_and(|last| self.renders(source, last))
    }

    pub fn row_at<'a>(
        &'a self,
        source: &'a str,
        owner: usize,
        start: Option<Position>,
        width: usize,
    ) -> (Content<'a>, Row, usize) {
        let content = self.content(source, owner);
        let prefix = content.prefix_width(start, width);

        // The position determines continuation and prefix context. No prepared row or prior wrapping state is required.
        let row = content
            .next_row(
                &mut State {
                    next: start,
                    soft: content.continuation(start),
                    ..State::default()
                },
                width,
            )
            .unwrap();
        (content, row, prefix)
    }

    pub fn row_starts(&self, source: &str, owner: usize, width: usize) -> Vec<Option<Position>> {
        let content = self.content(source, owner);
        let mut starts = Vec::new();
        let mut state = State::default();
        while let Some(row) = content.next_row(&mut state, width) {
            starts.push(row.start);
        }

        if owner == 0 && source.ends_with(['\r', '\n']) {
            // Pending text includes the empty physical line after a final newline, as plain-text rendering does.
            starts.push(Some(Position {
                node: 0,
                byte: source.len(),
            }));
        }
        starts
    }

    // TODO: this doesn't belong on Tree since this is renderer-specific!
    // Tree is supposed to be a data structure!
    pub fn cache_rows<'a>(
        &self,
        source: &str,
        owner: usize,
        width: usize,
        cache: &'a mut RowCache,
    ) -> &'a [Option<Position>] {
        cache
            .entry(owner)
            .or_insert_with(|| self.row_starts(source, owner, width))
    }
}
