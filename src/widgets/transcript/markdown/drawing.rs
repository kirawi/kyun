// TODO: remove this as a module because it shouldn't be one

use mdtext::Alignment;
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};
use unicode_width::UnicodeWidthStr;

use super::{
    ALT_STYLE, CODE_BLOCK_STYLE, CODE_SPAN_STYLE, HTML_STYLE, MATH_STYLE, MarkdownLink,
    MarkdownPosition, MarkdownRow, MdMessage, PREFIX_STYLE,
    table::{MIN_COLUMN_WIDTH, SEPARATOR},
    tree::{Element, Tree},
};
use crate::widgets::{
    SOFT_WRAP_INDICATOR,
    wrapping::{Content, Row},
};

fn style(style: Style, element: &Element) -> Style {
    style.patch(match element {
        Element::Heading(_) | Element::TableHead | Element::Strong => {
            Style::new().add_modifier(Modifier::BOLD)
        }
        Element::CodeBlock(_) => CODE_BLOCK_STYLE,
        Element::CodeSpan => CODE_SPAN_STYLE,
        Element::HtmlBlock => HTML_STYLE,
        Element::DisplayMath | Element::InlineMath(_) | Element::Math(_) => MATH_STYLE,
        Element::Emphasis => Style::new().add_modifier(Modifier::ITALIC),
        Element::Strikethrough => Style::new().add_modifier(Modifier::CROSSED_OUT),
        Element::Link { .. } => Style::new().add_modifier(Modifier::UNDERLINED),
        _ => Style::new(),
    })
}

impl Tree {
    pub fn append_spans<'a>(
        &'a self,
        source: &'a str,
        content: &Content,
        row: &Row,
        prefix_width: usize,
        rendered: &mut MarkdownRow<'a>,
    ) {
        for (id, range, spaces) in content.fragments(row, prefix_width) {
            let mut current = Style::new();
            let mut image = false;
            let mut link = None;
            for ancestor in self.path(id) {
                if let Element::Link { url, .. } = &self.nodes[ancestor].data {
                    link = Some((ancestor, url.resolve(source, &self.strings)));
                }
                image |= matches!(self.nodes[ancestor].data, Element::Image { .. });
                current = if image {
                    ALT_STYLE
                } else {
                    style(current, &self.nodes[ancestor].data)
                };
            }
            let text = if image
                && matches!(self.nodes[id].data, Element::SoftBreak | Element::HardBreak)
            {
                " "
            } else {
                self.leaf(source, id)
            };
            if let Some((node, url)) = link
                && !url.is_empty()
            {
                rendered.links.push(MarkdownLink {
                    span: rendered.line.spans.len(),
                    node,
                    url,
                });
            }
            rendered.line.spans.push(if spaces > 0 {
                Span::styled(" ".repeat(spaces), current)
            } else {
                Span::styled(&text[range], current)
            });
        }
    }
}

impl MdMessage {
    pub fn render<'a>(
        &'a mut self,
        source: &'a str,
        width: usize,
        position: &MarkdownPosition,
    ) -> MarkdownRow<'a> {
        match *position {
            MarkdownPosition::Gap { node } => {
                let mut path = self.tree.path(node);
                path.pop();
                MarkdownRow {
                    line: Line::from(Span::styled(
                        self.tree.prefix(&path, &[], None),
                        PREFIX_STYLE,
                    )),
                    links: Vec::new(),
                    background: None,
                }
            }
            MarkdownPosition::Text { node, start } => {
                let path = self.tree.path(node);
                if matches!(
                    self.tree.nodes[node].data,
                    Element::ThematicBreak | Element::ListItem
                ) {
                    let prefix =
                        self.tree
                            .prefix(&path, &self.tree.markers(source, node, &path), None);
                    let body = if matches!(self.tree.nodes[node].data, Element::ThematicBreak) {
                        "─".repeat(width.saturating_sub(prefix.width()))
                    } else {
                        String::new()
                    };
                    return MarkdownRow {
                        line: Line::from(vec![
                            Span::styled(prefix, PREFIX_STYLE),
                            Span::styled(body, PREFIX_STYLE),
                        ]),
                        links: Vec::new(),
                        background: None,
                    };
                }
                let first = self.tree.cache_rows(source, node, width, &mut self.rows)[0] == start;
                let (content, row, prefix_width) = self.tree.row_at(source, node, start, width);
                let soft = content.continuation(start);
                let heading = matches!(self.tree.nodes[node].data, Element::Heading(_))
                    && content.line_start(start) == 0;
                let markers = if first {
                    self.tree.markers(source, node, &path)
                } else {
                    Vec::new()
                };
                let mut prefix = self.tree.prefix(&path, &markers, heading.then_some(!soft));
                if soft && width.saturating_sub(prefix.width()) > SOFT_WRAP_INDICATOR.width() {
                    prefix.push_str(SOFT_WRAP_INDICATOR);
                }
                // TP1: Prefixes and resize information are derived at the cursor, rather than saved on every row.
                let mut rendered = MarkdownRow {
                    line: Line::from(Span::styled(prefix, PREFIX_STYLE)),
                    links: Vec::new(),
                    background: matches!(self.tree.nodes[node].data, Element::CodeBlock(_))
                        .then_some(CODE_BLOCK_STYLE),
                };
                self.tree
                    .append_spans(source, &content, &row, prefix_width, &mut rendered);
                rendered
            }
            MarkdownPosition::Table { first_cell, .. }
            | MarkdownPosition::Separator { first_cell } => {
                let (table, source_row, start) = self.tree.cell_location(first_cell);
                let separator = matches!(position, MarkdownPosition::Separator { .. });
                let line = if let MarkdownPosition::Table { line, .. } = *position {
                    line
                } else {
                    0
                };
                if !separator {
                    self.tree.table_row_height(
                        source,
                        first_cell,
                        width,
                        &mut self.rows,
                        &mut self.columns,
                    );
                }
                let first = !separator
                    && line == 0
                    && start == 0
                    && self.tree.table_rows(table).next() == Some(source_row);
                let path = self.tree.path(table);
                let markers = if first {
                    self.tree.markers(source, table, &path)
                } else {
                    Vec::new()
                };
                let prefix = self.tree.prefix(&path, &markers, None);
                let prefix_width = prefix.width();
                let mut rendered = MarkdownRow {
                    line: Line::from(Span::styled(prefix, PREFIX_STYLE)),
                    links: Vec::new(),
                    background: None,
                };
                if separator {
                    rendered.line.spans.push(Span::styled(
                        "─".repeat(width.saturating_sub(prefix_width).max(MIN_COLUMN_WIDTH)),
                        PREFIX_STYLE,
                    ));
                    return rendered;
                }
                let head = self
                    .tree
                    .path(source_row)
                    .iter()
                    .any(|&id| matches!(self.tree.nodes[id].data, Element::TableHead));
                let Element::Table(alignments) = &self.tree.nodes[table].data else {
                    unreachable!()
                };
                let end = (start + self.tree.columns_per_group(table, width)).min(alignments.len());
                for (column, cell) in self
                    .tree
                    .children(source_row)
                    .enumerate()
                    .skip(start)
                    .take(end - start)
                {
                    let cell_width = self.columns[&table][column];
                    if let Some(&start) = self.rows[&cell].get(line) {
                        // Cells select independent positions at this visual line.
                        // An exhausted cell contributes padding.
                        let (content, row, marker_width) =
                            self.tree.row_at(source, cell, start, cell_width);
                        let marker = if marker_width > 0 {
                            SOFT_WRAP_INDICATOR
                        } else {
                            ""
                        };
                        let padding = cell_width.saturating_sub(marker_width + row.width);
                        let before = match alignments[column] {
                            Alignment::Right => padding,
                            Alignment::Center => padding / 2,
                            Alignment::Left | Alignment::None => 0,
                        };
                        // Keep the continuation marker before alignment padding; tab columns use the cell's cursor.
                        if !marker.is_empty() {
                            rendered.line.spans.push(Span::styled(marker, PREFIX_STYLE));
                        }
                        if before > 0 {
                            rendered.line.spans.push(Span::raw(" ".repeat(before)));
                        }
                        self.tree
                            .append_spans(source, &content, &row, marker_width, &mut rendered);
                        if column + 1 < end && padding > before {
                            rendered
                                .line
                                .spans
                                .push(Span::raw(" ".repeat(padding - before)));
                        }
                    } else if column + 1 < end {
                        rendered.line.spans.push(Span::raw(" ".repeat(cell_width)));
                    }
                    if column + 1 < end {
                        rendered.line.spans.push(Span::styled(
                            SEPARATOR,
                            if head {
                                PREFIX_STYLE.add_modifier(Modifier::BOLD)
                            } else {
                                PREFIX_STYLE
                            },
                        ));
                    }
                }
                rendered
            }
        }
    }
}
