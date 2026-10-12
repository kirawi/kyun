//! Navigate from local text positions or a table's first cell. Boundary caches support backward scrolling.

use super::{Block, MarkdownPosition, MarkdownReflow, MdMessage, tree::Element};
use crate::widgets::wrapping::State;

impl MdMessage {
    fn first_body(&mut self, source: &str, width: usize, node: usize) -> MarkdownPosition {
        match self.tree.nodes[node].data {
            Element::Table(_) => {
                let row = self.tree.table_rows(node).next().unwrap();
                MarkdownPosition::Table {
                    first_cell: self.tree.children(row).next().unwrap(),
                    line: 0,
                }
            }
            Element::ThematicBreak | Element::ListItem => {
                MarkdownPosition::Text { node, start: None }
            }
            _ => MarkdownPosition::Text {
                node,
                start: self.tree.cache_rows(source, node, width, &mut self.rows)[0],
            },
        }
    }

    fn last_body(&mut self, source: &str, width: usize, node: usize) -> MarkdownPosition {
        match self.tree.nodes[node].data {
            Element::Table(ref alignments) => {
                let group_size = self.tree.columns_per_group(node, width);
                let column = (alignments.len() - 1) / group_size * group_size;
                let row = self.tree.table_rows(node).last().unwrap();
                let first_cell = self.tree.children(row).nth(column).unwrap();
                let height = self.tree.table_row_height(
                    source,
                    first_cell,
                    width,
                    &mut self.rows,
                    &mut self.columns,
                );
                MarkdownPosition::Table {
                    first_cell,
                    line: height - 1,
                }
            }
            Element::ThematicBreak | Element::ListItem => {
                MarkdownPosition::Text { node, start: None }
            }
            _ => MarkdownPosition::Text {
                node,
                start: *self
                    .tree
                    .cache_rows(source, node, width, &mut self.rows)
                    .last()
                    .unwrap(),
            },
        }
    }

    fn first_in(&mut self, source: &str, width: usize, block: Block) -> MarkdownPosition {
        if block.needs_gap {
            MarkdownPosition::Gap {
                node: block.node_idx,
            }
        } else {
            self.first_body(source, width, block.node_idx)
        }
    }

    fn last_in(&mut self, source: &str, width: usize, block: Block) -> MarkdownPosition {
        if block.has_content {
            self.last_body(source, width, block.node_idx)
        } else {
            MarkdownPosition::Gap {
                node: block.node_idx,
            }
        }
    }

    // TODO: Keep traversal context when moving between blocks. after_block() and before_block() restart at the root
    // to recover neighboring visible blocks and their separator state; the node position alone does not carry it.
    fn after_block(&mut self, source: &str, width: usize, node: usize) -> Option<MarkdownPosition> {
        let next = self
            .tree
            .blocks(source)
            .skip_while(|block| block.node_idx != node)
            .nth(1)?;
        Some(self.first_in(source, width, next))
    }

    fn before_block(
        &mut self,
        source: &str,
        width: usize,
        node: usize,
    ) -> Option<MarkdownPosition> {
        let previous = self
            .tree
            .blocks(source)
            .take_while(|block| block.node_idx != node)
            .last()?;
        Some(self.last_in(source, width, previous))
    }

    pub fn first_position(&mut self, source: &str, width: usize) -> Option<MarkdownPosition> {
        let block = self.tree.blocks(source).next()?;
        Some(self.first_in(source, width, block))
    }

    pub fn last_position(&mut self, source: &str, width: usize) -> Option<MarkdownPosition> {
        let block = self.tree.blocks(source).last()?;
        Some(self.last_in(source, width, block))
    }

    // Candidate invariant to check: at unchanged source/width, next(p) = Some(q) implies previous(q) = Some(p) for
    // reachable rows. Include empty rows, gaps, and table separators. Resize mapping needs separate checks because
    // changing width can merge or split rows, so movement across widths is not generally invertible.
    pub fn next(
        &mut self,
        source: &str,
        width: usize,
        position: &MarkdownPosition,
    ) -> Option<MarkdownPosition> {
        match *position {
            MarkdownPosition::Text { node, start } => {
                if !matches!(
                    self.tree.nodes[node].data,
                    Element::ThematicBreak | Element::ListItem
                ) {
                    // TODO: Advance the formatter from this boundary for forward traversal. Backward wrapping needs
                    // cached boundaries or replay, but does not require eagerly caching the whole block here.
                    let rows = self.tree.cache_rows(source, node, width, &mut self.rows);
                    let row = rows
                        .binary_search_by_key(&start.map(|p| (p.node, p.byte)), |p| {
                            p.map(|p| (p.node, p.byte))
                        })
                        .unwrap();
                    if let Some(&start) = rows.get(row + 1) {
                        return Some(MarkdownPosition::Text { node, start });
                    }
                }
                self.after_block(source, width, node)
            }
            MarkdownPosition::Gap { node } => {
                let block = self
                    .tree
                    .blocks(source)
                    .find(|block| block.node_idx == node)
                    .unwrap();
                if block.has_content {
                    Some(self.first_body(source, width, node))
                } else {
                    self.after_block(source, width, node)
                }
            }
            MarkdownPosition::Separator { first_cell } => Some(MarkdownPosition::Table {
                first_cell,
                line: 0,
            }),
            MarkdownPosition::Table { first_cell, line } => {
                let height = self.tree.table_row_height(
                    source,
                    first_cell,
                    width,
                    &mut self.rows,
                    &mut self.columns,
                );
                if line + 1 < height {
                    return Some(MarkdownPosition::Table {
                        first_cell,
                        line: line + 1,
                    });
                }
                let (table, row, column) = self.tree.cell_location(first_cell);
                if let Some(next) = self
                    .tree
                    .table_rows(table)
                    .skip_while(|&id| id != row)
                    .nth(1)
                {
                    return Some(MarkdownPosition::Table {
                        first_cell: self.tree.children(next).nth(column).unwrap(),
                        line: 0,
                    });
                }
                let Element::Table(alignments) = &self.tree.nodes[table].data else {
                    unreachable!()
                };
                let next_column = column + self.tree.columns_per_group(table, width);
                if next_column < alignments.len() {
                    let head = self.tree.table_rows(table).next().unwrap();
                    Some(MarkdownPosition::Separator {
                        first_cell: self.tree.children(head).nth(next_column).unwrap(),
                    })
                } else {
                    self.after_block(source, width, table)
                }
            }
        }
    }

    pub fn previous(
        &mut self,
        source: &str,
        width: usize,
        position: &MarkdownPosition,
    ) -> Option<MarkdownPosition> {
        let node = match *position {
            MarkdownPosition::Text { node, start } => {
                if !matches!(
                    self.tree.nodes[node].data,
                    Element::ThematicBreak | Element::ListItem
                ) {
                    let rows = self.tree.cache_rows(source, node, width, &mut self.rows);
                    let row = rows
                        .binary_search_by_key(&start.map(|p| (p.node, p.byte)), |p| {
                            p.map(|p| (p.node, p.byte))
                        })
                        .unwrap();
                    if row > 0 {
                        return Some(MarkdownPosition::Text {
                            node,
                            start: rows[row - 1],
                        });
                    }
                }
                node
            }
            MarkdownPosition::Gap { node } => return self.before_block(source, width, node),
            MarkdownPosition::Separator { first_cell } => {
                let (table, _, column) = self.tree.cell_location(first_cell);
                let row = self.tree.table_rows(table).last().unwrap();
                let first_cell = self
                    .tree
                    .children(row)
                    .nth(column - self.tree.columns_per_group(table, width))
                    .unwrap();
                let height = self.tree.table_row_height(
                    source,
                    first_cell,
                    width,
                    &mut self.rows,
                    &mut self.columns,
                );
                return Some(MarkdownPosition::Table {
                    first_cell,
                    line: height - 1,
                });
            }
            MarkdownPosition::Table { first_cell, line } => {
                if line > 0 {
                    return Some(MarkdownPosition::Table {
                        first_cell,
                        line: line - 1,
                    });
                }
                let (table, row, column) = self.tree.cell_location(first_cell);
                if let Some(previous) = self
                    .tree
                    .table_rows(table)
                    .take_while(|&id| id != row)
                    .last()
                {
                    let first_cell = self.tree.children(previous).nth(column).unwrap();
                    let height = self.tree.table_row_height(
                        source,
                        first_cell,
                        width,
                        &mut self.rows,
                        &mut self.columns,
                    );
                    return Some(MarkdownPosition::Table {
                        first_cell,
                        line: height - 1,
                    });
                }
                if column > 0 {
                    return Some(MarkdownPosition::Separator { first_cell });
                }
                table
            }
        };
        let block = self
            .tree
            .blocks(source)
            .find(|block| block.node_idx == node)
            .unwrap();
        if block.needs_gap {
            Some(MarkdownPosition::Gap { node })
        } else {
            self.before_block(source, width, node)
        }
    }

    /// Zero selects the pending suffix boundary, after every parsed block.
    pub fn position_before_node(
        &mut self,
        source: &str,
        width: usize,
        node: usize,
    ) -> Option<MarkdownPosition> {
        let previous = self
            .tree
            .blocks(source)
            .take_while(|block| block.node_idx != 0 && (node == 0 || block.node_idx < node))
            .last()?;
        Some(self.last_in(source, width, previous))
    }

    pub fn reflow(
        &mut self,
        source: &str,
        old_width: usize,
        width: usize,
        position: &MarkdownPosition,
    ) -> Option<MarkdownReflow> {
        match *position {
            MarkdownPosition::Text {
                node,
                start: Some(start),
            } => {
                // Resize needs the old row's end only here. Reconstruct it from the old width and byte position.
                let (content, old, prefix) = self.tree.row_at(source, node, Some(start), old_width);
                let line = content.line_start(Some(start));
                let end = content.line_end(start);
                let mut first = None;
                let mut last = None;
                let mut before = None;
                for &start in self.tree.cache_rows(source, node, width, &mut self.rows) {
                    if content.line_start(start) != line {
                        continue;
                    }
                    let position = MarkdownPosition::Text { node, start };
                    first.get_or_insert(position);
                    last = Some(position);
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
                    if row.start != row.end
                        && row
                            .end
                            .zip(old.end)
                            .is_some_and(|(end, old)| content.offset(end) <= content.offset(old))
                    {
                        before = Some(position);
                    }
                }
                if old_width <= prefix || old.end == end {
                    last.map(MarkdownReflow::At)
                } else if let Some(before) = before {
                    Some(MarkdownReflow::At(before))
                } else {
                    first.map(MarkdownReflow::Before)
                }
            }
            MarkdownPosition::Table { first_cell, .. }
            | MarkdownPosition::Separator { first_cell } => {
                let (table, _, column) = self.tree.cell_location(first_cell);
                let group_size = self.tree.columns_per_group(table, width);
                let column = column / group_size * group_size;
                let head = self.tree.table_rows(table).next()?;
                let first_cell = self.tree.children(head).nth(column).unwrap();
                Some(MarkdownReflow::Before(if column == 0 {
                    MarkdownPosition::Table {
                        first_cell,
                        line: 0,
                    }
                } else {
                    MarkdownPosition::Separator { first_cell }
                }))
            }
            _ => Some(MarkdownReflow::At(*position)),
        }
    }
}
