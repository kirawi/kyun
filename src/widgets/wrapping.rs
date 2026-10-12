//! Borrow resolved leaf text and map formatter boundaries back to tree positions.
//! Text stays in source/arena storage. This adapter contains no wrapping rules or retained styles.

use std::ops::Range;

use unicode_width::UnicodeWidthStr;

use super::formatter::{self, Chunk, Graphemes, SOFT_WRAP_INDICATOR, Source};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position {
    // Byte offsets are local to resolved leaf text, not Markdown source and not terminal columns.
    // Node zero uses absolute source bytes for the pending suffix; it is never traversed as Markdown.
    pub node: usize,
    pub byte: usize,
}

#[derive(Clone, Debug, Default)]
pub struct State {
    // A block or cell owns this cursor, even when the cursor points into one of its inline descendants.
    pub next: Option<Position>,
    pub rows: usize,
    pub soft: bool,
    pub finished: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    // The end is exclusive. An empty cell has no text position; its owning cell still identifies the row.
    pub start: Option<Position>,
    pub end: Option<Position>,
    pub width: usize,
}

struct Part<'a> {
    node: usize,
    byte: usize,
    offset: usize,
    text: &'a str,
}

pub struct Content<'a> {
    parts: Vec<Part<'a>>,
    len: usize,
    pub indent: usize,
    pub heading: usize,
}

impl Source for Content<'_> {
    fn len_bytes(&self) -> usize {
        self.len
    }

    fn chunk(&self, byte: usize) -> Chunk<'_> {
        let part = &self.parts[self.parts.partition_point(|part| part.offset <= byte) - 1];
        Chunk {
            content: part.text,
            offset: part.offset,
        }
    }

    fn prefixes(&self, state: &formatter::State, width: usize) -> (usize, usize) {
        let base = self.indent
            + if self.line_start(self.position(state.byte)) == 0 {
                self.heading
            } else {
                0
            };
        let marker = if width.saturating_sub(base) > SOFT_WRAP_INDICATOR.width() {
            SOFT_WRAP_INDICATOR.width()
        } else {
            0
        };
        (base + if state.soft { marker } else { 0 }, base + marker)
    }
}

impl<'a> Content<'a> {
    pub fn new(parts: impl IntoIterator<Item = (usize, &'a str)>) -> Self {
        Self::with_offsets(parts.into_iter().map(|(node, text)| (node, 0, text)))
    }

    pub fn with_offsets(parts: impl IntoIterator<Item = (usize, usize, &'a str)>) -> Self {
        let mut content = Self {
            parts: Vec::new(),
            len: 0,
            indent: 0,
            heading: 0,
        };
        for (node, byte, text) in parts {
            if text.is_empty() {
                continue;
            }
            content.parts.push(Part {
                node,
                byte,
                offset: content.len,
                text,
            });
            content.len += text.len();
        }
        content
    }

    pub fn position(&self, byte: usize) -> Option<Position> {
        // At an inline boundary, use the next leaf's byte zero. At the end, use the final leaf's length.
        self.parts
            .get(
                self.parts
                    .partition_point(|part| part.offset <= byte)
                    .saturating_sub(1),
            )
            .map(|part| Position {
                node: part.node,
                byte: part.byte + byte - part.offset,
            })
    }

    pub fn offset(&self, position: Position) -> usize {
        let part = &self.parts[self
            .parts
            .binary_search_by_key(&position.node, |part| part.node)
            .unwrap()];
        part.offset + position.byte - part.byte
    }

    pub fn prefix_width(&self, start: Option<Position>, width: usize) -> usize {
        self.prefixes(
            &formatter::State {
                byte: start.map_or(0, |start| self.offset(start)),
                soft: self.continuation(start),
                finished: false,
            },
            width,
        )
        .0
    }

    pub fn next_row(&self, state: &mut State, width: usize) -> Option<Row> {
        let mut cursor = formatter::State {
            byte: state.next.map_or(0, |position| self.offset(position)),
            soft: state.soft,
            finished: state.finished,
        };
        let row = cursor.next_row(self, width)?;
        state.next = self.position(cursor.byte);
        state.soft = cursor.soft;
        state.finished = cursor.finished;
        state.rows += 1;
        Some(Row {
            start: self.position(row.span.start),
            end: self.position(row.span.end),
            width: row.width,
        })
    }

    pub fn natural_width(&self, column: usize) -> usize {
        let mut widest = 0;
        let mut used = column;
        let mut graphemes = Graphemes::new(self, 0);
        while let Some((_, text)) = graphemes.next(self) {
            if matches!(text.as_ref(), "\n" | "\r" | "\r\n") {
                widest = widest.max(used - column);
                used = column;
            } else {
                used += if text == "\t" {
                    4 - used % 4
                } else {
                    text.width()
                };
            }
        }
        widest.max(used - column)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn line_end(&self, position: Position) -> Option<Position> {
        let mut byte = self.offset(position);
        while byte < self.len {
            let chunk = self.chunk(byte);
            if let Some(end) = chunk.content[byte - chunk.offset..].find(['\r', '\n']) {
                return self.position(byte + end);
            }
            byte = chunk.offset + chunk.content.len();
        }
        self.position(byte)
    }

    pub fn line_start(&self, position: Option<Position>) -> usize {
        let mut byte = position.map_or(0, |position| self.offset(position));
        while byte > 0 {
            let chunk = self.chunk(byte - 1);
            if let Some(newline) = chunk.content[..byte - chunk.offset].rfind(['\r', '\n']) {
                return chunk.offset + newline + 1;
            }
            byte = chunk.offset;
        }
        0
    }

    pub fn continuation(&self, position: Option<Position>) -> bool {
        position.is_some_and(|position| self.offset(position) != self.line_start(Some(position)))
    }

    pub fn fragments(&self, row: &Row, prefix_width: usize) -> Vec<(usize, Range<usize>, usize)> {
        let Some(start) = row.start else {
            return Vec::new();
        };
        let mut fragments: Vec<(usize, Range<usize>, usize)> = Vec::new();
        let mut column = prefix_width;
        let end = self.offset(row.end.unwrap());
        let mut graphemes = Graphemes::new(self, self.offset(start));
        while graphemes.byte() < end {
            let (range, text) = graphemes.next(self).unwrap();
            let spaces = if text == "\t" { 4 - column % 4 } else { 0 };
            // Measure the entire grapheme before splitting it into styled leaf ranges.
            let first = self
                .parts
                .partition_point(|part| part.offset <= range.start)
                - 1;
            for part in self.parts[first..]
                .iter()
                .take_while(|part| part.offset < range.end)
            {
                let bytes = part.byte + range.start.max(part.offset) - part.offset
                    ..part.byte + range.end.min(part.offset + part.text.len()) - part.offset;
                if let Some((node, previous, previous_spaces)) = fragments.last_mut()
                    && *node == part.node
                    && *previous_spaces == 0
                    && spaces == 0
                    && previous.end == bytes.start
                {
                    previous.end = bytes.end;
                } else {
                    fragments.push((part.node, bytes, spaces));
                }
            }
            column += if spaces > 0 { spaces } else { text.width() };
        }
        fragments
    }

    pub fn render(&self, row: &Row, prefix_width: usize) -> String {
        self.fragments(row, prefix_width)
            .into_iter()
            .map(|(node, range, spaces)| {
                if spaces > 0 {
                    return " ".repeat(spaces);
                }
                let part = &self.parts[self
                    .parts
                    .binary_search_by_key(&node, |part| part.node)
                    .unwrap()];
                part.text[range.start - part.byte..range.end - part.byte].to_owned()
            })
            .collect()
    }
}
