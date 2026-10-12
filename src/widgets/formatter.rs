//! Measure one visual row at a time. Sources borrow text; resumable state and row boundaries do not.

use std::{
    borrow::Cow,
    marker::PhantomData,
    ops::{Deref, Range},
    ptr::NonNull,
};

use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_segmentation::{GraphemeCursor, GraphemeIncomplete};
use unicode_width::UnicodeWidthStr;

pub struct Chunk<'a> {
    pub content: &'a str,
    /// Byte offset in the source's resolved text, independent of its underlying storage.
    pub offset: usize,
}

// TP2: Now that I consider it, it does not seem appropriate to have Highlight be defined as a grapheme
// unit. At least, not with the current architecture where Ranges are used. This points to a crucial understanding
// of the problem: the *content* is the concern of ratatui, not Walker's nor Transcript's caches. Therefore,
// the return type should be one that provides the data to derive the content from. For a plain string, this
// would just be a Range covering the text content. Wait... but continuation must actually be known to the renderer,
// so some content *does* have to be known that cannot be directly derived from the source. This is cyclical.
// I have incorrectly managed resource flow.
//
// Proposition (pending on review of how tables are rendered):
// - ChunkCollection, after the other changes are implemented, would be represent some content and express
// its continuation marker.
// - Therefore: it possesses the information that the renderer cares about except for knowledge on whether the
// continuation marker ought to be injected for its representation and the span up to which it should be rendered
// per the soft-wrapping algorithm.
// - What if formatter takes on the role of soft-wrapping exactly one line per the rules, according to the
// current state of the input it is operating on? What if a separate State type is maintained that is used
// alongside ChunkCollection? What if a separate Source type is used to make ChunkCollection a non-borrowing
// type that can be passed around? ChunkCollection could thus represent its own state.
// Implemented below: Source borrows text only during an operation, State is non-borrowing, and Row identifies content.
// TP1: Styles no longer pass through the formatter; drawing resolves them from the input's own representation.
pub trait Source {
    fn len_bytes(&self) -> usize;
    /// Return the nonempty chunk containing byte. Callers only request bytes below len_bytes().
    fn chunk(&self, byte: usize) -> Chunk<'_>;
    /// Current and continuation prefix widths. Prefix text and styles remain the source renderer's concern.
    fn prefixes(&self, state: &State, width: usize) -> (usize, usize);
}

pub struct PlainText<'a>(pub &'a str);

pub const SOFT_WRAP_INDICATOR: &str = "↪ ";

impl Source for PlainText<'_> {
    fn len_bytes(&self) -> usize {
        self.0.len()
    }
    fn chunk(&self, _: usize) -> Chunk<'_> {
        Chunk {
            content: self.0,
            offset: 0,
        }
    }
    fn prefixes(&self, state: &State, width: usize) -> (usize, usize) {
        let continuation = if width > SOFT_WRAP_INDICATOR.width() {
            SOFT_WRAP_INDICATOR.width()
        } else {
            0
        };
        (if state.soft { continuation } else { 0 }, continuation)
    }
}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub byte: usize,
    pub soft: bool,
    pub finished: bool,
}

// TP1: this should maybe be reconsidered? It's seemingly necessary by the current formatter design, but that
// doesn't equate to being necessary for the actual target task (knowing what text to render). For now, I'd prob
// leave it (but just as the Range, I don't think width is necessary as I don't believe the algorithm needs width
// to work correctly).
// The range identifies the content; width is transient measurement used for table-cell alignment, not cached per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub span: Range<usize>,
    pub width: usize,
}

#[derive(Clone)]
pub struct Graphemes {
    cursor: GraphemeCursor,
}

impl Graphemes {
    pub fn new(source: &impl Source, byte: usize) -> Self {
        Self {
            cursor: GraphemeCursor::new(byte, source.len_bytes(), true),
        }
    }

    pub fn byte(&self) -> usize {
        self.cursor.cur_cursor()
    }

    pub fn next<'a>(&mut self, source: &'a impl Source) -> Option<(Range<usize>, Cow<'a, str>)> {
        let start = self.byte();
        if start == source.len_bytes() {
            return None;
        }
        let mut chunk = source.chunk(start);
        let end = loop {
            match self.cursor.next_boundary(chunk.content, chunk.offset) {
                Ok(Some(end)) => break end,
                Ok(None) => break source.len_bytes(),
                Err(GraphemeIncomplete::NextChunk) => {
                    chunk = source.chunk(chunk.offset + chunk.content.len())
                }
                Err(GraphemeIncomplete::PreContext(end)) => {
                    let context = source.chunk(end - 1);
                    self.cursor
                        .provide_context(&context.content[..end - context.offset], context.offset);
                }
                Err(GraphemeIncomplete::PrevChunk | GraphemeIncomplete::InvalidOffset) => {
                    unreachable!()
                }
            }
        };
        let first = source.chunk(start);
        if end <= first.offset + first.content.len() {
            return Some((
                start..end,
                Cow::Borrowed(&first.content[start - first.offset..end - first.offset]),
            ));
        }
        // Style boundaries can divide an extended grapheme. Only that grapheme needs joined text for measurement.
        let mut text = String::with_capacity(end - start);
        let mut byte = start;
        while byte < end {
            let chunk = source.chunk(byte);
            let next = end.min(chunk.offset + chunk.content.len());
            text.push_str(&chunk.content[byte - chunk.offset..next - chunk.offset]);
            byte = next;
        }
        Some((start..end, Cow::Owned(text)))
    }
}

fn is_separator_character(character: char) -> bool {
    if character.is_ascii() {
        // ASCII whitespace and punctuation are separators. Symbols such as '$' and '+' stay within words.
        return matches!(character,
            '\t'..='\r' | ' ' | '!'..='#' | '%'..='*' | ','..='/' | ':'..=';'
                | '?'..='@' | '['..=']' | '_' | '{' | '}');
    }
    character.is_whitespace()
        || matches!(
            get_general_category(character),
            GeneralCategory::ClosePunctuation
                | GeneralCategory::ConnectorPunctuation
                | GeneralCategory::DashPunctuation
                | GeneralCategory::FinalPunctuation
                | GeneralCategory::InitialPunctuation
                | GeneralCategory::OpenPunctuation
                | GeneralCategory::OtherPunctuation
        )
}

impl State {
    pub fn next_row(&mut self, source: &impl Source, width: usize) -> Option<Row> {
        if self.finished {
            return None;
        }
        // TP1: `initial` and `continuation` must be data that the input implementation supplies opaquely.
        // Source now supplies their widths, with no styled prefix arguments or prefix-emission states.
        let (prefix, continuation) = source.prefixes(self, width);
        let mut row = Row {
            span: self.byte..self.byte,
            width: 0,
        };
        let mut graphemes = Graphemes::new(source, self.byte);
        let mut splitting_word = if self.byte > 0 {
            let chunk = source.chunk(self.byte - 1);
            chunk.content[..self.byte - chunk.offset]
                .chars()
                .next_back()
                .is_some_and(|c| !is_separator_character(c))
        } else {
            false
        };
        let mut committed_word_end = None;
        self.soft = false;
        while let Some((range, text)) = graphemes.next(source) {
            if matches!(text.as_ref(), "\n" | "\r" | "\r\n") {
                self.byte = range.end;
                self.finished = self.byte == source.len_bytes();
                return Some(row);
            }
            if width <= prefix {
                // A prefix-only row consumes the physical line without displaying its body.
                self.byte = range.end;
                continue;
            }
            let columns = if text == "\t" {
                4 - (prefix + row.width) % 4
            } else {
                text.width()
            };
            let separator = text.chars().any(is_separator_character);
            if committed_word_end == Some(range.start) {
                committed_word_end = None;
            }
            if committed_word_end.is_none() && !splitting_word && !separator {
                // Look ahead only far enough to choose the current row, a continuation row, or a split word.
                // Remember a fitting word's end so its remaining graphemes do not repeat the lookahead.
                let mut word = graphemes.clone();
                let mut word_width = columns;
                let current_capacity = width.saturating_sub(prefix + row.width);
                let continuation_capacity = width
                    .checked_sub(continuation)
                    .map(|capacity| capacity.min(40));
                let limit = if row.span.is_empty() {
                    current_capacity
                } else {
                    current_capacity.max(continuation_capacity.unwrap_or(0))
                };
                let complete = loop {
                    if word_width > limit {
                        break false;
                    }
                    let word_end = word.clone();
                    let Some((_, next)) = word.next(source) else {
                        break true;
                    };
                    if next.chars().any(is_separator_character) {
                        word = word_end;
                        break true;
                    }
                    word_width += next.width();
                };
                if complete && word_width <= current_capacity {
                    committed_word_end = Some(word.byte());
                } else if complete
                    && !row.span.is_empty()
                    && continuation_capacity.is_some_and(|capacity| word_width <= capacity)
                {
                    self.soft = true;
                    return Some(row);
                } else {
                    splitting_word = true;
                }
            }
            if prefix + row.width + columns > width && !row.span.is_empty() {
                self.soft = true;
                return Some(row);
            }
            splitting_word &= !separator;
            // An oversized first grapheme still advances. Never split a grapheme to fit a narrow row.
            row.width += columns;
            row.span.end = range.end;
            self.byte = range.end;
        }
        self.finished = true;
        Some(row)
    }
}
