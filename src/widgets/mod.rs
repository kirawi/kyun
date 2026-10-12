pub mod approval;
pub mod command_line;
pub mod prompt;
pub mod statusline;
pub mod transcript;

mod formatter;
pub mod wrapping;

use kyun_agent::ToolCall;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Color;
use std::borrow::Cow;
use std::ops::Range;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_segmentation::{GraphemeIndices, UnicodeSegmentation};
use unicode_width::UnicodeWidthStr;

pub use approval::Approval;
pub use command_line::CommandLine;
pub use prompt::Prompt;
pub use statusline::StatusLine;
pub use transcript::Transcript;

use crate::tools::ToolKind;

use formatter::SOFT_WRAP_INDICATOR;

fn soft_wrap_prefix_width(width: usize) -> usize {
    if width > SOFT_WRAP_INDICATOR.width() {
        SOFT_WRAP_INDICATOR.width()
    } else {
        0
    }
}

/// The column distance between tab stops. A tab always advances at least one column.
const TAB_STOP: usize = 4;

/// The number of columns a tab occupies when rendered at `column`: tabs advance to the next multiple of
/// [`TAB_STOP`].
fn tab_width(column: usize) -> usize {
    TAB_STOP - column % TAB_STOP
}

/// Expands tabs into the spaces they occupy so the rendered text matches the wrapping widths: ratatui does not
/// expand raw tabs into tab stops. `column` is the column the text starts at and must match the
/// `line_width` base used by the wrapping [`TextFormatter`]: 0 at a line start, or the soft-wrap indicator's width on
/// a continuation line.
pub fn expand_tabs(text: &str, column: usize) -> Cow<'_, str> {
    if !text.contains('\t') {
        return Cow::Borrowed(text);
    }

    let mut expanded = String::with_capacity(text.len() + TAB_STOP);
    let mut column = column;
    for grapheme in text.graphemes(true) {
        if grapheme == "\t" {
            let width = tab_width(column);
            expanded.extend(std::iter::repeat_n(' ', width));
            column += width;
        } else {
            expanded.push_str(grapheme);
            column += grapheme.width();
        }
    }
    Cow::Owned(expanded)
}

/// The width `text` occupies once its tabs are expanded from `column`.
pub fn display_width(text: &str, column: usize) -> usize {
    expand_tabs(text, column).width()
}

pub struct BaseAreas {
    pub transcript: Rect,
    pub prompt: Rect,
    pub status_line: Rect,
    pub command_line: Rect,
}

/// Certain components are specially laid out in relation to each other. This function provides the areas
/// for each respective UI element.
pub fn base_areas(area: Rect) -> BaseAreas {
    // TODO: cache it once and update as necessary on resize rather than recalculate every time
    let areas = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(6),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);
    BaseAreas {
        transcript: areas[0],
        prompt: areas[1],
        status_line: areas[2],
        command_line: areas[3],
    }
}

fn previous_word_boundary(line: &str, cursor: usize) -> usize {
    let mut boundary = cursor;
    for (idx, ch) in line[..cursor].char_indices().rev() {
        if ch.is_alphanumeric() || ch == '_' {
            break;
        }
        boundary = idx;
    }
    for (idx, ch) in line[..boundary].char_indices().rev() {
        if !ch.is_alphanumeric() && ch != '_' {
            break;
        }
        boundary = idx;
    }
    boundary
}

fn next_word_boundary(line: &str, cursor: usize) -> usize {
    let mut boundary = cursor;
    while let Some(ch) = line[boundary..].chars().next() {
        if !ch.is_alphanumeric() && ch != '_' {
            break;
        }
        boundary += ch.len_utf8();
    }
    while let Some(ch) = line[boundary..].chars().next() {
        if ch.is_alphanumeric() || ch == '_' {
            break;
        }
        boundary += ch.len_utf8();
    }
    boundary
}

pub fn real_lines(text: &str) -> impl Iterator<Item = &str> {
    let mut remaining = Some(text);
    std::iter::from_fn(move || {
        let line = remaining.take()?;
        let Some(ending_start) = memchr::memchr2(b'\r', b'\n', line.as_bytes()) else {
            return Some(line);
        };
        let mut next_line_start = ending_start;

        // Don't include preceding newlines in the next line to be emitted.
        // This means that each line is only returned with its content without any newlines.
        if line.as_bytes().get(next_line_start) == Some(&b'\r') {
            next_line_start += 1;
        }
        if line.as_bytes().get(next_line_start) == Some(&b'\n') {
            next_line_start += 1;
        }
        remaining = Some(&line[next_line_start..]);
        Some(&line[..ending_start])
    })
}

/// Soft-wrap spans of a real line, as absolute byte ranges into the text that `real_line` was sliced from.
/// `column` is the screen column the first segment starts at, so tabs advance to the same stops the
/// renderer will use; continuation segments always start at the soft-wrap indicator's column.
pub fn soft_spans(
    real_line: &str,
    mut start: usize,
    width: usize,
    column: usize,
) -> impl Iterator<Item = Range<usize>> {
    TextFormatter::new(real_line, width, column).map(move |(line, _)| {
        let span = start..start + line.len();
        start = span.end;
        span
    })
}

#[derive(Clone)]
struct WrapToken {
    range: Range<usize>,
    width: usize,
    separator: bool,
    tab: bool,
    line_ending: bool,
}

#[derive(Clone)]
struct StrGraphemes<'a> {
    inner: GraphemeIndices<'a>,
}

impl<'a> StrGraphemes<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            inner: text.grapheme_indices(true),
        }
    }
}

impl Iterator for StrGraphemes<'_> {
    type Item = WrapToken;

    fn next(&mut self) -> Option<Self::Item> {
        let (start, grapheme) = self.inner.next()?;
        Some(WrapToken {
            range: start..start + grapheme.len(),
            width: grapheme.width(),
            separator: is_word_separator(grapheme),
            tab: grapheme == "\t",
            line_ending: grapheme.starts_with(['\r', '\n']),
        })
    }
}

struct RangeFormatter<I: Iterator> {
    tokens: std::iter::Peekable<I>,
    text_len: usize,
    max_width: usize,
    /// The screen column the first segment starts at, which is also the base `expand_tabs` is given for it.
    start_column: usize,
    /// The starting column for continuation segments.
    continuation_width: usize,
    cursor: usize,
    splitting_word: bool,
    continuation: bool,
    finished: bool,
}

impl<I: Iterator<Item = WrapToken>> RangeFormatter<I> {
    fn new(tokens: I, text_len: usize, max_width: usize, start_column: usize) -> Self {
        Self {
            tokens: tokens.peekable(),
            text_len,
            max_width,
            start_column,
            continuation_width: soft_wrap_prefix_width(max_width),
            cursor: 0,
            splitting_word: false,
            continuation: false,
            finished: false,
        }
    }
}

fn is_word_separator(grapheme: &str) -> bool {
    grapheme.chars().any(|character| {
        if character.is_ascii() {
            // ASCII whitespace and Unicode punctuation; ASCII symbols such as '$' and '+' are not separators.
            // TODO: check if this behavior makes sense!
            return matches!(
                character,
                '\t'..='\r' | ' ' | '!'..='#' | '%'..='*' | ','..='/' | ':'..=';'
                    | '?'..='@' | '['..=']' | '_' | '{' | '}'
            );
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
    })
}

// TODO: need to reserve one final column for a whitespace char? or figure out a better way to handle splitting whitespace in general
// currently can be confusing.
impl<I> Iterator for RangeFormatter<I>
where
    I: Iterator<Item = WrapToken> + Clone,
{
    type Item = (Range<usize>, bool);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if self.text_len == 0 || self.max_width == 0 {
            // In such a case when no text can be rendered, this is the best interpretation.
            // That is to say: the whole *real* line gets consumed but no text can be rendered.
            self.finished = true;
            return Some((0..0, false));
        }

        let line_start = self.cursor;
        let is_continuation = self.continuation;

        let mut line_width = if is_continuation {
            self.continuation_width
        } else {
            self.start_column
        };
        loop {
            let Some(grapheme) = self.tokens.peek().cloned() else {
                self.finished = true;
                return Some((line_start..self.text_len, is_continuation));
            };
            if grapheme.line_ending {
                self.finished = true;
                return Some((line_start..grapheme.range.start, is_continuation));
            }
            // A tab advances to the next tab stop instead of the zero width unicode-width assigns control characters,
            // so wrapping agrees with `expand_tabs` at render time. `line_width` includes the continuation prefix,
            // which is also the base `expand_tabs` is given for continuation lines.
            let g_width = if grapheme.tab {
                tab_width(line_width)
            } else {
                grapheme.width
            };

            // Special-cased behavior for when we are inside a split word (i.e. the final word of the prior line exceeded
            // min(40, max_width - prefix) cells in width). Tries to render all graphemes to the end of each continuation
            // line without trying to fit the whole word on any single line.
            if self.splitting_word {
                // INVARIANT: It is possible for `line_width + g_width > self.max_width`.
                //
                // In such a case, the `self.cursor > line_start` guard ensures that the grapheme is vacuously added to the
                // next virtual line. `ratatui` will try to clip it based on its own definition of width, and then it's up
                // to the terminal to render the grapheme however it sees fit. Otherwise, an infinite loop would trigger.
                // A tab participates in this exception too: below `TAB_STOP` columns, a line-leading tab expands past
                // the budget and is clipped like any oversized grapheme.
                if (line_width + g_width > self.max_width) && grapheme.range.start > line_start {
                    self.continuation = true;
                    return Some((line_start..grapheme.range.start, is_continuation));
                }

                self.tokens.next();
                self.cursor = grapheme.range.end;
                line_width += g_width;
                if grapheme.separator {
                    // The entire split word has been consumed.
                    self.splitting_word = false;
                }
                continue;
            }

            if grapheme.separator {
                // Word separators are consumed to the end of each line without special-cased behavior.
                if line_width + g_width > self.max_width && grapheme.range.start > line_start {
                    self.continuation = true;
                    return Some((line_start..grapheme.range.start, is_continuation));
                }

                self.tokens.next();
                self.cursor = grapheme.range.end;
                line_width += g_width;
                continue;
            }

            // FIXME: Clone-based lookahead makes token sources either repeat tokenization or materialize every
            // token up front. Buffer the current word so the formatter can consume a source once from left to right.
            let mut word = self.tokens.clone();
            let mut word_end = grapheme.range.start;
            let mut word_width = 0;
            while let Some(grapheme) = word.peek().cloned() {
                if grapheme.separator || grapheme.line_ending {
                    break;
                }
                word_end = grapheme.range.end;
                word_width += grapheme.width;
                word.next();
            }

            if line_width + word_width <= self.max_width {
                self.tokens = word;
                self.cursor = word_end;
                line_width += word_width;
                continue;
            } // word doesn't fit on the current line

            // Check if the word can be put on the next line as a whole.
            if word_width <= 40
                && word_width + self.continuation_width <= self.max_width
                && self.cursor > line_start
            {
                self.continuation = true;
                return Some((line_start..self.cursor, is_continuation));
            } // needs to be split

            self.splitting_word = true;
        }
    }
}

struct TextFormatter<'a> {
    text: &'a str,
    ranges: RangeFormatter<StrGraphemes<'a>>,
}

impl<'a> TextFormatter<'a> {
    pub fn new(text: &'a str, max_width: usize, start_column: usize) -> Self {
        Self {
            text,
            ranges: RangeFormatter::new(
                StrGraphemes::new(text),
                text.len(),
                max_width,
                start_column,
            ),
        }
    }
}

impl<'a> Iterator for TextFormatter<'a> {
    type Item = (&'a str, bool);

    fn next(&mut self) -> Option<Self::Item> {
        self.ranges
            .next()
            .map(|(range, continuation)| (&self.text[range], continuation))
    }
}

pub const ASSISTANT: Color = Color::Rgb(0x35, 0xC8, 0xFF);
pub const USER: Color = Color::Rgb(0x5C, 0x63, 0x70);
pub const STATUS_ACCENT: Color = Color::Rgb(0xFF, 0x00, 0xCB);

// TUI theme
pub const TERM_BG: Color = Color::Rgb(0x0D, 0x0D, 0x0D);
pub const TERM_BORDER: Color = Color::Rgb(0x33, 0x33, 0x33);
pub const BOX_BORDER: Color = Color::Rgb(0x4A, 0x4A, 0x4A);
pub const TEXT: Color = Color::Rgb(0xE2, 0xE8, 0xF0);
pub const MUTED: Color = Color::Rgb(0x71, 0x80, 0x96);
pub const TURN: Color = Color::Rgb(0xE5, 0xC0, 0x7B);
pub const VIRTUAL: Color = Color::Rgb(0x4B, 0x52, 0x63);

// Tools
pub const TOOL: Color = Color::Rgb(0x98, 0xC3, 0x79);
pub const PENDING: Color = Color::Rgb(0xE5, 0xC0, 0x7B);
pub const DIFF_ADD: Color = Color::Rgb(0x8F, 0xB5, 0x73);
pub const DIFF_TEXT: Color = Color::Rgb(0xE6, 0xE6, 0xE6);
pub const JUMP: Color = Color::Rgb(0xE0, 0x6C, 0x75);
pub const DIFF_TRUNCATED: Color = Color::Rgb(0xD1, 0xD1, 0xD1);
pub const DIFF_TRUNCATED_BG: Color = Color::Rgb(0x13, 0x2C, 0x4F);

// Code
pub const CODE_BG: Color = Color::Rgb(0x16, 0x16, 0x16);

// Syntax highlighting
pub const SYN_KEYWORD: Color = Color::Rgb(0xC6, 0x78, 0xDD);
pub const SYN_FUNC: Color = Color::Rgb(0x61, 0xAF, 0xEF);
pub const SYN_STRING: Color = Color::Rgb(0x98, 0xC3, 0x79);
pub const SYN_BUILTIN: Color = Color::Rgb(0x56, 0xB6, 0xC2);
pub const SYN_COMMENT: Color = Color::Rgb(0x5C, 0x63, 0x70);

// Status bar segment backgrounds
pub const STATUS_BG: Color = Color::Rgb(0x29, 0x29, 0x29);
pub const STATUS_ACTIVE_BG: Color = Color::Rgb(0x13, 0x2c, 0x4F);
pub const STATUS_PENDING_BG: Color = Color::Rgb(0x3d, 0x35, 0x20);
pub const STATUS_MODE_BG: Color = Color::Rgb(0x3B, 0x00, 0x2C);
pub const STATUS_TASK_BG: Color = Color::Rgb(0x24, 0x28, 0x3B);
pub const STATUS_RIGHT_BG: Color = Color::Rgb(0x2F, 0x33, 0x4D);

/// Arguments text shown for a tool call, replacing large payloads with a compact summary.
pub fn get_tool_arguments(call: &ToolCall) -> String {
    match ToolKind::from_function_name(&call.function_name) {
        ToolKind::FileMutation(crate::tools::file_mutation::Kind::Write) => {
            serde_json::from_str::<crate::tools::write::WriteFileParams>(&call.arguments)
                .map(|params| {
                    format!(
                        r#"{{ "path": {} }}"#,
                        serde_json::to_string(&params.path).unwrap()
                    )
                })
                .unwrap_or_else(|_| call.arguments.clone())
        }
        ToolKind::FileMutation(crate::tools::file_mutation::Kind::ApplyPatch) => {
            crate::tools::apply_patch::summarize_input(&call.arguments)
        }
        ToolKind::FileMutation(crate::tools::file_mutation::Kind::Edit) => {
            crate::tools::edit::summarize_input(&call.arguments)
        }
        _ => call.arguments.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TAB_STOP, TextFormatter, display_width, expand_tabs, is_word_separator, real_lines,
    };
    use unicode_general_category::{GeneralCategory, get_general_category};

    #[test]
    fn ascii_separators_match_unicode_classification() {
        for character in '\0'..='\u{7f}' {
            let expected = character == '_'
                || character.is_whitespace()
                || matches!(
                    get_general_category(character),
                    GeneralCategory::ClosePunctuation
                        | GeneralCategory::ConnectorPunctuation
                        | GeneralCategory::DashPunctuation
                        | GeneralCategory::FinalPunctuation
                        | GeneralCategory::InitialPunctuation
                        | GeneralCategory::OpenPunctuation
                        | GeneralCategory::OtherPunctuation
                );
            let mut bytes = [0; 4];
            assert_eq!(
                is_word_separator(character.encode_utf8(&mut bytes)),
                expected,
                "{character:?}"
            );
        }
    }

    #[test]
    fn mixed_grapheme_separators_check_all_characters() {
        for (grapheme, expected) in [
            ("e\u{301}", false),
            ("!\u{301}", true),
            ("\u{600}a", false),
            ("\u{600}!", true),
        ] {
            assert_eq!(is_word_separator(grapheme), expected, "{grapheme:?}");
        }
    }

    #[test]
    fn tabs_expand_to_the_next_four_column_stop() {
        assert_eq!(expand_tabs("a\tbc", 0), "a   bc");
        assert_eq!(expand_tabs("\t", 2), "  ");
        assert_eq!(expand_tabs("a\tb\tc", 0), "a   b   c");
        assert_eq!(expand_tabs("no tabs", 3), "no tabs");
        assert_eq!(display_width("\t\tx", 0), 9);
    }

    #[test]
    fn tabs_occupy_stop_width_when_wrapping() {
        // The tab at column 2 advances to the 4-column stop, so "word" still fits an 8-wide line but wraps on a
        // 7-wide one.
        assert_eq!(
            TextFormatter::new("ab\tword", 8, 0).collect::<Vec<_>>(),
            [("ab\tword", false)]
        );
        assert_eq!(
            TextFormatter::new("ab\tword", 7, 0).collect::<Vec<_>>(),
            [("ab\t", false), ("word", true)]
        );
    }

    #[test]
    fn tabbed_segments_fit_the_wrap_budget_and_lose_no_text() {
        // Includes a word long enough to trigger split-word wrapping so tab behavior is exercised in every mode.
        let text =
            "ab\tc\td\tefgh\tij\tklmno\tp\tq rst\tuvwx\tverylongwordthatsplits\tz\tend\tword";
        for width in 1..=40 {
            let formatter = TextFormatter::new(text, width, 0);
            let mut expected = String::new();
            for (segment, is_continuation) in formatter {
                let base = if is_continuation && width > 2 { 2 } else { 0 };
                // Below `TAB_STOP`, a line-leading tab can exceed the budget like any oversized grapheme and is
                // clipped at render time; every usable width holds the expanded segment within the budget.
                if width >= TAB_STOP {
                    assert!(
                        display_width(segment, base) <= width,
                        "width={width} base={base} seg={segment:?}"
                    );
                }
                expected.push_str(segment);
            }
            assert_eq!(expected, text);
        }
    }

    #[test]
    fn hard_lines_recognize_cr_lf_and_crlf() {
        assert_eq!(
            real_lines("first\rsecond\r\nthird\n").collect::<Vec<_>>(),
            ["first", "second", "third", ""]
        );
    }

    #[test]
    fn short_words_move_to_the_next_line() {
        assert_eq!(
            TextFormatter::new("hello, world", 8, 0).collect::<Vec<_>>(),
            [("hello, ", false), ("world", true)]
        );
    }

    #[test]
    fn unicode_punctuation_and_underscores_separate_words() {
        assert_eq!(
            TextFormatter::new("hello—world", 8, 0).collect::<Vec<_>>(),
            [("hello—", false), ("world", true)]
        );
        assert_eq!(
            TextFormatter::new("foo_bar", 5, 0).collect::<Vec<_>>(),
            [("foo_", false), ("bar", true)]
        );
    }

    #[test]
    fn words_that_do_not_fit_on_a_continuation_line_are_split() {
        assert_eq!(
            TextFormatter::new("abcdefgh", 4, 0).collect::<Vec<_>>(),
            [("abcd", false), ("ef", true), ("gh", true)]
        );
    }

    #[test]
    fn words_wider_than_forty_cells_are_split_greedily() {
        let text = "a".repeat(45);
        assert_eq!(
            TextFormatter::new(&text, 20, 0).collect::<Vec<_>>(),
            [
                (&text[..20], false),
                (&text[20..38], true),
                (&text[38..], true),
            ]
        );
    }

    #[test]
    fn wrapping_preserves_unicode_graphemes_and_rendered_widths() {
        assert_eq!(
            TextFormatter::new("日本語", 5, 0).collect::<Vec<_>>(),
            [("日本", false), ("語", true)]
        );
        assert_eq!(
            TextFormatter::new("e\u{301}xy", 2, 0).collect::<Vec<_>>(),
            [("e\u{301}x", false), ("y", true)]
        );
    }

    #[test]
    fn separator_runs_fill_each_line() {
        assert_eq!(
            TextFormatter::new("ab    cd", 4, 0).collect::<Vec<_>>(),
            [("ab  ", false), ("  ", true), ("cd", true)]
        );
    }
}
