use std::ops::Range;

use ratatui::prelude::{Span, Style};
use unicode_width::UnicodeWidthStr;

use crate::widgets::{expand_tabs, soft_spans};

pub const TOOL_OUTPUT_LINE_NUMBER_WIDTH: usize = 4;

/// A source line with styles attached to byte ranges. The ranges partition `text` and remain valid after wrapping.
pub struct HighlightedLine<'a> {
    pub text: &'a str,
    pub highlights: Box<[(Style, Range<usize>)]>,
    // TODO: Remove this in favor of a better notion of highlights...
    pub marker_space: bool,
}

impl<'a> HighlightedLine<'a> {
    pub fn plain(text: &'a str, style: Style) -> Self {
        Self {
            text,
            highlights: Box::new([(style, 0..text.len())]),
            marker_space: false,
        }
    }

    pub fn wrap(&self, start: usize, width: usize) -> Vec<Range<usize>> {
        if self.marker_space {
            let mut ranges = soft_spans(&self.text[1..], start + 1, width, 2).collect::<Vec<_>>();
            ranges[0].start = start;
            ranges
        } else {
            soft_spans(self.text, start, width, 0).collect()
        }
    }

    pub fn render(&self, segment: Range<usize>, continuation_prefix: usize) -> Vec<Span<'a>> {
        let mut spans = Vec::new();
        let mut column = continuation_prefix;
        for (index, (style, highlight)) in self.highlights.iter().enumerate() {
            let start = segment.start.max(highlight.start);
            let end = segment.end.min(highlight.end);
            if start >= end {
                continue;
            }
            let expanded = expand_tabs(&self.text[start..end], column);
            column += expanded.as_ref().width();
            spans.push(Span::styled(expanded, *style));
            if self.marker_space && segment.start == 0 && index == 0 {
                spans.push(Span::styled(
                    " ",
                    self.highlights.get(1).map_or(*style, |(style, _)| *style),
                ));
                column += 1;
            }
        }
        spans
    }
}

#[cfg(test)]
mod tests {
    use ratatui::prelude::{Color, Style};

    use super::HighlightedLine;

    #[test]
    fn highlights_remain_attached_when_a_span_crosses_a_wrap_boundary() {
        let line = HighlightedLine {
            text: "abcdefghij",
            highlights: Box::new([
                (Style::new().fg(Color::Red), 0..3),
                (Style::new().fg(Color::Blue), 3..8),
                (Style::new().fg(Color::Green), 8..10),
            ]),
            marker_space: false,
        };
        let rows = line.wrap(0, 5);
        assert!(rows.len() > 1);
        let styled_chars = rows
            .iter()
            .flat_map(|row| line.render(row.clone(), 0))
            .flat_map(|span| {
                span.content
                    .chars()
                    .map(|character| (character, span.style.fg.unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            styled_chars
                .iter()
                .map(|(character, _)| *character)
                .collect::<String>(),
            line.text
        );
        assert_eq!(
            styled_chars
                .iter()
                .map(|(_, color)| *color)
                .collect::<Vec<_>>(),
            [
                Color::Red,
                Color::Red,
                Color::Red,
                Color::Blue,
                Color::Blue,
                Color::Blue,
                Color::Blue,
                Color::Blue,
                Color::Green,
                Color::Green,
            ]
        );
    }
}
