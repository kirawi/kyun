use std::ops::Range;

use kyun_agent::{AssistantMessage, Message};
use ratatui::prelude::{Modifier, Style};
use unicode_width::UnicodeWidthStr;

use super::{
    MarkdownCache, MarkdownPosition, MarkdownReflow, MdMessage, StreamState,
    tree::{Element, Text},
};
use crate::{
    test_support::agent_with,
    widgets::wrapping::{Content, Position, State},
};

fn positions(document: &mut MdMessage, source: &str, width: usize) -> Vec<MarkdownPosition> {
    let mut rows = Vec::new();
    let mut position = document.first_position(source, width);
    while let Some(current) = position {
        position = document.next(source, width, &current);
        rows.push(current);
    }
    rows
}

fn row_texts(document: &mut MdMessage, source: &str, width: usize) -> Vec<String> {
    positions(document, source, width)
        .iter()
        .map(|row| document.render(source, width, row).text())
        .collect()
}

/// Return the traversal-derived prefix, body, and styles at a width that keeps physical lines together.
fn render(markdown: &str) -> Vec<(String, String, Vec<(Style, Range<usize>)>)> {
    let mut document = MdMessage::parse(markdown);
    positions(&mut document, markdown, 10_000)
        .iter()
        .map(|position| {
            let table = matches!(
                position,
                MarkdownPosition::Table { .. } | MarkdownPosition::Separator { .. }
            );
            let rendered = document.render(markdown, 10_000, position);
            let prefix = if table {
                String::new()
            } else {
                rendered.line.spans[0].content.to_string()
            };
            let mut text = String::new();
            let mut styles: Vec<(Style, Range<usize>)> = Vec::new();
            for span in rendered.line.spans.into_iter().skip(usize::from(!table)) {
                if span.content.is_empty() {
                    continue;
                }
                let start = text.len();
                text.push_str(&span.content);
                if let Some((style, range)) = styles.last_mut()
                    && *style == span.style
                    && range.end == start
                {
                    range.end = text.len();
                } else {
                    styles.push((span.style, start..text.len()));
                }
            }
            (prefix, text, styles)
        })
        .collect()
}

fn line_texts(markdown: &str) -> Vec<String> {
    render(markdown)
        .into_iter()
        .map(|(_, text, _)| text)
        .collect()
}
fn line_prefixes(markdown: &str) -> Vec<String> {
    render(markdown)
        .into_iter()
        .map(|(prefix, _, _)| prefix)
        .collect()
}

#[test]
fn history_reset_retains_markdown_and_discards_removed_documents() {
    use crate::widgets::Transcript;
    use kyun_agent::{ToolCall, ToolCallKind, ToolResult};

    let mut agent = agent_with([
        Message::User("first question".into()),
        Message::Assistant(AssistantMessage {
            reasoning: Some("checking the file".into()),
            content: None,
            tool_calls: vec![ToolCall {
                id: "read".into(),
                kind: ToolCallKind::Function,
                function_name: "read".into(),
                arguments: "{\"path\":\"x\"}".into(),
            }],
        }),
        Message::Tool(vec![ToolResult {
            tool_call_id: "read".into(),
            content: "file contents".into(),
        }]),
        Message::Assistant(AssistantMessage {
            reasoning: None,
            content: Some("retained **paragraph**".into()),
            tool_calls: Vec::new(),
        }),
        Message::User("second question".into()),
        Message::Assistant(AssistantMessage {
            reasoning: None,
            content: Some("removed paragraph".into()),
            tool_calls: Vec::new(),
        }),
    ]);
    let mut transcript = Transcript::new(true);
    transcript.markdown.initialize_for_test(agent.messages());
    transcript.update_viewport(agent.messages(), 40, 8, agent.version(), false, None);
    let nodes = transcript.markdown.entries[0].tree.nodes.as_ptr();
    let boundaries: Vec<_> = transcript.markdown.entries[0]
        .rows
        .iter()
        .map(|(&node, rows)| (node, rows.as_ptr(), rows.len()))
        .collect();
    assert!(!boundaries.is_empty());
    assert_eq!(transcript.markdown.entries.len(), 2);

    assert_eq!(agent.undo(1), Some("second question"));
    transcript.reset_after_history_change(agent.messages());
    assert_eq!(transcript.markdown.entries.len(), 1);
    assert_eq!(transcript.markdown.entries[0].tree.nodes.as_ptr(), nodes);
    for (node, pointer, len) in boundaries {
        let rows = &transcript.markdown.entries[0].rows[&node];
        assert_eq!(rows.as_ptr(), pointer);
        assert_eq!(rows.len(), len);
    }

    agent.start_turn("replacement question".into());
    agent.submit_assistant_msg(AssistantMessage {
        reasoning: None,
        content: Some("replacement paragraph".into()),
        tool_calls: Vec::new(),
    });
    transcript.update_viewport(agent.messages(), 40, 8, agent.version(), false, None);
    assert_eq!(transcript.markdown.entries[0].tree.nodes.as_ptr(), nodes);
    assert_eq!(
        row_texts(
            &mut transcript.markdown.entries[1],
            "replacement paragraph",
            40
        ),
        ["replacement paragraph"]
    );
}

#[test]
fn zero_height_transcript_synchronizes_updates_and_completion() {
    use std::task::{Context, Poll};

    use crate::{test_support::StreamingBackend, widgets::Transcript};
    use kyun_agent::{Agent, BackendEvent, StepEvent, ToolCallDelta, ToolCallKind};

    for with_tools in [false, true] {
        let mut agent = Agent::new(StreamingBackend {
            events: vec![
                BackendEvent {
                    content: Some("```rs\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("let x = 1;\n".into()),
                    ..Default::default()
                },
                BackendEvent {
                    content: Some("```\nfinal tail".into()),
                    tool_calls: if with_tools {
                        vec![ToolCallDelta {
                            index: 0,
                            id: Some("call".into()),
                            kind: Some(ToolCallKind::Function),
                            function_name: Some("read".into()),
                            arguments: Some("{\"path\":\"x\"}".into()),
                        }]
                    } else {
                        Vec::new()
                    },
                    finished: true,
                    ..Default::default()
                },
            ],
        });
        agent.start_turn("question".into());
        agent.start_step(&[], true);
        let mut transcript = Transcript::new(true);
        let mut cx = Context::from_waker(futures_util::task::noop_waker_ref());
        for height in [8, 0] {
            let Poll::Ready(StepEvent::Updated {
                n_added_reasoning,
                n_added_content,
            }) = agent.poll_step(&mut cx)
            else {
                panic!("expected streamed content");
            };
            transcript.update_viewport(
                agent.messages(),
                40,
                height,
                agent.version(),
                true,
                Some((n_added_reasoning, n_added_content)),
            );
            assert_eq!(
                transcript
                    .markdown
                    .streaming
                    .as_ref()
                    .unwrap()
                    .state
                    .fed_len,
                super::get_assistant_content_for_view(agent.messages(), 1)
                    .unwrap()
                    .len()
            );
        }
        assert_eq!(
            agent.poll_step(&mut cx),
            Poll::Ready(if with_tools {
                StepEvent::ToolCalls
            } else {
                StepEvent::Finished
            })
        );
        transcript.update_viewport(agent.messages(), 40, 0, agent.version(), false, None);
        assert!(transcript.markdown.streaming.is_none());
        let document = &transcript.markdown.entries[0];
        assert_eq!(document.tree.open_container, 0);
        assert_eq!(
            document.tree.pending_start(),
            super::get_assistant_content_for_view(agent.messages(), 1)
                .unwrap()
                .len()
        );
    }
}

// These assertions verify retained boundary values and allocations. They do not detect parsed-tree traversal or
// temporary content rebuilding. TODO: Check those operations when enforcing unchanged/pending-only work limits.
// Possible validation: count node visits and content-index builds. For the same pending-suffix work, growing an
// unchanged parsed prefix should not add visits to that prefix. Compare warm caches with eviction for equal rows and
// update deltas. Test width changes, table widening, and closing-only visibility changes separately, since those
// cases legitimately require more work. Operation counts would distinguish cache retention from avoided computation.
#[test]
fn pending_updates_and_unchanged_feeds_retain_parsed_row_boundaries() {
    let mut source = "a **published** paragraph with several wrapped rows\n\n".to_owned();
    let mut document = MdMessage::new();
    let mut state = StreamState::default();
    document.update(&source, 12, false, &mut state);
    let owner = document
        .tree
        .blocks(&source)
        .find(|block| block.has_content)
        .unwrap()
        .node_idx;
    let rows = document.rows[&owner].as_ptr();
    let boundaries = document.rows[&owner].clone();
    let nodes = document.tree.nodes.len();
    source.push_str("pending text that also wraps");
    let update = document.update(&source, 12, false, &mut state);
    assert!(update.changed);
    assert_eq!(update.first_node, 0);
    assert_eq!(document.tree.nodes.len(), nodes);
    assert_eq!(document.rows[&owner].as_ptr(), rows);
    assert_eq!(document.rows[&owner], boundaries);
    assert_eq!(update.inserted_rows, document.rows[&0].len() + 1);
    let pending_rows = document.rows[&0].as_ptr();
    let unchanged = document.update(&source, 12, false, &mut state);
    assert!(!unchanged.changed);
    assert_eq!((unchanged.removed_rows, unchanged.inserted_rows), (0, 0));
    assert_eq!(document.rows[&0].as_ptr(), pending_rows);
}

#[test]
fn consumed_blank_input_moves_pending_positions_without_changing_either_length() {
    let mut document = MdMessage::new();
    let mut state = StreamState::default();
    document.update(" ", 20, false, &mut state);
    let nodes = document.tree.nodes.len();
    let update = document.update(" \n ", 20, false, &mut state);
    assert_eq!(document.tree.nodes.len(), nodes);
    assert_eq!(state.fed_len - document.tree.pending_start(), 1);
    assert_eq!(document.tree.pending_start(), 2);
    assert!(update.changed);
    assert_eq!(update.first_node, 0);
    assert_eq!(
        document.first_position(" \n ", 20),
        Some(MarkdownPosition::Text {
            node: 0,
            start: Some(Position { node: 0, byte: 2 })
        })
    );
}

#[test]
fn closing_empty_items_publishes_rows_without_adding_nodes_or_pending_text() {
    for source in ["- \n", "- -\n", "- >\n", "- ```\n  ```\n"] {
        let mut document = MdMessage::new();
        let mut state = StreamState::default();
        document.update(source, 12, false, &mut state);
        assert!(row_texts(&mut document, source, 12).is_empty());
        let nodes = document.tree.nodes.len();
        let pending = document.tree.pending_start();
        let update = document.update(source, 12, true, &mut state);
        assert_eq!(document.tree.nodes.len(), nodes);
        assert_eq!(document.tree.pending_start(), pending);
        assert!(update.changed);
        assert_eq!(update.removed_rows, 0);
        assert_eq!(update.inserted_rows, 1);
        assert_eq!(
            row_texts(&mut document, source, 12),
            [if source == "- -\n" {
                "• • "
            } else {
                "• "
            }]
        );
    }
}

#[test]
fn closing_list_containers_retains_completed_layout() {
    for source in [
        "- first paragraph with several wrapped rows\n\n",
        "- outer paragraph\n\n  - nested paragraph with several wrapped rows\n\n",
        "- | a | b |\n  |---|---|\n  | long first value | second |\n\n",
    ] {
        let mut document = MdMessage::new();
        let mut state = StreamState::default();
        document.update(source, 16, false, &mut state);
        assert!(matches!(
            document.tree.nodes[document.tree.open_container].data,
            Element::ListItem
        ));
        let before = row_texts(&mut document, source, 16);
        assert!(!before.is_empty());
        let rows: Vec<_> = document
            .rows
            .iter()
            .map(|(&node, rows)| (node, rows.as_ptr()))
            .collect();
        let columns: Vec<_> = document
            .columns
            .iter()
            .map(|(&node, columns)| (node, columns.as_ptr()))
            .collect();
        let nodes = document.tree.nodes.len();

        let change = document.update(source, 16, true, &mut state);
        assert!(!change.changed);
        assert_eq!(document.tree.nodes.len(), nodes);
        for (node, pointer) in rows {
            assert_eq!(document.rows[&node].as_ptr(), pointer);
        }
        for (node, pointer) in columns {
            assert_eq!(document.columns[&node].as_ptr(), pointer);
        }
        assert_eq!(row_texts(&mut document, source, 16), before);
    }
}

#[test]
fn new_list_siblings_keep_completed_paragraph_rows() {
    for sibling in ["- next paragraph\n", "- \n"] {
        let mut source = "- first paragraph with several wrapped rows\n\n".to_owned();
        let mut document = MdMessage::new();
        let mut state = StreamState::default();
        document.update(&source, 16, false, &mut state);
        let owner = document
            .tree
            .blocks(&source)
            .find(|block| block.has_content)
            .unwrap()
            .node_idx;
        let rows = document.rows[&owner].as_ptr();
        let before = row_texts(&mut document, &source, 16);
        let nodes = document.tree.nodes.len();

        source.push_str(sibling);
        let change = document.update(&source, 16, false, &mut state);
        assert!(change.first_node >= nodes);
        assert_eq!(change.removed_rows, 0);
        assert_eq!(document.rows[&owner].as_ptr(), rows);
        assert_eq!(
            before.len() + change.inserted_rows,
            row_texts(&mut document, &source, 16).len()
        );

        let old_height = state.height.unwrap();
        let finish = document.update(&source, 16, true, &mut state);
        assert_eq!(document.rows[&owner].as_ptr(), rows);
        let after = row_texts(&mut document, &source, 16);
        assert_eq!(
            old_height - finish.removed_rows + finish.inserted_rows,
            after.len()
        );
        assert_eq!(&after[..before.len()], before);
        assert_eq!(
            after,
            row_texts(&mut MdMessage::parse(&source), &source, 16)
        );
    }
}

#[test]
fn table_width_changes_rewrap_existing_cells_and_finish_retains_the_cache() {
    let mut source = "| a | b |\n|---|---|\n| 123456789 | x |\n".to_owned();
    let mut document = MdMessage::new();
    let mut state = StreamState::default();
    document.update(&source, 14, false, &mut state);
    let table = document
        .tree
        .blocks(&source)
        .find(|block| block.has_content)
        .unwrap()
        .node_idx;
    let row = document.tree.table_rows(table).last().unwrap();
    let cell = document.tree.children(row).next().unwrap();
    let old_starts = document.rows[&cell].clone();
    let old_widths = document.columns[&table].clone();
    let previous_height = state.height.unwrap();
    source.push_str("| longer words that wrap | another long cell |\n");
    let update = document.update(&source, 14, false, &mut state);
    assert_eq!(update.first_node, table);
    assert_ne!(document.rows[&cell], old_starts);
    assert_ne!(document.columns[&table], old_widths);
    assert_eq!(
        previous_height - update.removed_rows + update.inserted_rows,
        state.height.unwrap()
    );
    let mut complete = MdMessage::parse(&source);
    assert_eq!(
        row_texts(&mut document, &source, 14),
        row_texts(&mut complete, &source, 14)
    );
    let rows = document.rows[&cell].as_ptr();
    let columns = document.columns[&table].as_ptr();
    let finish = document.update(&source, 14, true, &mut state);
    assert!(!finish.changed);
    assert_eq!(document.rows[&cell].as_ptr(), rows);
    assert_eq!(document.columns[&table].as_ptr(), columns);
}

#[test]
fn table_column_reflow_and_forward_backward_walks_use_cached_boundaries() {
    let source = "| a | b | c | d |\n|---|---|---|---|\n| 1 | 2 | 3 | 4 |\n";
    let mut document = MdMessage::parse(source);
    let old = positions(&mut document, source, 10)
        .into_iter()
        .find(|position| match position {
            MarkdownPosition::Table { first_cell, .. } => {
                document.tree.cell_location(*first_cell).2 == 2
            }
            _ => false,
        })
        .unwrap();
    // The width change invalidates wrapping caches; in production MarkdownCache does this at its entry points.
    document.rows.clear();
    document.columns.clear();
    let MarkdownReflow::Before(MarkdownPosition::Separator { first_cell }) =
        document.reflow(source, 10, 4, &old).unwrap()
    else {
        unreachable!()
    };
    assert_eq!(document.tree.cell_location(first_cell).2, 2);
    assert_eq!(
        row_texts(&mut document, source, 4),
        [
            "a",
            "1",
            "────",
            "b",
            "2",
            "────",
            "c",
            "3",
            "────",
            "d",
            "4"
        ]
    );
    let forward = positions(&mut document, source, 4);
    let mut position = document.last_position(source, 4).unwrap();
    let mut backward = vec![position];
    while let Some(previous) = document.previous(source, 4, &position) {
        backward.push(previous);
        position = previous;
    }
    backward.reverse();
    assert_eq!(forward, backward);
}

#[test]
fn a_table_first_cell_and_line_select_independent_cell_progress() {
    let source = "| a | b |\n|---|---|\n| x | abcdefghijklmnop |\n";
    let mut document = MdMessage::parse(source);
    let cursors = positions(&mut document, source, 10);
    let table = document
        .tree
        .blocks(source)
        .find(|block| block.has_content)
        .unwrap()
        .node_idx;
    let row = document.tree.table_rows(table).last().unwrap();
    let cells = document.tree.children(row).collect::<Vec<_>>();
    let body = cursors
        .into_iter()
        .filter(|position| {
            matches!(position,
        MarkdownPosition::Table { first_cell, .. } if *first_cell == cells[0])
        })
        .collect::<Vec<_>>();
    assert_eq!(document.rows[&cells[0]].len(), 1);
    assert!(document.rows[&cells[1]].len() > 3);
    assert_eq!(body.len(), document.rows[&cells[1]].len());
    assert!(
        document
            .render(source, 10, &body[0])
            .text()
            .starts_with("x")
    );
    for position in &body[1..] {
        assert!(
            document
                .render(source, 10, position)
                .text()
                .starts_with("    │ ↪ ")
        );
    }
}

#[test]
fn wrapping_preserves_words_and_graphemes_across_inline_leaves() {
    let content = Content::new([(2, "e"), (4, "\u{301}x")]);
    let mut state = State::default();
    let first = content.next_row(&mut state, 1).unwrap();
    assert_eq!(first.end, Some(Position { node: 4, byte: 2 }));
    assert_eq!(content.render(&first, 0), "e\u{301}");
    assert_eq!(content.fragments(&first, 0), [(2, 0..1, 0), (4, 0..2, 0)]);
    let marked = "ab[**cdef**gh](https://example.com) ij*kl*mn\n";
    let plain = "abcdefgh ijklmn\n";
    let mut marked_document = MdMessage::parse(marked);
    let mut plain_document = MdMessage::parse(plain);
    for width in [1, 2, 3, 6, 9, 20] {
        // The width change invalidates wrapping caches; in production MarkdownCache does this at its entry points.
        marked_document.rows.clear();
        marked_document.columns.clear();
        plain_document.rows.clear();
        plain_document.columns.clear();
        assert_eq!(
            row_texts(&mut marked_document, marked, width),
            row_texts(&mut plain_document, plain, width)
        );
    }
}

#[test]
fn streamed_source_reallocation_and_resizing_match_complete_content() {
    let complete = "> - **nested** text that wraps\n\n```rs\nlet x = 123456789;\n```\n\n| a | b | c |\n|---|---|---|\n| short | **much longer** text | &amp; |\n";
    let mut source = String::new();
    let mut streamed = MdMessage::new();
    let mut state = StreamState::default();
    for character in complete.chars() {
        // Force source growth/moves while checking suffix counts. No node or cached row can retain a source borrow.
        source.shrink_to_fit();
        source.push(character);
        let old_height = state.height.unwrap_or(0);
        let change = streamed.update(&source, 12, false, &mut state);
        assert_eq!(
            old_height - change.removed_rows + change.inserted_rows,
            state.height.unwrap()
        );
        for position in positions(&mut streamed, &source, 12)
            .iter()
            .filter(|position| matches!(position, MarkdownPosition::Text { node: 0, .. }))
        {
            assert!(
                !streamed
                    .render(&source, 12, position)
                    .text()
                    .starts_with("│ ")
            );
        }
    }
    streamed.update(&source, 12, true, &mut state);
    let mut finished = MdMessage::parse(complete);
    for width in [4, 9, 25, 80, 9] {
        // The width change invalidates wrapping caches; in production MarkdownCache does this at its entry points.
        streamed.rows.clear();
        streamed.columns.clear();
        finished.rows.clear();
        finished.columns.clear();
        assert_eq!(
            row_texts(&mut streamed, &source, width),
            row_texts(&mut finished, complete, width)
        );
    }
    assert!(
        streamed
            .tree
            .nodes
            .iter()
            .any(|node| matches!(node.data, Element::Plain(Text::Source(_))))
    );
    assert!(
        streamed
            .tree
            .nodes
            .iter()
            .any(|node| matches!(node.data, Element::Plain(Text::Owned(_))))
    );
}

#[test]
fn prefixes_and_tab_columns_match_cursor_wrapping_without_repeating_item_markers() {
    for source in [
        "- > a quoted\titem with enough text to wrap several times\n",
        "> 9. [x] first\tword and a longer ending\n>\n>    next paragraph\n",
        "## heading\twith words and an ending that wraps\n",
        "- [x] earlier text\n\n  | a | b |\n  |---|---|\n  | 1 | 2 |\n",
    ] {
        let mut document = MdMessage::parse(source);
        for width in [0, 1, 4, 8, 20, 40] {
            // The width change invalidates wrapping caches; in production MarkdownCache does this at its entry points.
            document.rows.clear();
            document.columns.clear();
            let rows = row_texts(&mut document, source, width);
            if source.contains("[x]") {
                assert_eq!(rows.iter().filter(|row| row.contains("[x]")).count(), 1);
            }
            for position in positions(&mut document, source, width) {
                if let MarkdownPosition::Text { node, start } = position {
                    let (_, row, prefix) = document.tree.row_at(source, node, start, width);
                    let rendered = document.render(source, width, &position);
                    assert_eq!(rendered.line.spans[0].content.width(), prefix);
                    assert_eq!(rendered.text().width(), prefix + row.width);
                }
            }
        }
    }
}

#[test]
fn sync_builds_completed_trees_and_truncates() {
    let agent = agent_with([
        Message::User("q".into()),
        Message::Assistant(AssistantMessage {
            reasoning: None,
            content: Some("**answer**\n".into()),
            tool_calls: Vec::new(),
        }),
    ]);
    let mut cache = MarkdownCache::new(true);
    assert!(cache.sync(agent.messages(), false, 40).is_some());
    assert_eq!(cache.entries.len(), 1);
    assert!(cache.msg(agent.messages(), 1, 40).is_some());
    cache.truncate(agent_with([]).messages());
    assert!(cache.entries.is_empty());
    cache.initialize_for_test(agent.messages());
    assert!(cache.msg(agent.messages(), 1, 40).is_some());
    let mut disabled = MarkdownCache::default();
    assert!(disabled.msg(agent.messages(), 1, 40).is_none());
}

#[test]
fn paragraphs_publish_with_block_separation() {
    assert_eq!(
        line_texts("first para\n\nsecond para\n"),
        ["first para", "", "second para"]
    );
}

#[test]
fn pending_top_level_paragraph_reserves_its_published_separator() {
    let mut source = "first paragraph\n\nsecond paragraph".to_owned();
    let mut document = MdMessage::new();
    let mut state = StreamState::default();
    document.update(&source, 80, false, &mut state);
    assert_eq!(
        row_texts(&mut document, &source, 80),
        ["first paragraph", "", "second paragraph"]
    );

    source.push_str("\n\n");
    document.update(&source, 80, false, &mut state);
    assert_eq!(
        row_texts(&mut document, &source, 80),
        ["first paragraph", "", "second paragraph"]
    );
}

#[test]
fn soft_breaks_end_logical_lines() {
    assert_eq!(line_texts("one\ntwo\n\nnext"), ["one", "two", "", "next"]);
}

#[test]
fn headings_carry_level_markers_and_bold_text() {
    let lines = render("## Title with `code`\n");
    assert_eq!(lines[0].0, "## ");
    assert_eq!(lines[0].1, "Title with code");
    assert!(
        lines[0]
            .2
            .iter()
            .any(|(style, _)| style.add_modifier.contains(Modifier::BOLD))
    );
}

#[test]
fn setext_headings_split_soft_breaks_into_lines() {
    let lines = render("Title\nlines\n=====\n");
    assert_eq!(lines[0].1, "Title");
    assert_eq!(lines[1].1, "lines");
    for line in &lines[..2] {
        assert!(
            line.2
                .iter()
                .any(|(style, _)| style.add_modifier.contains(Modifier::BOLD))
        );
    }
}

#[test]
fn code_blocks_preserve_physical_lines_with_one_style_partition() {
    let lines = render("```\nlet a = 1;\nlet b = 2;\n```\n");
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0].1, "let a = 1;");
    assert_eq!(lines[1].1, "let b = 2;");
    for line in &lines {
        assert_eq!(line.2.len(), 1, "code lines carry exactly one style range");
    }
}

#[test]
fn crlf_code_lines_drop_their_line_endings() {
    assert_eq!(line_texts("```\r\nlet a = 1;\r\n```\r\n"), ["let a = 1;"]);
}

#[test]
fn empty_blocks_do_not_double_separators() {
    // The empty block publishes nothing itself, and its separator is the only one.
    assert_eq!(
        line_texts("before\n```\n```\nafter\n"),
        ["before", "", "after"]
    );
}

#[test]
fn lists_render_tight_with_bullets_and_indented_continuations() {
    let lines = render("- one two\n  three\n- four\n");
    assert_eq!(lines[0].0, "• ");
    assert_eq!(lines[0].1, "one two");
    assert_eq!(lines[1].0, "  ");
    assert_eq!(lines[1].1, "three");
    assert_eq!(lines[2].0, "• ");
    assert_eq!(lines[2].1, "four");
}

#[test]
fn ordered_lists_number_from_their_start() {
    assert_eq!(line_prefixes("3. three\n4. four\n"), ["3. ", "4. "]);
}

#[test]
fn task_markers_replace_bullets_and_widen_indents() {
    let lines = render("- [x] done\n- [ ] todo\n");
    assert_eq!(lines[0].0, "[x] ");
    assert_eq!(lines[1].0, "[ ] ");
    // The continuation of a task item aligns under the checkbox.
    let continued = render("- [x] done\n  and more\n");
    assert_eq!(continued[1].0, "    ");
}

#[test]
fn nested_lists_stack_markers_on_the_first_rendered_line() {
    let lines = render("- - inner\n");
    assert_eq!(lines[0].0, "• • ");
    assert_eq!(lines[0].1, "inner");
}

#[test]
fn code_first_items_put_the_marker_on_the_first_code_line() {
    let lines = render("1. ```\n   let a = 1;\n   ```\n");
    assert_eq!(lines[0].0, "1. ");
    assert_eq!(lines[0].1, "let a = 1;");
}

#[test]
fn empty_items_publish_marker_only_rows() {
    let lines = render("- a\n-\n- b\n");
    assert_eq!(lines[1].0, "• ");
    assert_eq!(lines[1].1, "");
}

#[test]
fn quotes_prefix_every_line_and_stack() {
    let lines = render("> outer\n> > inner\n");
    assert_eq!(lines[0].0, "│ ");
    assert_eq!(lines[0].1, "outer");
    assert_eq!(lines[1].0, "│ │ ");
    assert_eq!(lines[1].1, "inner");
}

#[test]
fn quote_content_separates_blocks_with_the_bar() {
    let lines = render("> first\n>\n> second\n");
    assert_eq!(lines[1].0, "│ ");
    assert_eq!(lines[1].1, "");
}

#[test]
fn lists_inside_quotes_carry_bars_and_bullets() {
    let lines = render("> - item\n");
    assert_eq!(lines[0].0, "│ • ");
}

#[test]
fn code_inside_items_indents_under_the_marker() {
    let lines = render("- item\n\n  ```\n  code\n  ```\n");
    assert_eq!(lines[0].0, "• ");
    assert_eq!(lines[2].0, "  ");
    assert_eq!(lines[2].1, "code");
}

#[test]
fn tables_align_columns() {
    // Column widths come from the widest cell across all rows; the last column keeps its
    // before-padding (here centered) but skips trailing padding.
    let lines = render("| a | b |\n|---|:-:|\n| 1 | 2 |\n| 3 | longer |\n");
    assert_eq!(lines[0].1, "a   │   b");
    assert_eq!(lines[1].1, "1   │   2");
    assert_eq!(lines[2].1, "3   │ longer");
    assert!(
        lines[0]
            .2
            .iter()
            .any(|(style, _)| style.add_modifier.contains(Modifier::BOLD))
    );
}

#[test]
fn table_alignment_hints_pad_their_columns() {
    let lines = render("| left | middle | right |\n|:-----|:------:|------:|\n| x | y | z |\n");
    assert_eq!(lines[0].1, "left │ middle │ right");
    // Left-aligned stays flush; centered splits its padding; right-aligned pads before.
    assert_eq!(lines[1].1, "x    │   y    │     z");
}

#[test]
fn inline_styles_compose_into_flat_ranges() {
    let lines = render("**bold with `code` inside** after\n");
    // "code" sits inside strong emphasis: one range with both the code foreground and bold.
    let code = lines[0]
        .2
        .iter()
        .find(|(_, range)| &lines[0].1[range.clone()] == "code")
        .unwrap();
    assert!(code.0.add_modifier.contains(Modifier::BOLD));
    assert_eq!(
        code.0.fg,
        Some(ratatui::prelude::Color::Rgb(0x82, 0xD2, 0xCE))
    );
    assert_eq!(code.0.bg, None);
    let plain = lines[0]
        .2
        .iter()
        .find(|(_, range)| &lines[0].1[range.clone()] == " after")
        .unwrap();
    assert!(!plain.0.add_modifier.contains(Modifier::BOLD));
}

#[test]
fn style_partitions_cover_the_text_exactly() {
    for markdown in [
        "plain\n",
        "**all bold**\n",
        "a *b* `c` **d** e\n",
        "> quote *with* spans\n",
        "- item `code` more\n",
        "| a | b |\n|---|---|\n| *x* | y |\n",
    ] {
        for (_, text, styles) in render(markdown) {
            let mut end = 0;
            for (_, range) in &styles {
                assert_eq!(
                    range.start, end,
                    "styles must partition {text:?} contiguously"
                );
                end = range.end;
            }
            assert_eq!(end, text.len(), "styles must cover {text:?}");
        }
    }
}

#[test]
fn adjacent_equal_styles_merge_into_one_range() {
    let lines = render("a *b* c *d* e\n");
    assert!(lines[0].2.len() <= 5, "{:?}", lines[0].2);
}

#[test]
fn image_alt_text_flattens_inline_markup() {
    let lines = render("before ![alt *text*](img.png) after\n");
    assert_eq!(lines[0].1, "before alt text after");
}

#[test]
fn raw_html_shows_literally() {
    assert_eq!(
        line_texts("<div>\ncontent\n</div>\n"),
        ["<div>", "content", "</div>"]
    );
    assert_eq!(line_texts("a <b>bold</b> c\n"), ["a <b>bold</b> c"]);
}

#[test]
fn links_render_labels_without_urls() {
    assert_eq!(line_texts("[label](https://example.com)\n"), ["label"]);
    assert_eq!(
        line_texts("visit https://example.com now\n"),
        ["visit https://example.com now"]
    );
}

#[test]
fn links_keep_targets_and_ids_across_wrapped_styles_and_table_cells() {
    for source in [
        "[alpha **bold** 界 e\u{301} tail](https://example.com) plain\n",
        "- [alpha **bold** 界 e\u{301} tail](https://example.com) plain\n",
        "| label | other |\n|---|---|\n| [alpha **bold** 界 e\u{301} tail](https://example.com) | x |\n",
    ] {
        let mut document = MdMessage::parse(source);
        let mut node = None;
        for width in [12, 30, 80] {
            // The width change invalidates wrapping caches; in production MarkdownCache does this at its entry points.
            document.rows.clear();
            document.columns.clear();
            let mut text = String::new();
            for position in positions(&mut document, source, width) {
                let rendered = document.render(source, width, &position);
                for link in &rendered.links {
                    assert_eq!(link.url, "https://example.com");
                    assert_eq!(*node.get_or_insert(link.node), link.node);
                    let span = &rendered.line.spans[link.span];
                    assert!(span.style.add_modifier.contains(Modifier::UNDERLINED));
                    text.push_str(&span.content);
                }
            }
            assert_eq!(
                text, "alpha bold 界 e\u{301} tail",
                "width {width}, source {source:?}"
            );
        }
    }
}

#[test]
fn quote_first_items_carry_the_bullet_then_the_bar() {
    let lines = render("- > quoted\n  more\n");
    assert_eq!(lines[0].0, "\u{2022} \u{2502} ");
    assert_eq!(lines[0].1, "quoted");
    // Continuations keep the item indent and the quote bar.
    assert_eq!(lines[1].0, "  \u{2502} ");
    assert_eq!(lines[1].1, "more");
}

#[test]
fn nested_empty_items_merge_their_markers_onto_one_row() {
    let lines = render("- -\n");
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].0, "\u{2022} \u{2022} ");
    assert_eq!(lines[0].1, "");
}

#[test]
fn lists_separate_from_preceding_content_once() {
    assert_eq!(
        line_texts("para\n\n- item\n- next\n"),
        ["para", "", "item", "next"]
    );
    // A nested list stays visually attached to its parent item.
    let lines = render("- a\n\n  - nested\n");
    assert_eq!(lines[0].0, "\u{2022} ");
    assert_eq!(lines[1].0, "  \u{2022} ");
    assert_eq!(lines[1].1, "nested");
}

#[test]
fn empty_table_rows_publish_with_countable_separators() {
    let lines = render("| a | b |\n|---|---|\n| | |\n");
    assert_eq!(lines.len(), 2);
    // Empty cells pad to their column widths, keeping the separator countable.
    assert_eq!(lines[1].1, "    \u{2502} ");
}

#[test]
fn inline_html_and_math_flatten_line_endings() {
    // An inline tag containing a line ending renders with a space, never a newline.
    assert_eq!(
        line_texts("a <b\nclass='x'>t</b> c\n"),
        ["a <b class='x'>t</b> c"]
    );
}

#[test]
fn pending_gap_changes_keep_cached_row_counts_equal_to_traversal() {
    let mut source = "first paragraph\n\n".to_owned();
    let mut document = MdMessage::new();
    let mut state = StreamState::default();
    document.update(&source, 80, false, &mut state);
    for chunk in [" ", "second", " paragraph", "\n\n"] {
        let old_rows = row_texts(&mut document, &source, 80).len();
        source.push_str(chunk);
        let change = document.update(&source, 80, false, &mut state);
        let rows = row_texts(&mut document, &source, 80).len();
        assert_eq!(old_rows - change.removed_rows + change.inserted_rows, rows);
    }
}
