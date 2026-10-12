use std::{borrow::Cow, num::NonZeroUsize, ops::Range};

#[cfg(not(test))]
use std::sync::{LazyLock, Mutex};

use mdtext::{Alignment, Event, ListKind, Parser, Tag};

// INVARIANT: Only one message may have unfinished parser state at a time. Completed messages retain their trees.
// Starting a new tree resets the parser, including after the preceding stream was cancelled.
#[cfg(not(test))]
static MD_PARSER: LazyLock<Mutex<Parser>> =
    LazyLock::new(|| Mutex::new(Parser::with_options(super::OPTIONS)));

#[derive(Debug)]
pub enum Element {
    // Container variants. Each opens a node with `subtree_end == None` and may hold descendant nodes.
    /// A sentinel value that does double-duty by holding the source offset to text for which markdown events have yet
    /// to be emitted (i.e. parsed).
    Root {
        pending_start: usize,
    },
    Paragraph,
    Heading(u8),
    CodeBlock(Option<Text>),
    CodeSpan,
    DisplayMath,
    HtmlBlock,
    Quote,
    List(ListKind),
    ListItem,
    Emphasis,
    Strong,
    Strikethrough,
    // An absent title (equiv. to None) is a `0..0` range.
    Link {
        url: Text,
        title: Text,
    },
    Image {
        url: Text,
        title: Text,
    },
    Table(Box<[Alignment]>),
    TableHead,
    TableBody,
    TableRow,
    TableCell,

    // Leaf variants. Each is appended to the tree as a terminal node with `subtree_end == Some(idx + 1)`.
    // Different from CommonMark definition of leaf (as emphasis can contain inline element nodes).
    Plain(Text),
    Code(Text),
    Html(Text),
    InlineMath(Text),
    Math(Text),
    SoftBreak,
    HardBreak,
    ThematicBreak,
    TaskListMarker(bool),
}

// `u32` instead of `usize` since hardware/architectural limitations prohibit long output sequences for a givne
// message anyway.
#[derive(Debug)]
pub enum Text {
    Source(Range<u32>),
    Owned(u32),
}

pub struct Node {
    pub data: Element,
    /// Points to the index of the parent node.
    pub parent: usize,
    /// The index representing the exclusive end of the span of nodes starting from the current node's index,
    /// with the span representing the entire subtree underneath the current node.
    ///
    /// A leaf closes with `idx + 1`, thus including only the leaf itself.
    ///
    /// `None` means that the subtree has not yet been finalized (i.e. it's still open).
    pub subtree_end: Option<NonZeroUsize>,
}

pub struct Tree {
    /// Nodes are appended in preorder, and never removed or reordered.
    pub nodes: Vec<Node>,
    /// An arena of owned strings (e.g. escaped content, entities)
    pub strings: Vec<Box<str>>,
    /// Index to the innermost open container (i.e. non-leaf) node, or 0 for the sentinel [`Element::Root`].
    pub open_container: usize,

    // Needed only for tests (but tests may need to be rewritten so this is no longer necessary)
    #[cfg(test)]
    parser: Option<Box<Parser>>,
}

impl Text {
    pub fn from_cow(text: Cow<'_, str>, source: &str, strings: &mut Vec<Box<str>>) -> Self {
        if let Cow::Borrowed(text) = &text
            && let Some(start) = (text.as_ptr() as usize).checked_sub(source.as_ptr() as usize)
            && start <= source.len()
            && text.len() <= source.len() - start
        {
            return Self::Source(start as u32..(start + text.len()) as u32);
        }

        // mdtext also borrows static synthesized text. Store it alongside owned/decoded parser output.
        let index = strings.len() as u32;
        strings.push(text.into_owned().into_boxed_str());
        Self::Owned(index)
    }

    pub fn resolve<'a>(&self, source: &'a str, strings: &'a [Box<str>]) -> &'a str {
        match self {
            Self::Source(range) => {
                let range = range.start as usize..range.end as usize;
                &source[range]
            }
            Self::Owned(index) => &strings[*index as usize],
        }
    }
}

impl Tree {
    pub fn new() -> Self {
        #[cfg(not(test))]
        MD_PARSER.lock().unwrap().reset();
        Self {
            nodes: vec![Node {
                data: Element::Root { pending_start: 0 },
                parent: 0,
                subtree_end: None,
            }],
            strings: Vec::new(),
            open_container: 0,
            #[cfg(test)]
            parser: Some(Box::new(Parser::with_options(super::OPTIONS))),
        }
    }

    pub fn pending_start(&self) -> usize {
        let Element::Root { pending_start } = self.nodes[0].data else {
            unreachable!()
        };
        pending_start
    }

    /// Returns whether the [`Element`] at `idx` is an open container block (leafs are always false).
    pub fn is_open_container(&self, idx: usize) -> bool {
        idx != 0 && self.nodes[idx].subtree_end.is_none()
    }

    /// Returns `Node::subtree_end` exclusive end, with `None` (representing an open element) defaulting to
    /// `self.nodes.len()` to encapsulate the idea that the rest of the nodes are part of its subtree (and thus
    /// the `.len()` is its range end).
    ///
    /// Recall: returns `idx + 1` if the [`Node`] at `idx` is a leaf.
    pub fn subtree_end(&self, idx: usize) -> usize {
        self.nodes[idx]
            .subtree_end
            .map_or(self.nodes.len(), NonZeroUsize::get)
    }

    /// Produces an iterator strictly through the children of the [`Node`] at `idx`.
    pub fn children(&self, parent_idx: usize) -> impl Iterator<Item = usize> {
        let end_idx = self.subtree_end(parent_idx);
        std::iter::successors(
            (parent_idx + 1 < end_idx).then_some(parent_idx + 1),
            move |&idx| {
                let sibling_idx = self.subtree_end(idx);
                (sibling_idx < end_idx).then_some(sibling_idx)
            },
        )
    }

    /// Iterates through descendants of the [`Node`] `top` at `top_idx` in preorder IFF `top` is NOT a leaf,
    /// in which case it short-circuits (as there is nothing to traverse down to). Iterates through each node
    /// twice, yielding `(node_idx, is_entering)` where `is_entering` distinguishes whether the iterator is
    /// entering (will explore its descendants) or exiting (has finished exploring; moving to a sibling or up to
    /// parent).
    pub fn traverse_subtree(&self, top_idx: usize) -> impl Iterator<Item = (usize, bool)> {
        // Construct the initial return tuple based on `top_idx`, treating it as an entry downwards.
        std::iter::successors(
            (top_idx + 1 < self.subtree_end(top_idx)).then_some((top_idx + 1, true)),
            move |&(idx, is_entering)| {
                if is_entering {
                    if idx + 1 < self.subtree_end(idx) {
                        Some((idx + 1, true))
                    } else {
                        Some((idx, false))
                    }
                } else {
                    // Exhausted the current subtree, so go to the next sibling subtree.
                    let parent_idx = self.nodes[idx].parent;
                    let sibling_idx = self.subtree_end(idx);
                    if sibling_idx < self.subtree_end(parent_idx) {
                        Some((sibling_idx, true))
                    } else if parent_idx != top_idx {
                        // No sibling subtrees; therefore simply "pop"
                        Some((parent_idx, false))
                    } else {
                        // Back at the top, thus terminate
                        None
                    }
                }
            },
        )
    }

    /// Process `mdtext` events and update the tree accordingly.
    pub fn process_events<'a>(
        &mut self,
        events: impl IntoIterator<Item = Event<'a>>,
        source: &str,
    ) {
        for event in events {
            if matches!(event, Event::End) {
                // "Pop" to close the currently open container element.
                let container_to_close_idx = self.open_container;
                debug_assert_ne!(
                    container_to_close_idx, 0,
                    "must never pop the sentinel root"
                );

                self.open_container = self.nodes[container_to_close_idx].parent;
                self.nodes[container_to_close_idx].subtree_end =
                    NonZeroUsize::new(self.nodes.len());

                continue;
            }

            let is_opener = matches!(event, Event::Start(_));
            let new_element = match event {
                Event::Start(tag) => match tag {
                    Tag::Paragraph => Element::Paragraph,
                    Tag::Heading(level) => Element::Heading(level),
                    Tag::CodeBlock(info) => Element::CodeBlock(
                        info.map(|text| Text::from_cow(text, source, &mut self.strings)),
                    ),
                    Tag::CodeSpan => Element::CodeSpan,
                    Tag::DisplayMath => Element::DisplayMath,
                    Tag::HtmlBlock => Element::HtmlBlock,
                    Tag::Quote => Element::Quote,
                    Tag::List(kind) => Element::List(kind),
                    Tag::Item => Element::ListItem,
                    Tag::Emphasis => Element::Emphasis,
                    Tag::Strong => Element::Strong,
                    Tag::Strikethrough => Element::Strikethrough,

                    // Trick: `0..0` can be used to avoid more memory for Option
                    Tag::Link { url, title } => Element::Link {
                        url: Text::from_cow(url, source, &mut self.strings),
                        title: title
                            .map(|text| Text::from_cow(text, source, &mut self.strings))
                            .unwrap_or(Text::Source(0..0)),
                    },
                    Tag::Image { url, title } => Element::Image {
                        url: Text::from_cow(url, source, &mut self.strings),
                        title: title
                            .map(|text| Text::from_cow(text, source, &mut self.strings))
                            .unwrap_or(Text::Source(0..0)),
                    },
                    Tag::Table(alignments) => Element::Table(alignments.into_boxed_slice()),
                    Tag::TableHead => Element::TableHead,
                    Tag::TableBody => Element::TableBody,
                    Tag::TableRow => Element::TableRow,
                    Tag::TableCell => Element::TableCell,
                },
                Event::Text(text) => {
                    Element::Plain(Text::from_cow(text, source, &mut self.strings))
                }
                Event::Code(text) => Element::Code(Text::from_cow(text, source, &mut self.strings)),
                Event::Html(text) => {
                    // Flatten inline HTML; technically a specification deviation but this seems most consistent with
                    // how newlines in paragraphs are handled.
                    let in_html_block =
                        matches!(self.nodes[self.open_container].data, Element::HtmlBlock);
                    let text = if !in_html_block && text.contains(['\r', '\n']) {
                        // NOTE: this is inefficient but whatever; unlikely the agent would return a lot of inline HTML
                        Cow::Owned(text.replace("\r\n", " ").replace(['\r', '\n'], " "))
                    } else {
                        text
                    };
                    Element::Html(Text::from_cow(text, source, &mut self.strings))
                }
                Event::InlineMath(text) => {
                    // Flatten inline math; same justification as for HTML
                    let text = if text.contains(['\r', '\n']) {
                        Cow::Owned(text.replace("\r\n", " ").replace(['\r', '\n'], " "))
                    } else {
                        text
                    };
                    Element::InlineMath(Text::from_cow(text, source, &mut self.strings))
                }
                Event::DisplayMath(text) => {
                    Element::Math(Text::from_cow(text, source, &mut self.strings))
                }
                Event::SoftBreak => Element::SoftBreak,
                Event::HardBreak => Element::HardBreak,
                Event::ThematicBreak => Element::ThematicBreak,
                Event::TaskListMarker(checked) => Element::TaskListMarker(checked),
                Event::End => unreachable!(),
            };

            let new_idx = self.nodes.len();
            self.nodes.push(Node {
                data: new_element,
                parent: self.open_container,
                subtree_end: if is_opener {
                    None
                } else {
                    NonZeroUsize::new(new_idx + 1)
                },
            });

            if is_opener {
                self.open_container = new_idx;
            }
        }
    }

    pub fn update(&mut self, source: &str, finalize: bool) {
        debug_assert!(
            self.nodes[0].subtree_end.is_none(),
            "a finalized Tree must never be updated"
        );

        #[cfg(not(test))]
        let mut parser = MD_PARSER.lock().unwrap();

        #[cfg(test)]
        let mut parser = self.parser.take().unwrap();

        // No guard against unchanged source is needed: the parser resumes from an internal cursor, and its retained
        // tail never contains a parseable line, so re-feeding identical input consumes nothing and emits no events.
        let start = self.pending_start();
        let (events, consumed) = parser.feed_chunk(&source[start..]);
        self.process_events(events, source);

        let Element::Root { pending_start } = &mut self.nodes[0].data else {
            unreachable!()
        };
        *pending_start += consumed;

        if finalize {
            let events = parser.finish(&source[self.pending_start()..]);
            self.process_events(events, source);

            // Need to currently update for blocks()
            self.nodes[0].data = Element::Root {
                pending_start: source.len(),
            };
            self.nodes[0].subtree_end = NonZeroUsize::new(self.nodes.len());
            self.nodes.shrink_to_fit();
            self.strings.shrink_to_fit();
        }

        #[cfg(test)]
        if !finalize {
            self.parser = Some(parser);
        }
    }
}
