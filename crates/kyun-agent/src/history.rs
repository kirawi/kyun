use crate::Message;
use jiff::Timestamp;
use std::num::NonZeroUsize;

/// Stores committed messages as an append-only revision tree.
///
/// Uncommitted messages are owned by [`super::Agent`]. A committed revision owns one message and its completion time
/// permanently; navigation changes only which revision is current.
pub struct History {
    /// The sentinel followed by every committed message in commit order.
    revisions: Vec<Revision>,
    /// IDs on the selected path permit direct indexed reads without copying message data.
    spine: Vec<NonZeroUsize>,
}

struct Revision {
    first_child: Option<NonZeroUsize>,
    next_sibling: Option<NonZeroUsize>,
    msg: Message,
    timestamp: Timestamp,
}

impl History {
    /// Creates a history with initial messages on the selected path, timestamped when they are loaded.
    // TODO: remove impl IntoIterator since we don't need it
    pub fn with_messages(messages: impl IntoIterator<Item = Message>) -> Self {
        // First revision is a sentinel
        let mut history = Self {
            revisions: vec![Revision {
                first_child: None,
                next_sibling: None,
                msg: Message::System(String::new()),
                timestamp: Timestamp::UNIX_EPOCH,
            }],
            spine: Vec::new(),
        };
        let mut messages = messages.into_iter().peekable();

        // TODO: replace asserts with error handling
        while let Some(msg) = messages.next() {
            match &msg {
                Message::Assistant(assistant) if !assistant.tool_calls.is_empty() => {
                    let Some(Message::Tool(results)) = messages.peek() else {
                        panic!(
                            "an assistant message with tool calls MUST be followed by its tool results"
                        );
                    };
                    assert_eq!(results.len(), assistant.tool_calls.len());
                    for (call, result) in assistant.tool_calls.iter().zip(results) {
                        assert_eq!(call.id, result.tool_call_id);
                    }
                }
                Message::Tool(_) => {
                    assert!(matches!(
                        history.revisions[history.spine.last().map_or(0, |revision| revision.get())].msg,
                        Message::Assistant(ref assistant) if !assistant.tool_calls.is_empty()
                    ));
                }
                _ => {}
            }
            history.commit([(msg, Timestamp::now())]);
        }
        history
    }

    /// Commits complete messages as a chain of immutable revisions, preserving their completion times.
    ///
    /// The constructor validates initial tool exchanges; agent transitions validate later ones.
    pub fn commit(&mut self, messages: impl IntoIterator<Item = (Message, Timestamp)>) {
        for (msg, timestamp) in messages {
            let parent = self.spine.last().map_or(0, |revision| revision.get());
            let id = NonZeroUsize::new(self.revisions.len()).unwrap();
            self.revisions.push(Revision {
                first_child: None,
                next_sibling: None,
                msg,
                timestamp,
            });
            if let Some(mut child) = self.revisions[parent].first_child {
                while let Some(next) = self.revisions[child.get()].next_sibling {
                    child = next;
                }
                self.revisions[child.get()].next_sibling = Some(id);
            } else {
                self.revisions[parent].first_child = Some(id);
            }
            self.spine.push(id);
        }
    }

    pub fn current_path(&self) -> &[NonZeroUsize] {
        &self.spine
    }

    pub fn message(&self, revision_idx: NonZeroUsize) -> &Message {
        &self.revisions[revision_idx.get()].msg
    }

    pub fn timestamp(&self, revision_idx: NonZeroUsize) -> Timestamp {
        self.revisions[revision_idx.get()].timestamp
    }

    /// Moves to the previous complete turn and returns its user message.
    pub fn undo(&mut self) -> Option<&Message> {
        let user_position = self
            .spine
            .iter()
            .rposition(|revision| matches!(self.message(*revision), Message::User(_)))?;
        let user_revision_idx = self.spine.drain(user_position..).next().unwrap();
        Some(self.message(user_revision_idx))
    }

    /// Moves through the newest child chain to the end of the current user turn.
    pub fn redo(&mut self) -> bool {
        let Some(mut child) =
            self.revisions[self.spine.last().map_or(0, |revision| revision.get())].first_child
        else {
            return false;
        };
        while let Some(next) = self.revisions[child.get()].next_sibling {
            child = next;
        }
        self.spine.push(child);

        while let Some(mut next) = self.revisions[child.get()].first_child {
            while let Some(sibling) = self.revisions[next.get()].next_sibling {
                next = sibling;
            }
            if matches!(self.revisions[next.get()].msg, Message::User(_)) {
                break;
            }
            child = next;
            self.spine.push(child);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::view::UncommittedMessages;
    use crate::{AssistantMessage, Message};
    use jiff::Timestamp;

    use super::History;

    fn answer(text: &str) -> Message {
        Message::Assistant(AssistantMessage {
            reasoning: None,
            content: Some(text.into()),
            tool_calls: Vec::new(),
        })
    }

    #[test]
    fn undo_then_commit_preserves_both_children_and_redo_selects_the_newest() {
        let mut history = History::with_messages([]);
        history.commit([
            (Message::User("first".into()), Timestamp::UNIX_EPOCH),
            (answer("old"), Timestamp::from_second(1).unwrap()),
        ]);
        let first = *history.current_path().last().unwrap();
        let first_timestamp = history.timestamp(first);
        assert_eq!(history.undo(), Some(&Message::User("first".into())));
        history.commit([
            (
                Message::User("second".into()),
                Timestamp::from_second(2).unwrap(),
            ),
            (answer("new"), Timestamp::from_second(3).unwrap()),
        ]);
        let second = *history.current_path().last().unwrap();
        assert_eq!(history.undo(), Some(&Message::User("second".into())));

        let first_child = history.revisions[0].first_child.unwrap();
        let second_child = history.revisions[first_child.get()].next_sibling.unwrap();
        assert_eq!(history.message(first_child), &Message::User("first".into()));
        assert_eq!(
            history.message(second_child),
            &Message::User("second".into())
        );
        assert!(history.redo());
        assert_eq!(history.current_path().last(), Some(&second));
        assert_eq!(history.message(first), &answer("old"));
        assert_eq!(history.timestamp(first), first_timestamp);
        assert_eq!(
            history.view(&UncommittedMessages::default()).timestamp(1),
            Some(history.timestamp(second))
        );
    }
}
