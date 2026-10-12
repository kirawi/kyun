use crate::{History, message::*};
use jiff::Timestamp;
use std::fmt;
use std::ops::Index;

pub struct MessageEntry {
    pub message: Message,
    pub timestamp: Option<Timestamp>,
}

#[derive(Default)]
pub struct UncommittedMessages {
    pub assistant: Option<MessageEntry>,
    pub tool_results: Option<MessageEntry>,
}

impl UncommittedMessages {
    pub fn is_empty(&self) -> bool {
        self.assistant.is_none() && self.tool_results.is_none()
    }

    pub fn get(&self, index: usize) -> Option<&MessageEntry> {
        match index {
            0 => self.assistant.as_ref(),
            1 => self.tool_results.as_ref(),
            _ => None,
        }
    }

    pub fn clear(&mut self) {
        self.assistant = None;
        self.tool_results = None;
    }

    pub fn take_completed(&mut self) -> impl Iterator<Item = (Message, Timestamp)> {
        self.assistant
            .take()
            .into_iter()
            .chain(self.tool_results.take())
            .map(|entry| (entry.message, entry.timestamp.unwrap()))
    }
}

/// Borrows committed messages and the uncommitted suffix without copying their contents. Indices below
/// `len()` are always present, and values are stable while the borrow is live.
#[derive(Clone, Copy)]
pub struct MessageView<'a> {
    history: &'a History,
    uncommitted: &'a UncommittedMessages,
}

impl<'a> MessageView<'a> {
    pub fn len(&self) -> usize {
        self.history.current_path().len()
            + usize::from(self.uncommitted.assistant.is_some())
            + usize::from(self.uncommitted.tool_results.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, msg_idx: usize) -> Option<&'a Message> {
        let path = self.history.current_path();
        if let Some(id) = path.get(msg_idx) {
            return Some(self.history.message(*id));
        }
        self.uncommitted
            .get(msg_idx - path.len())
            .map(|entry| &entry.message)
    }

    // TP2: For Transcript, it should maybe cache these results rather than try to calculate them each time. It only
    // changes on state transitions anyway. Though, it could complicate code for unknown perf benefit (needs benchmark
    // results to justify).
    /// Get references to the tool calls and corresponding results given a `msg_idx` pointing to the
    /// respective assistand or result message. This method will panic if `msg_idx` is incorrect.
    pub fn get_tool_calls(&self, msg_idx: usize) -> (&[ToolCall], &[ToolResult]) {
        match self.get(msg_idx).unwrap() {
            Message::Assistant(ass) => {
                let results = self
                    .get(msg_idx + 1)
                    .and_then(|msg| match msg {
                        Message::Tool(results) => Some(results.as_slice()),
                        _ => None,
                    })
                    .unwrap_or_default();
                (&ass.tool_calls, results)
            }
            Message::Tool(results) => {
                let Message::Assistant(ass) = self.get(msg_idx - 1).unwrap() else {
                    unreachable!()
                };
                (&ass.tool_calls, results)
            }
            _ => panic!("invalid message kind"),
        }
    }

    // TP3: Once transcript caches, this method will likely lose justification. Right now, it only saves on the
    // get_tool_calls, which is only possible because the results don't get persisted across rows.
    /// Get a tool call and its respective result (if exists). Will panic if `msg_idx` is incorrect.
    pub fn get_tool_call(
        &self,
        msg_idx: usize,
        call_idx: usize,
    ) -> (&ToolCall, Option<&ToolResult>) {
        let (calls, results) = self.get_tool_calls(msg_idx);
        (&calls[call_idx], results.get(call_idx))
    }

    /// Returns the completion time for a message, if it is complete.
    pub fn timestamp(&self, msg_idx: usize) -> Option<Timestamp> {
        let path = self.history.current_path();
        if let Some(id) = path.get(msg_idx) {
            return Some(self.history.timestamp(*id));
        }
        self.uncommitted
            .get(msg_idx - path.len())
            .and_then(|entry| entry.timestamp)
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &'a Message> + ExactSizeIterator {
        (0..self.len()).map(move |index| self.get(index).unwrap())
    }

    pub fn last(&self) -> Option<&'a Message> {
        self.iter().next_back()
    }
}

impl Index<usize> for MessageView<'_> {
    type Output = Message;

    fn index(&self, index: usize) -> &Message {
        self.get(index).unwrap()
    }
}

impl fmt::Debug for MessageView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl History {
    /// Presents committed messages on the selected path followed by `uncommitted` as one list.
    pub fn view<'a>(&'a self, uncommitted: &'a UncommittedMessages) -> MessageView<'a> {
        MessageView {
            history: self,
            uncommitted,
        }
    }
}
