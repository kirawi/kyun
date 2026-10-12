use crossterm::event::{Event as CrosstermEvent, KeyEvent, MouseEvent};

pub enum Event {
    FocusGained,
    FocusLost,
    Key(KeyEvent),
    Mouse(MouseEvent),
    Paste(String),
    Resize(u16, u16),
    Redraw,
    /// Byte counts appended to the streamed message's reasoning and content since the previous
    /// update, matching `StepEvent::Updated`'s payload.
    AgentUpdated {
        reasoning: usize,
        content: usize,
    },
}

impl From<CrosstermEvent> for Event {
    fn from(event: CrosstermEvent) -> Self {
        match event {
            CrosstermEvent::FocusGained => Self::FocusGained,
            CrosstermEvent::FocusLost => Self::FocusLost,
            CrosstermEvent::Key(event) => Self::Key(event),
            CrosstermEvent::Mouse(event) => Self::Mouse(event),
            CrosstermEvent::Paste(content) => Self::Paste(content),
            CrosstermEvent::Resize(width, height) => Self::Resize(width, height),
        }
    }
}
