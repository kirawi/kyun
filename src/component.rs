use std::any::Any;

use ratatui::{Frame, layout::Rect};

use crate::{compositor::Compositor, event::Event, session::Session};

pub type Callback = Box<dyn FnOnce(&mut Compositor, &mut Session)>;

pub enum EventResult {
    Ignored(Option<Callback>),
    Consumed(Option<Callback>),
}

pub trait Component: Any + AnyComponent {
    fn render(&mut self, frame: &mut Frame, area: Rect, cx: &mut Session);
    fn handle_event(&mut self, event: &Event, area: Rect, cx: &mut Session) -> EventResult;
}

pub trait AnyComponent {
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Component> AnyComponent for T {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
