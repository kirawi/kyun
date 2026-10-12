use ratatui::{Frame, layout::Rect};

use crate::{
    component::{Component, EventResult},
    event::Event,
    session::Session,
};

// TODO: Application is meant to drive the compositor by calling a method on it.
#[derive(Default)]
pub struct Compositor {
    layers: Vec<Box<dyn Component>>,
}

impl Compositor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, layer: Box<dyn Component>) {
        self.layers.push(layer);
    }

    pub fn pop(&mut self) -> Option<Box<dyn Component>> {
        self.layers.pop()
    }

    pub fn find<T: 'static>(&mut self) -> Option<&mut T> {
        self.layers
            .iter_mut()
            .find_map(|layer| layer.as_any_mut().downcast_mut())
    }

    pub fn handle_event(&mut self, event: &Event, area: Rect, cx: &mut Session) -> bool {
        let mut callbacks = Vec::new();
        let mut consumed = false;
        let broadcast = matches!(
            event,
            Event::Resize(_, _) | Event::Redraw | Event::AgentUpdated { .. }
        );

        // Most recently pushed layers take precedence. Callbacks run after the loop so they can
        // directly modify the layer list without conflicting with the active iteration.
        for layer in self.layers.iter_mut().rev() {
            match layer.handle_event(event, area, cx) {
                EventResult::Ignored(callback) => callbacks.extend(callback),
                EventResult::Consumed(callback) => {
                    callbacks.extend(callback);
                    consumed = true;
                    if !broadcast {
                        break;
                    }
                }
            }
        }

        for callback in callbacks {
            callback(self, cx);
        }

        consumed
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect, cx: &mut Session) {
        for layer in &mut self.layers {
            layer.render(frame, area, cx);
        }
    }
}

#[cfg(test)]
mod tests {
    use kyun_agent::{
        Agent,
        openai::{Config, OpenAiBackend},
    };
    use ratatui::{Frame, layout::Rect};

    use super::Compositor;
    use crate::{
        component::{Component, EventResult},
        event::Event,
        session::Session,
    };

    struct ClosingLayer;

    impl Component for ClosingLayer {
        fn render(&mut self, _frame: &mut Frame, _area: Rect, _cx: &mut Session) {}

        fn handle_event(&mut self, _event: &Event, _area: Rect, _cx: &mut Session) -> EventResult {
            EventResult::Consumed(Some(Box::new(|compositor, session| {
                compositor.pop();
                session.should_exit = true;
            })))
        }
    }

    #[test]
    fn callbacks_can_mutate_layers_after_event_dispatch() {
        let mut compositor = Compositor::new();
        compositor.push(Box::new(ClosingLayer));
        let mut session = Session::new(Agent::new(OpenAiBackend::new(Config {
            model: String::new(),
            base_url: String::new(),
            api_key: None,
            reasoning_effort: None,
            extra_body: None,
            ..kyun_agent::openai::Config::default()
        })));

        assert!(compositor.handle_event(&Event::FocusGained, Rect::default(), &mut session));
        assert!(compositor.layers.is_empty());
        assert!(session.should_exit);
    }
}
