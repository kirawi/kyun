use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use futures_util::future::OptionFuture;
use tokio::time::{Instant, Sleep};

use crate::{compositor::Compositor, session::Session};

pub const APPROVAL_TIMER_ID: &str = "approval";
pub const STATUSLINE_TIMER_ID: &str = "statusline";

// The bool indicates whether a redraw is necessary.
pub type Job = Box<dyn FnMut(&mut Compositor, &mut Session) -> (bool, Output)>;

#[derive(Default)]
pub struct TimerRegistry {
    timers: BinaryHeap<Reverse<Timer>>,
}

pub struct Timer {
    /// Stable identifier used to replace or cancel a timer.
    id: &'static str,
    callback: Job,
    deadline: Instant,
}

impl PartialEq for Timer {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline
    }
}

impl Eq for Timer {}

impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timer {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deadline.cmp(&other.deadline)
    }
}

pub enum Output {
    Reschedule(Instant),
    Stop,
}

impl TimerRegistry {
    pub fn schedule(
        &mut self,
        id: &'static str,
        deadline: Instant,
        callback: impl FnMut(&mut Compositor, &mut Session) -> (bool, Output) + 'static,
    ) {
        self.timers.retain(|Reverse(timer)| timer.id != id);
        self.timers.push(Reverse(Timer {
            id,
            callback: Box::new(callback),
            deadline,
        }));
    }

    pub fn cancel(&mut self, id: &'static str) {
        self.timers.retain(|Reverse(timer)| timer.id != id);
    }

    pub fn next_deadline(&self) -> Option<Instant> {
        self.timers.peek().map(|Reverse(timer)| timer.deadline)
    }

    pub fn wait_next_deadline(&self) -> OptionFuture<Sleep> {
        self.next_deadline().map(tokio::time::sleep_until).into()
    }

    pub fn pop_due(&mut self) -> Option<Timer> {
        if self
            .next_deadline()
            .is_none_or(|deadline| deadline > Instant::now())
        {
            return None;
        }
        Some(self.timers.pop().unwrap().0)
    }

    pub fn finish(&mut self, mut timer: Timer, output: Output) {
        if let Output::Reschedule(deadline) = output {
            timer.deadline = deadline;
            self.timers
                .retain(|Reverse(existing)| existing.id != timer.id);
            self.timers.push(Reverse(timer));
        }
    }
}

impl Timer {
    pub fn run(&mut self, compositor: &mut Compositor, session: &mut Session) -> (bool, Output) {
        (self.callback)(compositor, session)
    }
}
