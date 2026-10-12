// TODO: must handle (de)serialization
use etcetera::{BaseStrategy, choose_base_strategy};

use crate::{application::Application, config::Config};

mod application;
mod backup;
mod commands;
pub mod component;
mod compositor;
mod config;
mod event;
mod faccess;
mod file;
mod path;
mod session;
#[cfg(test)]
mod test_support;
mod timer;
mod tools;
pub mod widgets;

const TOOL_CALL_DENIED_MESSAGE: &str = "Error: tool call denied by user";
const TOOL_CALL_INTERRUPTED_MESSAGE: &str =
    "Error: tool result unavailable because execution was interrupted";
const TOOL_CALL_SKIPPED_MESSAGE: &str =
    "Error: tool call skipped because the batch was interrupted";

#[tokio::main]
async fn main() {
    let config_dir = choose_base_strategy().unwrap().config_dir().join("kyun");
    let config = Config::load(&config_dir);
    let mut app = Application::new(config);
    app.run().await;
}
