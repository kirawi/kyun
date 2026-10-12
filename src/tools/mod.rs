pub mod apply_patch;
pub mod content_change;
pub mod edit;
pub mod file_mutation;
pub mod read;
pub mod read_directory;
pub mod shell;
pub mod write;

use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use crate::compositor::Compositor;
use crate::config::Config;
use crate::widgets::Transcript;
use kyun_agent::{ToolCall, ToolResult};

pub type ToolFuture = Pin<Box<dyn Future<Output = ToolResult> + Send>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolKind {
    FileMutation(file_mutation::Kind),
    Read,
    ReadDirectory,
    Shell,
    Unknown,
}

impl ToolKind {
    pub fn from_function_name(function_name: &str) -> Self {
        match function_name {
            write::NAME => Self::FileMutation(file_mutation::Kind::Write),
            edit::NAME => Self::FileMutation(file_mutation::Kind::Edit),
            apply_patch::NAME => Self::FileMutation(file_mutation::Kind::ApplyPatch),
            read::NAME => Self::Read,
            read_directory::NAME => Self::ReadDirectory,
            shell::NAME => Self::Shell,
            _ => Self::Unknown,
        }
    }
}

/// Application state made available while a tool call is integrated into the user interface.
pub struct Context<'a> {
    pub compositor: &'a mut Compositor,
    /// Paths already targeted by earlier file mutations in the batch being prepared. A later call touching one
    /// of them chains onto the batch's own results instead of the pre-batch snapshot every call validated
    /// against, which no longer describes the disk by the time it executes.
    pub mutated_paths: &'a mut HashSet<PathBuf>,
}

pub async fn prepare_execution(
    call: &ToolCall,
    config: Arc<Config>,
    context: &mut Context<'_>,
) -> ToolFuture {
    match ToolKind::from_function_name(&call.function_name) {
        ToolKind::FileMutation(kind) => {
            let tool_call_id = call.id.clone();
            let arguments = call.arguments.clone();
            let mut prepared = file_mutation::prepare(kind, &call.arguments).await;
            let chained = prepared.as_ref().is_ok_and(|prepared| {
                prepared
                    .target_paths()
                    .iter()
                    .any(|path| context.mutated_paths.contains(path))
            });
            if let Ok(prepared) = &prepared {
                let paths = prepared.target_paths();
                context.mutated_paths.extend(paths);
            }
            // A chained call re-prepares at execute time against the state its predecessors left behind, so
            // its pre-batch preview would show a diff that never existed on disk; no preview is cached for
            // it. The executed result still carries the true diff for the model and the transcript export.
            let preview = match &mut prepared {
                Ok(prepared) if !chained => Some(prepared.preview_diff()),
                _ => None,
            };
            context
                .compositor
                .find::<Transcript>()
                .unwrap()
                .cache_diff(preview);
            // A call whose prepare failed may have failed only against the pre-batch state (an edit whose
            // context assumes an earlier call's write), so failed prepares retry at execute time too; a
            // genuinely broken call just fails the same way twice.
            let reprepare = chained || prepared.is_err();
            Box::pin(async move {
                // A re-prepared call runs against the current on-disk state (the file as this batch's
                // earlier calls left it): a write simply applies to it, an edit whose context no longer fits
                // errors instead of clobbering, and external changes are still detected for untouched paths.
                let outcome = if reprepare {
                    file_mutation::prepare(kind, &arguments).await
                } else {
                    prepared
                };
                let content = match outcome {
                    Ok(prepared) => prepared
                        .execute(&config.backup)
                        .await
                        .map(|output| serde_json::to_string(&output).unwrap())
                        .unwrap_or_else(|error| format!("Error: {error}")),
                    Err(error) => format!("Error: {error}"),
                };
                ToolResult {
                    tool_call_id,
                    content,
                }
            })
        }
        ToolKind::Read => {
            let tool_call_id = call.id.clone();
            let params = serde_json::from_str::<read::ReadFileParams>(&call.arguments);
            Box::pin(async move {
                let content = match params {
                    Ok(params) => read::execute(params)
                        .await
                        .unwrap_or_else(|error| format!("Error: {error}")),
                    Err(error) => format!("Error: {error}"),
                };
                ToolResult {
                    tool_call_id,
                    content,
                }
            })
        }
        ToolKind::ReadDirectory => {
            let tool_call_id = call.id.clone();
            let params =
                serde_json::from_str::<read_directory::ListDirectoryParams>(&call.arguments);
            Box::pin(async move {
                let content = match params {
                    Ok(params) => read_directory::execute(params)
                        .await
                        .unwrap_or_else(|error| format!("Error: {error}")),
                    Err(error) => format!("Error: {error}"),
                };
                ToolResult {
                    tool_call_id,
                    content,
                }
            })
        }
        ToolKind::Shell => {
            let tool_call_id = call.id.clone();
            let params = serde_json::from_str::<shell::ShellParams>(&call.arguments);
            Box::pin(async move {
                let content = match params {
                    Ok(params) => shell::execute(params, &config.shell)
                        .await
                        .unwrap_or_else(|error| format!("Error: {error}")),
                    Err(error) => format!("Error: {error}"),
                };
                ToolResult {
                    tool_call_id,
                    content,
                }
            })
        }
        ToolKind::Unknown => {
            let tool_call_id = call.id.clone();
            let function_name = call.function_name.clone();
            Box::pin(async move {
                ToolResult {
                    tool_call_id,
                    content: format!("Error: unknown tool {function_name:?}"),
                }
            })
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug, PartialEq, Eq)]
pub struct FileMutationOutput {
    pub summary: String,
    pub diff: Arc<str>,
}
