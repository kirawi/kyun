use anyhow::bail;
use kyun_agent::{FunctionDefinition, ToolDefinition};
use serde_json::json;

use crate::path::{current_working_dir, working_directory_root};

pub const NAME: &str = "list_directory";

#[derive(serde::Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ListDirectoryParams {
    pub path: String,
}

pub fn definition() -> ToolDefinition {
    ToolDefinition::Function(FunctionDefinition {
        name: NAME.into(),
        description:
            "List the immediate entries of a directory under the current working directory.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory, relative to the current working directory or an absolute path within it."
                }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
        strict: true,
    })
}

pub async fn execute(params: ListDirectoryParams) -> anyhow::Result<String> {
    let requested = std::path::Path::new(&params.path);
    let requested = if requested.is_absolute() {
        requested.to_owned()
    } else {
        current_working_dir().join(requested)
    };
    let path = tokio::fs::canonicalize(requested).await?;

    if !path.starts_with(working_directory_root()) {
        bail!("path is outside the current working directory");
    }

    let mut directory = tokio::fs::read_dir(path).await?;
    let mut entries = Vec::new();
    while let Some(entry) = directory.next_entry().await? {
        let mut name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type().await?.is_dir() {
            name.push('/');
        }
        entries.push(name);
    }
    entries.sort_unstable();
    Ok(entries.join("\n"))
}
