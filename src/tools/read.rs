use std::io::BufRead;

use anyhow::bail;
use kyun_agent::{FunctionDefinition, ToolDefinition};
use serde_json::json;

use crate::path::{current_working_dir, working_directory_root};

pub const NAME: &str = "read";

#[derive(serde::Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ReadFileParams {
    pub path: String,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

pub fn definition() -> ToolDefinition {
    ToolDefinition::Function(FunctionDefinition {
        name: NAME.into(),
        description: "Read the contents of a file under the current working directory.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to the current working directory or an absolute path within it."
                },
                "offset": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "description": "Number of lines to skip before reading, or null to start at the beginning."
                },
                "limit": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "description": "Maximum number of lines to read, or null to read the remainder of the file."
                }
            },
            "required": ["path", "offset", "limit"],
            "additionalProperties": false
        }),
        strict: true,
    })
}

pub async fn execute(params: ReadFileParams) -> anyhow::Result<String> {
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

    // Read through the same detection and decoding the mutation tools use, so UTF-16 and other non-UTF-8
    // files decode instead of erroring, and binary files are rejected consistently.
    let content = tokio::task::spawn_blocking({
        let path = path.clone();
        move || -> anyhow::Result<String> {
            let file = std::fs::File::open(&path)?;
            let mut reader = std::io::BufReader::new(file);
            if crate::file::is_binary(reader.fill_buf()?) {
                anyhow::bail!("{} is a binary file", path.display());
            }
            let (rope, _encoding, _has_bom) = crate::file::from_reader(&mut reader, None)?;
            Ok(rope.to_string())
        }
    })
    .await??;

    if params.offset.is_none() && params.limit.is_none() {
        return Ok(content);
    }

    Ok(content
        .split_inclusive('\n')
        .skip(params.offset.unwrap_or(0))
        .take(params.limit.unwrap_or(usize::MAX))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reads_a_file_in_the_current_working_directory() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("input.txt");
        tokio::fs::write(&path, "file content").await.unwrap();
        let content = execute(ReadFileParams {
            path: path.to_string_lossy().into_owned(),
            offset: None,
            limit: None,
        })
        .await
        .unwrap();

        assert_eq!(content, "file content");
    }

    #[tokio::test]
    async fn decodes_non_utf8_files_like_the_mutation_tools() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("utf16.txt");
        let mut bytes = vec![0xff, 0xfe];
        bytes.extend("utf-16 content\n".encode_utf16().flat_map(u16::to_le_bytes));
        tokio::fs::write(&path, bytes).await.unwrap();

        let content = execute(ReadFileParams {
            path: path.to_string_lossy().into_owned(),
            offset: None,
            limit: None,
        })
        .await
        .unwrap();

        assert_eq!(content, "utf-16 content\n");
    }

    #[tokio::test]
    async fn rejects_binary_files() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("blob.bin");
        tokio::fs::write(&path, [0u8, 1, 2, 3]).await.unwrap();

        let error = execute(ReadFileParams {
            path: path.to_string_lossy().into_owned(),
            offset: None,
            limit: None,
        })
        .await
        .unwrap_err();

        assert!(error.to_string().contains("binary file"), "{error}");
    }

    #[tokio::test]
    async fn rejects_paths_outside_the_current_working_directory() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let result = execute(ReadFileParams {
            path: file.path().to_string_lossy().into_owned(),
            offset: None,
            limit: None,
        })
        .await;

        assert_eq!(
            result.unwrap_err().to_string(),
            "path is outside the current working directory"
        );
    }

    #[tokio::test]
    async fn reads_a_range_of_lines() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("input.txt");
        tokio::fs::write(&path, "first\nsecond\nthird\nfourth\n")
            .await
            .unwrap();
        let path = path.to_string_lossy().into_owned();
        let content = execute(ReadFileParams {
            path: path.clone(),
            offset: Some(1),
            limit: Some(2),
        })
        .await
        .unwrap();

        assert_eq!(content, "second\nthird\n");

        let content = execute(ReadFileParams {
            path: path.clone(),
            offset: None,
            limit: Some(2),
        })
        .await
        .unwrap();

        assert_eq!(content, "first\nsecond\n");

        let content = execute(ReadFileParams {
            path,
            offset: Some(1),
            limit: None,
        })
        .await
        .unwrap();

        assert_eq!(content, "second\nthird\nfourth\n");
    }
}
