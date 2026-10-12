use std::io::{BufRead, BufReader, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context as _, bail};
use ropey::Rope;

use crate::backup::BackupConfig;
use crate::file;
use crate::path::{current_working_dir, working_directory_root};
use crate::tools::{FileMutationOutput, apply_patch, edit, write};
use encoding_rs::Encoding;

#[derive(Debug)]
pub enum FileSnapshot {
    Missing,
    Present {
        text: Rope,
        encoding: &'static Encoding,
        has_bom: bool,
        modified: SystemTime,
        size: u64,
    },
    Binary,
    Unreadable {
        message: String,
    },
}

impl FileSnapshot {
    pub async fn capture(path: &Path) -> Self {
        let path = path.to_owned();
        tokio::task::spawn_blocking(move || -> std::io::Result<Self> {
            let file = match std::fs::File::open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Self::Missing),
                Err(error) => return Err(error),
            };
            let mut reader = BufReader::new(file);
            if file::is_binary(reader.fill_buf()?) {
                return Ok(Self::Binary);
            }

            let (text, encoding, has_bom) = file::from_reader(&mut reader, None)?;
            let metadata = reader.get_ref().metadata()?;
            Ok(Self::Present {
                text,
                encoding,
                has_bom,
                modified: metadata.modified()?,
                size: metadata.len(),
            })
        })
        .await
        .map_err(|error| error.to_string())
        .and_then(|snapshot| snapshot.map_err(|error| error.to_string()))
        .unwrap_or_else(|message| Self::Unreadable { message })
    }

    pub async fn validate(&self, path: &Path, display: &str) -> Result<(), String> {
        let unchanged = match self {
            Self::Missing => matches!(
                tokio::fs::symlink_metadata(path).await,
                Err(error) if error.kind() == ErrorKind::NotFound
            ),
            Self::Present { modified, size, .. } => match tokio::fs::metadata(path).await {
                Ok(metadata) => metadata
                    .modified()
                    .is_ok_and(|current| *modified == current && *size == metadata.len()),
                Err(_) => false,
            },
            Self::Binary | Self::Unreadable { .. } => false,
        };
        if unchanged {
            return Ok(());
        }

        Err(format!(
            "{display} changed since the tool call was prepared"
        ))
    }
}

pub struct ResolvedPath {
    pub requested: PathBuf,
    pub target: PathBuf,
    pub existed: bool,
    pub is_symlink: bool,
}

impl ResolvedPath {
    pub fn pathname(&self) -> String {
        self.target
            .strip_prefix(working_directory_root())
            .unwrap()
            .to_string_lossy()
            .into_owned()
    }
}

pub async fn resolve_path(path: &str) -> anyhow::Result<ResolvedPath> {
    let root = working_directory_root();
    let requested = Path::new(path);
    let requested = if requested.is_absolute() {
        requested.to_owned()
    } else {
        current_working_dir().join(requested)
    };

    let requested_metadata = tokio::fs::symlink_metadata(&requested).await;
    let (target, existed, is_symlink) = match tokio::fs::canonicalize(&requested).await {
        Ok(target) => {
            let is_symlink = requested_metadata?.is_symlink();
            (target, true, is_symlink)
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            match requested_metadata {
                Ok(_) => return Err(error.into()),
                Err(error) if error.kind() != ErrorKind::NotFound => {
                    return Err(error.into());
                }
                Err(_) => {}
            }

            let mut ancestor = requested.clone();
            let mut missing = Vec::new();
            loop {
                let Some(parent) = ancestor.parent() else {
                    bail!("could not resolve path: {path}");
                };
                let Some(name) = ancestor.file_name() else {
                    bail!("path does not name a file: {path}");
                };
                missing.push(name.to_owned());
                ancestor = parent.to_owned();

                if let Ok(canonical) = tokio::fs::canonicalize(&ancestor).await {
                    let mut target = canonical;
                    for component in missing.iter().rev() {
                        target.push(component);
                    }
                    break (target, false, false);
                }
            }
        }
        Err(error) => return Err(error.into()),
    };

    if !target.starts_with(&root) {
        bail!("path is outside the current working directory: {path}");
    }

    Ok(ResolvedPath {
        requested,
        target,
        existed,
        is_symlink,
    })
}

enum PreparedOperation {
    Write(write::PreparedWrite),
    Edit(write::PreparedWrite),
    ApplyPatch(apply_patch::PreparedPatch),
}

pub struct PreparedFileMutation {
    operation: PreparedOperation,
    output: Option<FileMutationOutput>,
    /// The structured diff paired with the tool output.
    diff: Option<write::FileDiff>,
}

impl PreparedFileMutation {
    /// The files this mutation intends to touch. Used to detect later calls in the same batch that chain onto
    /// an earlier call's results.
    pub fn target_paths(&self) -> Vec<std::path::PathBuf> {
        match &self.operation {
            PreparedOperation::Write(prepared) | PreparedOperation::Edit(prepared) => {
                vec![prepared.resolved.target.clone()]
            }
            PreparedOperation::ApplyPatch(prepared) => prepared.target_paths(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Write,
    Edit,
    ApplyPatch,
}

pub async fn prepare(kind: Kind, arguments: &str) -> anyhow::Result<PreparedFileMutation> {
    match kind {
        Kind::Write => {
            let params = serde_json::from_str(arguments)?;
            write::prepare(params)
                .await
                .map(|prepared| PreparedFileMutation {
                    operation: PreparedOperation::Write(prepared),
                    output: None,
                    diff: None,
                })
        }
        Kind::Edit => edit::prepare(arguments)
            .await
            .map(|prepared| PreparedFileMutation {
                operation: PreparedOperation::Edit(prepared),
                output: None,
                diff: None,
            }),
        Kind::ApplyPatch => {
            apply_patch::prepare(arguments)
                .await
                .map(|prepared| PreparedFileMutation {
                    operation: PreparedOperation::ApplyPatch(prepared),
                    output: None,
                    diff: None,
                })
        }
    }
}

impl PreparedFileMutation {
    /// Materializes the operation's output and structured diff once; both are reused by `preview` and `execute`.
    fn materialize(&mut self) {
        if self.output.is_none() {
            let (output, diff) = match &mut self.operation {
                PreparedOperation::Write(prepared) => prepared.write_output(),
                PreparedOperation::Edit(prepared) => prepared.edit_output(),
                PreparedOperation::ApplyPatch(prepared) => prepared.output(),
            };
            self.output = Some(output);
            self.diff = Some(diff);
        }
    }

    /// The raw diff text computed from the file-state snapshot captured for the tool call.
    #[cfg(test)]
    pub fn raw_diff(&mut self) -> &str {
        self.materialize();
        &self.output.as_ref().unwrap().diff
    }

    /// Returns the structured diff that accompanies this mutation's provisional output before approval.
    pub fn preview_diff(&mut self) -> write::FileDiff {
        self.materialize();
        self.diff.take().unwrap()
    }

    /// The tool result built from the file-state snapshot captured during preparation.
    pub fn output(&mut self) -> &FileMutationOutput {
        self.materialize();
        self.output.as_ref().unwrap()
    }

    pub async fn execute(
        mut self,
        backup_config: &BackupConfig,
    ) -> anyhow::Result<FileMutationOutput> {
        let output = self.output().clone();
        match self.operation {
            PreparedOperation::Write(prepared) => {
                prepared.execute(backup_config).await?;
            }
            PreparedOperation::Edit(prepared) => prepared
                .execute(backup_config)
                .await
                .context("failed to write")?,
            PreparedOperation::ApplyPatch(prepared) => {
                prepared.execute(backup_config).await?;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{Kind, prepare};
    use crate::backup::BackupConfig;

    #[tokio::test]
    async fn prepared_diff_remains_based_on_the_tool_call_snapshot() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("file.txt");
        tokio::fs::write(&path, "before\n").await.unwrap();
        let arguments = serde_json::json!({
            "path": path.display().to_string(),
            "content": "after\n",
        })
        .to_string();
        let mut prepared = prepare(Kind::Write, &arguments).await.unwrap();
        tokio::fs::write(&path, "external change\n").await.unwrap();
        filetime::set_file_mtime(&path, filetime::FileTime::from_unix_time(2_000_000_000, 0))
            .unwrap();

        assert!(prepared.raw_diff().contains("-before\n+after\n"));
        assert!(!prepared.raw_diff().contains("external change"));
        assert!(
            prepared
                .execute(&BackupConfig::default())
                .await
                .unwrap_err()
                .to_string()
                .contains("changed since the tool call was prepared")
        );
        assert_eq!(
            tokio::fs::read_to_string(path).await.unwrap(),
            "external change\n"
        );
    }
}
