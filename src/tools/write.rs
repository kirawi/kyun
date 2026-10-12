use anyhow::{Result, anyhow, bail};
use imara_diff::{
    Algorithm, Diff, InternedInput, Interner, Token, UnifiedDiffConfig, UnifiedDiffPrinter,
};
use kyun_agent::{FunctionDefinition, ToolDefinition};
use ropey::Rope;
use serde_json::json;
use std::cell::RefCell;
use std::fmt::{self, Write as _};
use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;

use crate::backup::{Backup, BackupConfig, BackupKind, BackupMethod};
use crate::faccess;
use crate::file;
use crate::path::current_working_dir;
use crate::tools::FileMutationOutput;
use crate::tools::content_change::ChangeSet;
use crate::tools::file_mutation::{FileSnapshot, ResolvedPath, resolve_path};

pub const NAME: &str = "write";
const UNIFIED_DIFF_CONTEXT_LEN: usize = 3;

#[derive(serde::Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct WriteFileParams {
    pub path: String,
    pub content: String,
}

/// A single line of a file diff, parallel to the text emitted by imara's unified-diff printer. `content` is the byte
/// range within the file's diff text, excluding the `---`/`+++` headers that [`file_diff`] prepends. It includes the
/// leading `+`/`-`/space marker and excludes the trailing newline.
#[derive(Clone, Debug)]
pub struct DiffLine {
    pub content: Range<usize>,
    pub old_line: Option<NonZeroU32>,
    pub new_line: Option<NonZeroU32>,
    pub kind: DiffLineKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffLineKind {
    /// The collapsed `---`/`+++` file header; its content range points at the displayed path in the text.
    File,
    Addition,
    Removal,
    Context,
    /// A non-diff informational line (e.g. `Text diff unavailable: ...`); carries no line numbers.
    Note,
}

/// A file's diff: the unified-diff text for the tool result, plus the structured lines it was built from.
pub struct FileDiff {
    pub text: Arc<str>,
    pub lines: Box<[DiffLine]>,
}

struct RecordedDiffLine {
    token: Token,
    old_line: Option<NonZeroU32>,
    new_line: Option<NonZeroU32>,
    kind: DiffLineKind,
}

struct StructuredDiffState {
    before_cursor: usize,
    after_cursor: usize,
    trailing_context: usize,
    pending: Vec<RecordedDiffLine>,
    output_len: usize,
    lines: Vec<DiffLine>,
}

struct StructuredDiffPrinter<'a> {
    interner: &'a Interner<&'a str>,
    before: &'a [Token],
    state: RefCell<StructuredDiffState>,
}

impl<'a> StructuredDiffPrinter<'a> {
    fn record_line(&self, state: &mut StructuredDiffState, line: RecordedDiffLine) {
        let text = self.interner[line.token];
        let start = state.output_len;
        state.output_len += 1 + text.len() + usize::from(!text.ends_with('\n'));
        let content = text
            .strip_suffix('\n')
            .and_then(|text| text.strip_suffix('\r').or(Some(text)))
            .unwrap_or(text);
        state.lines.push(DiffLine {
            content: start..start + 1 + content.len(),
            old_line: line.old_line,
            new_line: line.new_line,
            kind: line.kind,
        });
    }

    fn next_line(cursor: &mut usize) -> NonZeroU32 {
        *cursor += 1;
        NonZeroU32::new(u32::try_from(*cursor).unwrap()).unwrap()
    }

    fn finish(self) -> (usize, Vec<DiffLine>) {
        let state = self.state.into_inner();
        assert!(state.pending.is_empty());
        (state.output_len, state.lines)
    }
}

impl UnifiedDiffPrinter for StructuredDiffPrinter<'_> {
    fn display_header(
        &self,
        mut f: impl fmt::Write,
        start_before: u32,
        start_after: u32,
        len_before: u32,
        len_after: u32,
    ) -> fmt::Result {
        let header = format!(
            "@@ -{},{} +{},{} @@",
            start_before + u32::from(len_before != 0),
            len_before,
            start_after + u32::from(len_after != 0),
            len_after
        );
        writeln!(f, "{header}")?;

        let mut state = self.state.borrow_mut();
        state.output_len += header.len() + 1;
        let leading_context = state
            .pending
            .iter()
            .take_while(|line| line.kind == DiffLineKind::Context)
            .count();
        let before_hunk_start = if start_before == 0 {
            leading_context
        } else {
            usize::try_from(start_before).unwrap() + UNIFIED_DIFF_CONTEXT_LEN
        };
        let after_hunk_start = if start_after == 0 {
            leading_context
        } else {
            usize::try_from(start_after).unwrap() + UNIFIED_DIFF_CONTEXT_LEN
        };
        let mut before_cursor = before_hunk_start - leading_context;
        let mut after_cursor = after_hunk_start - leading_context;
        for line in &mut state.pending {
            if matches!(line.kind, DiffLineKind::Context | DiffLineKind::Removal) {
                line.old_line = Some(Self::next_line(&mut before_cursor));
            }
            if matches!(line.kind, DiffLineKind::Context | DiffLineKind::Addition) {
                line.new_line = Some(Self::next_line(&mut after_cursor));
            }
        }
        state.before_cursor = before_cursor;
        state.after_cursor = after_cursor;
        for line in std::mem::take(&mut state.pending) {
            self.record_line(&mut state, line);
        }
        state.trailing_context = self
            .before
            .len()
            .min(state.before_cursor + UNIFIED_DIFF_CONTEXT_LEN)
            - state.before_cursor;
        Ok(())
    }

    fn display_context_token(&self, mut f: impl fmt::Write, token: Token) -> fmt::Result {
        let text = self.interner[token];
        write!(f, " {text}")?;
        if !text.ends_with('\n') {
            writeln!(f)?;
        }

        let mut state = self.state.borrow_mut();
        if state.trailing_context > 0 {
            state.trailing_context -= 1;
            let old_line = Self::next_line(&mut state.before_cursor);
            let new_line = Self::next_line(&mut state.after_cursor);
            self.record_line(
                &mut state,
                RecordedDiffLine {
                    token,
                    old_line: Some(old_line),
                    new_line: Some(new_line),
                    kind: DiffLineKind::Context,
                },
            );
        } else {
            state.pending.push(RecordedDiffLine {
                token,
                old_line: None,
                new_line: None,
                kind: DiffLineKind::Context,
            });
        }
        Ok(())
    }

    fn display_hunk(
        &self,
        mut f: impl fmt::Write,
        before: &[Token],
        after: &[Token],
    ) -> fmt::Result {
        let mut state = self.state.borrow_mut();
        assert_eq!(state.trailing_context, 0);
        for &token in before {
            let text = self.interner[token];
            write!(f, "-{text}")?;
            if !text.ends_with('\n') {
                writeln!(f)?;
            }
            state.pending.push(RecordedDiffLine {
                token,
                old_line: None,
                new_line: None,
                kind: DiffLineKind::Removal,
            });
        }
        for &token in after {
            let text = self.interner[token];
            write!(f, "+{text}")?;
            if !text.ends_with('\n') {
                writeln!(f)?;
            }
            state.pending.push(RecordedDiffLine {
                token,
                old_line: None,
                new_line: None,
                kind: DiffLineKind::Addition,
            });
        }
        Ok(())
    }
}

/// Builds unified text with imara's printer and records metadata for each displayed source line.
pub fn file_diff<'a>(
    before_path: Option<&str>,
    after_path: Option<&str>,
    before: &'a str,
    after: &'a str,
) -> FileDiff {
    if before_path == after_path && before == after {
        return FileDiff {
            text: String::new().into(),
            lines: Box::default(),
        };
    }

    let input = InternedInput::new(before, after);
    let mut diff = Diff::compute(Algorithm::Histogram, &input);
    diff.postprocess_lines(&input);

    // The displayed path is the `---` path when the file was deleted (`+++ /dev/null`), else the `+++` path.
    // The File line's content range points at that path within the header.
    let (before_path, after_path) = (
        before_path.unwrap_or("/dev/null"),
        after_path.unwrap_or("/dev/null"),
    );
    let mut text = format!("--- {before_path}\n+++ {after_path}\n");
    let path_offset = if after_path == "/dev/null" {
        4 // after "--- "
    } else {
        "--- ".len() + before_path.len() + "\n+++ ".len()
    };
    let path_len = if after_path == "/dev/null" {
        before_path.len()
    } else {
        after_path.len()
    };
    let printer = StructuredDiffPrinter {
        interner: &input.interner,
        before: &input.before,
        state: RefCell::new(StructuredDiffState {
            before_cursor: 0,
            after_cursor: 0,
            trailing_context: 0,
            pending: Vec::new(),
            output_len: text.len(),
            lines: Vec::new(),
        }),
    };
    let mut config = UnifiedDiffConfig::default();
    config.context_len(UNIFIED_DIFF_CONTEXT_LEN as u32);
    write!(&mut text, "{}", diff.unified_diff(&printer, config, &input)).unwrap();
    let (output_len, lines) = printer.finish();
    assert_eq!(text.len(), output_len);

    let mut all_lines = Vec::with_capacity(lines.len() + 1);
    all_lines.push(DiffLine {
        content: path_offset..path_offset + path_len,
        old_line: None,
        new_line: None,
        kind: DiffLineKind::File,
    });
    all_lines.extend(lines);
    FileDiff {
        text: text.into(),
        lines: all_lines.into_boxed_slice(),
    }
}

pub fn definition() -> ToolDefinition {
    ToolDefinition::Function(FunctionDefinition {
        name: NAME.into(),
        description: "Write content to a file under the current working directory.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to the current working directory or an absolute path within it."
                },
                "content": {
                    "type": "string",
                    "description": "Complete content to write to the file."
                }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
        strict: true,
    })
}

pub struct PreparedWrite {
    pub resolved: ResolvedPath,
    observed: FileSnapshot,
    pub text: Rope,
    encoding: &'static encoding_rs::Encoding,
    has_bom: bool,
    pub changed: bool,
}

fn validate_snapshot(resolved: &ResolvedPath, observed: &FileSnapshot) -> Result<()> {
    if resolved.existed && matches!(observed, FileSnapshot::Missing)
        || !resolved.existed
            && matches!(
                observed,
                FileSnapshot::Present { .. }
                    | FileSnapshot::Binary
                    | FileSnapshot::Unreadable { .. }
            )
    {
        bail!(
            "{} changed while its path was being resolved",
            resolved.pathname()
        );
    }
    Ok(())
}

pub async fn prepare(params: WriteFileParams) -> Result<PreparedWrite> {
    let resolved = resolve_path(&params.path).await?;
    let observed = FileSnapshot::capture(&resolved.target).await;
    PreparedWrite::from_resolved(params, resolved, observed)
}

impl PreparedWrite {
    pub fn from_change_set(
        resolved: ResolvedPath,
        observed: FileSnapshot,
        content: ChangeSet,
    ) -> Result<Self> {
        validate_snapshot(&resolved, &observed)?;

        let (mut text, encoding, has_bom) = match &observed {
            FileSnapshot::Missing => (Rope::new(), encoding_rs::UTF_8, false),
            FileSnapshot::Present {
                text,
                encoding,
                has_bom,
                ..
            } => (text.clone(), *encoding, *has_bom),
            FileSnapshot::Binary => {
                bail!("{} is a binary file", resolved.pathname());
            }
            FileSnapshot::Unreadable { message } => bail!("{message}"),
        };
        if !content.apply(&mut text) {
            bail!(
                "{} changed while its path was being resolved",
                resolved.pathname()
            );
        }
        let changed = content.is_changed() || !resolved.existed;
        Ok(Self {
            resolved,
            observed,
            text,
            encoding,
            has_bom,
            changed,
        })
    }

    pub fn from_text(
        resolved: ResolvedPath,
        observed: FileSnapshot,
        text: Rope,
        encoding: &'static encoding_rs::Encoding,
        has_bom: bool,
    ) -> Result<Self> {
        validate_snapshot(&resolved, &observed)?;

        let changed = match &observed {
            FileSnapshot::Missing => true,
            FileSnapshot::Present {
                text: before,
                encoding: before_encoding,
                has_bom: before_has_bom,
                ..
            } => before != &text || *before_encoding != encoding || *before_has_bom != has_bom,
            FileSnapshot::Binary => {
                bail!("{} is a binary file", resolved.pathname());
            }
            FileSnapshot::Unreadable { message } => bail!("{message}"),
        };

        Ok(Self {
            resolved,
            observed,
            text,
            encoding,
            has_bom,
            changed,
        })
    }

    pub fn from_resolved(
        params: WriteFileParams,
        resolved: ResolvedPath,
        observed: FileSnapshot,
    ) -> Result<Self> {
        let content = params.content;
        // Compare without creating a String: Rope's PartialEq<&str> walks chunks.
        let unchanged = match &observed {
            FileSnapshot::Present {
                text,
                encoding,
                has_bom,
                ..
            } if *text == content.as_str() => Some((text.clone(), *encoding, *has_bom)),
            _ => None,
        };
        if let Some((text, encoding, has_bom)) = unchanged {
            validate_snapshot(&resolved, &observed)?;
            return Ok(Self {
                resolved,
                observed,
                text,
                encoding,
                has_bom,
                changed: false,
            });
        }

        let len_before = match &observed {
            FileSnapshot::Missing => 0,
            FileSnapshot::Present { text, .. } => text.len_chars(),
            FileSnapshot::Binary => {
                bail!("{} is a binary file", resolved.pathname());
            }
            FileSnapshot::Unreadable { message } => bail!("{message}"),
        };
        Self::from_change_set(resolved, observed, ChangeSet::replace(len_before, content))
    }

    pub fn diff(&mut self, before_path: Option<&str>, after_path: Option<&str>) -> FileDiff {
        if !self.changed {
            return FileDiff {
                text: String::new().into(),
                lines: Box::default(),
            };
        }

        let before = match &self.observed {
            FileSnapshot::Missing => String::new(),
            FileSnapshot::Present { text, .. } => text.to_string(),
            FileSnapshot::Binary => {
                return unavailable_diff(before_path, after_path, "previous contents were binary");
            }
            FileSnapshot::Unreadable { message } => {
                return unavailable_diff(
                    before_path,
                    after_path,
                    &format!("could not read previous contents: {message}"),
                );
            }
        };
        let after = self.text.to_string();
        file_diff(before_path, after_path, &before, &after)
    }

    pub fn write_output(&mut self) -> (FileMutationOutput, FileDiff) {
        let pathname = self.resolved.pathname();
        let before_path = self.resolved.existed.then_some(pathname.as_str());
        let diff = self.diff(before_path, Some(&pathname));
        let output = FileMutationOutput {
            summary: if self.changed {
                // The encoded on-disk size, not the rope's UTF-8 length, so the count matches the file for
                // every encoding.
                if self.resolved.existed {
                    format!(
                        "Updated {} bytes.",
                        crate::file::encoded_len(&self.text, (self.encoding, self.has_bom))
                    )
                } else {
                    format!(
                        "Created {} bytes.",
                        crate::file::encoded_len(&self.text, (self.encoding, self.has_bom))
                    )
                }
            } else {
                format!(
                    "Unchanged ({} bytes).",
                    crate::file::encoded_len(&self.text, (self.encoding, self.has_bom))
                )
            },
            diff: diff.text.clone(),
        };
        (output, diff)
    }

    pub fn edit_output(&mut self) -> (FileMutationOutput, FileDiff) {
        let pathname = self.resolved.pathname();
        let diff = self.diff(Some(&pathname), Some(&pathname));
        let output = FileMutationOutput {
            summary: format!("Success. Updated the following files:\nM {pathname}\n"),
            diff: diff.text.clone(),
        };
        (output, diff)
    }

    pub async fn validate(&self) -> Result<()> {
        self.observed
            .validate(&self.resolved.target, &self.resolved.pathname())
            .await
            .map_err(anyhow::Error::msg)
    }

    pub async fn execute(self, backup_config: &BackupConfig) -> Result<()> {
        self.validate().await?;
        self.commit(backup_config).await
    }

    pub async fn commit(self, backup_config: &BackupConfig) -> Result<()> {
        if !self.changed {
            return Ok(());
        }

        if faccess::readonly(&self.resolved.target) {
            bail!("Path is read only");
        }

        if let Some(parent) = self.resolved.target.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Use a backup file
        let metadata = if self.resolved.existed {
            Some(tokio::fs::metadata(&self.resolved.target).await?)
        } else {
            None
        };

        let backup = if self.resolved.existed && backup_config.kind != BackupKind::None {
            Some(
                Backup::create(
                    &self.resolved.target,
                    &self.resolved.requested,
                    self.resolved.is_symlink,
                    &current_working_dir(),
                    backup_config,
                )
                .await
                .map_err(|error| anyhow!("could not create backup: {error}"))?,
            )
        } else {
            None
        };

        let write_result: Result<()> = async {
            let mut file = match backup.as_ref().map(|backup| (backup.method, backup)) {
                Some((BackupMethod::Rename, backup)) => {
                    // SECURITY: Ensure that the created file has the same perms as the original file
                    let metadata = metadata.as_ref().unwrap();
                    let mut open_options = tokio::fs::OpenOptions::new();
                    open_options.read(true).write(true).create_new(true);

                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let mode = metadata.permissions().mode();
                        open_options.mode(mode);
                    }

                    let file = open_options.open(&self.resolved.target).await?;

                    #[cfg(unix)]
                    {
                        use std::os::fd::AsFd;
                        use std::os::unix::fs::MetadataExt;

                        faccess::fchown(file.as_fd(), Some(metadata.uid()), Some(metadata.gid()))?;
                        file.set_permissions(metadata.permissions()).await?;

                        // RECALL: File at `target` is newly created, and file at `backup.path` is the original.
                        if let Err(error) = faccess::copy_xattr(&backup.path, &self.resolved.target)
                        {
                            let unsupported =
                                rustix::io::Errno::from_io_error(&error).is_some_and(|errno| {
                                    errno == rustix::io::Errno::NOTSUP
                                        || errno == rustix::io::Errno::OPNOTSUPP
                                });
                            if !unsupported {
                                return Err(error.into());
                            }
                        }
                    }

                    #[cfg(target_os = "macos")]
                    {
                        use std::fs::FileTimes;
                        use std::os::macos::fs::FileTimesExt;

                        let file = file.try_clone().await?.into_std().await;
                        let times = FileTimes::new().set_created(metadata.created()?);
                        file.set_times(times)?;
                    }

                    #[cfg(windows)]
                    {
                        let from = backup.path.clone();
                        let to = self.resolved.target.clone();
                        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                            faccess::copy_metadata(&from, &to)?;
                            Ok(())
                        })
                        .await??;
                    }

                    file
                }
                Some((BackupMethod::Copy, _)) => {
                    // INTEGRITY: Backup copy already exists. But possible TOCTOU race w/ file being changed underneath
                    let mut open_options = tokio::fs::File::options();
                    open_options
                        .create(false)
                        .write(true)
                        .truncate(true)
                        .open(&self.resolved.target)
                        .await?
                }
                None => tokio::fs::File::create(&self.resolved.target).await?,
            };

            file::to_writer(&mut file, (self.encoding, self.has_bom), &self.text).await?;
            match file.sync_all().await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::Unsupported => {}
                #[cfg(unix)]
                Err(error)
                    if rustix::io::Errno::from_io_error(&error).is_some_and(|errno| {
                        errno == rustix::io::Errno::NOTSUP || errno == rustix::io::Errno::OPNOTSUPP
                    }) => {}
                Err(error) => return Err(error.into()),
            }
            Ok(())
        }
        .await;

        // A failed write with a restorable backup leaves the file as it was; only when the restore itself
        // failed is the file actually in a partially written state.
        let mut restored_after_failure = false;
        if let Some(backup) = backup {
            let mut delete = true;

            if write_result.is_err()
                && backup
                    .restore(&self.resolved.target, metadata.as_ref().unwrap())
                    .await
                    .is_ok()
            {
                restored_after_failure = true;
            } else if write_result.is_err() {
                delete = false;
            }

            if delete {
                // Delete backup
                let _ = tokio::fs::remove_file(backup.path).await;
            }
        }

        write_result.map_err(|error| {
            if restored_after_failure {
                anyhow!("{error}; the file was restored to its previous contents")
            } else {
                anyhow!("{error}; the file may have been partially changed")
            }
        })
    }
}

/// A diff that carries only a human-readable reason (the previous contents were binary or unreadable). The
/// text matches the usual `---`/`+++` shape so the tool result stays uniform; the structured lines are just the
/// file header and the note.
fn unavailable_diff(before_path: Option<&str>, after_path: Option<&str>, reason: &str) -> FileDiff {
    let (before_path, after_path) = (
        before_path.unwrap_or("/dev/null"),
        after_path.unwrap_or("/dev/null"),
    );
    let header = format!("--- {before_path}\n+++ {after_path}\n");
    let note = format!("Text diff unavailable: {reason}");
    let path_offset = if after_path == "/dev/null" {
        4
    } else {
        "--- ".len() + before_path.len() + "\n+++ ".len()
    };
    let note_offset = header.len();
    FileDiff {
        text: format!("{header}{note}\n").into(),
        lines: vec![
            DiffLine {
                content: path_offset
                    ..path_offset
                        + if after_path == "/dev/null" {
                            before_path.len()
                        } else {
                            after_path.len()
                        },
                old_line: None,
                new_line: None,
                kind: DiffLineKind::File,
            },
            DiffLine {
                content: note_offset..note_offset + note.len(),
                old_line: None,
                new_line: None,
                kind: DiffLineKind::Note,
            },
        ]
        .into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::io::Cursor;
    use std::path::Path;

    use imara_diff::{Interner, Token, UnifiedDiffConfig, UnifiedDiffPrinter};

    use super::*;
    use crate::file;

    use filetime::FileTime;

    /// Every recorded body line's range must slice the emitted text at exactly that line, and splicing the
    /// lines and the `@@` headers back together in order must reproduce the hunk body byte-for-byte. This is
    /// the empirical check that the byte ranges and line numbers fall out of the token indices correctly.
    #[test]
    fn file_diff_line_ranges_index_the_emitted_text() {
        let before = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n";
        let after = "one\ntwo\nTHREE\nfour\nfive\nsix\nseven\n8\nnine\nten\neleven\n";
        let file = file_diff(Some("f"), Some("f"), before, after);

        // The first line is the File header; the rest are body lines.
        assert_eq!(file.lines[0].kind, DiffLineKind::File);
        let body_lines = &file.lines[1..];
        for line in body_lines {
            let text = &file.text[line.content.clone()];
            let marker = match line.kind {
                DiffLineKind::Addition => '+',
                DiffLineKind::Removal => '-',
                DiffLineKind::Context => ' ',
                DiffLineKind::File | DiffLineKind::Note => unreachable!(),
            };
            assert!(
                text.starts_with(marker),
                "{text:?} should start with {marker:?}"
            );
            assert!(!text.contains('\n'), "{text:?} should be one line");
        }

        // Reconstruct the body from the recorded lines: the bytes between consecutive lines (and before the
        // first) must be exactly the @@ headers, so splicing the headers and lines back together in order must
        // reproduce the body byte-for-byte.
        let body_start = file.text.find("@@").unwrap();
        let body = &file.text[body_start..];
        let mut reassembled = String::new();
        let mut offset = body_start;
        for line in body_lines {
            reassembled.push_str(&file.text[offset..line.content.start]);
            reassembled.push_str(&file.text[line.content.clone()]);
            reassembled.push('\n');
            offset = line.content.end + 1;
        }
        reassembled.push_str(&file.text[offset..]);
        assert_eq!(
            reassembled, body,
            "recorded lines must reassemble into the body"
        );
    }

    /// Checks imara's emitted text and the recorded line ranges across several input pairs.
    #[test]
    fn file_diff_matches_the_reference_unified_diff() {
        /// Imara's printer with the file-header convention used by `file_diff`.
        struct Reference<'a>(&'a Interner<&'a str>);
        impl UnifiedDiffPrinter for Reference<'_> {
            fn display_header(
                &self,
                mut f: impl fmt::Write,
                start_before: u32,
                start_after: u32,
                len_before: u32,
                len_after: u32,
            ) -> fmt::Result {
                writeln!(
                    f,
                    "@@ -{},{} +{},{} @@",
                    start_before + u32::from(len_before != 0),
                    len_before,
                    start_after + u32::from(len_after != 0),
                    len_after
                )
            }
            fn display_context_token(&self, f: impl fmt::Write, token: Token) -> fmt::Result {
                self.line(f, ' ', token)
            }
            fn display_hunk(
                &self,
                mut f: impl fmt::Write,
                before: &[Token],
                after: &[Token],
            ) -> fmt::Result {
                for &token in before {
                    self.line(&mut f, '-', token)?;
                }
                for &token in after {
                    self.line(&mut f, '+', token)?;
                }
                Ok(())
            }
        }
        impl Reference<'_> {
            fn line(&self, mut f: impl fmt::Write, prefix: char, token: Token) -> fmt::Result {
                let line = self.0[token];
                write!(f, "{prefix}{line}")?;
                if !line.ends_with('\n') {
                    writeln!(f)?;
                }
                Ok(())
            }
        }

        // (before, after): single hunk, multiple split hunks, pure insertion, pure deletion, CRLF lines, and a
        // missing final newline on both sides.
        let cases: &[(&str, &str)] = &[
            ("a\nb\nc\nd\ne\n", "a\nb\nX\nd\ne\n"),
            (
                "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20\n",
                "1\n2\nX\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\nY\n19\n20\n",
            ),
            ("a\nb\nc\n", "a\nb\ninserted\nc\n"),
            ("a\nb\nc\n", "a\nc\n"),
            ("a\r\nb\r\nc\r\n", "a\r\nX\r\nc\r\n"),
            ("a\nb\nc", "a\nX\nc"),
            ("a\nb\nc\n", "a\nX\nc"),
            // A deep first hunk (start beyond 2*context): imara clamps the leading context to the window.
            (
                "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12\nl13\nl14\nl15\nl16\nl17\nl18\nl19\nl20\n",
                "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nX10\nl11\nl12\nl13\nl14\nl15\nl16\nl17\nl18\nl19\nl20\n",
            ),
            // A first group adds a line, then a second group starts after a split: its leading context uses the
            // net +1 offset (context l14 is old 14 / new 15).
            (
                "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12\nl13\nl14\nl15\nl16\nl17\nl18\nl19\nl20\n",
                "l1\nl2\n3a\n3b\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12\nl13\nl14\nl15\nl16\nY17\nl18\nl19\nl20\n",
            ),
        ];
        for (before, after) in cases {
            let file = file_diff(Some("p"), Some("p"), before, after);

            let input = InternedInput::new(*before, *after);
            let mut diff = Diff::compute(Algorithm::Histogram, &input);
            diff.postprocess_lines(&input);
            let reference = format!(
                "--- p\n+++ p\n{}",
                diff.unified_diff(
                    &Reference(&input.interner),
                    UnifiedDiffConfig::default(),
                    &input
                )
            );
            assert_eq!(
                file.text.as_ref(),
                reference,
                "text mismatch for {before:?} -> {after:?}"
            );

            // Every body line's range slices the text at that line.
            for line in &file.lines[1..] {
                let text = &file.text[line.content.clone()];
                let marker = match line.kind {
                    DiffLineKind::Addition => '+',
                    DiffLineKind::Removal => '-',
                    DiffLineKind::Context => ' ',
                    DiffLineKind::File | DiffLineKind::Note => unreachable!(),
                };
                assert!(
                    text.starts_with(marker) && !text.contains('\n'),
                    "range {:?} gave {text:?} for {before:?} -> {after:?}",
                    line.content
                );
            }
        }
    }

    /// Body line numbers must be the line's true file position, even where imara's `@@` header start
    /// disagrees with the body's first line (imara does not clamp an over-long leading context window).
    #[test]
    fn file_diff_tracks_file_relative_line_numbers() {
        let numbers = |before: &str, after: &str| -> Vec<(String, Option<u32>, Option<u32>)> {
            let file = file_diff(Some("f"), Some("f"), before, after);
            file.lines[1..]
                .iter()
                .map(|line| {
                    (
                        file.text[line.content.clone()].to_string(),
                        line.old_line.map(NonZeroU32::get),
                        line.new_line.map(NonZeroU32::get),
                    )
                })
                .collect()
        };

        // When imara's header starts before the first displayed body line, git shows `@@ -3,5 +3,5 @@` with
        // context three,four,same; imara emits wider context, but the line numbers remain file positions.
        let n = numbers(
            "one\ntwo\nthree\nfour\nsame\nold\nseven\n",
            "one\ntwo\nthree\nfour\nsame\nnew\nseven\n",
        );
        assert_eq!(
            n.iter().map(|(text, ..)| text.clone()).collect::<Vec<_>>(),
            [
                " one", " two", " three", " four", " same", "-old", "+new", " seven"
            ]
        );
        let pair = |i: usize| (n[i].1, n[i].2);
        assert_eq!(pair(0), (Some(1), Some(1)), "{n:?}");
        assert_eq!(pair(4), (Some(5), Some(5)), "{n:?}");
        assert_eq!(pair(5), (Some(6), None), "{n:?}");
        assert_eq!(pair(6), (None, Some(6)), "{n:?}");
        assert_eq!(pair(7), (Some(7), Some(7)), "{n:?}");

        // Two hunks far enough apart to split into two groups; the second group's numbers follow the first
        // group's net change (here zero, one removal + one addition).
        let before = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20\n";
        let after = "1\n2\nX\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\nY\nZ\n19\n20\n";
        let n = numbers(before, after);
        let get = |text: &str| {
            let found = n.iter().find(|(t, ..)| t == text).unwrap();
            (found.1, found.2)
        };
        assert_eq!(get("-3"), (Some(3), None));
        assert_eq!(get("+X"), (None, Some(3)));
        // Second group: context 15,16,17 keep their numbers (net change so far is 0); after the insertion of
        // Z, the trailing context shifts by +1 on the new side.
        assert_eq!(get(" 15"), (Some(15), Some(15)));
        assert_eq!(get("-18"), (Some(18), None));
        assert_eq!(get("+Z"), (None, Some(19)));
        assert_eq!(get(" 19"), (Some(19), Some(20)));
        assert_eq!(get(" 20"), (Some(20), Some(21)));

        // A first group that adds a line (net +1), then a split second group: the second group's leading
        // context starts with the cumulative offset, so l14 is old 14 / new 15.
        let n = numbers(
            "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12\nl13\nl14\nl15\nl16\nl17\nl18\nl19\nl20\n",
            "l1\nl2\n3a\n3b\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12\nl13\nl14\nl15\nl16\nY17\nl18\nl19\nl20\n",
        );
        let get = |text: &str| {
            let found = n.iter().find(|(t, ..)| t == text).unwrap();
            (found.1, found.2)
        };
        assert_eq!(get(" l4"), (Some(4), Some(5)), "{n:?}");
        assert_eq!(get(" l14"), (Some(14), Some(15)), "{n:?}");
        assert_eq!(get("-l17"), (Some(17), None), "{n:?}");
        assert_eq!(get("+Y17"), (None, Some(18)), "{n:?}");
    }

    #[test]
    fn file_diff_does_not_assign_an_earlier_duplicate_line_number() {
        let mut before = String::from("line 1\n\n");
        for line in 3..=297 {
            writeln!(before, "line {line}").unwrap();
        }
        before.push('\n');
        before.push_str("#[cfg(test)]\nmod tests {\nuse super;\n");
        let after = before.replacen("use super;\n", "use unicode_width;\nuse super;\n", 1);

        let file = file_diff(Some("f"), Some("f"), &before, &after);
        let first_body_line = &file.lines[1];

        assert_eq!(&file.text[first_body_line.content.clone()], " ");
        assert_eq!(
            (
                first_body_line.old_line.map(NonZeroU32::get),
                first_body_line.new_line.map(NonZeroU32::get)
            ),
            (Some(298), Some(298))
        );
    }

    #[cfg(unix)]
    fn set_test_xattr(path: &Path, name: &std::ffi::CStr) -> bool {
        match rustix::fs::setxattr(path, name, b"metadata", rustix::fs::XattrFlags::empty()) {
            Ok(()) => true,
            Err(error)
                if error == rustix::io::Errno::NOTSUP
                    || error == rustix::io::Errno::OPNOTSUPP
                    || error == rustix::io::Errno::PERM
                    || error == rustix::io::Errno::ACCESS =>
            {
                false
            }
            Err(error) => panic!("could not set test xattr: {error}"),
        }
    }

    async fn write_file(path: &Path, content: &str) -> FileMutationOutput {
        write_file_with_config(path, content, &BackupConfig::default())
            .await
            .unwrap()
    }

    async fn write_file_with_config(
        path: &Path,
        content: &str,
        config: &BackupConfig,
    ) -> anyhow::Result<FileMutationOutput> {
        execute(
            WriteFileParams {
                path: path.to_string_lossy().into_owned(),
                content: content.into(),
            },
            config,
        )
        .await
    }

    async fn execute(
        params: WriteFileParams,
        backup_config: &BackupConfig,
    ) -> Result<FileMutationOutput> {
        let mut prepared = prepare(params).await?;
        let (output, _) = prepared.write_output();
        prepared.execute(backup_config).await?;
        Ok(output)
    }

    fn pathname(path: &Path) -> String {
        path.strip_prefix(current_working_dir())
            .unwrap()
            .display()
            .to_string()
    }

    #[tokio::test]
    async fn reports_created_updated_and_unchanged_files() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");

        assert_eq!(
            write_file(&path, "first").await,
            FileMutationOutput {
                summary: "Created 5 bytes.".into(),
                diff: format!(
                    "--- /dev/null\n+++ {}\n@@ -0,0 +1,1 @@\n+first\n",
                    pathname(&path)
                )
                .into(),
            }
        );
        assert_eq!(
            write_file(&path, "replacement").await,
            FileMutationOutput {
                summary: "Updated 11 bytes.".into(),
                diff: format!(
                    "--- {0}\n+++ {0}\n@@ -1,1 +1,1 @@\n-first\n+replacement\n",
                    pathname(&path)
                )
                .into(),
            }
        );
        assert_eq!(
            write_file(&path, "replacement").await,
            FileMutationOutput {
                summary: "Unchanged (11 bytes).".into(),
                diff: String::new().into(),
            }
        );

        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "replacement"
        );
    }

    #[tokio::test]
    async fn creates_an_empty_file() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("empty.txt");

        let output = write_file(&path, "").await;

        assert_eq!(output.summary, "Created 0 bytes.");
        assert!(tokio::fs::metadata(&path).await.unwrap().is_file());
        assert!(tokio::fs::read(&path).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn diff_shows_a_changed_final_newline() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "hello").await.unwrap();

        assert_eq!(
            write_file(&path, "hello\n").await,
            FileMutationOutput {
                summary: "Updated 6 bytes.".into(),
                diff: format!(
                    "--- {0}\n+++ {0}\n@@ -1,1 +1,1 @@\n-hello\n+hello\n",
                    pathname(&path)
                )
                .into(),
            }
        );
    }

    #[tokio::test]
    async fn diff_headers_use_line_zero_for_empty_ranges() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "").await.unwrap();

        assert_eq!(
            write_file(&path, "hello\n").await,
            FileMutationOutput {
                summary: "Updated 6 bytes.".into(),
                diff: format!(
                    "--- {0}\n+++ {0}\n@@ -0,0 +1,1 @@\n+hello\n",
                    pathname(&path)
                )
                .into(),
            }
        );
        assert_eq!(
            write_file(&path, "").await,
            FileMutationOutput {
                summary: "Updated 0 bytes.".into(),
                diff: format!(
                    "--- {0}\n+++ {0}\n@@ -1,1 +0,0 @@\n-hello\n",
                    pathname(&path)
                )
                .into(),
            }
        );
    }

    #[tokio::test]
    async fn decodes_non_utf8_files_and_preserves_text_diff() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, [0xff]).await.unwrap();

        assert_eq!(
            write_file(&path, "replacement").await,
            FileMutationOutput {
                summary: "Updated 11 bytes.".into(),
                diff: format!(
                    "--- {0}\n+++ {0}\n@@ -1,1 +1,1 @@\n-ÿ\n+replacement\n",
                    pathname(&path)
                )
                .into(),
            }
        );
        assert_eq!(
            tokio::fs::read_to_string(path).await.unwrap(),
            "replacement"
        );
    }

    #[tokio::test]
    async fn preserves_utf16_encoding_and_bom() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");
        let mut original = vec![0xff, 0xfe];
        original.extend("before\n".encode_utf16().flat_map(u16::to_le_bytes));
        tokio::fs::write(&path, original).await.unwrap();

        write_file(&path, "after\n").await;

        let bytes = tokio::fs::read(&path).await.unwrap();
        assert!(bytes.starts_with(&[0xff, 0xfe]));
        let (text, encoding, has_bom) = file::from_reader(&mut Cursor::new(bytes), None).unwrap();
        assert_eq!(text.to_string(), "after\n");
        assert_eq!(encoding, encoding_rs::UTF_16LE);
        assert!(has_bom);
    }

    #[tokio::test]
    async fn rejects_binary_files() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.bin");
        tokio::fs::write(&path, [0, 1, 2]).await.unwrap();

        let result = write_file_with_config(&path, "replacement", &BackupConfig::default()).await;
        assert!(result.unwrap_err().to_string().contains("binary"));
        assert_eq!(tokio::fs::read(path).await.unwrap(), [0, 1, 2]);
    }

    #[tokio::test]
    async fn unchanged_content_does_not_write_the_file() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "unchanged").await.unwrap();
        let old_time = FileTime::from_unix_time(1_000_000, 0);
        filetime::set_file_mtime(&path, old_time).unwrap();

        assert_eq!(
            write_file(&path, "unchanged").await,
            FileMutationOutput {
                summary: "Unchanged (9 bytes).".into(),
                diff: String::new().into(),
            }
        );

        let metadata = tokio::fs::metadata(path).await.unwrap();
        assert_eq!(FileTime::from_last_modification_time(&metadata), old_time);
    }

    #[tokio::test]
    async fn backup_kind_none_does_not_require_a_backup_directory() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::None,
            directories: Vec::new(),
            extension: String::new(),
        };

        write_file_with_config(&path, "new", &config).await.unwrap();

        assert_eq!(tokio::fs::read_to_string(path).await.unwrap(), "new");
    }

    #[tokio::test]
    async fn backup_creation_failure_aborts_before_writing() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Copy,
            directories: Vec::new(),
            extension: "bck".into(),
        };

        let result = write_file_with_config(&path, "new", &config).await;

        assert_eq!(
            result.unwrap_err().to_string(),
            "could not create backup: no backup directories are configured"
        );
        assert_eq!(tokio::fs::read_to_string(path).await.unwrap(), "old");
    }

    #[tokio::test]
    async fn successful_writes_remove_copy_and_automatic_backups() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        for (number, kind) in [BackupKind::Copy, BackupKind::Auto].into_iter().enumerate() {
            let backup_directory = directory.path().join(format!("backups-{number}"));
            tokio::fs::create_dir(&backup_directory).await.unwrap();
            let path = directory.path().join(format!("output-{number}.txt"));
            tokio::fs::write(&path, "old").await.unwrap();
            let config = BackupConfig {
                kind,
                directories: vec![backup_directory.clone()],
                extension: "bck".into(),
            };

            write_file_with_config(&path, "new", &config).await.unwrap();

            assert!(
                tokio::fs::read_dir(backup_directory)
                    .await
                    .unwrap()
                    .next_entry()
                    .await
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn automatic_write_preserves_a_symlink() {
        use std::os::unix::fs::{MetadataExt, symlink};

        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let target = directory.path().join("target.txt");
        let link = directory.path().join("link.txt");
        tokio::fs::write(&target, "old").await.unwrap();
        symlink(&target, &link).unwrap();
        let inode = tokio::fs::metadata(&target).await.unwrap().ino();

        write_file(&link, "new").await;

        assert!(
            tokio::fs::symlink_metadata(&link)
                .await
                .unwrap()
                .is_symlink()
        );
        assert_eq!(tokio::fs::metadata(&target).await.unwrap().ino(), inode);
        assert_eq!(tokio::fs::read_to_string(target).await.unwrap(), "new");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn automatic_write_preserves_hardlinks() {
        use std::os::unix::fs::MetadataExt;

        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("output.txt");
        let link = directory.path().join("link.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        std::fs::hard_link(&path, &link).unwrap();
        let inode = tokio::fs::metadata(&path).await.unwrap().ino();

        write_file(&path, "new").await;

        assert_eq!(tokio::fs::metadata(&path).await.unwrap().ino(), inode);
        assert_eq!(tokio::fs::metadata(&link).await.unwrap().ino(), inode);
        assert_eq!(tokio::fs::read_to_string(link).await.unwrap(), "new");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn automatic_rename_preserves_unix_metadata() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let backup_directory = directory.path().join("backups");
        tokio::fs::create_dir(&backup_directory).await.unwrap();
        let path = directory.path().join("output.txt");
        tokio::fs::write(&path, "old").await.unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        let xattr_name = c"user.kyun-write-test";
        let xattr_supported = set_test_xattr(&path, xattr_name);
        let before = tokio::fs::metadata(&path).await.unwrap();
        let config = BackupConfig {
            kind: BackupKind::Auto,
            directories: vec![backup_directory],
            extension: "bck".into(),
        };

        write_file_with_config(&path, "new", &config).await.unwrap();

        let after = tokio::fs::metadata(&path).await.unwrap();
        assert_ne!(after.ino(), before.ino());
        assert_eq!(after.mode(), before.mode());
        assert_eq!(after.uid(), before.uid());
        assert_eq!(after.gid(), before.gid());
        if xattr_supported {
            let mut value = [0; 8];
            let length = rustix::fs::getxattr(&path, xattr_name, &mut value).unwrap();
            assert_eq!(&value[..length], b"metadata");
        }
    }

    #[tokio::test]
    async fn rejects_paths_outside_the_current_working_directory() {
        let directory = tempfile::tempdir().unwrap();
        let result = execute(
            WriteFileParams {
                path: directory
                    .path()
                    .join("output.txt")
                    .to_string_lossy()
                    .into_owned(),
                content: "content".into(),
            },
            &BackupConfig::default(),
        )
        .await;

        assert!(
            result
                .unwrap_err()
                .to_string()
                .starts_with("path is outside the current working directory")
        );
    }
}
