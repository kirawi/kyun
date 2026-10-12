use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, bail};
use kyun_agent::ToolDefinition;

use crate::backup::{Backup, BackupConfig, BackupKind, BackupMethod};
use crate::faccess;
use crate::path::current_working_dir;
use crate::tools::FileMutationOutput;
use crate::tools::content_change::ChangeSet;
use crate::tools::file_mutation::{FileSnapshot, ResolvedPath, resolve_path};
use crate::tools::write::{self, WriteFileParams};

pub const NAME: &str = "apply_patch";

const BEGIN_PATCH_MARKER: &str = "*** Begin Patch";

/// The Lark grammar constraining the model's patch output on APIs that support grammar sampling.
const APPLY_PATCH_LARK_GRAMMAR: &str = r#"start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: change_context? change_line+ (change_context change_line+)* eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF"#;

pub fn definition() -> ToolDefinition {
    ToolDefinition::Freeform {
        name: NAME.into(),
        description: "Apply a patch to one or more files under the current working directory. Use this instead of `write` \
            for targeted edits: it can add, update, move, and delete several files in one call and does not need the full \
            file contents. Every `-`/context line is matched against the file, so keep hunks small and include enough \
            surrounding context to locate each change. This is a FREEFORM tool: the input is the raw patch text, not JSON."
            .into(),
        grammar: Some(APPLY_PATCH_LARK_GRAMMAR.into()),
    }
}

/// Returns a compact single-line description of the files touched by `patch` for display in the UI.
pub fn summarize_patch(patch: &str) -> String {
    let paths = patch
        .lines()
        .map(str::trim)
        .filter_map(|line| {
            [
                "*** Add File: ",
                "*** Delete File: ",
                "*** Update File: ",
                "*** Move to: ",
            ]
            .iter()
            .find_map(|marker| line.strip_prefix(marker))
        })
        .collect::<Vec<_>>();
    let summary = paths.join(", ");

    format!(
        "{{ \"patch\": {} }}",
        serde_json::to_string(&summary).unwrap()
    )
}

/// Returns the patch text from a tool call's input, accepting either freeform patch text or a JSON object with a `patch`
/// field (models habituated to JSON tool arguments sometimes wrap freeform input). Returns `None` when the input is not
/// patch text.
pub fn patch_text(input: &str) -> Option<String> {
    if input.trim_start().starts_with(BEGIN_PATCH_MARKER) {
        return Some(input.to_owned());
    }

    serde_json::from_str::<serde_json::Value>(input)
        .ok()
        .and_then(|value| value.get("patch")?.as_str().map(str::to_owned))
}

/// Returns a compact single-line description of a tool call's input for display in the UI.
pub fn summarize_input(input: &str) -> String {
    if let Some(patch) = patch_text(input) {
        return summarize_patch(&patch);
    }

    input.to_owned()
}

#[derive(Debug, PartialEq, Eq)]
enum Hunk {
    Add {
        path: String,
        content: String,
    },
    Delete {
        path: String,
    },
    Update {
        path: String,
        move_to: Option<String>,
        chunks: Vec<Chunk>,
    },
}

#[derive(Debug, PartialEq, Eq)]
struct Chunk {
    context: Option<String>,
    old_lines: Vec<String>,
    new_lines: Vec<String>,
    end_of_file: bool,
}

fn is_hunk_marker(line: &str) -> bool {
    line.starts_with("*** Add File: ")
        || line.starts_with("*** Delete File: ")
        || line.starts_with("*** Update File: ")
        || line.starts_with("*** End Patch")
}

fn parse_patch(patch: &str) -> Result<Vec<Hunk>, String> {
    let patch = patch.trim();
    let lines: Vec<&str> = patch.lines().collect();

    match (lines.first(), lines.last()) {
        (Some(first), Some(last))
            if first.trim() == "*** Begin Patch" && last.trim() == "*** End Patch" => {}
        (Some(first), _) if first.trim() != "*** Begin Patch" => {
            return Err("The first line of the patch must be '*** Begin Patch'".into());
        }
        _ => return Err("The last line of the patch must be '*** End Patch'".into()),
    }
    let body = &lines[1..lines.len() - 1];

    let mut hunks = Vec::new();
    let mut i = 0;
    while i < body.len() {
        let line = body[i];

        if line.is_empty() {
            i += 1;
            continue;
        }

        if let Some(path) = line.strip_prefix("*** Add File: ") {
            let path = path.trim().to_string();
            if path.is_empty() {
                return Err("Add file hunk is missing a path".into());
            }
            i += 1;

            let mut content = String::new();
            while i < body.len() && body[i].starts_with('+') {
                content.push_str(&body[i][1..]);
                content.push('\n');
                i += 1;
            }

            hunks.push(Hunk::Add { path, content });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            let path = path.trim().to_string();
            if path.is_empty() {
                return Err("Delete file hunk is missing a path".into());
            }
            i += 1;

            hunks.push(Hunk::Delete { path });
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            let path = path.trim().to_string();
            if path.is_empty() {
                return Err("Update file hunk is missing a path".into());
            }
            i += 1;

            let move_to = if i < body.len() && body[i].starts_with("*** Move to: ") {
                let destination = body[i]["*** Move to: ".len()..].trim().to_string();
                if destination.is_empty() {
                    return Err("Update file hunk has an empty move destination".into());
                }
                i += 1;

                Some(destination)
            } else {
                None
            };

            let mut chunks = Vec::new();
            while i < body.len() && !is_hunk_marker(body[i]) && !body[i].is_empty() {
                let context = if let Some(rest) = body[i].strip_prefix("@@") {
                    let context = if rest.is_empty() {
                        None
                    } else if let Some(context) = rest.strip_prefix(' ') {
                        Some(context.to_string())
                    } else {
                        return Err(format!("Invalid change context: {:?}", body[i]));
                    };
                    i += 1;

                    context
                } else {
                    None
                };

                let mut chunk = Chunk {
                    context,
                    old_lines: Vec::new(),
                    new_lines: Vec::new(),
                    end_of_file: false,
                };
                while i < body.len() {
                    let line = body[i];
                    if line.is_empty() || is_hunk_marker(line) || line.starts_with("@@") {
                        break;
                    }
                    if line == "*** End of File" {
                        chunk.end_of_file = true;
                        i += 1;
                        break;
                    }

                    if let Some(rest) = line.strip_prefix('+') {
                        chunk.new_lines.push(rest.to_string());
                    } else if let Some(rest) = line.strip_prefix('-') {
                        chunk.old_lines.push(rest.to_string());
                    } else if let Some(rest) = line.strip_prefix(' ') {
                        chunk.old_lines.push(rest.to_string());
                        chunk.new_lines.push(rest.to_string());
                    } else {
                        return Err(format!("Unexpected line inside an update hunk: {line:?}"));
                    }
                    i += 1;
                }

                chunks.push(chunk);
            }

            if chunks.is_empty() {
                return Err(format!("Update file hunk for path '{path}' is empty"));
            }

            hunks.push(Hunk::Update {
                path,
                move_to,
                chunks,
            });
        } else {
            return Err(format!("Unexpected line outside of a hunk: {line:?}"));
        }
    }

    Ok(hunks)
}

/// Maps typographic punctuation to its ASCII counterpart, mirroring `git apply`. ASCII and all
/// characters outside this table are left unchanged.
fn normalize_punctuation(character: char) -> char {
    match character {
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' => '-',
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        '\u{00A0}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}' | '\u{2006}'
        | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200A}' | '\u{202F}' | '\u{205F}'
        | '\u{3000}' => ' ',
        other => other,
    }
}

/// Locates `pattern` in `lines` starting at or after `start`, returning its starting index.
fn seek_sequence<T: AsRef<str>>(
    lines: &[T],
    pattern: &[String],
    start: usize,
    eof: bool,
) -> Option<usize> {
    if pattern.is_empty() {
        return Some(start.min(lines.len()));
    }
    if pattern.len() > lines.len() {
        return None;
    }

    let search_start = if eof {
        lines.len() - pattern.len()
    } else {
        start
    };
    let end = lines.len() - pattern.len();

    for i in search_start..=end {
        if pattern
            .iter()
            .enumerate()
            .all(|(offset, line)| lines[i + offset].as_ref() == line)
        {
            return Some(i);
        }
    }

    for i in search_start..=end {
        if pattern
            .iter()
            .enumerate()
            .all(|(offset, line)| lines[i + offset].as_ref().trim_end() == line.trim_end())
        {
            return Some(i);
        }
    }

    for i in search_start..=end {
        if pattern
            .iter()
            .enumerate()
            .all(|(offset, line)| lines[i + offset].as_ref().trim() == line.trim())
        {
            return Some(i);
        }
    }

    // Tolerate ASCII patches applied to files containing typographic punctuation, mirroring
    // `git apply`. When neither the pattern nor any candidate line contains a mappable character,
    // normalized comparison is identical to the trim comparison above.
    // Windows are anchored at `search_start` and cover `lines.len() - pattern.len() + 1`
    // positions, so `lines[search_start..]` is exactly the union of candidate lines.
    fn is_unmappable(line: &str) -> bool {
        line.chars().all(|c| normalize_punctuation(c) == c)
    }
    if pattern.iter().all(|line| is_unmappable(line))
        && lines[search_start..]
            .iter()
            .all(|line| is_unmappable(line.as_ref()))
    {
        return None;
    }

    (search_start..=end).find(|&i| {
        pattern.iter().enumerate().all(|(offset, line)| {
            let normalized: String = lines[i + offset]
                .as_ref()
                .trim()
                .chars()
                .map(normalize_punctuation)
                .collect();
            normalized
                == line
                    .trim()
                    .chars()
                    .map(normalize_punctuation)
                    .collect::<String>()
        })
    })
}

struct SourceLine {
    content: std::ops::Range<usize>,
    full: std::ops::Range<usize>,
}

fn source_lines(original: &str) -> Vec<SourceLine> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (index, &byte) in original.as_bytes().iter().enumerate() {
        if byte != b'\n' {
            continue;
        }

        let content_end = if index > start && original.as_bytes()[index - 1] == b'\r' {
            index - 1
        } else {
            index
        };
        lines.push(SourceLine {
            content: start..content_end,
            full: start..index + 1,
        });
        start = index + 1;
    }
    if start < original.len() {
        lines.push(SourceLine {
            content: start..original.len(),
            full: start..original.len(),
        });
    }
    lines
}

fn replacement_text(lines: &[String], ending: &str, separator: bool) -> String {
    let mut replacement = String::new();
    if separator && !lines.is_empty() {
        replacement.push_str(ending);
    }
    replacement.push_str(&lines.join(ending));
    if !lines.is_empty() && !replacement.ends_with(ending) {
        replacement.push_str(ending);
    }
    replacement
}

fn output_ends_with_newline(
    original: &[u8],
    replacements: &[(std::ops::Range<usize>, String)],
) -> bool {
    let mut suffix_start = original.len();
    for (range, replacement) in replacements.iter().rev() {
        if range.end < suffix_start {
            break;
        }
        if let Some(&byte) = replacement.as_bytes().last() {
            return byte == b'\n';
        }
        suffix_start = range.start;
    }
    suffix_start > 0 && original[suffix_start - 1] == b'\n'
}

/// Resolves patch chunks to source ranges without constructing the resulting file.
fn resolve_chunks(original: &str, chunks: &[Chunk], path: &str) -> Result<ChangeSet, String> {
    let source_lines = source_lines(original);
    let lines = source_lines
        .iter()
        .map(|line| &original[line.content.clone()])
        .collect::<Vec<_>>();
    let ending = if original.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let mut replacements: Vec<(usize, usize, Vec<String>)> = Vec::new();
    let mut line_index = 0;
    for chunk in chunks {
        if let Some(context) = &chunk.context {
            match seek_sequence(&lines, std::slice::from_ref(context), line_index, false) {
                Some(index) => line_index = index + 1,
                None => return Err(format!("Failed to find context '{context}' in {path}")),
            }
        }

        if chunk.old_lines.is_empty() {
            replacements.push((lines.len(), 0, chunk.new_lines.clone()));
            continue;
        }

        let mut pattern = chunk.old_lines.clone();
        let mut new_lines = chunk.new_lines.clone();
        let mut found = seek_sequence(&lines, &pattern, line_index, chunk.end_of_file);
        if found.is_none() && pattern.last().is_some_and(String::is_empty) {
            // A trailing empty line represents the file's final newline and is not present in `lines`.
            pattern.pop();
            if new_lines.last().is_some_and(String::is_empty) {
                new_lines.pop();
            }
            found = seek_sequence(&lines, &pattern, line_index, chunk.end_of_file);
        }

        match found {
            Some(start) => {
                replacements.push((start, pattern.len(), new_lines));
                line_index = start + pattern.len();
            }
            None => {
                return Err(format!(
                    "Failed to find expected lines in {path}:\n{}",
                    chunk.old_lines.join("\n")
                ));
            }
        }
    }

    replacements.sort_by_key(|(start, _, _)| *start);

    let mut byte_replacements = Vec::with_capacity(replacements.len() + 1);
    for (start, old_len, new_lines) in replacements {
        let start_byte = if start == source_lines.len() {
            original.len()
        } else {
            source_lines[start].full.start
        };
        let end_byte = if old_len == 0 {
            start_byte
        } else {
            source_lines[start + old_len - 1].full.end
        };
        byte_replacements.push((
            start_byte..end_byte,
            replacement_text(
                &new_lines,
                ending,
                old_len == 0 && start_byte == original.len() && !original.ends_with('\n'),
            ),
        ));
    }

    if ending == "\r\n" {
        for line in &source_lines {
            let line_ending = &original[line.content.end..line.full.end];
            if line_ending == "\n"
                && !byte_replacements.iter().any(|(range, _)| {
                    range.start <= line.content.start && line.full.end <= range.end
                })
            {
                byte_replacements.push((line.content.end..line.full.end, ending.into()));
            }
        }
        byte_replacements.sort_by_key(|(range, _)| range.start);
    }

    if !output_ends_with_newline(original.as_bytes(), &byte_replacements) {
        byte_replacements.push((original.len()..original.len(), ending.into()));
    }

    ChangeSet::from_replacements(original, byte_replacements)
}

/// Computes the new contents of a file by applying `chunks` to `original`.
#[cfg(test)]
#[cfg(test)]
fn apply_chunks(original: &str, chunks: &[Chunk], path: &str) -> Result<String, String> {
    let changes = resolve_chunks(original, chunks, path)?;
    let mut text = ropey::Rope::from(original);
    assert!(changes.apply(&mut text));
    Ok(text.to_string())
}

async fn delete_file(resolved: &ResolvedPath, backup_config: &BackupConfig) -> anyhow::Result<()> {
    if resolved.is_symlink {
        tokio::fs::remove_file(&resolved.requested).await?;
        return Ok(());
    }

    let path = &resolved.target;
    if faccess::readonly(path) {
        bail!("{} is read only", path.display());
    }

    let metadata = tokio::fs::metadata(path).await?;
    let requested = path
        .strip_prefix(current_working_dir())
        .map_or_else(|_| path.to_owned(), Path::to_owned);

    let backup = if backup_config.kind != BackupKind::None {
        Some(
            Backup::create(
                path,
                &requested,
                false,
                &current_working_dir(),
                backup_config,
            )
            .await
            .map_err(|error| anyhow!("could not create backup: {error}"))?,
        )
    } else {
        None
    };

    // A rename backup has already moved the file out of the way, so only a copy backup leaves the original to remove.
    let removal = match &backup {
        Some(backup) if backup.method == BackupMethod::Rename => Ok(()),
        _ => tokio::fs::remove_file(path).await,
    };

    match removal {
        Ok(()) => {
            if let Some(backup) = backup {
                let _ = tokio::fs::remove_file(backup.path).await;
            }
            Ok(())
        }
        Err(error) => {
            if let Some(backup) = backup {
                let _ = backup.restore(path, &metadata).await;
            }
            Err(error.into())
        }
    }
}

enum Change {
    Add {
        write: write::PreparedWrite,
    },
    Update {
        write: write::PreparedWrite,
    },
    Move {
        source: ResolvedPath,
        source_snapshot: FileSnapshot,
        destination: write::PreparedWrite,
    },
    Delete {
        resolved: ResolvedPath,
        observed: FileSnapshot,
    },
}

impl Change {
    fn description(&self) -> String {
        match self {
            Self::Add { write } => format!("add {}", write.resolved.pathname()),
            Self::Update { write } => format!("update {}", write.resolved.pathname()),
            Self::Move {
                source,
                destination,
                ..
            } => format!(
                "move {} to {}",
                source.pathname(),
                destination.resolved.pathname()
            ),
            Self::Delete { resolved, .. } => format!("delete {}", resolved.pathname()),
        }
    }
}

pub struct PreparedPatch {
    changes: Vec<Change>,
}

/// Appends one file's diff to the multi-file diff being built. Matches the text layout of separate `---`/`+++`
/// sections joined by a blank line; the appended file's line ranges are offset by the accumulated text length
/// (plus the separator) so they keep indexing into the combined text.
fn append_file_diff(
    text: &mut String,
    lines: &mut Vec<write::DiffLine>,
    file_diff: write::FileDiff,
) {
    if file_diff.text.is_empty() {
        return;
    }

    let mut offset = text.len();
    if !text.is_empty() {
        text.push('\n');
        offset += 1;
    }
    text.push_str(&file_diff.text);
    for mut line in file_diff.lines {
        line.content.start += offset;
        line.content.end += offset;
        lines.push(line);
    }
}

fn append_diff(
    text: &mut String,
    lines: &mut Vec<write::DiffLine>,
    before_path: Option<&str>,
    after_path: Option<&str>,
    before: &str,
    after: &str,
) {
    append_file_diff(
        text,
        lines,
        write::file_diff(before_path, after_path, before, after),
    );
}

pub async fn prepare(input: &str) -> anyhow::Result<PreparedPatch> {
    let patch = patch_text(input).unwrap_or_else(|| input.to_owned());
    let hunks = parse_patch(&patch).map_err(anyhow::Error::msg)?;
    if hunks.is_empty() {
        bail!("No files were modified.");
    }

    // Verify every hunk and compute its result before writing anything so a malformed patch changes nothing.
    let mut changes = Vec::new();
    let mut targets = HashSet::new();
    for hunk in hunks {
        match hunk {
            Hunk::Add { path, content } => {
                let resolved = resolve_path(&path).await?;
                if !targets.insert(resolved.target.clone()) {
                    bail!("multiple operations target {path}");
                }

                let observed = FileSnapshot::capture(&resolved.target).await;
                let write = write::PreparedWrite::from_resolved(
                    WriteFileParams {
                        path: path.clone(),
                        content,
                    },
                    resolved,
                    observed,
                )?;
                if !write.changed {
                    continue;
                }
                changes.push(Change::Add { write });
            }
            Hunk::Delete { path } => {
                let resolved = resolve_path(&path).await?;
                if !targets.insert(resolved.target.clone()) {
                    bail!("multiple operations target {path}");
                }

                let observed = FileSnapshot::capture(&resolved.target).await;
                match &observed {
                    FileSnapshot::Present { .. } => {}
                    FileSnapshot::Missing => {
                        bail!("failed to read {path}: file not found");
                    }
                    FileSnapshot::Binary => {
                        bail!("failed to read {path}: file appears to be binary");
                    }
                    FileSnapshot::Unreadable { message } => {
                        bail!("failed to read {path}: {message}");
                    }
                };

                changes.push(Change::Delete { resolved, observed });
            }
            Hunk::Update {
                path,
                move_to,
                chunks,
            } => {
                let resolved = resolve_path(&path).await?;
                if !targets.insert(resolved.target.clone()) {
                    bail!("multiple operations target {path}");
                }

                let observed = FileSnapshot::capture(&resolved.target).await;
                let original = match &observed {
                    FileSnapshot::Present { text, .. } => text.to_string(),
                    FileSnapshot::Missing => {
                        bail!("failed to read {path}: file not found");
                    }
                    FileSnapshot::Binary => {
                        bail!("failed to read {path}: file appears to be binary");
                    }
                    FileSnapshot::Unreadable { message } => {
                        bail!("failed to read {path}: {message}");
                    }
                };

                let content_changes =
                    resolve_chunks(&original, &chunks, &path).map_err(anyhow::Error::msg)?;

                match move_to {
                    Some(destination) => {
                        let destination_resolved = resolve_path(&destination).await?;
                        if destination_resolved.target == resolved.target {
                            let write = write::PreparedWrite::from_change_set(
                                resolved,
                                observed,
                                content_changes,
                            )?;
                            if !write.changed {
                                continue;
                            }
                            changes.push(Change::Update { write });
                        } else {
                            if !targets.insert(destination_resolved.target.clone()) {
                                bail!("multiple operations target {destination}");
                            }

                            let destination_snapshot =
                                FileSnapshot::capture(&destination_resolved.target).await;
                            let mut content = match &observed {
                                FileSnapshot::Present { text, .. } => text.clone(),
                                _ => unreachable!(),
                            };
                            assert!(content_changes.apply(&mut content));
                            let (encoding, has_bom) = match &observed {
                                FileSnapshot::Present {
                                    encoding, has_bom, ..
                                } => (*encoding, *has_bom),
                                _ => unreachable!(),
                            };
                            let destination_write = write::PreparedWrite::from_text(
                                destination_resolved,
                                destination_snapshot,
                                content,
                                encoding,
                                has_bom,
                            )?;
                            changes.push(Change::Move {
                                source: resolved,
                                source_snapshot: observed,
                                destination: destination_write,
                            });
                        }
                    }
                    None => {
                        let write = write::PreparedWrite::from_change_set(
                            resolved,
                            observed,
                            content_changes,
                        )?;
                        if !write.changed {
                            continue;
                        }
                        changes.push(Change::Update { write });
                    }
                }
            }
        }
    }

    if changes.is_empty() {
        bail!("No files were modified.");
    }

    Ok(PreparedPatch { changes })
}

impl PreparedPatch {
    /// The files this patch intends to touch, including move sources.
    pub fn target_paths(&self) -> Vec<std::path::PathBuf> {
        self.changes
            .iter()
            .flat_map(|change| match change {
                Change::Add { write } | Change::Update { write } => {
                    vec![write.resolved.target.clone()]
                }
                Change::Move {
                    source,
                    destination,
                    ..
                } => vec![source.target.clone(), destination.resolved.target.clone()],
                Change::Delete { resolved, .. } => vec![resolved.target.clone()],
            })
            .collect()
    }

    pub fn output(&mut self) -> (FileMutationOutput, write::FileDiff) {
        let mut summary = String::from("Success. Updated the following files:\n");
        let mut diff_text = String::new();
        let mut diff_lines = Vec::new();
        for change in &mut self.changes {
            match change {
                Change::Add { write } => {
                    let pathname = write.resolved.pathname();
                    summary.push_str(&format!("A {pathname}\n"));
                    let before_path = write.resolved.existed.then_some(pathname.as_str());
                    append_file_diff(
                        &mut diff_text,
                        &mut diff_lines,
                        write.diff(before_path, Some(&pathname)),
                    );
                }
                Change::Update { write } => {
                    let pathname = write.resolved.pathname();
                    summary.push_str(&format!("M {pathname}\n"));
                    append_file_diff(
                        &mut diff_text,
                        &mut diff_lines,
                        write.diff(Some(&pathname), Some(&pathname)),
                    );
                }
                Change::Move {
                    source,
                    source_snapshot,
                    destination,
                } => {
                    let source_pathname = source.pathname();
                    let destination_pathname = destination.resolved.pathname();
                    summary.push_str(&format!("M {destination_pathname}\n"));
                    if destination.resolved.existed {
                        let original = match source_snapshot {
                            FileSnapshot::Present { text, .. } => text.to_string(),
                            _ => unreachable!(),
                        };
                        append_diff(
                            &mut diff_text,
                            &mut diff_lines,
                            Some(&source_pathname),
                            None,
                            &original,
                            "",
                        );
                    }
                    if destination.resolved.existed {
                        let before_path = destination_pathname.clone();
                        append_file_diff(
                            &mut diff_text,
                            &mut diff_lines,
                            destination.diff(Some(&before_path), Some(&destination_pathname)),
                        );
                    } else {
                        let original = match source_snapshot {
                            FileSnapshot::Present { text, .. } => text.to_string(),
                            _ => unreachable!(),
                        };
                        let content = destination.text.to_string();
                        append_diff(
                            &mut diff_text,
                            &mut diff_lines,
                            Some(&source_pathname),
                            Some(&destination_pathname),
                            &original,
                            &content,
                        );
                    }
                }
                Change::Delete { resolved, observed } => {
                    let pathname = resolved.pathname();
                    summary.push_str(&format!("D {pathname}\n"));
                    let original = match observed {
                        FileSnapshot::Present { text, .. } => text.to_string(),
                        _ => unreachable!(),
                    };
                    append_diff(
                        &mut diff_text,
                        &mut diff_lines,
                        Some(&pathname),
                        None,
                        &original,
                        "",
                    );
                }
            }
        }

        let diff = write::FileDiff {
            text: diff_text.into(),
            lines: diff_lines.into_boxed_slice(),
        };
        let output = FileMutationOutput {
            summary,
            diff: diff.text.clone(),
        };
        (output, diff)
    }

    pub async fn execute(self, backup_config: &BackupConfig) -> anyhow::Result<()> {
        for change in &self.changes {
            match change {
                Change::Add { write, .. } | Change::Update { write, .. } => {
                    write.validate().await?;
                }
                Change::Move {
                    source,
                    source_snapshot,
                    destination,
                    ..
                } => {
                    source_snapshot
                        .validate(&source.target, &source.pathname())
                        .await
                        .map_err(anyhow::Error::msg)?;
                    destination.validate().await?;
                }
                Change::Delete { resolved, observed } => observed
                    .validate(&resolved.target, &resolved.pathname())
                    .await
                    .map_err(anyhow::Error::msg)?,
            }
        }

        let mut completed = Vec::new();
        for change in self.changes {
            let description = change.description();
            let result = match change {
                Change::Add { write, .. } | Change::Update { write, .. } => {
                    write.commit(backup_config).await
                }
                Change::Move {
                    source,
                    destination,
                    ..
                } => {
                    async {
                        destination.commit(backup_config).await?;
                        delete_file(&source, backup_config).await
                    }
                    .await
                }
                Change::Delete { resolved, .. } => delete_file(&resolved, backup_config).await,
            };
            if let Err(error) = result {
                let completed = if completed.is_empty() {
                    String::new()
                } else {
                    format!(" Completed operations: {}.", completed.join(", "))
                };
                return Err(anyhow!(
                    "{description} failed: {error}.{completed} The current operation may also have partially changed the filesystem."
                ));
            }
            completed.push(description);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch(body: &str) -> String {
        format!("*** Begin Patch\n{body}\n*** End Patch")
    }

    fn pathname(path: &Path) -> String {
        path.strip_prefix(current_working_dir())
            .unwrap()
            .display()
            .to_string()
    }

    async fn execute(input: &str, backup_config: &BackupConfig) -> anyhow::Result<String> {
        let mut prepared = prepare(input).await?;
        let (output, _) = prepared.output();
        prepared.execute(backup_config).await?;
        Ok(serde_json::to_string(&output)?)
    }

    async fn apply(patch: &str, backup: &BackupConfig) -> anyhow::Result<String> {
        execute(patch, backup).await
    }

    #[test]
    fn parses_add_update_delete_and_move_hunks() {
        let source = patch(
            "*** Add File: add.txt\n+one\n+two\n\
             *** Delete File: old.txt\n\
             *** Update File: src.txt\n*** Move to: dst.txt\n@@ fn main()\n-let x = 1;\n+let x = 2;\n",
        );
        assert_eq!(
            parse_patch(&source).unwrap(),
            vec![
                Hunk::Add {
                    path: "add.txt".into(),
                    content: "one\ntwo\n".into(),
                },
                Hunk::Delete {
                    path: "old.txt".into(),
                },
                Hunk::Update {
                    path: "src.txt".into(),
                    move_to: Some("dst.txt".into()),
                    chunks: vec![Chunk {
                        context: Some("fn main()".into()),
                        old_lines: vec!["let x = 1;".into()],
                        new_lines: vec!["let x = 2;".into()],
                        end_of_file: false,
                    }],
                },
            ]
        );
    }

    #[test]
    fn rejects_patches_without_boundaries() {
        assert!(parse_patch("*** Add File: add.txt\n+x").is_err());
        assert!(parse_patch("*** Begin Patch\n*** Add File: add.txt\n+x").is_err());
    }

    #[test]
    fn multiple_chunks_in_one_update_apply_to_separate_regions() {
        let hunks = parse_patch(&patch(
            "*** Update File: multi.txt\n@@\n a\n-b\n+B\n@@\n c\n-d\n+D",
        ))
        .unwrap();
        let Hunk::Update { chunks, .. } = &hunks[0] else {
            panic!("expected an update hunk")
        };

        let content = apply_chunks("a\nb\nc\nd\n", chunks, "multi.txt").unwrap();

        assert_eq!(content, "a\nB\nc\nD\n");
    }

    #[test]
    fn matching_ignores_trailing_whitespace_and_unicode_punctuation() {
        let hunks = parse_patch(&patch("*** Update File: f.txt\n@@\n-hello\n+world")).unwrap();
        let Hunk::Update { chunks, .. } = &hunks[0] else {
            panic!("expected an update hunk")
        };
        assert_eq!(
            apply_chunks("hello   \n", chunks, "f.txt").unwrap(),
            "world\n"
        );

        let hunks = parse_patch(&patch("*** Update File: f.txt\n@@\n-a - b\n+c")).unwrap();
        let Hunk::Update { chunks, .. } = &hunks[0] else {
            panic!("expected an update hunk")
        };
        assert_eq!(
            apply_chunks("a \u{2013} b\n", chunks, "f.txt").unwrap(),
            "c\n"
        );
    }

    #[test]
    fn end_of_file_marker_matches_the_final_line() {
        let hunks = parse_patch(&patch(
            "*** Update File: f.txt\n@@\n-a\n+b\n*** End of File",
        ))
        .unwrap();
        let Hunk::Update { chunks, .. } = &hunks[0] else {
            panic!("expected an update hunk")
        };

        // Without the marker the first `a` would be replaced; the marker must select the final occurrence.
        assert_eq!(
            apply_chunks("a\nx\na\n", chunks, "f.txt").unwrap(),
            "a\nx\nb\n"
        );
    }

    #[test]
    fn updates_preserve_crlf_line_endings() {
        let hunks = parse_patch(&patch("*** Update File: f.txt\n@@\n-old\n+new")).unwrap();
        let Hunk::Update { chunks, .. } = &hunks[0] else {
            panic!("expected an update hunk")
        };

        assert_eq!(
            apply_chunks("keep\r\nold\r\n", chunks, "f.txt").unwrap(),
            "keep\r\nnew\r\n"
        );
    }

    #[test]
    fn insertion_at_eof_separates_an_unterminated_line() {
        let hunks = parse_patch(&patch("*** Update File: f.txt\n@@\n+new")).unwrap();
        let Hunk::Update { chunks, .. } = &hunks[0] else {
            panic!("expected an update hunk")
        };

        assert_eq!(apply_chunks("old", chunks, "f.txt").unwrap(), "old\nnew\n");
        assert_eq!(
            apply_chunks("old\r\nlast", chunks, "f.txt").unwrap(),
            "old\r\nlast\r\nnew\r\n"
        );
    }

    #[test]
    fn mixed_line_endings_follow_the_existing_crlf_normalization() {
        let hunks = parse_patch(&patch("*** Update File: f.txt\n@@\n-first\n+changed")).unwrap();
        let Hunk::Update { chunks, .. } = &hunks[0] else {
            panic!("expected an update hunk")
        };

        assert_eq!(
            apply_chunks("first\r\nsecond\nlast\r\n", chunks, "f.txt").unwrap(),
            "changed\r\nsecond\r\nlast\r\n"
        );
    }

    #[test]
    fn deleting_a_middle_line_restores_the_final_newline() {
        let hunks = parse_patch(&patch("*** Update File: f.txt\n@@\n-b")).unwrap();
        let Hunk::Update { chunks, .. } = &hunks[0] else {
            panic!("expected an update hunk")
        };

        assert_eq!(apply_chunks("a\nb\nc", chunks, "f.txt").unwrap(), "a\nc\n");
    }

    #[test]
    fn summarizes_hunks_for_display() {
        let source = patch("*** Add File: add.txt\n+one\n*** Update File: src.txt\n@@\n-a\n+b");
        assert_eq!(
            summarize_patch(&source),
            r#"{ "patch": "add.txt, src.txt" }"#
        );
    }

    #[test]
    fn patch_text_detects_freeform_and_json_wrapped_patch_input() {
        let raw = patch("*** Add File: a.txt\n+x");

        assert_eq!(patch_text(&raw).as_deref(), Some(raw.as_str()));

        let wrapped = serde_json::json!({ "patch": raw }).to_string();
        assert_eq!(patch_text(&wrapped).as_deref(), Some(raw.as_str()));

        let source = patch("*** Update File: src.txt\n@@\n-a\n+b");
        assert_eq!(summarize_input(&source), r#"{ "patch": "src.txt" }"#);
    }

    #[tokio::test]
    async fn adds_updates_and_deletes_files() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let add = directory.path().join("add.txt");
        let update = directory.path().join("update.txt");
        let delete = directory.path().join("delete.txt");
        tokio::fs::write(&update, "keep\nold\n").await.unwrap();
        tokio::fs::write(&delete, "bye\n").await.unwrap();

        let source = patch(&format!(
            "*** Add File: {}\n+first\n+second\n\
             *** Update File: {}\n@@\n keep\n-old\n+new\n\
             *** Delete File: {}\n",
            add.display(),
            update.display(),
            delete.display(),
        ));
        let output = apply(&source, &BackupConfig::default()).await.unwrap();
        let output: FileMutationOutput = serde_json::from_str(&output).unwrap();

        assert_eq!(
            tokio::fs::read_to_string(&add).await.unwrap(),
            "first\nsecond\n"
        );
        assert_eq!(
            tokio::fs::read_to_string(&update).await.unwrap(),
            "keep\nnew\n"
        );
        assert!(!delete.exists());
        assert!(output.summary.contains(&format!("A {}", pathname(&add))));
        assert!(output.summary.contains(&format!("M {}", pathname(&update))));
        assert!(output.summary.contains(&format!("D {}", pathname(&delete))));
        assert!(
            output
                .diff
                .contains(&format!("--- /dev/null\n+++ {}", pathname(&add)))
        );
        assert!(output.diff.contains("+first\n+second\n"));
        assert!(
            output
                .diff
                .contains(&format!("--- {0}\n+++ {0}", pathname(&update)))
        );
        assert!(output.diff.contains("-old\n+new\n"));
        assert!(
            output
                .diff
                .contains(&format!("--- {}\n+++ /dev/null", pathname(&delete)))
        );
        assert!(output.diff.contains("-bye\n"));
    }

    #[tokio::test]
    async fn adds_an_empty_file() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("empty.txt");
        let source = patch(&format!("*** Add File: {}\n", path.display()));

        apply(&source, &BackupConfig::default()).await.unwrap();

        assert!(tokio::fs::metadata(&path).await.unwrap().is_file());
        assert!(tokio::fs::read(&path).await.unwrap().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn deleting_a_symlink_removes_the_link_without_deleting_its_target() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let target = directory.path().join("target.txt");
        let link = directory.path().join("link.txt");
        tokio::fs::write(&target, "keep\n").await.unwrap();
        symlink(&target, &link).unwrap();
        let source = patch(&format!("*** Delete File: {}\n", link.display()));

        apply(&source, &BackupConfig::default()).await.unwrap();

        assert!(target.exists());
        assert!(tokio::fs::symlink_metadata(&link).await.is_err());
    }

    #[tokio::test]
    async fn moves_a_file_and_creates_missing_parent_directories() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let source = directory.path().join("src.txt");
        let destination = directory.path().join("nested/dst.txt");
        tokio::fs::write(&source, "before\n").await.unwrap();

        let patch = patch(&format!(
            "*** Update File: {}\n*** Move to: {}\n@@\n-before\n+after\n",
            source.display(),
            destination.display(),
        ));
        let output = apply(&patch, &BackupConfig::default()).await.unwrap();
        let output: FileMutationOutput = serde_json::from_str(&output).unwrap();

        assert!(!source.exists());
        assert_eq!(
            tokio::fs::read_to_string(&destination).await.unwrap(),
            "after\n"
        );
        assert!(
            output
                .summary
                .contains(&format!("M {}", pathname(&destination)))
        );
        assert!(output.diff.contains(&format!(
            "--- {}\n+++ {}",
            pathname(&source),
            pathname(&destination)
        )));
        assert!(output.diff.contains("-before\n+after\n"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn moving_to_an_identical_read_only_destination_does_not_rewrite_it() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let source = directory.path().join("source.txt");
        let destination = directory.path().join("destination.txt");
        tokio::fs::write(&source, "same\n").await.unwrap();
        tokio::fs::write(&destination, "same\n").await.unwrap();
        let old_time = filetime::FileTime::from_unix_time(1_000_000, 0);
        filetime::set_file_mtime(&destination, old_time).unwrap();
        tokio::fs::set_permissions(&destination, std::fs::Permissions::from_mode(0o400))
            .await
            .unwrap();
        let patch = patch(&format!(
            "*** Update File: {}\n*** Move to: {}\n@@\n same\n",
            source.display(),
            destination.display(),
        ));

        let output = apply(&patch, &BackupConfig::default()).await.unwrap();
        let output: FileMutationOutput = serde_json::from_str(&output).unwrap();

        assert!(!source.exists());
        assert_eq!(
            tokio::fs::read_to_string(&destination).await.unwrap(),
            "same\n"
        );
        assert_eq!(
            filetime::FileTime::from_last_modification_time(
                &tokio::fs::metadata(&destination).await.unwrap()
            ),
            old_time
        );
        assert!(
            output
                .diff
                .contains(&format!("--- {}\n+++ /dev/null", pathname(&source)))
        );
        assert!(
            !output
                .diff
                .contains(&format!("--- {0}\n+++ {0}", pathname(&destination)))
        );
    }

    #[tokio::test]
    async fn diffs_existing_add_and_move_destinations_against_their_actual_contents() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let add = directory.path().join("add.txt");
        tokio::fs::write(&add, "before\n").await.unwrap();
        let source = patch(&format!("*** Add File: {}\n+after\n", add.display()));
        let output = apply(&source, &BackupConfig::default()).await.unwrap();
        let output: FileMutationOutput = serde_json::from_str(&output).unwrap();

        assert!(
            output
                .diff
                .contains(&format!("--- {0}\n+++ {0}", pathname(&add)))
        );
        assert!(output.diff.contains("-before\n+after\n"));

        let source = directory.path().join("source.txt");
        let destination = directory.path().join("destination.txt");
        tokio::fs::write(&source, "source before\n").await.unwrap();
        tokio::fs::write(&destination, "destination before\n")
            .await
            .unwrap();
        let patch = patch(&format!(
            "*** Update File: {}\n*** Move to: {}\n@@\n-source before\n+source after\n",
            source.display(),
            destination.display(),
        ));
        let output = apply(&patch, &BackupConfig::default()).await.unwrap();
        let output: FileMutationOutput = serde_json::from_str(&output).unwrap();

        assert!(
            output
                .diff
                .contains(&format!("--- {}\n+++ /dev/null", pathname(&source)))
        );
        assert!(output.diff.contains("-source before\n"));
        assert!(
            output
                .diff
                .contains(&format!("--- {0}\n+++ {0}", pathname(&destination)))
        );
        assert!(output.diff.contains("-destination before\n+source after\n"));
    }

    #[tokio::test]
    async fn rejects_binary_add_and_move_destinations() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let add = directory.path().join("add.bin");
        tokio::fs::write(&add, [0, 1]).await.unwrap();
        let source = patch(&format!("*** Add File: {}\n+replacement\n", add.display()));
        assert!(apply(&source, &BackupConfig::default()).await.is_err());

        assert_eq!(tokio::fs::read(&add).await.unwrap(), [0, 1]);

        let source = directory.path().join("source.txt");
        let destination = directory.path().join("destination.bin");
        tokio::fs::write(&source, "source before\n").await.unwrap();
        tokio::fs::write(&destination, [0, 1]).await.unwrap();
        let patch = patch(&format!(
            "*** Update File: {}\n*** Move to: {}\n@@\n-source before\n+source after\n",
            source.display(),
            destination.display(),
        ));
        assert!(apply(&patch, &BackupConfig::default()).await.is_err());

        assert!(source.exists());
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), [0, 1]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_unreadable_add_and_move_destinations() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let add = directory.path().join("add.txt");
        tokio::fs::write(&add, "before\n").await.unwrap();
        tokio::fs::set_permissions(&add, std::fs::Permissions::from_mode(0o200))
            .await
            .unwrap();
        let source = patch(&format!("*** Add File: {}\n+after\n", add.display()));
        assert!(apply(&source, &BackupConfig::default()).await.is_err());

        tokio::fs::set_permissions(&add, std::fs::Permissions::from_mode(0o600))
            .await
            .unwrap();
        assert_eq!(tokio::fs::read_to_string(&add).await.unwrap(), "before\n");

        let source = directory.path().join("source.txt");
        let destination = directory.path().join("destination.txt");
        tokio::fs::write(&source, "source before\n").await.unwrap();
        tokio::fs::write(&destination, "destination before\n")
            .await
            .unwrap();
        tokio::fs::set_permissions(&destination, std::fs::Permissions::from_mode(0o200))
            .await
            .unwrap();
        let patch = patch(&format!(
            "*** Update File: {}\n*** Move to: {}\n@@\n-source before\n+source after\n",
            source.display(),
            destination.display(),
        ));
        assert!(apply(&patch, &BackupConfig::default()).await.is_err());

        tokio::fs::set_permissions(&destination, std::fs::Permissions::from_mode(0o600))
            .await
            .unwrap();
        assert!(source.exists());
        assert_eq!(
            tokio::fs::read_to_string(&destination).await.unwrap(),
            "destination before\n"
        );
    }

    #[tokio::test]
    async fn reports_when_add_and_update_hunks_make_no_changes() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let add = directory.path().join("add.txt");
        tokio::fs::write(&add, "same\n").await.unwrap();
        let source = patch(&format!("*** Add File: {}\n+same\n", add.display()));

        assert_eq!(
            apply(&source, &BackupConfig::default())
                .await
                .unwrap_err()
                .to_string(),
            "No files were modified."
        );

        let update = directory.path().join("update.txt");
        tokio::fs::write(&update, "same\n").await.unwrap();
        let source = patch(&format!(
            "*** Update File: {}\n@@\n same\n",
            update.display()
        ));

        assert_eq!(
            apply(&source, &BackupConfig::default())
                .await
                .unwrap_err()
                .to_string(),
            "No files were modified."
        );
    }

    #[tokio::test]
    async fn rejects_paths_outside_the_working_directory_and_writes_nothing() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let outside_directory = tempfile::tempdir().unwrap();
        let inside = directory.path().join("inside.txt");
        let outside = outside_directory.path().join("outside.txt");
        tokio::fs::write(&outside, "keep\n").await.unwrap();
        let source = patch(&format!(
            "*** Add File: {}\n+ok\n*** Delete File: {}\n",
            inside.display(),
            outside.display(),
        ));

        let result = apply(&source, &BackupConfig::default()).await;

        assert!(result.is_err());
        assert!(!inside.exists());
        assert_eq!(tokio::fs::read_to_string(&outside).await.unwrap(), "keep\n");
    }

    #[tokio::test]
    async fn rejects_two_hunks_targeting_the_same_file() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("dup.txt");
        tokio::fs::write(&path, "value\n").await.unwrap();
        let source = patch(&format!(
            "*** Update File: {0}\n@@\n-value\n+one\n*** Update File: {0}\n@@\n-one\n+two\n",
            path.display(),
        ));

        assert_eq!(
            apply(&source, &BackupConfig::default())
                .await
                .unwrap_err()
                .to_string(),
            format!("multiple operations target {}", path.display())
        );
        assert_eq!(tokio::fs::read_to_string(path).await.unwrap(), "value\n");
    }

    #[tokio::test]
    async fn failed_matching_leaves_existing_files_untouched() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("file.txt");
        tokio::fs::write(&path, "actual\n").await.unwrap();
        let source = patch(&format!(
            "*** Update File: {}\n@@\n-missing\n+replacement\n",
            path.display(),
        ));

        assert!(apply(&source, &BackupConfig::default()).await.is_err());
        assert_eq!(tokio::fs::read_to_string(path).await.unwrap(), "actual\n");
    }
}
