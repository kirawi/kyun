use std::ops::Range;

use anyhow::{Result, bail};
use kyun_agent::{FunctionDefinition, ToolDefinition};
use memchr::memchr;
use serde_json::json;

use crate::tools::content_change::ChangeSet;
use crate::tools::file_mutation::{FileSnapshot, resolve_path};
use crate::tools::write;

pub const NAME: &str = "edit";

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EditFileParams {
    path: String,
    edits: Vec<Edit>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Edit {
    old_text: String,
    new_text: String,
}

pub fn definition() -> ToolDefinition {
    ToolDefinition::Function(FunctionDefinition {
        name: NAME.into(),
        description: "Edit a single existing file under the current working directory using exact text replacement. Every \
            edits[].oldText must match a unique, non-overlapping region of the original file; edits are matched against the \
            original contents, not incrementally. If two changes affect the same block or nearby lines, merge them into one \
            edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes."
            .into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to an existing file, relative to the current working directory or an absolute path within it."
                },
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."
                            },
                            "newText": {
                                "type": "string",
                                "description": "Replacement text for this targeted edit."
                            }
                        },
                        "required": ["oldText", "newText"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["path", "edits"],
            "additionalProperties": false
        }),
        strict: true,
    })
}

/// Returns a compact single-line description of a tool call's input for display in the UI.
pub fn summarize_input(input: &str) -> String {
    if let Ok(params) = serde_json::from_str::<EditFileParams>(input) {
        return format!(
            r#"{{ "path": {}, "edits": {} }}"#,
            serde_json::to_string(&params.path).unwrap(),
            params.edits.len()
        );
    }

    input.to_owned()
}

fn line_ending(text: &str) -> Option<&'static str> {
    let bytes = text.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'\r' if bytes.get(index + 1) == Some(&b'\n') => return Some("\r\n"),
            b'\r' => return Some("\r"),
            b'\n' => return Some("\n"),
            _ => {}
        }
    }

    None
}

enum NormalizedMatch {
    NotFound,
    Unique(Range<usize>),
    Ambiguous,
}

/// Finds the unique occurrence of `needle` in `haystack`, where `needle` is already \r-normalized and each `\n`
/// in it consumes `\n`, `\r`, or `\r\n` from `haystack`. The returned range is in  `haystack` byte offsets,
/// so it spans the original line endings.
fn find_normalized(haystack: &str, needle: &str) -> NormalizedMatch {
    let bytes = haystack.as_bytes();
    let first = needle.split('\n').next().unwrap();
    let mut result = NormalizedMatch::NotFound;
    let mut from = 0;

    while let Some(start) = if first.is_empty() {
        (from <= bytes.len()).then_some(from)
    } else {
        bytes
            .get(from..)
            .and_then(|window| memchr(first.as_bytes()[0], window))
            .map(|index| from + index)
    } {
        let mut matched = bytes[start..].starts_with(first.as_bytes());
        let mut end = start + first.len();
        if matched {
            for segment in needle[first.len()..].split('\n').skip(1) {
                match bytes.get(end) {
                    Some(b'\r') => {
                        end += 1;
                        if bytes.get(end) == Some(&b'\n') {
                            end += 1;
                        }
                    }
                    Some(b'\n') => end += 1,
                    _ => {
                        matched = false;
                        break;
                    }
                }
                if !bytes[end..].starts_with(segment.as_bytes()) {
                    matched = false;
                    break;
                }
                end += segment.len();
            }
        }

        if matched {
            if matches!(result, NormalizedMatch::Unique(_)) {
                return NormalizedMatch::Ambiguous;
            }
            result = NormalizedMatch::Unique(start..end);
        }
        // Advance to the next position that exists in the normalized space: the `\n` of a `\r\n` pair
        // is consumed together with the `\r`, so a match can never start there.
        from = start + 1;
        if bytes.get(start) == Some(&b'\r') && bytes.get(from) == Some(&b'\n') {
            from += 1;
        }
    }

    result
}

fn resolve_edits(
    original: &str,
    edits: &[Edit],
    path: &str,
) -> std::result::Result<ChangeSet, String> {
    let mut matched = Vec::new();

    for (index, edit) in edits.iter().enumerate() {
        let old_text = edit.old_text.replace("\r\n", "\n").replace('\r', "\n");
        if old_text.is_empty() {
            return Err(format!(
                "edits[{index}].oldText must not be empty in {path}."
            ));
        }

        let range = match find_normalized(original, &old_text) {
            NormalizedMatch::NotFound => {
                return Err(format!(
                    "Could not find edits[{index}] in {path}. The oldText must match exactly including all whitespace and newlines."
                ));
            }
            NormalizedMatch::Ambiguous => {
                return Err(format!(
                    "edits[{index}].oldText is not unique in {path}. Please provide more context to make it unique."
                ));
            }
            NormalizedMatch::Unique(range) => range,
        };

        let ending = line_ending(&original[range.clone()])
            .or_else(|| line_ending(&original[range.end..]))
            .or_else(|| line_ending(&original[..range.start]))
            .unwrap_or("\n");
        let new_text = edit
            .new_text
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\n', ending);
        matched.push((index, range.start, range.end - range.start, new_text));
    }

    matched.sort_by_key(|(_, start, _, _)| *start);
    for window in matched.windows(2) {
        let (previous_index, previous_start, previous_length, _) = &window[0];
        let (current_index, current_start, _, _) = &window[1];
        if previous_start + previous_length > *current_start {
            return Err(format!(
                "edits[{previous_index}] and edits[{current_index}] overlap in {path}. Merge them into one edit or target disjoint regions."
            ));
        }
    }

    let changes = ChangeSet::from_replacements(
        original,
        matched
            .into_iter()
            .map(|(_, start, length, new_text)| (start..start + length, new_text)),
    )?;
    if !changes.is_changed() {
        return Err(format!(
            "No changes made to {path}. The replacements produced identical content."
        ));
    }

    Ok(changes)
}

pub async fn prepare(input: &str) -> Result<write::PreparedWrite> {
    let params = serde_json::from_str::<EditFileParams>(input)?;
    if params.edits.is_empty() {
        bail!("No edits were provided.");
    }

    let resolved = resolve_path(&params.path).await?;
    let observed = FileSnapshot::capture(&resolved.target).await;
    let original = match &observed {
        FileSnapshot::Present { text, .. } => text.to_string(),
        FileSnapshot::Missing => {
            bail!("failed to read {}: file not found", params.path);
        }
        FileSnapshot::Binary => {
            bail!("failed to read {}: file appears to be binary", params.path);
        }
        FileSnapshot::Unreadable { message } => {
            bail!("failed to read {}: {message}", params.path);
        }
    };

    let changes =
        resolve_edits(&original, &params.edits, &params.path).map_err(anyhow::Error::msg)?;
    write::PreparedWrite::from_change_set(resolved, observed, changes)
}

#[cfg(test)]
mod tests {
    use ropey::Rope;

    use crate::backup::BackupConfig;
    use crate::{path::current_working_dir, tools::FileMutationOutput};

    use super::*;

    fn apply_edits(
        original: &str,
        edits: &[Edit],
        path: &str,
    ) -> std::result::Result<String, String> {
        let changes = resolve_edits(original, edits, path)?;
        let mut text = Rope::from(original);
        assert!(changes.apply(&mut text));
        Ok(text.to_string())
    }

    fn edits(pairs: &[(&str, &str)]) -> Vec<Edit> {
        pairs
            .iter()
            .map(|(old, new)| Edit {
                old_text: (*old).into(),
                new_text: (*new).into(),
            })
            .collect()
    }

    fn pathname(path: &std::path::Path) -> String {
        path.strip_prefix(current_working_dir())
            .unwrap()
            .display()
            .to_string()
    }

    async fn execute(input: &str, backup_config: &BackupConfig) -> Result<String> {
        let mut prepared = prepare(input).await?;
        let (output, _) = prepared.edit_output();
        prepared.execute(backup_config).await?;
        Ok(serde_json::to_string(&output).unwrap())
    }

    #[test]
    fn summarizes_the_path_and_edit_count() {
        let input = r#"{ "path": "src/main.rs", "edits": [ { "oldText": "a", "newText": "b" }, { "oldText": "c", "newText": "d" } ] }"#;

        assert_eq!(
            summarize_input(input),
            r#"{ "path": "src/main.rs", "edits": 2 }"#
        );
    }

    #[test]
    fn applies_disjoint_edits_against_the_original() {
        let original = "one\ntwo\nthree\nfour\n";
        let result = apply_edits(
            original,
            &edits(&[("four", "FOUR"), ("one", "ONE")]),
            "f.txt",
        )
        .unwrap();

        assert_eq!(result, "ONE\ntwo\nthree\nFOUR\n");
    }

    #[test]
    fn rejects_missing_duplicate_empty_and_overlapping_edits() {
        assert!(apply_edits("actual\n", &edits(&[("missing", "x")]), "f.txt").is_err());
        assert!(apply_edits("same\nsame\n", &edits(&[("same", "x")]), "f.txt").is_err());
        assert!(apply_edits("ababa", &edits(&[("aba", "x")]), "f.txt").is_err());
        assert!(apply_edits("value", &edits(&[("", "x")]), "f.txt").is_err());
        assert!(apply_edits("abcdef\n", &edits(&[("abc", "x"), ("bcd", "y")]), "f.txt").is_err());
    }

    #[test]
    fn preserves_bom_and_untouched_mixed_line_endings() {
        let original = "\u{feff}keep\r\ncafé\r\nunix\n";
        let result = apply_edits(original, &edits(&[("café", "new\nadded")]), "f.txt").unwrap();

        assert_eq!(result, "\u{feff}keep\r\nnew\r\nadded\r\nunix\n");
    }

    #[tokio::test]
    async fn executes_json_edits_and_reports_the_path() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("file.txt");
        tokio::fs::write(&path, "first\nsecond\n").await.unwrap();
        let input = serde_json::json!({
            "path": path.display().to_string(),
            "edits": [
                { "oldText": "first", "newText": "1st" },
                { "oldText": "second", "newText": "2nd" },
            ],
        })
        .to_string();

        let output = execute(&input, &BackupConfig::default()).await.unwrap();
        let output: FileMutationOutput = serde_json::from_str(&output).unwrap();

        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "1st\n2nd\n"
        );
        assert_eq!(
            output.summary,
            format!(
                "Success. Updated the following files:\nM {}\n",
                pathname(&path)
            )
        );
        assert_eq!(
            output.diff,
            format!(
                "--- {0}\n+++ {0}\n@@ -1,2 +1,2 @@\n-first\n-second\n+1st\n+2nd\n",
                pathname(&path)
            )
            .into()
        );
    }

    #[tokio::test]
    async fn failed_edit_leaves_the_file_untouched() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("file.txt");
        tokio::fs::write(&path, "actual\n").await.unwrap();
        let input = serde_json::json!({
            "path": path.display().to_string(),
            "edits": [{ "oldText": "missing", "newText": "replacement" }],
        })
        .to_string();

        assert!(execute(&input, &BackupConfig::default()).await.is_err());
        assert_eq!(tokio::fs::read_to_string(path).await.unwrap(), "actual\n");
    }

    #[tokio::test]
    async fn rejects_paths_outside_the_working_directory() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("file.txt");
        tokio::fs::write(&path, "old\n").await.unwrap();
        let input = serde_json::json!({
            "path": path.display().to_string(),
            "edits": [{ "oldText": "old", "newText": "new" }],
        })
        .to_string();

        assert!(execute(&input, &BackupConfig::default()).await.is_err());
        assert_eq!(tokio::fs::read_to_string(path).await.unwrap(), "old\n");
    }
}
