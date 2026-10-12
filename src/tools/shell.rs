use std::process::Stdio;

use anyhow::{Result, bail};
use kyun_agent::{FunctionDefinition, ToolDefinition};
use serde_json::json;
use tokio::process::Command;

pub const NAME: &str = "shell";

#[derive(serde::Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ShellParams {
    pub command: String,
}

pub fn definition() -> ToolDefinition {
    ToolDefinition::Function(FunctionDefinition {
        name: NAME.into(),
        description: "Run a command through the configured shell in the current working directory."
            .into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute through the configured shell."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        }),
        strict: true,
    })
}

/// Caps a captured stream so a chatty command (build logs, test output) cannot flood the conversation
/// history, which has no context-window management yet.
const OUTPUT_LIMIT: usize = 64 * 1024;

fn bounded(text: &str) -> String {
    if text.chars().count() <= OUTPUT_LIMIT {
        text.to_owned()
    } else {
        let mut bounded = text.chars().take(OUTPUT_LIMIT).collect::<String>();
        bounded.push_str("\n...[output truncated]");
        bounded
    }
}

pub async fn execute(params: ShellParams, shell: &[String]) -> Result<String> {
    let Some(program) = shell.first() else {
        bail!("No shell set");
    };

    let output = Command::new(program)
        .args(&shell[1..])
        .arg(params.command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // An interrupted or abandoned call must not leave its command running. Only the direct child is
        // signaled: a compound command's forked subshells may survive (sh -c "sleep 300 && mv a b").
        .kill_on_drop(true)
        .output()
        .await?;

    let stdout = bounded(&String::from_utf8_lossy(&output.stdout));
    let stderr = bounded(&String::from_utf8_lossy(&output.stderr));
    if output.status.success() {
        return Ok(if stderr.is_empty() {
            stdout
        } else {
            format!("{stdout}\n[stderr]\n{stderr}")
        });
    }

    // The exit status and both streams: the model cannot distinguish failure from success, or diagnose
    // either, without them.
    bail!(
        "command failed: {}\n[stdout]\n{stdout}\n[stderr]\n{stderr}",
        output.status
    )
}

#[cfg(test)]
mod tests {
    use super::{ShellParams, execute};

    #[tokio::test]
    async fn executes_a_command_through_the_configured_shell() {
        let shell = if cfg!(windows) {
            vec!["cmd".into(), "/C".into()]
        } else {
            vec!["sh".into(), "-c".into()]
        };

        let output = execute(
            ShellParams {
                command: "echo shell output".into(),
            },
            &shell,
        )
        .await
        .unwrap();

        assert_eq!(output.trim(), "shell output");
    }

    #[tokio::test]
    async fn rejects_an_empty_shell_configuration() {
        let error = execute(
            ShellParams {
                command: "echo unused".into(),
            },
            &[],
        )
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), "No shell set");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn success_with_stderr_reports_both_streams() {
        let shell = ["sh".to_owned(), "-c".to_owned()];

        let output = execute(
            ShellParams {
                command: "echo stdout text; echo stderr text 1>&2".into(),
            },
            &shell,
        )
        .await
        .unwrap();

        assert!(output.contains("stdout text"), "{output}");
        assert!(output.contains("[stderr]"), "{output}");
        assert!(output.contains("stderr text"), "{output}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failure_reports_the_exit_status_and_both_streams() {
        let shell = ["sh".to_owned(), "-c".to_owned()];

        let error = execute(
            ShellParams {
                command: "echo stdout text; echo stderr text 1>&2; exit 3".into(),
            },
            &shell,
        )
        .await
        .unwrap_err();
        let error = error.to_string();

        assert!(error.contains("exit status: 3"), "{error}");
        assert!(error.contains("[stdout]"), "{error}");
        assert!(error.contains("stdout text"), "{error}");
        assert!(error.contains("[stderr]"), "{error}");
        assert!(error.contains("stderr text"), "{error}");
    }
}
