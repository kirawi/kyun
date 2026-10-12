use std::{fs, path::Path, time::Duration};

use anyhow::Context;
use kyun_agent::{
    Agent, Message, ToolDefinition,
    openai::{Config as OpenAiConfig, OpenAiBackend},
};
use serde::{Deserialize, Deserializer, de::Error as _};
use serde_json::{Value, json};

use crate::backup::BackupConfig;
use crate::tools::{
    ToolKind, apply_patch, edit, file_mutation, read, read_directory, shell, write,
};

// TODO: Need to extract certain components into per-provider and per-model config.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub backup: BackupConfig,
    /// Maximum number of tool-call iterations per turn.
    pub max_steps: usize,
    /// Render completed real lines while an assistant response is streaming.
    pub stream_assistant_messages: bool,
    /// Render assistant message content as Markdown in the transcript.
    pub render_markdown: bool,
    /// Provider request timeout, in seconds: bounds connecting (capped at 10s regardless of the value), how
    /// long a request may sit idle without receiving data (including between streamed events), and the whole
    /// duration of non-streaming requests. Zero disables all of these, the previous behavior. A value must
    /// exceed the provider's worst-case pause between streamed events, or slow but legitimate streams will
    /// be cut off.
    pub per_step_timeout: u64,
    /// Maximum context window for entire session.
    pub max_tokens: usize,
    /// Maximum number of source lines rendered for shell output before showing an ellipsis.
    pub tool_output_line_limit: usize,
    /// Maximum number of source lines rendered for file diffs before showing an ellipsis.
    pub tool_diff_line_limit: usize,
    /// Whether the provider supports custom (freeform/grammar-constrained) tools. This selects the custom `apply_patch`
    /// tool when true and the JSON `edit` function when false.
    pub custom_tools: bool,
    /// Tools to make available to the agent. The `edit` entry selects the editing tool configured by `custom_tools`.
    #[serde(default = "default_tools", deserialize_with = "deserialize_tools")]
    pub tools: Vec<ToolKind>,
    /// Whether to execute all tool calls without waiting for user approval.
    pub skip_approval: bool,
    /// Shell command and arguments used to run shell tools. Defaults to `sh -c` on Unix and `cmd /C` on Windows.
    pub shell: Vec<String>,
    pub provider: ProviderConfig,
    // TODO: Replace with a default AgentPermissions
    // pub default_approval_policy: ApprovalPolicy,
}

/// Tool names used when `tools` is absent from the configuration.
const DEFAULT_TOOLS: [&str; 3] = [write::NAME, edit::NAME, shell::NAME];
// const DEFAULT_TOOLS: [&str; 2] = [read::NAME, read_directory::NAME];

fn default_tools() -> Vec<ToolKind> {
    DEFAULT_TOOLS
        .iter()
        .map(|name| ToolKind::from_function_name(name))
        .collect()
}

fn deserialize_tools<'de, D>(deserializer: D) -> Result<Vec<ToolKind>, D::Error>
where
    D: Deserializer<'de>,
{
    Vec::<String>::deserialize(deserializer)?
        .into_iter()
        .map(|name| match ToolKind::from_function_name(&name) {
            ToolKind::Unknown => Err(D::Error::custom(format!("unknown tool {name:?}"))),
            kind => Ok(kind),
        })
        .collect()
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub api_key_env: Option<String>,
    pub base_url: String,
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub extra_body: Option<Value>,
}

/// The opening message of every conversation kyun starts with a provider.
const SYSTEM_PROMPT: &str = r#"You are kyun, an agentic coding assistant operating in a terminal workspace. Be precise and direct.

## Capabilities

You receive user requests and workspace context. You communicate with the user through brief progress updates and final
responses. You can call only the tools made available by the current configuration; tool availability is authoritative.
Do not assume that an unavailable tool exists.

## Responsiveness

Before a group of tool calls, briefly state what you are about to do and why. Keep these updates to one or two concise
sentences, connect them to the work already completed, and avoid updates for trivial reads. For longer tasks, provide
occasional progress updates that state what is complete and what you will do next.

## Planning

For non-trivial, multi-step, ambiguous, or dependency-heavy work, make a short plan with concrete steps that can be
verified. Do not create plans for simple tasks, pad plans with obvious steps, or repeat a plan unnecessarily. Adjust the
plan when the task changes or new information changes the approach.

## Task execution

Understand the request and inspect relevant context before acting. Keep working until the request is completely resolved;
do not guess or invent results. Ask for clarification when a missing choice would materially change the result;
otherwise make a reasonable assumption and state it when useful.

When changing code, fix the root cause when practical, avoid unnecessary complexity, keep changes minimal and consistent
with the existing codebase, and update documentation when the change requires it. Use the file-editing tools for file
changes rather than ad hoc shell scripts. Preserve unrelated user work, do not broaden the task, and do not commit,
create branches, or make other external changes unless the user explicitly requests them. Do not add unrelated fixes,
copyright headers, or inline comments unless requested.

When asked to change or build, implement the request and verify it. When asked to explain, diagnose, review, or report,
do not mutate files unless the user also asks for a change. When blocked, perform safe in-scope checks and alternatives
before reporting the exact blocker. Do not stop while required work remains.

## Tool use and validation

Tool calls may require user approval depending on the current configuration. Wait for each completed result, and do not
treat a pending or denied call as executed. Destructive or externally visible actions require clear authorization. When
searching for files or text,
prefer `rg` or `rg --files`; do not use scripts to dump large files.

When validating code, start with checks targeted to the change and then broaden validation as confidence grows. Use the
tests, builds, or other checks supported by the repository. Do not fix unrelated failures while validating, and never
claim that a check passed without its completed result.

## Communication

Lead with the outcome. Be concise and factual, state assumptions and prerequisites when they matter, and report errors,
uncertainty, blockers, and remaining work accurately. The user can see the same workspace, so do not reproduce large
files or ask them to copy or save changes made through tools. End with a self-contained handoff that states what changed,
what was verified, and the next useful step when one remains. Never claim success without evidence."#;

impl Config {
    /// Creates an agent configured from provider settings. Called at startup and again whenever a `:new`
    /// session replaces the current one.
    pub fn make_agent(&self) -> Agent<OpenAiBackend> {
        let backend = OpenAiBackend::new(OpenAiConfig {
            model: self.provider.model.clone(),
            base_url: self.provider.base_url.clone(),
            api_key: self
                .provider
                .api_key_env
                .as_deref()
                .and_then(|name| std::env::var(name).ok()),
            reasoning_effort: self.provider.reasoning_effort.clone(),
            extra_body: self.provider.extra_body.clone(),
            timeout: (self.per_step_timeout > 0)
                .then(|| Duration::from_secs(self.per_step_timeout)),
            ..OpenAiConfig::default()
        });
        Agent::with_messages(backend, [Message::System(SYSTEM_PROMPT.into())])
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|tool| match tool {
                ToolKind::FileMutation(file_mutation::Kind::Write) => write::definition(),
                ToolKind::FileMutation(file_mutation::Kind::Edit) if self.custom_tools => {
                    apply_patch::definition()
                }
                ToolKind::FileMutation(file_mutation::Kind::Edit) => edit::definition(),
                ToolKind::FileMutation(file_mutation::Kind::ApplyPatch) => {
                    apply_patch::definition()
                }
                ToolKind::Read => read::definition(),
                ToolKind::ReadDirectory => read_directory::definition(),
                ToolKind::Shell => shell::definition(),
                ToolKind::Unknown => unreachable!(),
            })
            .collect()
    }

    pub fn load(path: &Path) -> Self {
        let path = if path.is_dir() {
            path.join("config.toml")
        } else {
            path.to_owned()
        };

        let contents = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(error) => panic!(
                "could not read configuration file '{}': {error}",
                path.display()
            ),
        };

        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("could not parse configuration file '{}'", path.display()))
            .unwrap();
        if let Some(extra_body) = &config.provider.extra_body
            && !extra_body.is_object()
        {
            panic!(
                "provider extra_body in '{}' must be a JSON object",
                path.display()
            );
        }
        config
    }
}

impl Default for Config {
    fn default() -> Self {
        // These are the author's dev-workstation defaults, not release defaults: a shell-enabled toolset with
        // approval skipped, pointed at an ollama-hosted provider. The auto-approval and shell access are
        // deliberate (FINDINGS M-j); any shared or release configuration re-enables approval.
        Self {
            backup: BackupConfig::default(),
            max_steps: 500,
            stream_assistant_messages: true,
            render_markdown: true,
            per_step_timeout: Default::default(),
            max_tokens: 1_000_000,
            tool_output_line_limit: 10,
            tool_diff_line_limit: 100,
            custom_tools: false,
            tools: default_tools(),
            skip_approval: false,
            shell: if cfg!(windows) {
                vec!["cmd".into(), "/C".into()]
            } else {
                vec!["sh".into(), "-c".into()]
            },
            provider: ProviderConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use kyun_agent::ToolDefinition;

    use super::Config;
    use crate::tools::{ToolKind, apply_patch, edit, file_mutation, read, read_directory};

    #[test]
    fn parses_configured_tools() {
        let config: Config = toml::from_str(&format!(
            "tools = [\"{}\", \"{}\"]",
            read::NAME,
            read_directory::NAME
        ))
        .unwrap();

        assert_eq!(config.tools, [ToolKind::Read, ToolKind::ReadDirectory]);
    }

    #[test]
    fn parses_configured_transcript_limits() {
        let config: Config = toml::from_str(
            r#"
                tool_output_line_limit = 20
                tool_diff_line_limit = 200
            "#,
        )
        .unwrap();

        assert_eq!(config.tool_output_line_limit, 20);
        assert_eq!(config.tool_diff_line_limit, 200);
    }

    #[test]
    fn rejects_unknown_tools() {
        let error = toml::from_str::<Config>(
            r#"
                tools = ["writ"]
            "#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("unknown tool"));
    }

    #[test]
    fn custom_tools_does_not_change_the_configured_tool_kind() {
        let config: Config = toml::from_str(&format!(
            "custom_tools = true\ntools = [\"{}\"]",
            edit::NAME
        ))
        .unwrap();

        assert_eq!(
            config.tools,
            [ToolKind::FileMutation(file_mutation::Kind::Edit)]
        );
    }

    #[test]
    fn custom_tools_selects_the_freeform_edit_definition() {
        let config: Config = toml::from_str(&format!(
            "custom_tools = true\ntools = [\"{}\"]",
            edit::NAME
        ))
        .unwrap();

        let definitions = config.tool_definitions();
        let [ToolDefinition::Freeform { name, .. }] = definitions.as_slice() else {
            unreachable!()
        };
        assert_eq!(name, apply_patch::NAME);
    }

    #[test]
    fn absent_tools_use_the_default_toolset() {
        let config: Config = toml::from_str("custom_tools = true").unwrap();

        assert_eq!(
            config.tools,
            [
                ToolKind::FileMutation(file_mutation::Kind::Write),
                ToolKind::FileMutation(file_mutation::Kind::Edit),
                ToolKind::Shell,
            ]
        );
    }

    #[test]
    fn an_explicitly_empty_toolset_is_not_replaced_by_the_default() {
        let config: Config = toml::from_str("tools = []").unwrap();

        assert!(config.tools.is_empty());
    }

    #[test]
    fn default_tools_exclude_filesystem_reads() {
        assert_eq!(
            Config::default().tools,
            [
                ToolKind::FileMutation(file_mutation::Kind::Write),
                ToolKind::FileMutation(file_mutation::Kind::Edit),
                ToolKind::Shell,
            ]
        );
    }

    #[test]
    fn parses_skip_approval() {
        let config: Config = toml::from_str("skip_approval = true").unwrap();

        assert!(config.skip_approval);
    }
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_key_env: Some("OLLAMA_API_KEY".into()),
            base_url: "https://ollama.com/v1".into(),
            model: "glm-5.3-flash:cloud".into(),
            reasoning_effort: Some("max".into()),
            extra_body: Some(json!({
                "chat_template_kwargs": {
                    "enable_thinking": true
                }
            })),
        }
    }
}
