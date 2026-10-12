use reqwest::{Client, header::CONTENT_TYPE};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::backend_stream::serialize_to_json;
use crate::{
    AssistantMessage, Backend, BackendError, BackendFuture, BackendRequest, Message, ToolCall,
    ToolCallKind, ToolDefinition,
};

pub struct OpenAiResponsesBackend {
    client: Client,
    model: String,
    base_url: String,
    api_key: Option<String>,
}

impl OpenAiResponsesBackend {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
        }
    }

    fn request_body(&self, request: BackendRequest<'_>) -> Value {
        let mut input = Vec::new();
        for message in request.messages.iter() {
            match message {
                Message::System(content) => input.push(json!({
                    "role": "system",
                    "content": content,
                })),
                Message::User(content) => input.push(json!({
                    "role": "user",
                    "content": content,
                })),
                Message::Assistant(message) => {
                    if let Some(content) = &message.content {
                        input.push(json!({
                            "role": "assistant",
                            "content": content,
                        }));
                    }
                    input.extend(message.tool_calls.iter().map(|call| match call.kind {
                        ToolCallKind::Function => json!({
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.function_name,
                            "arguments": call.arguments,
                        }),
                        ToolCallKind::Custom => json!({
                            "type": "custom_tool_call",
                            "call_id": call.id,
                            "name": call.function_name,
                            "input": call.arguments,
                        }),
                    }));
                }
                Message::Tool(results) => input.extend(results.iter().map(|res| {
                    let is_custom = request.messages.iter().any(|message| match message {
                        Message::Assistant(message) => message.tool_calls.iter().any(|call| {
                            call.id == res.tool_call_id && call.kind == ToolCallKind::Custom
                        }),
                        _ => false,
                    });

                    if is_custom {
                        json!({
                            "type": "custom_tool_call_output",
                            "call_id": res.tool_call_id,
                            "output": res.content,
                        })
                    } else {
                        json!({
                            "type": "function_call_output",
                            "call_id": res.tool_call_id,
                            "output": res.content,
                        })
                    }
                })),
            }
        }

        let tools = request
            .tools
            .iter()
            .map(|tool| match tool {
                ToolDefinition::Function(function) => json!({
                    "type": "function",
                    "name": function.name,
                    "description": function.description,
                    "parameters": function.parameters,
                    "strict": function.strict,
                }),
                ToolDefinition::Freeform {
                    name,
                    description,
                    grammar,
                } => {
                    let mut custom = Map::from_iter([
                        ("type".into(), Value::String("custom".into())),
                        ("name".into(), Value::String(name.clone())),
                        ("description".into(), Value::String(description.clone())),
                    ]);

                    // An absent format leaves the input unconstrained free-form text.
                    if let Some(definition) = grammar {
                        custom.insert(
                            "format".into(),
                            json!({
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": definition,
                            }),
                        );
                    }

                    Value::Object(custom)
                }
            })
            .collect::<Vec<_>>();

        let mut body = Map::from_iter([
            ("model".into(), Value::String(self.model.clone())),
            ("input".into(), Value::Array(input)),
            ("stream".into(), Value::Bool(false)),
        ]);

        if !tools.is_empty() {
            body.insert("tools".into(), Value::Array(tools));
        }

        Value::Object(body)
    }
}

impl Backend for OpenAiResponsesBackend {
    fn request(&mut self, request: BackendRequest<'_>) -> BackendFuture {
        let body = self.request_body(request);
        let url = format!("{}/responses", self.base_url);
        let mut request = self
            .client
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .body(serialize_to_json(body));

        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        Box::pin(async move {
            // The Responses backend maps errors without the chat backend's retry and timeout machinery;
            // it is exercised only by its own tests today.
            let response = request
                .send()
                .await
                .map_err(|error| BackendError::Transport {
                    message: error.to_string(),
                })?;
            let status = response.status();
            if !status.is_success() {
                let message = response
                    .text()
                    .await
                    .unwrap_or_default()
                    .chars()
                    .take(2_000)
                    .collect();
                return Err(BackendError::Http {
                    status: status.as_u16(),
                    message,
                });
            }
            let response = response
                .json::<ResponsesResponse>()
                .await
                .map_err(|error| BackendError::Protocol {
                    message: format!("invalid response body: {error}"),
                })?;
            response.into_assistant_message()
        })
    }
}

#[derive(Deserialize)]
struct ResponsesResponse {
    output: Vec<ResponseOutput>,
}

impl ResponsesResponse {
    fn into_assistant_message(self) -> Result<AssistantMessage, BackendError> {
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for output in self.output {
            match output {
                ResponseOutput::Message { content: parts } => {
                    for part in parts {
                        match part {
                            ResponseContent::OutputText { text } => content.push_str(&text),
                            ResponseContent::Refusal { refusal } => content.push_str(&refusal),
                            ResponseContent::Other => {}
                        }
                    }
                }
                ResponseOutput::FunctionCall {
                    call_id,
                    name,
                    arguments,
                } => tool_calls.push(ToolCall {
                    id: call_id,
                    kind: ToolCallKind::Function,
                    function_name: name,
                    arguments,
                }),
                ResponseOutput::CustomToolCall {
                    call_id,
                    name,
                    input,
                } => tool_calls.push(ToolCall {
                    id: call_id,
                    kind: ToolCallKind::Custom,
                    function_name: name,
                    arguments: input,
                }),
                ResponseOutput::Other => {}
            }
        }

        Ok(AssistantMessage {
            reasoning: None,
            content: (!content.is_empty()).then_some(content),
            tool_calls,
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ResponseOutput {
    #[serde(rename = "message")]
    Message { content: Vec<ResponseContent> },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "custom_tool_call")]
    CustomToolCall {
        call_id: String,
        name: String,
        input: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ResponseContent {
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(other)]
    Other,
}

#[cfg(test)]
mod tests {
    use super::super::test_support::with_messages;
    use super::*;
    use crate::{FunctionDefinition, ToolResult};

    const CUSTOM_TOOL_NAME: &str = "foo";

    #[test]
    fn converts_all_request_message_types_and_functions() {
        let messages = vec![
            Message::System("Use functions when needed.".into()),
            Message::User("Call the test function.".into()),
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: Some("Calling the function.".into()),
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    kind: ToolCallKind::Function,
                    function_name: "test_function".into(),
                    arguments: r#"{"input":"value"}"#.into(),
                }],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: "call-1".into(),
                content: "result".into(),
            }]),
        ];
        let functions = vec![ToolDefinition::Function(FunctionDefinition {
            name: "test_function".into(),
            description: "A function used by this test.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                },
                "required": ["input"],
                "additionalProperties": false
            }),
            strict: true,
        })];
        let backend = OpenAiResponsesBackend::new("test-model", "", None);

        let body = with_messages(messages, |messages| {
            backend.request_body(BackendRequest {
                messages,
                tools: &functions,
            })
        });

        assert_eq!(
            body,
            json!({
                "model": "test-model",
                "input": [
                    {
                        "role": "system",
                        "content": "Use functions when needed."
                    },
                    {
                        "role": "user",
                        "content": "Call the test function."
                    },
                    {
                        "role": "assistant",
                        "content": "Calling the function."
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "test_function",
                        "arguments": "{\"input\":\"value\"}"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "result"
                    }
                ],
                "stream": false,
                "tools": [{
                    "type": "function",
                    "name": "test_function",
                    "description": "A function used by this test.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": { "type": "string" }
                        },
                        "required": ["input"],
                        "additionalProperties": false
                    },
                    "strict": true
                }]
            })
        );
    }

    #[test]
    fn converts_custom_tool_messages_and_lark_grammar() {
        let messages = vec![
            Message::Assistant(AssistantMessage {
                reasoning: None,
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    kind: ToolCallKind::Custom,
                    function_name: CUSTOM_TOOL_NAME.into(),
                    arguments: "*** Begin Patch\n*** End Patch".into(),
                }],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: "call-1".into(),
                content: "Applied patch".into(),
            }]),
        ];
        let tools = vec![ToolDefinition::Freeform {
            name: CUSTOM_TOOL_NAME.into(),
            description: "Applies a patch.".into(),
            grammar: Some(r#"start: "hello""#.into()),
        }];
        let backend = OpenAiResponsesBackend::new("test-model", "", None);

        let body = with_messages(messages, |messages| {
            backend.request_body(BackendRequest {
                messages,
                tools: &tools,
            })
        });

        assert_eq!(
            body,
            json!({
                "model": "test-model",
                "input": [
                    {
                        "type": "custom_tool_call",
                        "call_id": "call-1",
                        "name": CUSTOM_TOOL_NAME,
                        "input": "*** Begin Patch\n*** End Patch"
                    },
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call-1",
                        "output": "Applied patch"
                    }
                ],
                "stream": false,
                "tools": [{
                    "type": "custom",
                    "name": CUSTOM_TOOL_NAME,
                    "description": "Applies a patch.",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": r#"start: "hello""#
                    }
                }]
            })
        );
    }

    #[test]
    fn parses_text_refusals_and_multiple_raw_tool_calls() {
        let response = serde_json::from_value::<ResponsesResponse>(json!({
            "output": [
                {
                    "id": "reasoning-1",
                    "type": "reasoning",
                    "summary": []
                },
                {
                    "id": "message-1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "partial text",
                            "annotations": []
                        },
                        {
                            "type": "refusal",
                            "refusal": "refusal text"
                        }
                    ]
                },
                {
                    "id": "function-1",
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "first",
                    "arguments": "not json"
                },
                {
                    "id": "function-2",
                    "type": "function_call",
                    "call_id": "call-2",
                    "name": "second",
                    "arguments": "{\"valid\":true}"
                },
                {
                    "id": "custom-1",
                    "type": "custom_tool_call",
                    "call_id": "call-3",
                    "name": CUSTOM_TOOL_NAME,
                    "input": "*** Begin Patch\n*** End Patch"
                }
            ]
        }))
        .unwrap()
        .into_assistant_message()
        .unwrap();

        assert_eq!(
            response,
            AssistantMessage {
                reasoning: None,
                content: Some("partial textrefusal text".into()),
                tool_calls: vec![
                    ToolCall {
                        id: "call-1".into(),
                        kind: ToolCallKind::Function,
                        function_name: "first".into(),
                        arguments: "not json".into(),
                    },
                    ToolCall {
                        id: "call-2".into(),
                        kind: ToolCallKind::Function,
                        function_name: "second".into(),
                        arguments: r#"{"valid":true}"#.into(),
                    },
                    ToolCall {
                        id: "call-3".into(),
                        kind: ToolCallKind::Custom,
                        function_name: CUSTOM_TOOL_NAME.into(),
                        arguments: "*** Begin Patch\n*** End Patch".into(),
                    },
                ],
            }
        );
    }
}
