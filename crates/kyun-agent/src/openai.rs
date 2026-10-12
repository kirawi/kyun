use std::collections::VecDeque;
use std::time::Duration;

use futures_util::{StreamExt, stream};
use reqwest::{
    Client, StatusCode,
    header::{CONTENT_TYPE, RETRY_AFTER},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

pub use crate::RetryPolicy;
use crate::backend_stream::serialize_to_json;
use crate::{
    AssistantMessage, Backend, BackendError, BackendEvent, BackendFuture, BackendRequest,
    BackendStream, Message, TokenUsage, ToolCall, ToolCallDelta, ToolCallKind, ToolDefinition,
};

mod responses;

#[cfg(test)]
mod test_support {
    use crate::{Message, MessageView, history::History};

    pub fn with_messages<T>(
        messages: impl IntoIterator<Item = Message>,
        f: impl for<'a> FnOnce(MessageView<'a>) -> T,
    ) -> T {
        let history = History::with_messages(messages);
        f(history.view(&Default::default()))
    }
}

pub use responses::OpenAiResponsesBackend;

#[derive(Default)]
pub struct Config {
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub reasoning_effort: Option<String>,
    pub extra_body: Option<Value>,
    /// Connect and idle-read timeout for provider requests; `None` disables both (the previous behavior).
    pub timeout: Option<Duration>,
    pub retries: RetryPolicy,
}

pub struct OpenAiBackend {
    client: Client,
    config: Config,
}

impl OpenAiBackend {
    pub fn new(mut config: Config) -> Self {
        config.base_url = config.base_url.trim_end_matches('/').to_owned();
        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout {
            // Connecting should never take as long as a legitimate idle gap between streamed events, so it
            // gets the tighter of the configured timeout and ten seconds.
            builder = builder
                .connect_timeout(timeout.min(Duration::from_secs(10)))
                .read_timeout(timeout);
        }
        Self {
            client: builder.build().unwrap(),
            config,
        }
    }

    fn request_body(
        &self,
        request: BackendRequest<'_>,
        stream: bool,
    ) -> Result<Value, BackendError> {
        let mut messages = Vec::new();
        for message in request.messages.iter() {
            match message {
                Message::Tool(results) => messages.extend(results.iter().map(|result| {
                    json!({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    })
                })),
                _ => messages.push(message_to_value(message)),
            }
        }
        let tools = request.tools.iter().map(tool_to_value).collect::<Vec<_>>();

        let mut body = Map::from_iter([
            ("model".into(), Value::String(self.config.model.clone())),
            ("messages".into(), Value::Array(messages)),
        ]);

        if !tools.is_empty() {
            body.insert("tools".into(), Value::Array(tools));
        }

        if let Some(reasoning_effort) = &self.config.reasoning_effort {
            body.insert(
                "reasoning_effort".into(),
                Value::String(reasoning_effort.clone()),
            );
        }

        if let Some(extra_body) = &self.config.extra_body {
            let Some(extra_body) = extra_body.as_object() else {
                return Err(BackendError::Config {
                    message: format!(
                        "provider extra_body must be a JSON object, found {}",
                        type_name(extra_body)
                    ),
                });
            };
            body.extend(extra_body.clone());
        }
        if stream {
            body.insert("stream_options".into(), json!({ "include_usage": true }));
        }
        body.insert("stream".into(), Value::Bool(stream));

        Ok(Value::Object(body))
    }

    fn request_builder(
        &self,
        body: Value,
        timeout: Option<Duration>,
    ) -> impl FnMut() -> reqwest::RequestBuilder + Send + 'static {
        let client = self.client.clone();
        let url = format!("{}/chat/completions", self.config.base_url);
        let api_key = self.config.api_key.clone();
        move || {
            let mut request = client
                .post(&url)
                .header(CONTENT_TYPE, "application/json")
                .body(serialize_to_json(body.clone()));
            if let Some(api_key) = &api_key {
                request = request.bearer_auth(api_key);
            }
            if let Some(timeout) = timeout {
                request = request.timeout(timeout);
            }
            request
        }
    }
}

/// The outcome of a successful send phase.
enum Sent {
    /// A response whose body is read as a whole (non-streaming requests).
    Complete(reqwest::Response),
    /// A streamed response whose first bytes already arrived; the connection proved itself alive.
    Started {
        response: reqwest::Response,
        pending: Vec<u8>,
    },
}

fn is_retryable_status(status: StatusCode) -> bool {
    // All 5xx are included (Cloudflare-style 520-524 and 529 included) since a retry can only help and the
    // status is the provider's word that the failure is on its side.
    status.as_u16() == 408 || status.as_u16() == 429 || status.is_server_error()
}

fn is_retryable_transport(error: &reqwest::Error) -> bool {
    // Builder errors (a malformed URL, mostly) can never succeed on retry; neither can body decode errors.
    error.is_connect() || error.is_timeout() || error.is_request()
}

/// `Retry-After` in seconds, saturating at `u64::MAX` seconds so an absurd value still compares as "longer
/// than any policy" and fails fast; HTTP-date values are not parsed and fall back to computed backoff.
fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u128>()
        .ok()
        .map(|seconds| Duration::from_secs(seconds.min(u64::MAX as u128) as u64))
}

/// The provider's error body, preferring its `error.message` field and truncated: full bodies serve no one
/// past the first screen.
async fn error_body(response: reqwest::Response) -> String {
    let body = response.text().await.unwrap_or_default();
    #[derive(Deserialize)]
    struct ErrorBody {
        error: Option<ErrorObject>,
    }
    #[derive(Deserialize)]
    struct ErrorObject {
        message: Option<String>,
    }
    let body = serde_json::from_str::<ErrorBody>(&body)
        .ok()
        .and_then(|body| body.error)
        .and_then(|error| error.message)
        .unwrap_or(body);
    if body.chars().count() > 2_000 {
        body.chars().take(2_000).collect()
    } else {
        body
    }
}

/// Sends the request, retrying transient failures (rate limits, server errors, connect and timeout failures)
/// with exponential backoff. For streaming requests the attempt only succeeds once the first bytes of the body
/// have arrived; a connection that dies before any data was received is retried like any other transport
/// failure. A retry may reach a server that already processed the request, so a duplicated completion is
/// possible in exchange for the turn surviving.
async fn send_with_retries(
    mut request: impl FnMut() -> reqwest::RequestBuilder + Send + 'static,
    policy: RetryPolicy,
    streaming: bool,
) -> Result<Sent, BackendError> {
    let mut retries_used = 0;
    loop {
        let delay = 'retry: {
            match request().send().await {
                Ok(mut response) => {
                    let status = response.status();
                    if !status.is_success() {
                        let retry_after = retry_after(&response);
                        let message = error_body(response).await;
                        if !is_retryable_status(status) || retries_used == policy.max_retries {
                            return Err(BackendError::Http {
                                status: status.as_u16(),
                                message: with_retry_after(message, retry_after),
                            });
                        }
                        match policy.delay(retries_used, retry_after) {
                            // The server demands a longer quiet window than the policy allows: fail instead of
                            // retrying into a window that cannot have elapsed.
                            None => {
                                return Err(BackendError::Http {
                                    status: status.as_u16(),
                                    message: with_retry_after(message, retry_after),
                                });
                            }
                            Some(delay) => delay,
                        }
                    } else if !streaming {
                        return Ok(Sent::Complete(response));
                    } else {
                        // Read until the first non-empty chunk: an accepted request whose connection dies
                        // before producing data is retried; after that, failures are mid-stream and terminal.
                        let mut pending = Vec::new();
                        loop {
                            match response.chunk().await {
                                Ok(Some(chunk)) if chunk.is_empty() => {
                                    pending.extend_from_slice(&chunk);
                                }
                                Ok(Some(chunk)) => {
                                    pending.extend_from_slice(&chunk);
                                    return Ok(Sent::Started { response, pending });
                                }
                                Ok(None) => {
                                    if retries_used == policy.max_retries {
                                        return Err(BackendError::Transport {
                                            message:
                                                "connection closed before any data was received"
                                                    .into(),
                                        });
                                    }
                                    // Dead before any data: retry immediately, budget permitting.
                                    break 'retry Duration::ZERO;
                                }
                                Err(error) => {
                                    if retries_used == policy.max_retries {
                                        return Err(BackendError::Transport {
                                            message: error.to_string(),
                                        });
                                    }
                                    break 'retry policy.delay(retries_used, None).unwrap();
                                }
                            }
                        }
                    }
                }
                Err(error) => {
                    if !is_retryable_transport(&error) || retries_used == policy.max_retries {
                        return Err(BackendError::Transport {
                            message: error.to_string(),
                        });
                    }
                    policy.delay(retries_used, None).unwrap()
                }
            }
        };
        retries_used += 1;
        tokio::time::sleep(delay).await;
    }
}

fn with_retry_after(message: String, retry_after: Option<Duration>) -> String {
    match retry_after {
        Some(retry_after) => format!("{message} (retry after {}s)", retry_after.as_secs()),
        None => message,
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

impl Backend for OpenAiBackend {
    fn request(&mut self, request: BackendRequest<'_>) -> BackendFuture {
        let body = match self.request_body(request, false) {
            Ok(body) => body,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        // No deltas are involved, so the whole response may be bounded by one overall timeout.
        let request = self.request_builder(body, self.config.timeout);
        let retries = self.config.retries;
        Box::pin(async move {
            let response = match send_with_retries(request, retries, false).await? {
                Sent::Complete(response) => response,
                Sent::Started { .. } => {
                    unreachable!("non-streaming sends never wait for a first byte")
                }
            };
            let response =
                response
                    .json::<ChatResponse>()
                    .await
                    .map_err(|error| BackendError::Protocol {
                        message: format!("invalid response body: {error}"),
                    })?;
            response.into_assistant_message()
        })
    }

    fn request_stream(&mut self, request: BackendRequest<'_>) -> BackendStream {
        let body = match self.request_body(request, true) {
            Ok(body) => body,
            Err(error) => return Box::pin(stream::once(async move { Err(error) })),
        };
        let request = self.request_builder(body, None);
        let retries = self.config.retries;
        Box::pin(
            stream::once(async move {
                match send_with_retries(request, retries, true).await {
                    Ok(Sent::Started { response, pending }) => {
                        ChatEventStream::into_stream(response, pending)
                    }
                    Ok(Sent::Complete(_)) => {
                        unreachable!("streaming sends always wait for a first byte")
                    }
                    Err(error) => {
                        Box::pin(stream::once(async move { Err(error) })) as BackendStream
                    }
                }
            })
            .flatten(),
        )
    }
}

#[derive(Default)]
struct SseDecoder {
    input: Vec<u8>,
    data: Vec<u8>,
    saw_data: bool,
    ready: VecDeque<Result<String, std::str::Utf8Error>>,
}

impl SseDecoder {
    fn feed(&mut self, chunk: &[u8]) {
        self.input.extend_from_slice(chunk);
        self.parse_input(false);
    }

    /// Flushes a partial final event; call once the response body has ended.
    fn finish(&mut self) {
        self.parse_input(true);
    }

    /// Drops a partially received line and event without dispatching either.
    fn discard_partial(&mut self) {
        self.input.clear();
        self.data.clear();
        self.saw_data = false;
    }

    fn next_event(&mut self) -> Option<Result<String, std::str::Utf8Error>> {
        self.ready.pop_front()
    }

    fn finish_event(&mut self) {
        if !self.saw_data {
            return;
        }

        self.data.pop();
        self.ready.push_back(
            String::from_utf8(std::mem::take(&mut self.data)).map_err(|error| error.utf8_error()),
        );
        self.saw_data = false;
    }

    fn process_line(&mut self, line: &[u8]) {
        if line.is_empty() {
            self.finish_event();
            return;
        }
        if line[0] == b':' {
            return;
        }

        let colon = line.iter().position(|byte| *byte == b':');
        let field = &line[..colon.unwrap_or(line.len())];
        if field == b"data" {
            let mut value = colon.map_or(&[][..], |colon| &line[colon + 1..]);
            if value.first() == Some(&b' ') {
                value = &value[1..];
            }
            self.data.extend_from_slice(value);
            self.data.push(b'\n');
            self.saw_data = true;
        }
    }

    fn parse_input(&mut self, eof: bool) {
        let mut consumed = 0;
        while let Some(line_end) = self.input[consumed..]
            .iter()
            .position(|byte| matches!(byte, b'\r' | b'\n'))
            .map(|offset| consumed + offset)
        {
            let ending_len = if self.input[line_end] == b'\r' {
                if self.input.get(line_end + 1) == Some(&b'\n') {
                    2
                } else if line_end + 1 < self.input.len() || eof {
                    1
                } else {
                    break;
                }
            } else {
                1
            };
            let line = self.input[consumed..line_end].to_vec();
            self.process_line(&line);
            consumed = line_end + ending_len;
        }

        if eof && consumed < self.input.len() {
            let line = self.input[consumed..].to_vec();
            self.process_line(&line);
            consumed = self.input.len();
        }
        self.input.drain(..consumed);
        if eof {
            self.finish_event();
        }
    }
}

/// How long a stream may sit idle after a finish reason before it is closed on the provider's behalf: some
/// servers send `finish_reason` and then neither `[DONE]` nor end-of-body, which would otherwise hang the turn
/// until the user interrupts.
const FINISH_GRACE: Duration = Duration::from_secs(10);

fn finish_reason_is_truncation(reason: &str) -> Result<bool, BackendError> {
    match reason {
        "stop" | "tool_calls" | "function_call" => Ok(false),
        "length" | "content_filter" => Ok(true),
        _ => Err(BackendError::Protocol {
            message: format!("unknown finish reason from provider: {reason}"),
        }),
    }
}

/// A provider `{"error": ...}` object delivered as a stream event; there is no HTTP status to attach it to.
#[derive(Deserialize)]
struct ProviderError {
    error: ProviderErrorMessage,
}

#[derive(Deserialize)]
struct ProviderErrorMessage {
    message: String,
}

struct ChatEventStream {
    response: Option<reqwest::Response>,
    decoder: SseDecoder,
    /// Set once the terminal event (or error) has been emitted; the stream ends afterwards.
    finished: bool,
    finish_reason: Option<String>,
    /// Whether any tool-call delta was seen, so truncation mid-tool-call can be treated as an error.
    has_tool_calls: bool,
    /// A truncation notice to emit as one final content delta before the stream finishes.
    truncation: Option<String>,
    eof: bool,
}

impl ChatEventStream {
    fn into_stream(response: reqwest::Response, pending: Vec<u8>) -> BackendStream {
        let mut events = Self {
            response: Some(response),
            decoder: SseDecoder::default(),
            finished: false,
            finish_reason: None,
            has_tool_calls: false,
            truncation: None,
            eof: false,
        };
        events.decoder.feed(&pending);
        Box::pin(stream::unfold(events, |mut events| async move {
            events.next_event().await.map(|event| (event, events))
        }))
    }

    async fn next_event(&mut self) -> Option<Result<BackendEvent, BackendError>> {
        loop {
            if self.finished {
                return None;
            }
            if let Some(truncation) = self.truncation.take() {
                return Some(Ok(BackendEvent {
                    content: Some(truncation),
                    ..Default::default()
                }));
            }
            if let Some(data) = self.decoder.next_event() {
                match self.process_event(data) {
                    Ok(Some(event)) => return Some(Ok(event)),
                    Ok(None) => {}
                    Err(error) => {
                        self.finished = true;
                        return Some(Err(error));
                    }
                }
                continue;
            }
            if self.eof {
                self.finished = true;
                if self.finish_reason.is_some() {
                    // The server completed the message but skipped `[DONE]`; accept its finish reason.
                    return Some(Ok(BackendEvent {
                        finished: true,
                        ..Default::default()
                    }));
                }
                return Some(Err(BackendError::Protocol {
                    message: "stream ended before it finished".into(),
                }));
            }

            let chunk = if self.finish_reason.is_some() {
                match tokio::time::timeout(FINISH_GRACE, self.read_chunk()).await {
                    Ok(chunk) => chunk,
                    Err(_) => {
                        self.eof = true;
                        // A finish reason already endorses completion, so a partial trailing event (say half
                        // of "[DONE]") is discarded rather than flushed and reported as malformed.
                        self.decoder.discard_partial();
                        continue;
                    }
                }
            } else {
                self.read_chunk().await
            };
            match chunk {
                Ok(Some(chunk)) => self.decoder.feed(&chunk),
                Ok(None) => {
                    self.eof = true;
                    self.decoder.finish();
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(BackendError::Transport {
                        message: error.to_string(),
                    }));
                }
            }
        }
    }

    async fn read_chunk(&mut self) -> Result<Option<bytes::Bytes>, reqwest::Error> {
        match self.response.as_mut() {
            Some(response) => response.chunk().await,
            // An absent response has no more bytes: it was handed to the drain task or never existed
            // (tests drive the decoder directly).
            None => Ok(None),
        }
    }

    /// After a clean `[DONE]`, reads the body to its end in the background so the connection returns to the
    /// pool instead of being dropped mid-body (one fewer TCP+TLS handshake per turn). Streams that ended at
    /// EOF, through the finish grace, or in an error keep their connections dropped: their bodies are
    /// exhausted, unwanted, or suspect.
    fn drain(&mut self) {
        if self.eof {
            return;
        }
        if let Some(mut response) = self.response.take() {
            tokio::spawn(async move {
                // A server that sends [DONE] but never ends its body must not pin a task and a socket for
                // the life of the process, so the drain gives up after the finish grace.
                while matches!(
                    tokio::time::timeout(FINISH_GRACE, response.chunk()).await,
                    Ok(Ok(Some(_)))
                ) {}
            });
        }
    }

    fn process_event(
        &mut self,
        data: Result<String, std::str::Utf8Error>,
    ) -> Result<Option<BackendEvent>, BackendError> {
        let data = data.map_err(|_| BackendError::Protocol {
            message: "event is not valid UTF-8".into(),
        })?;
        if data.is_empty() {
            // An empty `data:` event is a legal SSE keep-alive.
            return Ok(None);
        }
        if data == "[DONE]" {
            self.finished = true;
            self.drain();
            return Ok(Some(BackendEvent {
                finished: true,
                ..Default::default()
            }));
        }

        // The provider error shape is checked first: with `choices` defaulted, an error object would
        // otherwise parse as an empty chunk and the provider's report would be swallowed.
        if let Ok(error) = serde_json::from_str::<ProviderError>(&data) {
            return Err(BackendError::Protocol {
                message: format!("provider reported: {}", error.error.message),
            });
        }
        let chunk = match serde_json::from_str::<ChatChunk>(&data) {
            Ok(chunk) => chunk,
            Err(_) => {
                let excerpt = data.chars().take(120).collect::<String>();
                return Err(BackendError::Protocol {
                    message: format!("malformed event: {excerpt}"),
                });
            }
        };
        // TODO: Distinguish interim usage from final accounting if providers report usage on each delta.
        let usage = chunk.usage.map(TokenUsage::from);
        let Some(choice) = chunk.choices.into_iter().find(|choice| choice.index == 0) else {
            if usage.is_some() {
                return Ok(Some(BackendEvent {
                    usage,
                    ..Default::default()
                }));
            }
            return Ok(None);
        };
        self.has_tool_calls |= !choice.delta.tool_calls.is_empty();
        let tool_calls = choice
            .delta
            .tool_calls
            .into_iter()
            .map(StreamToolCall::into_delta)
            .collect();
        if let Some(reason) = choice.finish_reason.as_deref()
            && self.finish_reason.is_none()
        {
            let is_truncation = finish_reason_is_truncation(reason)?;
            self.finish_reason = Some(reason.to_owned());
            if is_truncation {
                if self.has_tool_calls {
                    // Truncated tool-call arguments must never enter history as a real call.
                    return Err(BackendError::Protocol {
                        message: format!(
                            "response truncated by the provider mid-tool-call ({reason})"
                        ),
                    });
                }
                self.truncation = Some(format!("\n[response truncated by the provider: {reason}]"));
            }
        }
        Ok(Some(BackendEvent {
            reasoning: choice.delta.reasoning,
            content: choice.delta.content,
            tool_calls,
            usage,
            // Some providers send usage in a separate chunk after finish_reason. Wait for [DONE] so that
            // chunk is not discarded.
            finished: false,
        }))
    }
}

fn message_to_value(message: &Message) -> Value {
    match message {
        Message::System(content) => json!({
            "role": "system",
            "content": content,
        }),
        Message::User(content) => json!({
            "role": "user",
            "content": content,
        }),
        Message::Assistant(message) => {
            let mut value = Map::from_iter([
                ("role".into(), Value::String("assistant".into())),
                (
                    "content".into(),
                    message
                        .content
                        .as_ref()
                        .map_or(Value::Null, |content| Value::String(content.clone())),
                ),
            ]);

            if !message.tool_calls.is_empty() {
                value.insert(
                    "tool_calls".into(),
                    Value::Array(message.tool_calls.iter().map(tool_call_to_value).collect()),
                );
            }

            if let Some(reasoning) = &message.reasoning {
                value.insert("reasoning".into(), Value::String(reasoning.clone()));
            }

            Value::Object(value)
        }
        Message::Tool(_) => unreachable!(),
    }
}

fn tool_call_to_value(call: &ToolCall) -> Value {
    match call.kind {
        ToolCallKind::Function => json!({
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function_name,
                "arguments": call.arguments,
            },
        }),
        ToolCallKind::Custom => json!({
            "id": call.id,
            "type": "custom",
            "custom": {
                "name": call.function_name,
                "input": call.arguments,
            },
        }),
    }
}

fn tool_to_value(tool: &ToolDefinition) -> Value {
    match tool {
        ToolDefinition::Function(function) => json!({
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": function.parameters,
                "strict": function.strict,
            },
        }),
        ToolDefinition::Freeform {
            name,
            description,
            grammar,
        } => {
            let mut custom = Map::from_iter([
                ("name".into(), Value::String(name.clone())),
                ("description".into(), Value::String(description.clone())),
            ]);

            // An absent format leaves the input unconstrained free-form text.
            if let Some(definition) = grammar {
                custom.insert(
                    "format".into(),
                    json!({
                        "type": "grammar",
                        "grammar": { "definition": definition, "syntax": "lark" },
                    }),
                );
            }

            json!({
                "type": "custom",
                "custom": custom,
            })
        }
    }
}

#[derive(Deserialize)]
struct ChatChunk {
    // Some servers omit `choices` entirely on the final usage-only event instead of sending an empty array.
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

impl From<ChatUsage> for TokenUsage {
    fn from(usage: ChatUsage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

#[derive(Default, Deserialize)]
struct ChunkDelta {
    content: Option<String>,
    #[serde(alias = "reasoning_content")]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<StreamToolCall>,
}

#[derive(Deserialize)]
struct StreamToolCall {
    index: usize,
    id: Option<String>,
    function: Option<StreamFunctionCall>,
    custom: Option<StreamCustomToolCall>,
}

#[derive(Deserialize)]
struct StreamFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct StreamCustomToolCall {
    name: Option<String>,
    input: Option<String>,
}

impl StreamToolCall {
    /// Custom tool calls are detected by the presence of their `custom` object; the streaming schema only documents
    /// `function` deltas, so providers may omit an explicit `type` discriminant. Function deltas therefore report no
    /// kind and leave the call's default (function) kind in place.
    fn into_delta(self) -> ToolCallDelta {
        if let Some(custom) = self.custom {
            return ToolCallDelta {
                index: self.index,
                id: self.id,
                kind: Some(ToolCallKind::Custom),
                function_name: custom.name,
                arguments: custom.input,
            };
        }

        let (function_name, arguments) = self
            .function
            .map_or((None, None), |function| (function.name, function.arguments));

        ToolCallDelta {
            index: self.index,
            id: self.id,
            kind: None,
            function_name,
            arguments,
        }
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

impl ChatResponse {
    fn into_assistant_message(self) -> Result<AssistantMessage, BackendError> {
        let choice = self
            .choices
            .into_iter()
            .next()
            .ok_or(BackendError::Protocol {
                message: "response contained no choices".into(),
            })?;
        let message = choice.message;
        let mut tool_calls = Vec::with_capacity(message.tool_calls.len());
        for call in message.tool_calls {
            tool_calls.push(call.into_tool_call()?);
        }
        let mut assistant = AssistantMessage {
            reasoning: message.reasoning,
            content: message.content,
            tool_calls,
        };
        // The same truncation rule as the streaming path: a marker the model and user can see, unless tool
        // calls were cut short, in which case the response cannot be trusted at all.
        if let Some(reason) = choice.finish_reason.as_deref()
            && finish_reason_is_truncation(reason)?
        {
            if !assistant.tool_calls.is_empty() {
                return Err(BackendError::Protocol {
                    message: format!("response truncated by the provider mid-tool-call ({reason})"),
                });
            }
            assistant
                .content
                .get_or_insert_with(String::new)
                .push_str(&format!("\n[response truncated by the provider: {reason}]"));
        }
        Ok(assistant)
    }
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

#[derive(Deserialize)]
struct ResponseToolCall {
    id: String,
    function: Option<ResponseFunctionCall>,
    custom: Option<ResponseCustomToolCall>,
}

impl ResponseToolCall {
    fn into_tool_call(self) -> Result<ToolCall, BackendError> {
        let (kind, function_name, arguments) = match (self.function, self.custom) {
            (Some(function), _) => (ToolCallKind::Function, function.name, function.arguments),
            (None, Some(custom)) => (ToolCallKind::Custom, custom.name, custom.input),
            // A silently dropped call would end the turn as an empty, successful assistant message.
            (None, None) => {
                return Err(BackendError::Protocol {
                    message: format!(
                        "tool call {} has neither function nor custom payload",
                        self.id
                    ),
                });
            }
        };

        Ok(ToolCall {
            id: self.id,
            kind,
            function_name,
            arguments,
        })
    }
}

#[derive(Deserialize)]
struct ResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct ResponseCustomToolCall {
    name: String,
    input: String,
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::test_support::with_messages;
    use super::*;
    use crate::{FunctionDefinition, ToolResult};

    const TOOL_NAME: &str = "foo";

    fn dummy_function() -> FunctionDefinition {
        FunctionDefinition {
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
        }
    }

    #[test]
    fn converts_all_request_message_types_and_functions() {
        let call = ToolCall {
            id: "call-1".into(),
            kind: ToolCallKind::Function,
            function_name: "test_function".into(),
            arguments: r#"{"input":"value"}"#.into(),
        };
        let messages = vec![
            Message::System("Use functions when needed.".into()),
            Message::User("Call the test function.".into()),
            Message::Assistant(AssistantMessage {
                reasoning: Some("Need to call the function.".into()),
                content: None,
                tool_calls: vec![call],
            }),
            Message::Tool(vec![ToolResult {
                tool_call_id: "call-1".into(),
                content: "result".into(),
            }]),
        ];
        let functions = vec![ToolDefinition::Function(dummy_function())];
        let backend = OpenAiBackend::new(Config {
            model: "test-model".into(),
            base_url: "".into(),
            api_key: None,
            reasoning_effort: Some("high".into()),
            extra_body: Some(json!({
                "stream": false,
                "chat_template_kwargs": {
                    "enable_thinking": true
                }
            })),
            ..Config::default()
        });

        let body = with_messages(messages, |messages| {
            backend.request_body(
                BackendRequest {
                    messages,
                    tools: &functions,
                },
                true,
            )
        })
        .unwrap();

        assert_eq!(
            body,
            json!({
                "model": "test-model",
                "messages": [
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
                        "content": null,
                        "reasoning": "Need to call the function.",
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "test_function",
                                "arguments": "{\"input\":\"value\"}"
                            }
                        }]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "content": "result"
                    }
                ],
                "stream": true,
                "stream_options": {
                    "include_usage": true
                },
                "reasoning_effort": "high",
                "chat_template_kwargs": {
                    "enable_thinking": true
                },
                "tools": [{
                    "type": "function",
                    "function": {
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
                    }
                }]
            })
        );
    }

    #[test]
    fn extra_body_that_is_not_an_object_is_a_configuration_error() {
        let backend = OpenAiBackend::new(Config {
            model: "test-model".into(),
            extra_body: Some(json!(1)),
            ..Config::default()
        });

        let error = with_messages([], |messages| {
            backend.request_body(
                BackendRequest {
                    messages,
                    tools: &[],
                },
                false,
            )
        })
        .unwrap_err();

        assert_eq!(
            error,
            BackendError::Config {
                message: "provider extra_body must be a JSON object, found a number".into()
            }
        );
    }

    #[test]
    fn parses_nullable_content_and_multiple_raw_function_calls() {
        let response = serde_json::from_value::<ChatResponse>(json!({
            "choices": [{
                "message": {
                    "content": null,
                    "reasoning": "checking available functions",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "first",
                                "arguments": "not json"
                            }
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "second",
                                "arguments": "{\"valid\":true}"
                            }
                        }
                    ]
                }
            }]
        }))
        .unwrap()
        .into_assistant_message()
        .unwrap();

        assert_eq!(
            response,
            AssistantMessage {
                reasoning: Some("checking available functions".into()),
                content: None,
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
                ],
            }
        );
    }

    #[test]
    fn invalid_chat_responses_are_rejected() {
        let cases = [
            (
                json!({ "choices": [] }),
                BackendError::Protocol {
                    message: "response contained no choices".into(),
                },
            ),
            (
                json!({
                    "choices": [{
                        "message": {
                            "tool_calls": [{ "id": "call-1" }]
                        }
                    }]
                }),
                BackendError::Protocol {
                    message: "tool call call-1 has neither function nor custom payload".into(),
                },
            ),
        ];

        for (value, expected) in cases {
            let error = serde_json::from_value::<ChatResponse>(value)
                .unwrap()
                .into_assistant_message()
                .unwrap_err();
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn retryable_statuses_are_408_429_and_server_errors() {
        let retryable: Vec<u16> = [408, 429, 500, 502, 503, 504, 520, 529].into();
        let terminal = [200, 301, 400, 401, 403, 404, 409, 413, 422];
        for status in retryable
            .iter()
            .chain(&(500..=599).collect::<Vec<u16>>())
            .map(|status| StatusCode::from_u16(*status).unwrap())
        {
            assert!(is_retryable_status(status), "{status}");
        }
        for status in terminal
            .iter()
            .map(|status| StatusCode::from_u16(*status).unwrap())
        {
            assert!(!is_retryable_status(status), "{status}");
        }
    }

    #[test]
    fn computed_backoff_doubles_and_is_jittered_within_bounds() {
        let policy = RetryPolicy {
            max_retries: 4,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
        };

        for retry in 0..4 {
            let delay = policy.delay(retry, None).unwrap();
            let base = Duration::from_secs(1) * (1_u32 << retry);
            assert!(
                delay >= base / 2 && delay <= base.mul_f64(1.5),
                "retry {retry}: {delay:?} not within [half, 1.5x] of {base:?}"
            );
        }
        // The exponential growth saturates: every delay stays at or under max_delay.
        for retry in 4..10 {
            assert!(policy.delay(retry, None).unwrap() <= Duration::from_secs(30));
        }
    }

    #[test]
    fn retry_after_is_honored_verbatim_and_rejected_when_too_long() {
        let policy = RetryPolicy {
            max_retries: 4,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
        };

        assert_eq!(
            policy.delay(0, Some(Duration::from_secs(5))),
            Some(Duration::from_secs(5))
        );
        assert_eq!(policy.delay(0, Some(Duration::ZERO)), Some(Duration::ZERO));
        // A server demanding a longer window than the policy ever sleeps cannot be retried into success.
        assert_eq!(policy.delay(0, Some(Duration::from_secs(31))), None);
    }

    async fn decode(chunks: &[&[u8]]) -> Vec<Result<String, std::str::Utf8Error>> {
        let mut decoder = SseDecoder::default();
        for chunk in chunks {
            decoder.feed(chunk);
        }
        decoder.finish();
        std::iter::from_fn(|| decoder.next_event()).collect()
    }

    #[tokio::test]
    async fn sse_decoder_handles_transport_boundaries_and_event_fields() {
        let events = decode(&[
            b": keepalive\r".as_slice(),
            b"\ndata: first\r\ndata: sec".as_slice(),
            b"ond\r\nid: ignored\r\n\r".as_slice(),
            b"\ndata: final".as_slice(),
        ])
        .await;

        assert_eq!(events, [Ok("first\nsecond".into()), Ok("final".into())]);
    }

    #[tokio::test]
    async fn non_utf8_events_surface_as_decode_errors() {
        let events = decode(&[b"data: \xff\xfe\n\n".as_slice()]).await;

        assert!(events[0].is_err());
    }

    /// Drives the event-processing state machine from raw SSE bytes, as if the response body had delivered
    /// them all and ended.
    async fn collect_stream(data: &[u8]) -> Vec<Result<BackendEvent, BackendError>> {
        let mut events = ChatEventStream {
            response: None,
            decoder: SseDecoder::default(),
            finished: false,
            finish_reason: None,
            has_tool_calls: false,
            truncation: None,
            eof: false,
        };
        events.decoder.feed(data);
        let stream: BackendStream = Box::pin(stream::unfold(events, |mut events| async move {
            events.next_event().await.map(|event| (event, events))
        }));
        stream.collect().await
    }

    #[tokio::test]
    async fn streaming_chunks_map_content_reasoning_tools_and_finish() {
        let data = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"think\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\",\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"foo\",\"arguments\":\"{\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream(data.as_bytes()).await;

        assert_eq!(
            events,
            [
                Ok(BackendEvent {
                    reasoning: Some("think".into()),
                    ..Default::default()
                }),
                Ok(BackendEvent {
                    content: Some("answer".into()),
                    tool_calls: vec![ToolCallDelta {
                        index: 0,
                        id: Some("call-1".into()),
                        kind: None,
                        function_name: Some(TOOL_NAME.into()),
                        arguments: Some("{".into()),
                    }],
                    ..Default::default()
                }),
                Ok(BackendEvent {
                    tool_calls: vec![ToolCallDelta {
                        index: 0,
                        id: None,
                        kind: None,
                        function_name: None,
                        arguments: Some("}".into()),
                    }],
                    ..Default::default()
                }),
                Ok(BackendEvent {
                    usage: Some(TokenUsage {
                        input_tokens: 12,
                        output_tokens: 3,
                        total_tokens: 15,
                    }),
                    ..Default::default()
                }),
                Ok(BackendEvent {
                    finished: true,
                    ..Default::default()
                }),
            ]
        );
    }

    #[tokio::test]
    async fn usage_events_are_preserved_with_or_without_choices() {
        let cases = [
            (
                concat!(
                    "data: {\"choices\":[{\"index\":0,\"delta\":{}}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\"total_tokens\":15}}\n\n",
                    "data: [DONE]\n\n",
                ),
                vec![
                    Ok(BackendEvent {
                        usage: Some(TokenUsage {
                            input_tokens: 12,
                            output_tokens: 3,
                            total_tokens: 15,
                        }),
                        ..Default::default()
                    }),
                    Ok(BackendEvent {
                        finished: true,
                        ..Default::default()
                    }),
                ],
            ),
            (
                concat!(
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: {\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n",
                    "data: [DONE]\n\n",
                ),
                vec![
                    Ok(BackendEvent {
                        content: Some("ok".into()),
                        ..Default::default()
                    }),
                    Ok(BackendEvent {
                        usage: Some(TokenUsage {
                            input_tokens: 7,
                            output_tokens: 2,
                            total_tokens: 9,
                        }),
                        ..Default::default()
                    }),
                    Ok(BackendEvent {
                        finished: true,
                        ..Default::default()
                    }),
                ],
            ),
        ];

        for (data, expected) in cases {
            assert_eq!(collect_stream(data.as_bytes()).await, expected);
        }
    }

    #[tokio::test]
    async fn empty_data_lines_and_events_without_a_choice_are_skipped() {
        let data = concat!(
            ": ping\n\n",
            "data:\n\n",
            "data: {\"choices\":[],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream(data.as_bytes()).await;

        assert_eq!(
            events,
            [
                Ok(BackendEvent {
                    content: Some("ok".into()),
                    ..Default::default()
                }),
                Ok(BackendEvent {
                    finished: true,
                    ..Default::default()
                }),
            ]
        );
    }

    #[tokio::test]
    async fn finish_reasons_publish_completion_with_or_without_a_delta() {
        let cases = [
            (
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n",
                vec![
                    Ok(BackendEvent {
                        content: Some("answer".into()),
                        ..Default::default()
                    }),
                    Ok(BackendEvent {
                        finished: true,
                        ..Default::default()
                    }),
                ],
            ),
            (
                concat!(
                    "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
                    "data: [DONE]\n\n",
                ),
                vec![
                    Ok(BackendEvent::default()),
                    Ok(BackendEvent {
                        finished: true,
                        ..Default::default()
                    }),
                ],
            ),
        ];

        for (data, expected) in cases {
            assert_eq!(collect_stream(data.as_bytes()).await, expected);
        }
    }

    #[tokio::test]
    async fn a_stream_ending_without_a_finish_reason_or_done_is_an_error() {
        let events = collect_stream(
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
        )
        .await;

        assert_eq!(
            events,
            [
                Ok(BackendEvent {
                    content: Some("partial".into()),
                    ..Default::default()
                }),
                Err(BackendError::Protocol {
                    message: "stream ended before it finished".into()
                }),
            ]
        );
    }

    #[tokio::test]
    async fn length_truncation_appends_a_marker_to_the_content() {
        let data = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"the ans\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream(data.as_bytes()).await;

        assert_eq!(
            events,
            [
                Ok(BackendEvent {
                    content: Some("the ans".into()),
                    ..Default::default()
                }),
                Ok(BackendEvent::default()),
                Ok(BackendEvent {
                    content: Some("\n[response truncated by the provider: length]".into()),
                    ..Default::default()
                }),
                Ok(BackendEvent {
                    finished: true,
                    ..Default::default()
                }),
            ]
        );
    }

    #[tokio::test]
    async fn length_truncation_mid_tool_call_is_an_error() {
        let data = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"foo\",\"arguments\":\"{\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream(data.as_bytes()).await;

        assert_eq!(
            events,
            [
                Ok(BackendEvent {
                    tool_calls: vec![ToolCallDelta {
                        index: 0,
                        id: Some("call-1".into()),
                        kind: None,
                        function_name: Some(TOOL_NAME.into()),
                        arguments: Some("{".into()),
                    }],
                    ..Default::default()
                }),
                Err(BackendError::Protocol {
                    message: "response truncated by the provider mid-tool-call (length)".into()
                }),
            ]
        );
    }

    #[tokio::test]
    async fn a_provider_error_event_ends_the_stream_with_that_error() {
        let data = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n",
            "data: {\"error\":{\"message\":\"insufficient quota\",\"type\":\"insufficient_quota\"}}\n\n",
        );
        let events = collect_stream(data.as_bytes()).await;

        assert_eq!(
            events,
            [
                Ok(BackendEvent {
                    content: Some("par".into()),
                    ..Default::default()
                }),
                Err(BackendError::Protocol {
                    message: "provider reported: insufficient quota".into()
                }),
            ]
        );
    }

    #[tokio::test]
    async fn malformed_stream_events_are_rejected() {
        let cases = [
            (
                b"data: not json\n\n".as_slice(),
                "malformed event: not json",
            ),
            (b"data: \xff\xfe\n\n".as_slice(), "event is not valid UTF-8"),
        ];

        for (data, message) in cases {
            assert_eq!(
                collect_stream(data).await,
                [Err(BackendError::Protocol {
                    message: message.into(),
                })]
            );
        }
    }

    /// Serves scripted HTTP/1.1 responses on a background thread: each connection is read to its request end
    /// (headers plus any Content-Length body), answered with the next canned response, and closed.
    struct ScriptedServer {
        address: std::net::SocketAddr,
        requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ScriptedServer {
        fn spawn(responses: Vec<String>) -> Self {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let requests = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counter = std::sync::Arc::clone(&requests);
            std::thread::spawn(move || {
                for response in responses {
                    let (mut stream, _) = listener.accept().unwrap();
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    read_request(&stream);
                    use std::io::Write as _;
                    stream.write_all(response.as_bytes()).unwrap();
                }
            });
            Self { address, requests }
        }

        fn url(&self) -> String {
            format!("http://{}/v1", self.address)
        }
    }

    /// Reads one full request (headers plus any Content-Length body) so the response is not written into a
    /// receive buffer the client has not drained yet.
    fn read_request(stream: &std::net::TcpStream) {
        use std::io::{BufRead, Read};

        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.trim_end().is_empty() {
                break;
            }
            let lowered = line.to_ascii_lowercase();
            let Some(value) = lowered.strip_prefix("content-length:").map(str::trim) else {
                continue;
            };
            content_length = value.parse().unwrap();
        }
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body).unwrap();
    }

    fn http_response(status: &str, headers: &[(&str, &str)], body: &str) -> String {
        let mut response = format!("HTTP/1.1 {status}\r\n");
        for (name, value) in headers {
            response.push_str(&format!("{name}: {value}\r\n"));
        }
        response.push_str(&format!(
            "Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ));
        response
    }

    fn fast_policy() -> RetryPolicy {
        RetryPolicy {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
        }
    }

    fn backend_for(url: String) -> OpenAiBackend {
        OpenAiBackend::new(Config {
            model: "test-model".into(),
            base_url: url,
            api_key: None,
            retries: fast_policy(),
            ..Config::default()
        })
    }

    #[tokio::test]
    async fn retries_before_first_response_byte_and_reports_terminal_errors() {
        let cases = [
            (
                vec![
                    http_response(
                        "429 Too Many Requests",
                        &[("Retry-After", "0")],
                        "{\"error\":{\"message\":\"rate limited\"}}",
                    ),
                    http_response(
                        "200 OK",
                        &[("Content-Type", "application/json")],
                        "{\"choices\":[{\"message\":{\"content\":\"hello\"}}]}",
                    ),
                ],
                Ok(AssistantMessage {
                    reasoning: None,
                    content: Some("hello".into()),
                    tool_calls: Vec::new(),
                }),
                2,
            ),
            (
                vec![
                    String::new(),
                    http_response(
                        "200 OK",
                        &[("Content-Type", "application/json")],
                        "{\"choices\":[{\"message\":{\"content\":\"hello\"}}]}",
                    ),
                ],
                Ok(AssistantMessage {
                    reasoning: None,
                    content: Some("hello".into()),
                    tool_calls: Vec::new(),
                }),
                2,
            ),
            (
                vec![
                    http_response("503 Service Unavailable", &[], "busy"),
                    http_response("503 Service Unavailable", &[], "busy"),
                    http_response("503 Service Unavailable", &[], "busy"),
                ],
                Err(BackendError::Http {
                    status: 503,
                    message: "busy".into(),
                }),
                3,
            ),
            (
                vec![http_response(
                    "429 Too Many Requests",
                    &[("Retry-After", "30")],
                    "{\"error\":{\"message\":\"rate limited\"}}",
                )],
                Err(BackendError::Http {
                    status: 429,
                    message: "rate limited (retry after 30s)".into(),
                }),
                1,
            ),
            (
                vec![http_response(
                    "401 Unauthorized",
                    &[],
                    "{\"error\":{\"message\":\"bad key\"}}",
                )],
                Err(BackendError::Http {
                    status: 401,
                    message: "bad key".into(),
                }),
                1,
            ),
        ];

        for (responses, expected, request_count) in cases {
            let server = ScriptedServer::spawn(responses);
            let mut backend = backend_for(server.url());
            let result = with_messages([Message::User("hi".into())], |messages| {
                backend.request(BackendRequest {
                    messages,
                    tools: &[],
                })
            })
            .await;

            assert_eq!(result, expected);
            assert_eq!(
                server.requests.load(std::sync::atomic::Ordering::SeqCst),
                request_count
            );
        }
    }

    #[tokio::test]
    async fn streaming_requests_retry_before_first_byte_but_not_after() {
        let cases = [
            (
                vec![
                    // A complete 200 whose body is empty: the connection proved dead before any data arrived.
                    http_response("200 OK", &[("Content-Type", "text/event-stream")], ""),
                    http_response(
                        "200 OK",
                        &[("Content-Type", "text/event-stream")],
                        concat!(
                            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        ),
                    ),
                ],
                vec![
                    Ok(BackendEvent {
                        content: Some("hi".into()),
                        ..Default::default()
                    }),
                    Ok(BackendEvent {
                        finished: true,
                        ..Default::default()
                    }),
                ],
                2,
            ),
            (
                vec![http_response(
                    "200 OK",
                    &[("Content-Type", "text/event-stream")],
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n",
                )],
                vec![
                    Ok(BackendEvent {
                        content: Some("par".into()),
                        ..Default::default()
                    }),
                    Err(BackendError::Protocol {
                        message: "stream ended before it finished".into(),
                    }),
                ],
                1,
            ),
        ];

        for (responses, expected, request_count) in cases {
            let server = ScriptedServer::spawn(responses);
            let mut backend = backend_for(server.url());
            let events = with_messages([Message::User("hi".into())], |messages| {
                backend.request_stream(BackendRequest {
                    messages,
                    tools: &[],
                })
            })
            .collect::<Vec<_>>()
            .await;

            assert_eq!(events, expected);
            assert_eq!(
                server.requests.load(std::sync::atomic::Ordering::SeqCst),
                request_count
            );
        }
    }
}
