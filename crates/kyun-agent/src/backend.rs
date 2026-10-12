use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{Stream, stream};

use crate::{AssistantMessage, MessageView, ToolCallKind, ToolDefinition};

/// How requests retry transient failures. Providers decide which failures are retryable and when a response
/// stream has progressed too far to retry safely.
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    /// Retries after the first attempt. Zero retries each request exactly once.
    pub max_retries: u32,
    /// The delay before the first retry; later retries double it.
    pub initial_delay: Duration,
    /// The largest delay ever slept, including one taken from `Retry-After`.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 4,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
        }
    }
}

impl RetryPolicy {
    /// The backoff delay before retry number `retry` (zero-based). Only computed backoff is jittered: a delay
    /// the server dictated through `Retry-After` is honored verbatim once it fits under `max_delay`.
    pub fn delay(&self, retry: u32, retry_after: Option<Duration>) -> Option<Duration> {
        match retry_after {
            // A server demanding longer than we are ever willing to wait cannot be retried into a window that
            // cannot have elapsed.
            Some(retry_after) if retry_after > self.max_delay => None,
            Some(retry_after) => Some(retry_after),
            None => {
                let base = self.initial_delay.saturating_mul(1_u32 << retry.min(16));
                // Equal-jitter-ish: keeps at least half of the delay so retrying clients stay polite, while
                // decorrelating identical clients that failed on the same request. The clamp is applied after
                // jittering so `max_delay` stays a true upper bound.
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|since_epoch| since_epoch.subsec_nanos())
                    .unwrap_or(0);
                let factor = 0.5 + f64::from(nanos % 1_000) / 1_000.0;
                Some(Duration::from_secs_f64(
                    (base.as_secs_f64() * factor).min(self.max_delay.as_secs_f64()),
                ))
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BackendRequest<'a> {
    pub messages: MessageView<'a>,
    pub tools: &'a [ToolDefinition],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

/// One provider chunk. Reasoning precedes content, though one boundary chunk may contain both. Fields are applied
/// in the order reasoning, content, tools, then usage. `finished` must occur exactly once on the final event.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BackendEvent {
    pub reasoning: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallDelta>,
    pub usage: Option<TokenUsage>,
    pub finished: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    /// The tool kind reported by this delta; `None` keeps the call's existing kind (defaulting to function).
    pub kind: Option<ToolCallKind>,
    pub function_name: Option<String>,
    pub arguments: Option<String>,
}

/// Why a backend request failed. Every path that used to panic on provider or network anomalies produces one of
/// these instead; retries have already been exhausted (or were never applicable) by the time an error is seen.
#[derive(Clone, Debug, PartialEq)]
pub enum BackendError {
    /// The provider answered with a non-success HTTP status. `message` is the provider's error body, if any.
    Http { status: u16, message: String },
    /// The request could not be sent or the response body could not be read (connect, TLS, timeout, mid-body IO).
    Transport { message: String },
    /// A successful response violated the protocol: malformed or undecodable events, a provider error event, a
    /// stream that ended before completing, or a response with no choices.
    Protocol { message: String },
    /// The backend's local configuration is invalid.
    Config { message: String },
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http { status, message } => {
                write!(f, "provider returned HTTP {status}: {message}")
            }
            Self::Transport { message } => write!(f, "network error: {message}"),
            Self::Protocol { message } => write!(f, "malformed response stream: {message}"),
            Self::Config { message } => write!(f, "invalid configuration: {message}"),
        }
    }
}

/// A stream of backend events. An `Err` item is terminal: the stream must end immediately after emitting it,
/// and consumers drop the stream without polling further.
pub type BackendStream =
    Pin<Box<dyn Stream<Item = Result<BackendEvent, BackendError>> + Send + 'static>>;

pub type BackendFuture =
    Pin<Box<dyn Future<Output = Result<AssistantMessage, BackendError>> + Send + 'static>>;

pub trait Backend: Send + 'static {
    fn request(&mut self, request: BackendRequest<'_>) -> BackendFuture;

    /// Returns a stream immediately; request setup runs when the stream is polled.
    /// Backends without incremental output emit the complete response as one final event.
    fn request_stream(&mut self, request: BackendRequest<'_>) -> BackendStream {
        let response = self.request(request);
        Box::pin(stream::once(async move {
            response.await.map(|assistant| BackendEvent {
                reasoning: assistant.reasoning,
                content: assistant.content,
                tool_calls: assistant
                    .tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, call)| ToolCallDelta {
                        index,
                        id: Some(call.id),
                        kind: Some(call.kind),
                        function_name: Some(call.function_name),
                        arguments: Some(call.arguments),
                    })
                    .collect(),
                usage: None,
                finished: true,
            })
        }))
    }
}
