use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    System(String),
    User(String),
    Assistant(AssistantMessage),
    Tool(Vec<ToolResult>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssistantMessage {
    pub reasoning: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCallKind {
    /// A call to a JSON-schema function tool.
    Function,
    /// A call to a freeform custom tool whose input is raw text.
    Custom,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub kind: ToolCallKind,
    pub function_name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolDefinition {
    Function(FunctionDefinition),
    /// A freeform tool whose input is raw text rather than JSON arguments.
    Freeform {
        name: String,
        description: String,
        /// Optional Lark/regex grammar constraining the generated input.
        grammar: Option<String>,
    },
}
