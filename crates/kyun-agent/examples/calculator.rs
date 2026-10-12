use std::env;

use kyun_agent::{
    Agent, FunctionDefinition, Message, StepEvent, ToolDefinition, ToolResult,
    openai::{Config, OpenAiBackend},
};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct CalculatorArguments {
    left: f64,
    operator: String,
    right: f64,
}

fn calculator_definition() -> FunctionDefinition {
    FunctionDefinition {
        name: "calculate".into(),
        description: "Evaluate one arithmetic operation.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "left": { "type": "number" },
                "operator": { "type": "string", "enum": ["+", "-", "*", "/"] },
                "right": { "type": "number" }
            },
            "required": ["left", "operator", "right"],
            "additionalProperties": false
        }),
        strict: true,
    }
}

fn calculate(arguments: &CalculatorArguments) -> f64 {
    match arguments.operator.as_str() {
        "+" => arguments.left + arguments.right,
        "-" => arguments.left - arguments.right,
        "*" => arguments.left * arguments.right,
        "/" => arguments.left / arguments.right,
        operator => panic!("unsupported operator: {operator}"),
    }
}

#[tokio::main]
async fn main() {
    pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
    let model = env::var("OPENAI_MODEL").unwrap();
    let api_key = env::var("OPENAI_API_KEY").ok();
    let base_url = env::var("OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
    let backend = OpenAiBackend::new(Config {
        model,
        base_url,
        api_key,
        ..Config::default()
    });
    let mut agent = Agent::with_messages(
        backend,
        [Message::System(
            "Use the calculator function for arithmetic.".into(),
        )],
    );
    let functions = [ToolDefinition::Function(calculator_definition())];
    let mut calculation_count = 0;

    agent.start_turn("Calculate 15 multiplied by 7.".into());

    for _ in 0..8 {
        match agent.step(&functions).await {
            StepEvent::Finished => break,
            StepEvent::Error(error) => {
                eprintln!("backend error: {error}");
                break;
            }
            StepEvent::ToolCalls => {
                // The assistant message that made the calls is the last message when the event arrives.
                let Message::Assistant(assistant) = agent.messages().last().unwrap() else {
                    unreachable!()
                };
                let results = assistant
                    .tool_calls
                    .iter()
                    .map(|call| {
                        assert_eq!(call.function_name, "calculate");
                        let arguments =
                            serde_json::from_str::<CalculatorArguments>(&call.arguments).unwrap();
                        calculation_count += 1;
                        ToolResult {
                            tool_call_id: call.id.clone(),
                            content: calculate(&arguments).to_string(),
                        }
                    })
                    .collect::<Vec<_>>();
                for result in results {
                    agent.append_tool_result(result);
                }
            }
            _ => unreachable!(),
        }
    }

    for message in agent.messages().iter() {
        println!("{message:#?}");
    }
    println!("calculator calls: {calculation_count}");
}
