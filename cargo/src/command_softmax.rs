use nu_plugin::PluginCommand;
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::get_kind_from_call;
use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

pub struct CommandSoftmax;

impl PluginCommand for CommandSoftmax {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch softmax"
    }

    fn description(&self) -> &str {
        "Compute the softmax of a tensor along a specified dimension. (similar to tensor.softmax() in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch softmax")
            // tensor id may come from pipeline or from a single argument
            .input_output_types(vec![
                (Type::String, Type::String),  // pipeline-in
                (Type::Nothing, Type::String), // arg-in
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "ID of the tensor (if not supplied by pipeline)",
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to compute softmax (default: last dimension)",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the output tensor (default: inherits input dtype)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Compute softmax over the last dimension (pipeline input)",
                example: "let t = (torch linspace 0 5 6 | torch repeat 2 1); $t | torch softmax | torch value",
                result: None,
            },
            Example {
                description: "Compute softmax along dim 1 (argument input)",
                example: "let t = (torch linspace 0 5 6 | torch repeat 2 1); torch softmax $t --dim 1 | torch value",
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // ── dual input: pipeline OR argument (not both) ───────────────
        // Supports: $t | torch softmax   OR   torch softmax $t
        let piped = match input {
            PipelineData::Empty => None,
            PipelineData::Value(v, _) => Some(v),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or single Value inputs are supported", call.head))
            }
        };
        let arg0 = call.nth(0);

        // ── extract tensor ID from exactly one source ─────────────────
        let tensor_id = match (piped, arg0) {
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide tensor ID via pipeline OR argument, not both",
                    call.head,
                ))
            }
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Tensor ID must be supplied via pipeline or argument",
                    call.head,
                ))
            }
            (Some(v), None) => v.as_str().map(|s| s.to_string()).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Pipeline input must be a tensor ID (string)", call.head)
            })?,
            (None, Some(a)) => a.as_str().map(|s| s.to_string()).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Argument must be a tensor ID (string)", call.head)
            })?,
        };

        // ── fetch tensor from registry ────────────────────────────────
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor = registry
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        // ── optional dtype flag ───────────────────────────────────────
        let kind = get_kind_from_call(call)?;

        // ── optional dim flag (default: last dimension) ───────────────
        let dim = match call.get_flag::<i64>("dim")? {
            Some(d) => {
                let n = tensor.size().len() as i64;
                if d < 0 || d >= n {
                    return Err(LabeledError::new("Invalid dimension").with_label(
                        format!("Dimension {d} out of bounds for tensor with {n} dimensions"),
                        call.head,
                    ));
                }
                d
            }
            None => (tensor.size().len() as i64) - 1,
        };

        // ── compute softmax ───────────────────────────────────────────
        let result_tensor = tensor.softmax(dim, kind);

        // ── store & return ────────────────────────────────────────────
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}