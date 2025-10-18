use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

// Repeat command to replicate a tensor into a multidimensional structure
pub struct CommandRepeat;

impl PluginCommand for CommandRepeat {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch repeat"
    }

    fn description(&self) -> &str {
        "Repeat a tensor along each dimension. Automatically expands dimensions if needed. (similar to tensor.repeat() in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch repeat")
            .rest(
                "sizes",
                SyntaxShape::Int,
                "Number of times to repeat along each dimension",
            )
            .input_output_types(vec![(Type::String, Type::String)])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Repeat a tensor 3 times along the first dimension",
                example: "torch linspace 0.0 1.0 4 | torch repeat 3 | torch value",
                result: None,
            },
            Example {
                description: "Repeat a tensor 2 times along first dim and 2 times along second dim (creates new dim if needed)",
                example: "torch linspace 0.0 1.0 4 | torch repeat 2 2 | torch value",
                result: None,
            }
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // ── source tensor must come through pipeline ───────────────────
        // Pipeline-only design matches other shape ops (squeeze, unsqueeze, reshape)
        let input_value = input.into_value(call.head)?;
        let tensor_id = input_value.as_str()?;

        // ── get repeat sizes (variable number via .rest()) ────────────
        // Using .rest() allows: torch repeat 2 3 4  (repeats along 3 dims)
        let sizes: Vec<i64> = call
            .rest(0)
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Unable to parse repeat sizes", call.head)
            })?
            .into_iter()
            .map(|v: Value| v.as_int())
            .collect::<Result<Vec<i64>, _>>()?;
        // ── validate sizes before processing ───────────────────────────
        if sizes.is_empty() {
            return Err(LabeledError::new("Invalid input")
                .with_label("At least one repeat size must be provided", call.head));
        }
        if sizes.iter().any(|&n| n < 1) {
            return Err(LabeledError::new("Invalid input")
                .with_label("All repeat sizes must be at least 1", call.head));
        }

        // ── fetch tensor ───────────────────────────────────────────────
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor = registry
            .get(tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();
        // ── auto-expand dimensions if needed ───────────────────────────
        // If sizes.len() > tensor.dim(), unsqueeze leading dimensions
        // Example: [3] tensor with sizes [2, 4] → unsqueeze to [1, 3] → repeat to [2, 12]
        let dims = tensor.size();
        let mut working_tensor = tensor;
        let target_dims = sizes.len();
        let current_dims = dims.len();

        if target_dims > current_dims {
            // Add leading singleton dimensions to match sizes length
            for _ in 0..(target_dims - current_dims) {
                working_tensor = working_tensor.unsqueeze(0);
            }
        }

        // ── build repeat dimensions vector ─────────────────────────────
        // If tensor has more dims than sizes, pad sizes with 1s (no repeat on trailing dims)
        let final_dims = working_tensor.size();
        let mut repeat_dims = vec![1; final_dims.len()];
        for (i, &size) in sizes.iter().enumerate() {
            repeat_dims[i] = size;
        }

        // ── perform repeat operation ───────────────────────────────────
        let result_tensor = working_tensor.repeat(&repeat_dims);

        // ── store & return ─────────────────────────────────────────────
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
