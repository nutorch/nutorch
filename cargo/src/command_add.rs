use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

pub struct CommandAdd;

impl PluginCommand for CommandAdd {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch add"
    }

    fn description(&self) -> &str {
        "Compute the element-wise sum of two tensors with broadcasting (similar to torch.add or + operator)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch add")
            .input_output_types(vec![(Type::String, Type::String), (Type::Nothing, Type::String)])
            .optional("tensor1_id", SyntaxShape::String, "ID of the first tensor (if not using pipeline input)")
            .optional("tensor2_id", SyntaxShape::String, "ID of the second tensor")
            .named(
                "alpha",
                SyntaxShape::String,
                "ID of a tensor to use as a multiplier for the second tensor before addition (default: tensor with value 1.0)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Add two tensors using pipeline and argument",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); $t1 | torch add $t2 | torch value",
                result: None,
            },
            Example {
                description: "Add two tensors using arguments only",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); torch add $t1 $t2 | torch value",
                result: None,
            },
            Example {
                description: "Add two tensors with alpha scaling tensor on the second tensor",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); let alpha = (torch full 0.5 1); $t1 | torch add $t2 --alpha $alpha | torch value",
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
        // Check for pipeline input
        let pipeline_input = match input {
            PipelineData::Empty => None,
            PipelineData::Value(val, _) => Some(val),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs are supported", call.head));
            }
        };

        // Check for positional argument inputs
        let arg1 = call.nth(0);
        let arg2 = call.nth(1);

        // Validate the number of tensor inputs (exactly 2 tensors required for addition)
        let pipeline_count = if pipeline_input.is_some() { 1 } else { 0 };
        let arg_count = if arg1.is_some() { 1 } else { 0 } + if arg2.is_some() { 1 } else { 0 };
        let total_count = pipeline_count + arg_count;

        if total_count != 2 {
            return Err(LabeledError::new("Invalid input count").with_label(
                "Exactly two tensors must be provided via pipeline and/or arguments",
                call.head,
            ));
        }

        // Determine the two tensor IDs based on input sources
        let (tensor1_id, tensor2_id) = match (pipeline_input, arg1, arg2) {
            (Some(input_val), Some(arg_val), None) => {
                let input_id = input_val.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Unable to parse tensor ID from pipeline input", call.head)
                })?;
                let arg_id = arg_val.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Unable to parse tensor ID from argument", call.head)
                })?;
                (input_id, arg_id)
            }
            (None, Some(arg1_val), Some(arg2_val)) => {
                let arg1_id = arg1_val.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Unable to parse first tensor ID from argument", call.head)
                })?;
                let arg2_id = arg2_val.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Unable to parse second tensor ID from argument", call.head)
                })?;
                (arg1_id, arg2_id)
            }
            _ => {
                return Err(LabeledError::new("Invalid input configuration").with_label(
                    "Must provide exactly two tensors via pipeline and/or arguments",
                    call.head,
                ));
            }
        };

        // Look up tensors in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor1 = registry
            .get(&tensor1_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid first tensor ID", call.head)
            })?
            .shallow_clone();
        let tensor2 = registry
            .get(&tensor2_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid second tensor ID", call.head)
            })?
            .shallow_clone();

        // Validate device compatibility
        if tensor1.device() != tensor2.device() {
            return Err(LabeledError::new("Device mismatch").with_label(
                format!(
                    "Tensors must be on the same device. tensor1: {:?}, tensor2: {:?}",
                    tensor1.device(),
                    tensor2.device()
                ),
                call.head,
            ));
        }

        // Handle optional alpha argument (as a tensor ID)
        let result_tensor = match call.get_flag::<String>("alpha")? {
            Some(alpha_id) => {
                let alpha_tensor = registry
                    .get(&alpha_id)
                    .ok_or_else(|| {
                        LabeledError::new("Tensor not found")
                            .with_label("Invalid alpha tensor ID", call.head)
                    })?
                    .shallow_clone();
                tensor1 + (alpha_tensor * tensor2)
            }
            None => {
                // Default to a scalar tensor with value 1.0
                tensor1 + tensor2
            }
        };

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
