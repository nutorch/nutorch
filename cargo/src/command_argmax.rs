use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;
// torch argmax  -------------------------------------------------------------
//   $x | torch argmax                    # flatten, return scalar index
//   torch argmax $x --dim 1              # along dim 1
//   $x | torch argmax --dim 1 --keepdim  # keepdim = true
// ---------------------------------------------------------------------------

pub struct CommandArgmax;

impl PluginCommand for CommandArgmax {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str { "torch argmax" }

    fn description(&self) -> &str {
        "Return indices of the maximum values of a tensor, optionally along a \
         specified dimension (wraps Tensor::argmax in tch-rs)."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch argmax")
            .input_output_types(vec![
                (Type::String,  Type::String),   // tensor id via pipeline
                (Type::Nothing, Type::String),   // tensor id via arg
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "Tensor ID (if not supplied by pipeline)",
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension to reduce (default: flatten)",
                None,
            )
            .named(
                "keepdim",
                SyntaxShape::Boolean,
                "Keep reduced dimension (default false)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Argmax of a flattened tensor",
                example: "([1 5 3] | torch tensor) | torch argmax | torch value",
                result: None,
            },
            Example {
                description: "Argmax along dim 1 with keepdim",
                example: r#"
let a = ([[1 5] [7 0]] | torch tensor)
torch argmax $a --dim 1 --keepdim true | torch value
"#,
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin : &NutorchPlugin,
        _engine : &nu_plugin::EngineInterface,
        call    : &nu_plugin::EvaluatedCall,
        input   : PipelineData,
    ) -> Result<PipelineData, LabeledError>
    {
        //---------------- obtain tensor ID ------------------------------
        let piped = match input {
            PipelineData::Value(v, _) => Some(v),
            PipelineData::Empty       => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs supported", call.head))
            }
        };
        let arg0  = call.nth(0);

        match (&piped, &arg0) {
            (None, None) =>
                return Err(LabeledError::new("Missing input")
                    .with_label("Must supply tensor ID via pipeline or argument", call.head)),
            (Some(_), Some(_)) =>
                return Err(LabeledError::new("Conflicting input")
                    .with_label("Provide tensor ID through pipeline OR argument, not both", call.head)),
            _ => {}
        }

        let id_val   = piped.or(arg0).unwrap();
        let tensor_id = id_val.as_str()?.to_string();

        //---------------- optional flags --------------------------------
        let dim_opt: Option<i64>   = call.get_flag("dim")?;
        let keepdim: bool          = call.get_flag("keepdim")?.unwrap_or(false);

        //---------------- fetch tensor ----------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let t = reg.get(&tensor_id).ok_or_else(|| {
            LabeledError::new("Tensor not found")
                .with_label("Invalid tensor ID", call.head)
        })?.shallow_clone();

        //---------------- argmax ----------------------------------------
        // Validate dimension if provided
        if let Some(dim) = dim_opt {
            let num_dims = t.size().len() as i64;
            if dim < 0 || dim >= num_dims {
                return Err(LabeledError::new("Invalid dimension").with_label(
                    format!(
                        "Dimension {dim} out of bounds for tensor with {num_dims} dimensions"
                    ),
                    call.head,
                ));
            }
        }

        // tch-rs exposes argmax(dim, keepdim) where dim: Option<i64>
        let result = t.argmax(dim_opt, keepdim);

        //---------------- store & return -------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
