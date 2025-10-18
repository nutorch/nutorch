use nu_plugin::PluginCommand;
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Tensor;
use uuid::Uuid;

use crate::add_grad_from_call;
use crate::get_device_from_call;
use crate::get_kind_from_call;
use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

// torch arange  -------------------------------------------------------------
// Create a 1-D tensor with evenly spaced values.
//   torch arange end
//   torch arange start end
//   torch arange start end step
//
// Optional flags handled by helpers already available:
//   --dtype <kind>   --device <cpu|cuda:N>   --requires_grad
// ---------------------------------------------------------------------------
pub struct CommandArange;

impl PluginCommand for CommandArange {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch arange"
    }

    fn description(&self) -> &str {
        "Return a 1-D tensor with values in [start, end) and the given step \
         (like torch.arange in PyTorch)."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch arange")
            .required(
                "end_or_start",
                SyntaxShape::Number,
                "If only one number is given, it is `end`; if two or three are given, it is `start`",
            )
            .optional(
                "end",
                SyntaxShape::Number,
                "End value (exclusive) if start supplied",
            )
            .optional(
                "step",
                SyntaxShape::Number,
                "Step (default 1) if start and end supplied",
            )
            .named(
                "device",
                SyntaxShape::String,
                "Device to create the tensor on ('cpu', 'cuda', 'mps', default: 'cpu')",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor ('float32', 'float64', 'int32', 'int64')",
                None,
            )
            .named(
                "requires_grad",
                SyntaxShape::Boolean,
                "Whether the tensor requires gradient tracking for autograd (default: false)",
                None,
            )
            // we already declared global flags for dtype/device/grad earlier,
            // so they are parsed by the helper functions; no need to repeat.
            .input_output_types(vec![(Type::Nothing, Type::String)])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "arange(5)  -> 0 1 2 3 4",
                example: "torch arange 5 | torch value",
                result: None,
            },
            Example {
                description: "arange(2, 7)  -> 2 3 4 5 6",
                example: "torch arange 2 7 | torch value",
                result: None,
            },
            Example {
                description: "arange(1, 5, 0.5)  -> 1 1.5 … 4.5 (float)",
                example: "torch arange 1 5 0.5 --dtype float | torch value",
                result: None,
            },
            Example {
                description: "Create a tensor with gradient tracking enabled",
                example: "torch arange 0 10 --requires_grad true",
                result: None,
            },
            Example {
                description: "Create a tensor on CPU device",
                example: "torch arange 0 5 --device cpu | torch value",
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        //------------------------------------------------------------------
        // 1. parse positional numbers
        //------------------------------------------------------------------
        let argc = call.positional.iter().count();
        if !(1..=3).contains(&argc) {
            return Err(LabeledError::new("Invalid arange usage")
                .with_label("Require 1, 2 or 3 numeric arguments", call.head));
        }

        // helper to convert a Value to f64
        let to_f64 = |v: &Value| -> Result<f64, LabeledError> {
            if let Ok(i) = v.as_int() {
                Ok(i as f64)
            } else if let Ok(f) = v.as_float() {
                Ok(f)
            } else {
                Err(LabeledError::new("Expected number")
                    .with_label("Argument must be int or float", v.span()))
            }
        };

        let arg0 = call.nth(0).unwrap(); // safe: argc>=1
        let a0 = to_f64(&arg0)?; // end OR start

        let (start, end, step) = match argc {
            1 => (0.0, a0, 1.0),
            2 => {
                let arg1 = call.nth(1).unwrap();
                (a0, to_f64(&arg1)?, 1.0)
            }
            3 => {
                let arg1 = call.nth(1).unwrap();
                let arg2 = call.nth(2).unwrap();
                (a0, to_f64(&arg1)?, to_f64(&arg2)?)
            }
            _ => unreachable!(),
        };

        if step == 0.0 {
            return Err(LabeledError::new("Step cannot be zero").with_label("step", call.head));
        }

        //------------------------------------------------------------------
        // 2. dtype, device, requires_grad flags
        //------------------------------------------------------------------
        let device = get_device_from_call(call)?;
        let kind = get_kind_from_call(call)?;

        //------------------------------------------------------------------
        // 3. build tensor with tch-rs
        //------------------------------------------------------------------
        let options = (kind, device);
        let mut t = if (start.fract() == 0.0) && (end.fract() == 0.0) && (step.fract() == 0.0) {
            // integer path
            let s = start as i64;
            let e = end as i64;
            let k = step as i64;
            match argc {
                1 => Tensor::arange(e, options),
                2 => Tensor::arange_start(s, e, options),
                _ => Tensor::arange_start_step(s, e, k, options),
            }
        } else {
            // floating path
            match argc {
                1 => Tensor::arange(end, options),
                2 => Tensor::arange_start(start, end, options),
                _ => Tensor::arange_start_step(start, end, step, options),
            }
        };

        // handle --requires_grad
        t = add_grad_from_call(call, t)?;

        //------------------------------------------------------------------
        // 4. store tensor & return id
        //------------------------------------------------------------------
        let id = Uuid::new_v4().to_string();
        TENSOR_REGISTRY.lock().unwrap().insert(id.clone(), t);

        Ok(PipelineData::Value(Value::string(id, call.head), None))
    }
}
