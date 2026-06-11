//! JSON↔tensor conversion, ported from v1 (v1/cargo/src/lib.rs
//! value_to_tensor / tensor_to_value) with serde_json::Value in place of
//! Nushell Value, and fallible tch calls (`f_*`) in place of panicking ones —
//! a daemon must return errors, not die.

use tch::{Device, Kind, Tensor};

/// tch errors stringify with a full C++ backtrace; keep the first line only —
/// good error messages are a carried-forward principle, stack frames are not.
pub fn tch_error(e: tch::TchError) -> String {
    e.to_string()
        .lines()
        .next()
        .unwrap_or("internal torch error")
        .to_string()
}

/// v1 fidelity: default dtype is float32 (v1/cargo/src/lib.rs:197), not
/// Python torch.tensor's int64 inference.
pub fn parse_kind(dtype: Option<&str>) -> Result<Kind, String> {
    match dtype {
        None | Some("float32") | Some("float") => Ok(Kind::Float),
        Some("float64") | Some("double") => Ok(Kind::Double),
        Some("int32") | Some("int") => Ok(Kind::Int),
        Some("int64") | Some("long") => Ok(Kind::Int64),
        Some(s) => Err(format!(
            "invalid dtype: {s} (expected float32, float64, int32, or int64)"
        )),
    }
}

/// Recursively convert a JSON number / (nested) array of numbers to a tensor.
pub fn json_to_tensor(
    value: &serde_json::Value,
    kind: Kind,
    device: Device,
) -> Result<Tensor, String> {
    match value {
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                return Err("list cannot be empty".to_string());
            }
            if items[0].is_array() {
                // Nested list: convert each sublist and stack along dim 0.
                // f_stack (not stack) so ragged shapes return an error.
                let subtensors: Result<Vec<Tensor>, String> = items
                    .iter()
                    .map(|v| json_to_tensor(v, kind, device))
                    .collect();
                let subtensors = subtensors?;
                Tensor::f_stack(&subtensors, 0)
                    .map_err(|e| format!("ragged or mismatched nested list: {}", tch_error(e)))
            } else {
                // Flat list. v1 fidelity: all-integers build an i64 buffer,
                // anything float builds f64; both then cast to `kind`.
                let all_ints = items.iter().all(|v| v.is_i64() || v.is_u64());
                let tensor = if all_ints {
                    let data: Result<Vec<i64>, String> = items
                        .iter()
                        .map(|v| {
                            v.as_i64()
                                .ok_or_else(|| format!("expected integer, got {v}"))
                        })
                        .collect();
                    Tensor::from_slice(&data?)
                } else {
                    let data: Result<Vec<f64>, String> = items
                        .iter()
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| format!("expected number, got {v}"))
                        })
                        .collect();
                    Tensor::from_slice(&data?)
                };
                tensor
                    .f_to_kind(kind)
                    .map_err(tch_error)?
                    .f_to_device(device)
                    .map_err(tch_error)
            }
        }
        serde_json::Value::Number(n) => {
            let tensor = if let Some(i) = n.as_i64() {
                Tensor::from(i)
            } else if let Some(f) = n.as_f64() {
                Tensor::from(f)
            } else {
                return Err(format!("unsupported number: {n}"));
            };
            tensor
                .f_to_kind(kind)
                .map_err(tch_error)?
                .f_to_device(device)
                .map_err(tch_error)
        }
        other => Err(format!(
            "input must be a number or a (nested) list of numbers, got {other}"
        )),
    }
}

/// Recursively convert a tensor to JSON (0-D → number, N-D → nested arrays).
/// The caller is expected to hand in a CPU tensor; `value` requests copy the
/// tensor off-device first.
pub fn tensor_to_json(tensor: &Tensor) -> Result<serde_json::Value, String> {
    let dims = tensor.size();
    let kind = tensor.kind();

    if dims.is_empty() {
        return match kind {
            Kind::Int | Kind::Int8 | Kind::Int16 | Kind::Int64 | Kind::Uint8 => {
                Ok(serde_json::Value::from(tensor.int64_value(&[])))
            }
            Kind::Float | Kind::Double | Kind::Half => {
                Ok(serde_json::Value::from(tensor.double_value(&[])))
            }
            _ => Err(format!("cannot convert tensor of type {kind:?}")),
        };
    }

    let first_dim = dims[0];
    let mut items: Vec<serde_json::Value> = Vec::with_capacity(first_dim as usize);
    for i in 0..first_dim {
        let element = tensor.f_get(i).map_err(tch_error)?;
        items.push(tensor_to_json(&element)?);
    }
    Ok(serde_json::Value::Array(items))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flat_int_list_round_trips_as_float32_by_default() {
        let kind = parse_kind(None).unwrap();
        let tensor = json_to_tensor(&json!([1, 2, 3]), kind, Device::Cpu).unwrap();
        assert_eq!(tensor.kind(), Kind::Float);
        assert_eq!(tensor_to_json(&tensor).unwrap(), json!([1.0, 2.0, 3.0]));
    }

    #[test]
    fn nested_list_round_trips() {
        let tensor = json_to_tensor(&json!([[1, 2], [3, 4]]), Kind::Float, Device::Cpu).unwrap();
        assert_eq!(tensor.size(), vec![2, 2]);
        assert_eq!(
            tensor_to_json(&tensor).unwrap(),
            json!([[1.0, 2.0], [3.0, 4.0]])
        );
    }

    #[test]
    fn scalar_round_trips() {
        let tensor = json_to_tensor(&json!(7), Kind::Float, Device::Cpu).unwrap();
        assert_eq!(tensor.size(), Vec::<i64>::new());
        assert_eq!(tensor_to_json(&tensor).unwrap(), json!(7.0));
    }

    #[test]
    fn int64_dtype_round_trips_as_integers() {
        let kind = parse_kind(Some("int64")).unwrap();
        let tensor = json_to_tensor(&json!([1, 2, 3]), kind, Device::Cpu).unwrap();
        assert_eq!(tensor.kind(), Kind::Int64);
        assert_eq!(tensor_to_json(&tensor).unwrap(), json!([1, 2, 3]));
    }

    #[test]
    fn ragged_nested_list_is_an_error_not_a_panic() {
        let result = json_to_tensor(&json!([[1, 2], [3]]), Kind::Float, Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn empty_list_is_an_error() {
        assert!(json_to_tensor(&json!([]), Kind::Float, Device::Cpu).is_err());
    }

    #[test]
    fn non_numeric_input_is_an_error() {
        assert!(json_to_tensor(&json!("hello"), Kind::Float, Device::Cpu).is_err());
        assert!(json_to_tensor(&json!([1, "two"]), Kind::Float, Device::Cpu).is_err());
    }

    #[test]
    fn invalid_dtype_is_an_error() {
        assert!(parse_kind(Some("float16")).is_err());
    }
}
