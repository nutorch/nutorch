//! Request parsing and dispatch: bespoke ops (tensor/value/lifecycle) plus
//! the table-driven op execution (issue 0005). The op table lives in the
//! shared `nutorch-ops` crate; this module maps each table row to its
//! fallible tch call.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use nutorch_ops::{Arity, OpSpec, ParamKind, ResultKind};
use tch::{Device, Kind, Tensor};

use crate::convert;
use crate::lifecycle::{self, Lifecycle};
use crate::protocol::{Bespoke, Request, Response};
use crate::registry::Registry;

/// Reject a request that still carries the removed `device` option (issue
/// 0003), then route: bespoke names deserialize by tag; anything else is
/// looked up in the op table; unknown names get `unknown_op`.
pub fn parse_request(line: &str) -> Result<Request, Response> {
    let raw: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| Response::error("bad_request", format!("bad request: {e}")))?;
    if raw.get("device").is_some() {
        return Err(Response::error(
            "bad_request",
            "the device option was removed (issue 0003): tensors always live on the GPU (mps)",
        ));
    }
    let name = raw
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Response::error("bad_request", "request has no op"))?
        .to_string();
    match name.as_str() {
        "tensor" | "value" | "status" | "set_ttl" | "shutdown" => {
            let bespoke: Bespoke = serde_json::from_value(raw)
                .map_err(|e| Response::error("bad_request", format!("bad request: {e}")))?;
            Ok(Request::Bespoke(bespoke))
        }
        _ => {
            if nutorch_ops::find(&name).is_none() {
                return Err(Response::error(
                    "unknown_op",
                    format!("unknown op: {name} (see `torch ops`)"),
                ));
            }
            let tensors = match raw.get("tensors") {
                None => Vec::new(),
                Some(serde_json::Value::Array(items)) => items
                    .iter()
                    .map(|v| {
                        v.as_str().map(str::to_string).ok_or_else(|| {
                            Response::error("bad_request", "tensors must be handle strings")
                        })
                    })
                    .collect::<Result<_, _>>()?,
                Some(_) => {
                    return Err(Response::error(
                        "bad_request",
                        "tensors must be an array of handle strings",
                    ))
                }
            };
            let params = match raw.get("params") {
                None => serde_json::Map::new(),
                Some(serde_json::Value::Object(map)) => map.clone(),
                Some(_) => return Err(Response::error("bad_request", "params must be an object")),
            };
            Ok(Request::Table {
                name,
                tensors,
                params,
            })
        }
    }
}

/// Dispatch one request. Returns the response plus a shutdown flag (the
/// serve loop writes and flushes the response BEFORE acting on the flag).
///
/// Tensor work (bespoke tensor/value and every table op) resets the idle
/// clock; `status`/`set_ttl`/`shutdown` do not.
pub fn handle_request(
    registry: &mut Registry,
    lifecycle: &Mutex<Lifecycle>,
    socket: &Path,
    request: Request,
) -> (Response, bool) {
    match request {
        Request::Table {
            name,
            tensors,
            params,
        } => {
            lifecycle.lock().unwrap().touch();
            let spec = nutorch_ops::find(&name).expect("parse_request validated the name");
            (execute_table(registry, spec, &tensors, &params), false)
        }
        Request::Bespoke(bespoke) => match bespoke {
            Bespoke::Tensor { data, dtype } => {
                lifecycle.lock().unwrap().touch();
                let kind = match convert::parse_kind(dtype.as_deref()) {
                    Ok(k) => k,
                    Err(e) => return (Response::error("bad_dtype", e), false),
                };
                match convert::json_to_tensor(&data, kind, Device::Mps) {
                    Ok(tensor) => (Response::handle(registry.insert(tensor)), false),
                    Err(e) => (Response::error("bad_argument", e), false),
                }
            }
            Bespoke::Value { handle } => {
                lifecycle.lock().unwrap().touch();
                match registry.get(&handle) {
                    Some(tensor) => {
                        let cpu = match tensor.f_to_device(Device::Cpu) {
                            Ok(t) => t,
                            Err(e) => {
                                return (
                                    Response::error("torch_error", convert::tch_error(e)),
                                    false,
                                )
                            }
                        };
                        match convert::tensor_to_json(&cpu) {
                            Ok(value) => (Response::value(value), false),
                            Err(e) => (Response::error("torch_error", e), false),
                        }
                    }
                    None => (
                        Response::error("unknown_handle", format!("unknown handle: {handle}")),
                        false,
                    ),
                }
            }
            Bespoke::Status => {
                let state = lifecycle.lock().unwrap();
                (
                    Response::value(serde_json::json!({
                        "pid": std::process::id(),
                        "uptime_secs": state.uptime_secs(),
                        "device": "mps",
                        "ttl_secs": state.ttl_secs(),
                        "idle_secs": state.idle_secs(),
                        "remaining_secs": state.remaining_secs(),
                        "tensors": registry.len(),
                        "approx_bytes": registry.approx_bytes(),
                        "socket": socket.display().to_string(),
                        "log": log_path_for(socket).display().to_string(),
                    })),
                    false,
                )
            }
            Bespoke::SetTtl { ttl } => match lifecycle::parse_ttl(&ttl) {
                Ok(parsed) => {
                    let mut state = lifecycle.lock().unwrap();
                    state.set_ttl(parsed);
                    (
                        Response::value(serde_json::json!({ "ttl_secs": state.ttl_secs() })),
                        false,
                    )
                }
                Err(e) => (Response::error("bad_argument", e), false),
            },
            Bespoke::Shutdown => (Response::value(serde_json::json!("shutting down")), true),
        },
    }
}

/// The conventional log path: the socket path with its extension replaced
/// by `.log` (nutorchd.sock -> nutorchd.log).
pub fn log_path_for(socket: &Path) -> PathBuf {
    socket.with_extension("log")
}

/// nutorchd is GPU-only (issue 0003): Mac-only for now, so the GPU is MPS.
pub fn require_mps() -> Result<(), String> {
    if tch::utils::has_mps() {
        Ok(())
    } else {
        Err("nutorchd requires an Apple-silicon Mac with MPS (GPU-only by design)".to_string())
    }
}

// ---------- table op execution ----------

/// PyTorch broadcastability: right-aligned walk; dims equal or either is 1.
fn broadcastable(a: &[i64], b: &[i64]) -> bool {
    let mut ai = a.iter().rev();
    let mut bi = b.iter().rev();
    loop {
        match (ai.next(), bi.next()) {
            (Some(&x), Some(&y)) if x != y && x != 1 && y != 1 => return false,
            (None, _) | (_, None) => return true,
            _ => continue,
        }
    }
}

/// A typed view over the request's params, validated against the spec.
struct Params<'a> {
    map: &'a serde_json::Map<String, serde_json::Value>,
}

enum Scalar {
    Int(i64),
    Float(f64),
}

impl<'a> Params<'a> {
    fn validate(
        spec: &OpSpec,
        map: &'a serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, Response> {
        for key in map.keys() {
            if !spec.params.iter().any(|p| p.name == key) {
                return Err(Response::error(
                    "bad_argument",
                    format!("{}: unknown parameter: {key}", spec.name),
                ));
            }
        }
        for param in spec.params {
            let value = map.get(param.name);
            if param.required && value.is_none() {
                return Err(Response::error(
                    "bad_argument",
                    format!("{}: missing required parameter: {}", spec.name, param.name),
                ));
            }
            if let Some(value) = value {
                let ok = match param.kind {
                    ParamKind::Int => value.is_i64(),
                    ParamKind::Float => value.is_f64() || value.is_i64(),
                    ParamKind::Scalar => value.is_number(),
                    ParamKind::IntList => {
                        value.is_array() && value.as_array().unwrap().iter().all(|v| v.is_i64())
                    }
                    ParamKind::Bool => value.is_boolean(),
                    ParamKind::Str => value.is_string(),
                };
                if !ok {
                    return Err(Response::error(
                        "bad_argument",
                        format!(
                            "{}: parameter {} must be {:?}",
                            spec.name, param.name, param.kind
                        ),
                    ));
                }
            }
        }
        Ok(Params { map })
    }

    fn int(&self, name: &str) -> Option<i64> {
        self.map.get(name).and_then(|v| v.as_i64())
    }

    fn float(&self, name: &str) -> Option<f64> {
        self.map.get(name).and_then(|v| v.as_f64())
    }

    fn scalar(&self, name: &str) -> Option<Scalar> {
        let v = self.map.get(name)?;
        if let Some(i) = v.as_i64() {
            Some(Scalar::Int(i))
        } else {
            v.as_f64().map(Scalar::Float)
        }
    }

    fn int_list(&self, name: &str) -> Option<Vec<i64>> {
        self.map
            .get(name)?
            .as_array()
            .map(|items| items.iter().filter_map(|v| v.as_i64()).collect())
    }

    fn bool(&self, name: &str) -> bool {
        self.map
            .get(name)
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    fn str(&self, name: &str) -> Option<&str> {
        self.map.get(name).and_then(|v| v.as_str())
    }
}

enum Applied {
    Tensors(Vec<Tensor>),
    Value(serde_json::Value),
    Nothing,
}

type OpError = (&'static str, String);

fn tch(op: &str, e: tch::TchError) -> OpError {
    ("torch_error", format!("{op}: {}", convert::tch_error(e)))
}

pub fn execute_table(
    registry: &mut Registry,
    spec: &OpSpec,
    tensor_handles: &[String],
    params: &serde_json::Map<String, serde_json::Value>,
) -> Response {
    // Arity.
    let arity_ok = match spec.tensors {
        Arity::Exactly(n) => tensor_handles.len() == n,
        Arity::AtLeast(n) => tensor_handles.len() >= n,
    };
    if !arity_ok {
        let expected = match spec.tensors {
            Arity::Exactly(n) => format!("{n}"),
            Arity::AtLeast(n) => format!("at least {n}"),
        };
        return Response::error(
            "bad_argument",
            format!(
                "{}: expected {expected} tensor(s), got {}",
                spec.name,
                tensor_handles.len()
            ),
        );
    }

    // Params.
    let params = match Params::validate(spec, params) {
        Ok(p) => p,
        Err(response) => return response,
    };

    // Handles.
    let mut tensors: Vec<&Tensor> = Vec::with_capacity(tensor_handles.len());
    for handle in tensor_handles {
        match registry.get(handle) {
            Some(t) => tensors.push(t),
            None => return Response::error("unknown_handle", format!("unknown handle: {handle}")),
        }
    }
    debug_assert!(
        tensors.iter().all(|t| t.device() == Device::Mps),
        "registry invariant violated: all tensors live on MPS"
    );

    // Broadcasting pre-check for elementwise ops (quality errors; tch
    // broadcasts natively on the happy path).
    if spec.broadcasts && tensors.len() == 2 {
        let (sa, sb) = (tensors[0].size(), tensors[1].size());
        if !broadcastable(&sa, &sb) {
            return Response::error(
                "shape_mismatch",
                format!(
                    "{}: shapes {sa:?} and {sb:?} are not broadcastable",
                    spec.name
                ),
            );
        }
    }

    match apply(spec, &tensors, &params) {
        Ok(Applied::Tensors(outputs)) => {
            debug_assert!(matches!(spec.results, ResultKind::Handles(n) if n == outputs.len()));
            Response::handles(outputs.into_iter().map(|t| registry.insert(t)).collect())
        }
        Ok(Applied::Value(value)) => Response::value(value),
        Ok(Applied::Nothing) => Response::handles(Vec::new()),
        Err((code, message)) => Response::error(code, message),
    }
}

fn one(op: &str, result: Result<Tensor, tch::TchError>) -> Result<Applied, OpError> {
    result
        .map(|t| Applied::Tensors(vec![t]))
        .map_err(|e| tch(op, e))
}

fn apply(spec: &OpSpec, t: &[&Tensor], p: &Params) -> Result<Applied, OpError> {
    let op = spec.name;
    match op {
        // add: a + alpha*b; sub: a - alpha*b (PyTorch semantics; tch's f_add
        // has no alpha parameter, so alpha scales the right operand first).
        "add" | "sub" => {
            let result = match p.scalar("alpha") {
                None | Some(Scalar::Int(1)) => {
                    if op == "add" {
                        t[0].f_add(t[1])
                    } else {
                        t[0].f_sub(t[1])
                    }
                }
                Some(alpha) => {
                    let scaled = match alpha {
                        Scalar::Int(i) => t[1].f_mul_scalar(i),
                        Scalar::Float(f) => t[1].f_mul_scalar(f),
                    }
                    .map_err(|e| tch(op, e))?;
                    if op == "add" {
                        t[0].f_add(&scaled)
                    } else {
                        t[0].f_sub(&scaled)
                    }
                }
            };
            one(op, result)
        }
        "sin" => one(op, t[0].f_sin()),
        // --- pointwise sweep (issue 0005 exp 2): unary ---
        "abs" => one(op, t[0].f_abs()),
        "acos" => one(op, t[0].f_acos()),
        "acosh" => one(op, t[0].f_acosh()),
        "asin" => one(op, t[0].f_asin()),
        "asinh" => one(op, t[0].f_asinh()),
        "atan" => one(op, t[0].f_atan()),
        "atanh" => one(op, t[0].f_atanh()),
        "ceil" => one(op, t[0].f_ceil()),
        "cos" => one(op, t[0].f_cos()),
        "cosh" => one(op, t[0].f_cosh()),
        "deg2rad" => one(op, t[0].f_deg2rad()),
        "digamma" => one(op, t[0].f_digamma()),
        "erf" => one(op, t[0].f_erf()),
        "erfc" => one(op, t[0].f_erfc()),
        "exp" => one(op, t[0].f_exp()),
        "exp2" => one(op, t[0].f_exp2()),
        "expm1" => one(op, t[0].f_expm1()),
        "floor" => one(op, t[0].f_floor()),
        "frac" => one(op, t[0].f_frac()),
        "i0" => one(op, t[0].f_i0()),
        "lgamma" => one(op, t[0].f_lgamma()),
        "log" => one(op, t[0].f_log()),
        "log10" => one(op, t[0].f_log10()),
        "log1p" => one(op, t[0].f_log1p()),
        "log2" => one(op, t[0].f_log2()),
        "logit" => one(op, t[0].f_logit(None::<f64>)),
        "neg" => one(op, t[0].f_neg()),
        "rad2deg" => one(op, t[0].f_rad2deg()),
        "reciprocal" => one(op, t[0].f_reciprocal()),
        "relu" => one(op, t[0].f_relu()),
        "round" => one(op, t[0].f_round()),
        "rsqrt" => one(op, t[0].f_rsqrt()),
        "sgn" => one(op, t[0].f_sgn()),
        "sigmoid" => one(op, t[0].f_sigmoid()),
        "sign" => one(op, t[0].f_sign()),
        "sinc" => one(op, t[0].f_sinc()),
        "sinh" => one(op, t[0].f_sinh()),
        "sqrt" => one(op, t[0].f_sqrt()),
        "square" => one(op, t[0].f_square()),
        "tan" => one(op, t[0].f_tan()),
        "tanh" => one(op, t[0].f_tanh()),
        "trunc" => one(op, t[0].f_trunc()),
        "softmax" => one(
            op,
            t[0].f_softmax(p.int("dim").expect("required"), Kind::Float),
        ),
        "log_softmax" => one(
            op,
            t[0].f_log_softmax(p.int("dim").expect("required"), Kind::Float),
        ),
        "nan_to_num" => one(
            op,
            t[0].f_nan_to_num(p.float("nan"), p.float("posinf"), p.float("neginf")),
        ),
        // --- pointwise sweep: binary, broadcasting ---
        "mul" => one(op, t[0].f_mul(t[1])),
        "div" => one(op, t[0].f_div(t[1])),
        "maximum" => one(op, t[0].f_maximum(t[1])),
        "minimum" => one(op, t[0].f_minimum(t[1])),
        "atan2" => one(op, t[0].f_atan2(t[1])),
        "fmod" => one(op, t[0].f_fmod_tensor(t[1])),
        "remainder" => one(op, t[0].f_remainder_tensor(t[1])),
        "floor_divide" => one(op, t[0].f_floor_divide(t[1])),
        "hypot" => one(op, t[0].f_hypot(t[1])),
        "copysign" => one(op, t[0].f_copysign(t[1])),
        "xlogy" => one(op, t[0].f_xlogy(t[1])),
        "logaddexp" => one(op, t[0].f_logaddexp(t[1])),
        "pow" => match p.scalar("exponent").expect("required") {
            Scalar::Int(i) => one(op, t[0].f_pow_tensor_scalar(i)),
            Scalar::Float(f) => one(op, t[0].f_pow_tensor_scalar(f)),
        },
        "clamp" => {
            let to_scalar = |s: Scalar| -> tch::Scalar {
                match s {
                    Scalar::Int(i) => i.into(),
                    Scalar::Float(f) => f.into(),
                }
            };
            match (p.scalar("min"), p.scalar("max")) {
                (None, None) => Err((
                    "bad_argument",
                    "clamp: at least one of --min/--max is required".to_string(),
                )),
                (Some(min), Some(max)) => one(op, t[0].f_clamp(to_scalar(min), to_scalar(max))),
                (Some(min), None) => one(op, t[0].f_clamp_min(to_scalar(min))),
                (None, Some(max)) => one(op, t[0].f_clamp_max(to_scalar(max))),
            }
        }
        "sum" => match p.int("dim") {
            Some(dim) => one(
                op,
                t[0].f_sum_dim_intlist(Some(&[dim][..]), p.bool("keepdim"), None::<Kind>),
            ),
            None => one(op, t[0].f_sum(None::<Kind>)),
        },
        "mean" => match p.int("dim") {
            // v1 fidelity: mean reduces in float32 regardless of input kind.
            Some(dim) => one(
                op,
                t[0].f_mean_dim(Some(&[dim][..]), p.bool("keepdim"), Kind::Float),
            ),
            None => one(op, t[0].f_mean(Kind::Float)),
        },
        "eq" => one(op, t[0].f_eq_tensor(t[1])),
        "allclose" => {
            let rtol = p.float("rtol").unwrap_or(1e-5);
            let atol = p.float("atol").unwrap_or(1e-8);
            Ok(Applied::Value(serde_json::Value::Bool(
                t[0].allclose(t[1], rtol, atol, false),
            )))
        }
        "sort" => {
            let dim = p.int("dim").unwrap_or(-1);
            t[0].f_sort(dim, p.bool("descending"))
                .map(|(values, indices)| Applied::Tensors(vec![values, indices]))
                .map_err(|e| tch(op, e))
        }
        "mm" => {
            // Ported v1 validation: both rank-2, inner dims equal.
            let (sa, sb) = (t[0].size(), t[1].size());
            if sa.len() != 2 || sb.len() != 2 {
                return Err((
                    "shape_mismatch",
                    format!("mm: requires two 2-D tensors, got shapes {sa:?} and {sb:?}"),
                ));
            }
            if sa[1] != sb[0] {
                return Err((
                    "shape_mismatch",
                    format!("mm: inner dimensions must match, got {sa:?} and {sb:?}"),
                ));
            }
            one(op, t[0].f_mm(t[1]))
        }
        "cat" => {
            let dim = p.int("dim").unwrap_or(0);
            one(op, Tensor::f_cat(t, dim))
        }
        "full" => {
            let shape = p.int_list("shape").expect("required");
            validate_shape(op, &shape)?;
            let kind = parse_table_kind(op, p.str("dtype"))?;
            match p.scalar("value").expect("required") {
                Scalar::Int(i) => one(op, Tensor::f_full(&shape, i, (kind, Device::Mps))),
                Scalar::Float(f) => one(op, Tensor::f_full(&shape, f, (kind, Device::Mps))),
            }
        }
        "randn" => {
            let shape = p.int_list("shape").expect("required");
            validate_shape(op, &shape)?;
            let kind = parse_table_kind(op, p.str("dtype"))?;
            if !matches!(kind, Kind::Float | Kind::Half) {
                return Err((
                    "bad_dtype",
                    format!(
                        "randn: requires float32 or float16 (float64 is unsupported on MPS), got {kind:?}"
                    ),
                ));
            }
            // Generate on the seeded CPU generator, then transfer: tch's
            // manual_seed does NOT reach the MPS generator (discovered in
            // issue 0005 exp 1), and the CPU generator is the one Python's
            // torch.manual_seed drives too — so this is what makes randn
            // both deterministic and golden-comparable.
            one(
                op,
                Tensor::f_randn(&shape, (kind, Device::Cpu))
                    .and_then(|t| t.f_to_device(Device::Mps)),
            )
        }
        "manual_seed" => {
            tch::manual_seed(p.int("seed").expect("required"));
            Ok(Applied::Nothing)
        }
        other => Err((
            "unknown_op",
            format!("table op {other} has no apply mapping (bug)"),
        )),
    }
}

fn validate_shape(op: &str, shape: &[i64]) -> Result<(), OpError> {
    if shape.is_empty() {
        return Err(("bad_argument", format!("{op}: shape cannot be empty")));
    }
    if let Some(bad) = shape.iter().find(|d| **d < 1) {
        return Err((
            "bad_argument",
            format!("{op}: every shape dimension must be >= 1, got {bad}"),
        ));
    }
    Ok(())
}

fn parse_table_kind(op: &str, dtype: Option<&str>) -> Result<Kind, OpError> {
    convert::parse_kind(dtype).map_err(|e| ("bad_dtype", format!("{op}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_socket() -> &'static Path {
        Path::new("/tmp/nutorchd-unit-test.sock")
    }

    fn run(registry: &mut Registry, line: serde_json::Value) -> Response {
        let lifecycle = Mutex::new(Lifecycle::new(None));
        let request = match parse_request(&line.to_string()) {
            Ok(r) => r,
            Err(error_response) => return error_response,
        };
        let (response, _) = handle_request(registry, &lifecycle, test_socket(), request);
        response
    }

    fn expect_handles(response: Response) -> Vec<String> {
        match response {
            Response::Handles { handles, .. } => handles,
            Response::Handle { handle, .. } => vec![handle],
            other => panic!("expected handles, got {other:?}"),
        }
    }

    fn expect_value(response: Response) -> serde_json::Value {
        match response {
            Response::Value { value, .. } => value,
            other => panic!("expected value, got {other:?}"),
        }
    }

    fn expect_error(response: Response) -> (&'static str, String) {
        match response {
            Response::Error { code, error, .. } => (code, error),
            other => panic!("expected error, got {other:?}"),
        }
    }

    fn tensor_of(registry: &mut Registry, data: serde_json::Value) -> String {
        expect_handles(run(registry, json!({"op":"tensor","data":data})))
            .pop()
            .unwrap()
    }

    fn value_of(registry: &mut Registry, handle: &str) -> serde_json::Value {
        expect_value(run(registry, json!({"op":"value","handle":handle})))
    }

    #[test]
    fn require_mps_holds_on_this_machine() {
        assert!(require_mps().is_ok());
    }

    #[test]
    fn created_tensors_live_on_mps() {
        let mut registry = Registry::new();
        let h = tensor_of(&mut registry, json!([1, 2, 3]));
        assert_eq!(registry.get(&h).unwrap().device(), Device::Mps);
    }

    #[test]
    fn add_broadcasts_like_pytorch() {
        let mut registry = Registry::new();
        let a = tensor_of(&mut registry, json!([[1, 2, 3], [4, 5, 6]]));
        let b = tensor_of(&mut registry, json!([10, 20, 30]));
        let sum = expect_handles(run(&mut registry, json!({"op":"add","tensors":[a, b]})))
            .pop()
            .unwrap();
        assert_eq!(
            value_of(&mut registry, &sum),
            json!([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])
        );
    }

    #[test]
    fn add_rejects_non_broadcastable_shapes_by_name() {
        let mut registry = Registry::new();
        let a = tensor_of(&mut registry, json!([[1, 2, 3], [4, 5, 6]]));
        let b = tensor_of(&mut registry, json!([1, 2, 3, 4]));
        let (code, message) =
            expect_error(run(&mut registry, json!({"op":"add","tensors":[a, b]})));
        assert_eq!(code, "shape_mismatch");
        assert!(
            message.contains("[2, 3]") && message.contains("[4]"),
            "{message}"
        );
    }

    #[test]
    fn sort_returns_two_handles() {
        let mut registry = Registry::new();
        let t = tensor_of(&mut registry, json!([3, 1, 2]));
        let handles = expect_handles(run(
            &mut registry,
            json!({"op":"sort","tensors":[t],"params":{"descending":true}}),
        ));
        assert_eq!(handles.len(), 2);
        assert_eq!(value_of(&mut registry, &handles[0]), json!([3.0, 2.0, 1.0]));
        assert_eq!(value_of(&mut registry, &handles[1]), json!([0, 2, 1]));
    }

    #[test]
    fn eq_returns_bool_tensor() {
        let mut registry = Registry::new();
        let a = tensor_of(&mut registry, json!([1, 2, 3]));
        let b = tensor_of(&mut registry, json!([1, 0, 3]));
        let result = expect_handles(run(&mut registry, json!({"op":"eq","tensors":[a, b]})))
            .pop()
            .unwrap();
        assert_eq!(value_of(&mut registry, &result), json!([true, false, true]));
    }

    #[test]
    fn allclose_returns_a_plain_value() {
        let mut registry = Registry::new();
        let a = tensor_of(&mut registry, json!([1.0, 2.0]));
        let b = tensor_of(&mut registry, json!([1.0, 2.0]));
        assert_eq!(
            expect_value(run(
                &mut registry,
                json!({"op":"allclose","tensors":[a, b]})
            )),
            json!(true)
        );
    }

    #[test]
    fn mean_reduces_in_float32_even_for_int_input() {
        let mut registry = Registry::new();
        let t = expect_handles(run(
            &mut registry,
            json!({"op":"tensor","data":[1, 2, 3, 4],"dtype":"int64"}),
        ))
        .pop()
        .unwrap();
        let mean = expect_handles(run(&mut registry, json!({"op":"mean","tensors":[t]})))
            .pop()
            .unwrap();
        assert_eq!(value_of(&mut registry, &mean), json!(2.5));
    }

    #[test]
    fn manual_seed_makes_randn_deterministic() {
        let mut registry = Registry::new();
        let randn = |registry: &mut Registry| -> serde_json::Value {
            run(
                registry,
                json!({"op":"manual_seed","tensors":[],"params":{"seed":42}}),
            );
            let h = expect_handles(run(
                registry,
                json!({"op":"randn","params":{"shape":[2, 2]}}),
            ))
            .pop()
            .unwrap();
            value_of(registry, &h)
        };
        let first = randn(&mut registry);
        let second = randn(&mut registry);
        assert_eq!(first, second);
    }

    #[test]
    fn randn_rejects_int_dtypes() {
        let mut registry = Registry::new();
        let (code, _) = expect_error(run(
            &mut registry,
            json!({"op":"randn","params":{"shape":[2],"dtype":"int64"}}),
        ));
        assert_eq!(code, "bad_dtype");
    }

    #[test]
    fn mm_rejects_bad_shapes_with_codes() {
        let mut registry = Registry::new();
        let a = expect_handles(run(
            &mut registry,
            json!({"op":"full","params":{"shape":[2, 3],"value":1}}),
        ))
        .pop()
        .unwrap();
        let (code, message) = expect_error(run(
            &mut registry,
            json!({"op":"mm","tensors":[a.clone(), a]}),
        ));
        assert_eq!(code, "shape_mismatch");
        assert!(message.contains("[2, 3]"), "{message}");
    }

    #[test]
    fn unknown_things_get_specific_codes() {
        let mut registry = Registry::new();
        let (code, _) = expect_error(run(&mut registry, json!({"op":"frobnicate"})));
        assert_eq!(code, "unknown_op");
        let (code, _) = expect_error(run(&mut registry, json!({"op":"sin","tensors":["nope"]})));
        assert_eq!(code, "unknown_handle");
        let (code, _) = expect_error(run(
            &mut registry,
            json!({"op":"sin","tensors":[],"params":{"bogus":1}}),
        ));
        assert_eq!(code, "bad_argument");
    }

    #[test]
    fn clamp_requires_a_bound_and_honors_them() {
        let mut registry = Registry::new();
        let t = tensor_of(&mut registry, json!([-5, 0, 5]));
        let (code, _) = expect_error(run(
            &mut registry,
            json!({"op":"clamp","tensors":[t.clone()]}),
        ));
        assert_eq!(code, "bad_argument");
        let clamped = expect_handles(run(
            &mut registry,
            json!({"op":"clamp","tensors":[t],"params":{"min":-1,"max":1}}),
        ))
        .pop()
        .unwrap();
        assert_eq!(value_of(&mut registry, &clamped), json!([-1.0, 0.0, 1.0]));
    }

    #[test]
    fn device_field_is_rejected_with_removal_message() {
        let mut registry = Registry::new();
        let (code, message) = expect_error(run(
            &mut registry,
            json!({"op":"tensor","data":[1],"device":"cpu"}),
        ));
        assert_eq!(code, "bad_request");
        assert!(message.contains("device option was removed"), "{message}");
    }

    #[test]
    fn cat_concatenates_variadically() {
        let mut registry = Registry::new();
        let a = tensor_of(&mut registry, json!([1]));
        let b = tensor_of(&mut registry, json!([2]));
        let c = tensor_of(&mut registry, json!([3]));
        let joined = expect_handles(run(&mut registry, json!({"op":"cat","tensors":[a, b, c]})))
            .pop()
            .unwrap();
        assert_eq!(value_of(&mut registry, &joined), json!([1.0, 2.0, 3.0]));
    }
}

#[cfg(test)]
mod nan_to_num_semantics {
    use super::*;
    use serde_json::json;

    /// Golden inputs must be finite JSON, so the real NaN/inf replacement
    /// semantics are proven here with directly constructed non-finite values.
    #[test]
    fn nan_to_num_replaces_non_finite_values() {
        let mut registry = Registry::new();
        let numerator =
            convert::json_to_tensor(&json!([0.0, 1.0, -1.0]), Kind::Float, Device::Mps).unwrap();
        let zero = convert::json_to_tensor(&json!([0.0]), Kind::Float, Device::Mps).unwrap();
        let non_finite = numerator.f_div(&zero).unwrap(); // [NaN, inf, -inf]
        let handle = registry.insert(non_finite);
        let spec = nutorch_ops::find("nan_to_num").unwrap();
        let mut params = serde_json::Map::new();
        params.insert("nan".into(), json!(0.5));
        params.insert("posinf".into(), json!(100.0));
        params.insert("neginf".into(), json!(-100.0));
        let response = execute_table(&mut registry, spec, &[handle], &params);
        let out = match response {
            Response::Handles { handles, .. } => handles[0].clone(),
            other => panic!("expected handles, got {other:?}"),
        };
        let cpu = registry
            .get(&out)
            .unwrap()
            .f_to_device(Device::Cpu)
            .unwrap();
        assert_eq!(
            convert::tensor_to_json(&cpu).unwrap(),
            json!([0.5, 100.0, -100.0])
        );
    }
}
