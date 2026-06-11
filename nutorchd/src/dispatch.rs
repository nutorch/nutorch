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
        "tensor" | "value" | "free" | "tensors" | "nn" | "forward" | "nn_parameters" | "step"
        | "nn_zero_grad" | "nn_set_lr" | "nn_mode" | "nn_save" | "nn_load" | "nn_info"
        | "status" | "set_ttl" | "shutdown" => {
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
            Bespoke::Tensor {
                data,
                dtype,
                requires_grad,
            } => {
                lifecycle.lock().unwrap().touch();
                match build_input_tensor(&data, dtype.as_deref(), requires_grad.unwrap_or(false)) {
                    Ok(tensor) => (Response::handle(registry.insert_tensor(tensor)), false),
                    Err((code, message)) => (Response::error(code, message), false),
                }
            }
            Bespoke::Value { handle, meta } => {
                lifecycle.lock().unwrap().touch();
                registry.touch(&handle);
                match registry.get_tensor(&handle) {
                    Ok(tensor) => {
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
                            Ok(data) => {
                                // --meta: the envelope a round-trip can carry
                                // its dtype in (issue 0006).
                                let value = if meta.unwrap_or(false) {
                                    serde_json::json!({
                                        "dtype": convert::kind_name(cpu.kind()),
                                        "shape": cpu.size(),
                                        "data": data,
                                    })
                                } else {
                                    data
                                };
                                (Response::value(value), false)
                            }
                            Err(e) => (Response::error("torch_error", e), false),
                        }
                    }
                    Err(lookup) => (Response::error(lookup.code(), lookup.message()), false),
                }
            }
            Bespoke::Free { handles, all } => {
                lifecycle.lock().unwrap().touch();
                let all = all.unwrap_or(false);
                let handles = handles.unwrap_or_default();
                if all && !handles.is_empty() {
                    return (
                        Response::error(
                            "bad_request",
                            "free: handles and all are mutually exclusive",
                        ),
                        false,
                    );
                }
                if all {
                    let freed = registry.clear();
                    return (
                        Response::value(serde_json::json!({ "freed": freed })),
                        false,
                    );
                }
                if handles.is_empty() {
                    return (
                        Response::error("bad_request", "free: no handles given"),
                        false,
                    );
                }
                // Atomic: validate ALL handles before removing ANY.
                for handle in &handles {
                    if let Err(lookup) = registry.check(handle) {
                        return (Response::error(lookup.code(), lookup.message()), false);
                    }
                }
                for handle in &handles {
                    registry.remove(handle);
                }
                (
                    Response::value(serde_json::json!({ "freed": handles.len() })),
                    false,
                )
            }
            Bespoke::Tensors => {
                // Analysis, like status: no lease touch, no tensor touch.
                let rows: Vec<serde_json::Value> = registry
                    .list()
                    .into_iter()
                    .map(|row| {
                        serde_json::json!({
                            "handle": row.handle,
                            "shape": row.shape,
                            "dtype": convert::kind_name(row.dtype),
                            "bytes": row.bytes,
                            "age_secs": row.age_secs,
                            "idle_secs": row.idle_secs,
                        })
                    })
                    .collect();
                (Response::value(serde_json::Value::Array(rows)), false)
            }
            Bespoke::Nn { kind, args } => {
                lifecycle.lock().unwrap().touch();
                let args = args.unwrap_or_default();
                if matches!(kind.as_str(), "sgd" | "adam" | "adamw" | "rmsprop") {
                    match build_optimizer(registry, &kind, &args) {
                        Ok(optimizer) => (
                            Response::handle(registry.insert_optimizer(optimizer)),
                            false,
                        ),
                        Err((code, message)) => (Response::error(code, message), false),
                    }
                } else {
                    match build_module(registry, &kind, &args) {
                        Ok(module) => (Response::handle(registry.insert_module(module)), false),
                        Err((code, message)) => (Response::error(code, message), false),
                    }
                }
            }
            Bespoke::Forward { module, tensor } => {
                lifecycle.lock().unwrap().touch();
                registry.touch(&module);
                registry.touch(&tensor);
                let result = match registry.get_module(&module) {
                    Ok(m) => match registry.get_tensor(&tensor) {
                        Ok(x) => m.forward(x),
                        Err(lookup) => {
                            return (Response::error(lookup.code(), lookup.message()), false)
                        }
                    },
                    Err(lookup) => {
                        return (Response::error(lookup.code(), lookup.message()), false)
                    }
                };
                match result {
                    Ok(output) => (Response::handle(registry.insert_tensor(output)), false),
                    Err(message) => (Response::error("torch_error", message), false),
                }
            }
            Bespoke::NnParameters { module } => {
                lifecycle.lock().unwrap().touch();
                registry.touch(&module);
                // Live views (issue 0009 decision 4): shallow clones share
                // the TensorImpl — storage, requires_grad, and .grad — so
                // grad/backward work through these handles, and later
                // in-place optimizer steps will be visible through them.
                let params: Vec<Tensor> = match registry.get_module(&module) {
                    Ok(m) => m
                        .parameters()
                        .into_iter()
                        .map(|t| t.shallow_clone())
                        .collect(),
                    Err(lookup) => {
                        return (Response::error(lookup.code(), lookup.message()), false)
                    }
                };
                let handles: Vec<String> = params
                    .into_iter()
                    .map(|t| registry.insert_tensor(t))
                    .collect();
                (Response::handles(handles), false)
            }
            Bespoke::NnInfo { module } => {
                lifecycle.lock().unwrap().touch();
                match registry.get_module(&module) {
                    Ok(m) => (
                        Response::value(serde_json::Value::Array(
                            m.describe()
                                .into_iter()
                                .map(serde_json::Value::String)
                                .collect(),
                        )),
                        false,
                    ),
                    Err(lookup) => (Response::error(lookup.code(), lookup.message()), false),
                }
            }
            Bespoke::Step { optimizer } => {
                lifecycle.lock().unwrap().touch();
                match registry.get_optimizer_mut(&optimizer) {
                    Ok(opt) => {
                        // In-place mutation of grad-requiring leaves is
                        // illegal outside no_grad.
                        let result = tch::no_grad(|| opt.step());
                        match result {
                            Ok(()) => (Response::value(serde_json::json!("stepped")), false),
                            Err(message) => (Response::error("torch_error", message), false),
                        }
                    }
                    Err(lookup) => (Response::error(lookup.code(), lookup.message()), false),
                }
            }
            Bespoke::NnZeroGrad { handle } => {
                lifecycle.lock().unwrap().touch();
                registry.touch(&handle);
                // Both kinds are natural here: an optimizer zeroes its
                // captured params; a module zeroes its own.
                let result = if handle.starts_with("optim://") {
                    registry
                        .get_optimizer_mut(&handle)
                        .map_err(|l| (l.code(), l.message()))
                        .and_then(|opt| opt.zero_grad().map_err(|m| ("torch_error", m)))
                } else {
                    registry
                        .get_module(&handle)
                        .map_err(|l| (l.code(), l.message()))
                        .and_then(|m| {
                            let params: Vec<Tensor> = m
                                .parameters()
                                .into_iter()
                                .map(|t| t.shallow_clone())
                                .collect();
                            crate::nn::zero_grads(&params).map_err(|m| ("torch_error", m))
                        })
                };
                match result {
                    Ok(()) => (Response::value(serde_json::json!("zeroed")), false),
                    Err((code, message)) => (Response::error(code, message), false),
                }
            }
            Bespoke::NnSetLr { optimizer, lr } => {
                lifecycle.lock().unwrap().touch();
                match registry.get_optimizer_mut(&optimizer) {
                    Ok(opt) => {
                        // PyTorch permits lr = 0 (a deliberate freeze);
                        // only negative/non-finite is rejected.
                        if !(lr.is_finite() && lr >= 0.0) {
                            return (
                                Response::error(
                                    "bad_argument",
                                    format!("set_lr: lr must be a non-negative number, got {lr}"),
                                ),
                                false,
                            );
                        }
                        opt.lr = lr;
                        (Response::value(serde_json::json!({ "lr": lr })), false)
                    }
                    Err(lookup) => (Response::error(lookup.code(), lookup.message()), false),
                }
            }
            Bespoke::NnSave { module, path } => {
                lifecycle.lock().unwrap().touch();
                registry.touch(&module);
                let result = registry
                    .get_module(&module)
                    .map_err(|l| (l.code(), l.message()))
                    .and_then(|m| {
                        // Save from CPU (safetensors is host-memory).
                        let entries: Result<Vec<(String, Tensor)>, _> = m
                            .named_state()
                            .into_iter()
                            .map(|(name, t)| {
                                t.f_detach()
                                    .and_then(|d| d.f_to_device(Device::Cpu))
                                    .map(|cpu| (name, cpu))
                            })
                            .collect();
                        let entries =
                            entries.map_err(|e| ("torch_error", convert::tch_error(e)))?;
                        Tensor::write_safetensors(&entries, &path).map_err(|e| {
                            // Unwritable path = caller mistake.
                            ("bad_argument", format!("save: cannot write {path}: {e}"))
                        })
                    });
                match result {
                    Ok(()) => (Response::value(serde_json::json!({ "saved": path })), false),
                    Err((code, message)) => (Response::error(code, message), false),
                }
            }
            Bespoke::NnLoad { module, path } => {
                lifecycle.lock().unwrap().touch();
                registry.touch(&module);
                if !std::path::Path::new(&path).exists() {
                    return (
                        Response::error("bad_argument", format!("load: no such file: {path}")),
                        false,
                    );
                }
                let result = Tensor::read_safetensors(&path)
                    .map_err(|e| ("torch_error", convert::tch_error(e)))
                    .and_then(|entries| {
                        registry
                            .get_module(&module)
                            .map_err(|l| (l.code(), l.message()))
                            .and_then(|m| {
                                m.load_state(&entries).map_err(|msg| ("bad_argument", msg))
                            })
                    });
                match result {
                    Ok(()) => (
                        Response::value(serde_json::json!({ "loaded": path })),
                        false,
                    ),
                    Err((code, message)) => (Response::error(code, message), false),
                }
            }
            Bespoke::NnMode { module, train } => {
                lifecycle.lock().unwrap().touch();
                match registry.get_module_mut(&module) {
                    Ok(m) => {
                        m.set_training(train);
                        (
                            Response::value(serde_json::json!({ "training": train })),
                            false,
                        )
                    }
                    Err(lookup) => (Response::error(lookup.code(), lookup.message()), false),
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

/// Build a tensor from `torch tensor` input: a bare JSON array/scalar, or
/// the `--meta` envelope `{"dtype":…,"shape":…,"data":…}` (issue 0006).
/// Envelope recognition is unambiguous: json_to_tensor rejects ALL objects,
/// so no legitimate non-envelope object input exists.
fn build_input_tensor(
    data: &serde_json::Value,
    dtype_flag: Option<&str>,
    requires_grad: bool,
) -> Result<Tensor, (&'static str, String)> {
    let (payload, envelope_dtype, envelope_shape) = match data.as_object() {
        Some(object) if object.contains_key("data") => {
            let envelope_dtype = match object.get("dtype") {
                None => None,
                Some(serde_json::Value::String(s)) => Some(s.clone()),
                Some(other) => {
                    return Err((
                        "bad_argument",
                        format!("envelope dtype must be a string, got {other}"),
                    ))
                }
            };
            let envelope_shape = match object.get("shape") {
                None => None,
                Some(serde_json::Value::Array(dims)) => {
                    let dims: Result<Vec<i64>, _> = dims
                        .iter()
                        .map(|d| {
                            d.as_i64().ok_or((
                                "bad_argument",
                                format!("envelope shape must be integers, got {d}"),
                            ))
                        })
                        .collect();
                    Some(dims?)
                }
                Some(other) => {
                    return Err((
                        "bad_argument",
                        format!("envelope shape must be an array, got {other}"),
                    ))
                }
            };
            (&object["data"], envelope_dtype, envelope_shape)
        }
        Some(_) => {
            return Err((
                "bad_argument",
                "object input must be a {\"dtype\", \"shape\", \"data\"} envelope".to_string(),
            ))
        }
        None => (data, None, None),
    };

    // Explicit over implicit: a --dtype flag that CONFLICTS with the
    // envelope's dtype is ambiguity, and ambiguity errors.
    let effective_dtype = match (envelope_dtype.as_deref(), dtype_flag) {
        (Some(a), Some(b)) if a != b => {
            return Err((
                "bad_argument",
                format!("envelope dtype {a} conflicts with --dtype {b}"),
            ))
        }
        (Some(a), _) => Some(a.to_string()),
        (None, Some(b)) => Some(b.to_string()),
        (None, None) => None,
    };
    let explicit = match effective_dtype.as_deref() {
        Some(name) => Some(convert::parse_kind(Some(name)).map_err(|e| ("bad_dtype", e))?),
        None => None,
    };
    let kind = convert::resolve_kind(payload, explicit).map_err(|e| ("bad_argument", e))?;
    let tensor =
        convert::json_to_tensor(payload, kind, Device::Mps).map_err(|e| ("bad_argument", e))?;
    if let Some(expected) = envelope_shape {
        let actual = tensor.size();
        if actual != expected {
            return Err((
                "bad_argument",
                format!("envelope shape {expected:?} does not match data shape {actual:?}"),
            ));
        }
    }
    if requires_grad {
        return mark_requires_grad(tensor);
    }
    Ok(tensor)
}

/// Make a tensor a tracked autograd leaf. MUST be the last construction
/// step, applied to the post-transfer MPS tensor: set before a device
/// move, the moved tensor is a NON-leaf whose .grad stays None forever
/// while gradients land on a hidden pre-transfer leaf (the .to() trap,
/// issue 0008). Float dtypes only — PyTorch's own rule.
fn mark_requires_grad(tensor: Tensor) -> Result<Tensor, (&'static str, String)> {
    if !matches!(tensor.kind(), Kind::Float | Kind::Double | Kind::Half) {
        return Err((
            "bad_dtype",
            format!(
                "only floating point tensors can require gradients, got {}",
                convert::kind_name(tensor.kind())
            ),
        ));
    }
    Ok(tensor.set_requires_grad(true))
}

/// Construct a module from `torch nn <kind>` args (issue 0009).
fn build_module(
    registry: &mut Registry,
    kind: &str,
    args: &serde_json::Map<String, serde_json::Value>,
) -> Result<crate::nn::NnModule, (&'static str, String)> {
    use crate::nn::NnModule;
    let int_arg = |name: &str| -> Option<i64> { args.get(name).and_then(|v| v.as_i64()) };
    let str_arg = |name: &str| -> Option<&str> { args.get(name).and_then(|v| v.as_str()) };
    let bool_arg =
        |name: &str| -> bool { args.get(name).and_then(|v| v.as_bool()).unwrap_or(false) };
    match kind {
        "linear" => {
            let in_features = int_arg("in_features").ok_or((
                "bad_argument",
                "nn linear: usage: torch nn linear <in> <out> [--no-bias] [--weight T] [--bias-tensor T]".to_string(),
            ))?;
            let out_features = int_arg("out_features").ok_or((
                "bad_argument",
                "nn linear: missing out_features".to_string(),
            ))?;
            if in_features < 1 || out_features < 1 {
                return Err((
                    "bad_argument",
                    format!("nn linear: features must be >= 1, got {in_features} and {out_features}"),
                ));
            }
            let no_bias = bool_arg("no_bias");
            if no_bias && str_arg("bias_tensor").is_some() {
                return Err((
                    "bad_argument",
                    "nn linear: --bias-tensor contradicts --no-bias".to_string(),
                ));
            }
            // Explicit weights are DEEP-COPIED (state_dict-load semantics:
            // the caller's tensor is never aliased or mutated) and the
            // module's parameter gets requires_grad set LAST on the
            // post-copy tensor regardless of the source's setting.
            let copy_param = |registry: &Registry,
                              handle: &str,
                              expected_shape: &[i64],
                              what: &str|
             -> Result<Tensor, (&'static str, String)> {
                let source = registry
                    .get_tensor(handle)
                    .map_err(|lookup| (lookup.code(), lookup.message()))?;
                let actual = source.size();
                if actual != expected_shape {
                    return Err((
                        "shape_mismatch",
                        format!(
                            "nn linear: {what} must have shape {expected_shape:?}, got {actual:?}"
                        ),
                    ));
                }
                let detached = source.f_detach().map_err(|e| tch("nn", e))?;
                let mut copy = detached.f_zeros_like().map_err(|e| tch("nn", e))?;
                copy.f_copy_(&detached).map_err(|e| tch("nn", e))?;
                Ok(copy.set_requires_grad(true))
            };
            let weight = match str_arg("weight") {
                Some(handle) => {
                    copy_param(registry, handle, &[out_features, in_features], "weight")?
                }
                None => init_linear_param(&[out_features, in_features], in_features)?,
            };
            let bias = if no_bias {
                None
            } else {
                Some(match str_arg("bias_tensor") {
                    Some(handle) => copy_param(registry, handle, &[out_features], "bias")?,
                    None => init_linear_param(&[out_features], in_features)?,
                })
            };
            Ok(NnModule::Linear { weight, bias })
        }
        "conv1d" | "conv2d" | "conv_transpose2d" => {
            let req = |name: &str| -> Result<i64, (&'static str, String)> {
                int_arg(name).ok_or((
                    "bad_argument",
                    format!("nn {kind}: usage: torch nn {kind} <in_channels> <out_channels> <kernel_size> [flags]"),
                ))
            };
            let (in_ch, out_ch, k) = (req("in_channels")?, req("out_channels")?, req("kernel_size")?);
            if in_ch < 1 || out_ch < 1 || k < 1 {
                return Err((
                    "bad_argument",
                    format!("nn {kind}: channels and kernel_size must be >= 1"),
                ));
            }
            let stride = int_arg("stride").unwrap_or(1);
            let padding = int_arg("padding").unwrap_or(0);
            let dilation = int_arg("dilation").unwrap_or(1);
            let groups = int_arg("groups").unwrap_or(1);
            // Validate BEFORE the division below: groups = 0 would panic
            // the connection thread (divide by zero), and non-divisible
            // channels would silently truncate the weight shape.
            if groups < 1 {
                return Err((
                    "bad_argument",
                    format!("nn {kind}: groups must be a positive integer, got {groups}"),
                ));
            }
            let (divided, label) = if kind == "conv_transpose2d" {
                (out_ch, "out_channels")
            } else {
                (in_ch, "in_channels")
            };
            if divided % groups != 0 {
                return Err((
                    "bad_argument",
                    format!("nn {kind}: {label} ({divided}) must be divisible by groups ({groups})"),
                ));
            }
            let no_bias = bool_arg("no_bias");
            if no_bias && str_arg("bias_tensor").is_some() {
                return Err((
                    "bad_argument",
                    format!("nn {kind}: --bias-tensor contradicts --no-bias"),
                ));
            }
            // Weight shapes: conv = [out, in/groups, k(,k)];
            // conv_transpose = [in, out/groups, k, k].
            let weight_shape: Vec<i64> = match kind {
                "conv1d" => vec![out_ch, in_ch / groups, k],
                "conv2d" => vec![out_ch, in_ch / groups, k, k],
                _ => vec![in_ch, out_ch / groups, k, k],
            };
            let fan_in = (in_ch / groups) * if kind == "conv1d" { k } else { k * k };
            let weight = match str_arg("weight") {
                Some(handle) => copy_module_param(registry, handle, &weight_shape, "weight")?,
                None => init_linear_param(&weight_shape, fan_in)?,
            };
            let bias = if no_bias {
                None
            } else {
                Some(match str_arg("bias_tensor") {
                    Some(handle) => copy_module_param(registry, handle, &[out_ch], "bias")?,
                    None => init_linear_param(&[out_ch], fan_in)?,
                })
            };
            Ok(match kind {
                "conv1d" => NnModule::Conv1d {
                    weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                },
                "conv2d" => NnModule::Conv2d {
                    weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                },
                _ => NnModule::ConvTranspose2d {
                    weight,
                    bias,
                    stride,
                    padding,
                    output_padding: int_arg("output_padding").unwrap_or(0),
                    groups,
                    dilation,
                },
            })
        }
        "embedding" => {
            let num = int_arg("num_embeddings").ok_or((
                "bad_argument",
                "nn embedding: usage: torch nn embedding <num_embeddings> <embedding_dim>"
                    .to_string(),
            ))?;
            let dim = int_arg("embedding_dim").ok_or((
                "bad_argument",
                "nn embedding: missing embedding_dim".to_string(),
            ))?;
            let weight = match str_arg("weight") {
                Some(handle) => copy_module_param(registry, handle, &[num, dim], "weight")?,
                None => {
                    // PyTorch init: N(0, 1) — seeded CPU randn convention.
                    let t = Tensor::f_randn([num, dim], (Kind::Float, Device::Cpu))
                        .and_then(|t| t.f_to_device(Device::Mps))
                        .map_err(|e| tch("nn", e))?;
                    t.set_requires_grad(true)
                }
            };
            Ok(NnModule::Embedding { weight })
        }
        "layer_norm" => {
            let shape = match args.get("normalized_shape") {
                Some(serde_json::Value::Array(dims)) => dims
                    .iter()
                    .map(|d| {
                        d.as_i64().ok_or((
                            "bad_argument",
                            "nn layer_norm: normalized_shape must be integers".to_string(),
                        ))
                    })
                    .collect::<Result<Vec<i64>, _>>()?,
                _ => {
                    return Err((
                        "bad_argument",
                        "nn layer_norm: usage: torch nn layer_norm '[shape]'".to_string(),
                    ))
                }
            };
            let eps = float_module_arg(args, "eps", 1e-5);
            let weight = match str_arg("weight") {
                Some(handle) => copy_module_param(registry, handle, &shape, "weight")?,
                None => ones_param(&shape)?,
            };
            let bias = match str_arg("bias_tensor") {
                Some(handle) => copy_module_param(registry, handle, &shape, "bias")?,
                None => zeros_param(&shape)?,
            };
            Ok(NnModule::LayerNorm {
                shape,
                weight,
                bias,
                eps,
            })
        }
        "batch_norm" => {
            let features = int_arg("num_features").ok_or((
                "bad_argument",
                "nn batch_norm: usage: torch nn batch_norm <num_features>".to_string(),
            ))?;
            let eps = float_module_arg(args, "eps", 1e-5);
            let momentum = float_module_arg(args, "momentum", 0.1);
            let weight = ones_param(&[features])?;
            let bias = zeros_param(&[features])?;
            // Buffers, not parameters: no requires_grad.
            let running_mean = Tensor::f_zeros([features], (Kind::Float, Device::Mps))
                .map_err(|e| tch("nn", e))?;
            let running_var = Tensor::f_ones([features], (Kind::Float, Device::Mps))
                .map_err(|e| tch("nn", e))?;
            let num_batches_tracked = Tensor::f_zeros([], (Kind::Int64, Device::Mps))
                .map_err(|e| tch("nn", e))?;
            Ok(NnModule::BatchNorm {
                weight,
                bias,
                running_mean,
                running_var,
                num_batches_tracked,
                eps,
                momentum,
                training: true,
            })
        }
        "group_norm" => {
            let groups = int_arg("num_groups").ok_or((
                "bad_argument",
                "nn group_norm: usage: torch nn group_norm <num_groups> <num_channels>"
                    .to_string(),
            ))?;
            let channels = int_arg("num_channels").ok_or((
                "bad_argument",
                "nn group_norm: missing num_channels".to_string(),
            ))?;
            if groups < 1 || channels < 1 || channels % groups != 0 {
                return Err((
                    "bad_argument",
                    format!("nn group_norm: num_channels ({channels}) must be divisible by num_groups ({groups})"),
                ));
            }
            Ok(NnModule::GroupNorm {
                num_groups: groups,
                weight: ones_param(&[channels])?,
                bias: zeros_param(&[channels])?,
                eps: float_module_arg(args, "eps", 1e-5),
            })
        }
        "dropout" => {
            let p = float_module_arg(args, "p", 0.5);
            if !(0.0..=1.0).contains(&p) {
                return Err((
                    "bad_argument",
                    format!("nn dropout: p must be in [0, 1], got {p}"),
                ));
            }
            Ok(NnModule::Dropout { p, training: true })
        }
        "leaky_relu" => Ok(NnModule::LeakyRelu {
            slope: float_module_arg(args, "negative_slope", 0.01),
        }),
        "softmax" => {
            let dim = int_arg("dim").ok_or((
                "bad_argument",
                "nn softmax: usage: torch nn softmax <dim>".to_string(),
            ))?;
            Ok(NnModule::Softmax { dim })
        }
        "max_pool2d" | "avg_pool2d" => {
            let kernel = int_arg("kernel_size").ok_or((
                "bad_argument",
                format!("nn {kind}: usage: torch nn {kind} <kernel_size> [--stride S --padding P]"),
            ))?;
            let stride = int_arg("stride").unwrap_or(kernel);
            let padding = int_arg("padding").unwrap_or(0);
            Ok(if kind == "max_pool2d" {
                NnModule::MaxPool2d {
                    kernel,
                    stride,
                    padding,
                }
            } else {
                NnModule::AvgPool2d {
                    kernel,
                    stride,
                    padding,
                }
            })
        }
        "flatten" => Ok(NnModule::Flatten {
            // nn.Flatten defaults (1, -1) — NOT the table op's 0.
            start_dim: int_arg("start_dim").unwrap_or(1),
            end_dim: int_arg("end_dim").unwrap_or(-1),
        }),
        "relu" => Ok(NnModule::Relu),
        "sigmoid" => Ok(NnModule::Sigmoid),
        "tanh" => Ok(NnModule::Tanh),
        "gelu" => Ok(NnModule::Gelu),
        "sequential" => {
            let children = match args.get("children") {
                Some(serde_json::Value::Array(items)) => items
                    .iter()
                    .map(|v| {
                        v.as_str().map(str::to_string).ok_or((
                            "bad_argument",
                            "nn sequential: children must be module handles".to_string(),
                        ))
                    })
                    .collect::<Result<Vec<String>, _>>()?,
                _ => Vec::new(),
            };
            if children.is_empty() {
                return Err((
                    "bad_argument",
                    "nn sequential: needs at least one child module".to_string(),
                ));
            }
            // ATOMIC consume (the free invariant): validate ALL children
            // — including duplicates, which fail the seen-twice check —
            // BEFORE removing any from the registry.
            let mut seen = std::collections::HashSet::new();
            for handle in &children {
                registry
                    .get_module(handle)
                    .map_err(|lookup| (lookup.code(), lookup.message()))?;
                if !seen.insert(handle.clone()) {
                    return Err((
                        "bad_argument",
                        format!("nn sequential: duplicate child handle {handle}"),
                    ));
                }
            }
            let mut modules = Vec::with_capacity(children.len());
            for handle in &children {
                let entry = registry.remove(handle).expect("validated above");
                match entry.object {
                    crate::registry::Object::Module(module) => modules.push(module),
                    _ => unreachable!("validated as module"),
                }
            }
            Ok(NnModule::Sequential { children: modules })
        }
        other => Err((
            "bad_argument",
            format!(
                "unknown module kind: {other} (expected linear, relu, sigmoid, tanh, gelu, or sequential)"
            ),
        )),
    }
}

/// Deep-copy an explicit weight into a module parameter: never aliases or
/// mutates the caller's tensor; requires_grad set LAST post-copy.
fn copy_module_param(
    registry: &Registry,
    handle: &str,
    expected_shape: &[i64],
    what: &str,
) -> Result<Tensor, (&'static str, String)> {
    let source = registry
        .get_tensor(handle)
        .map_err(|lookup| (lookup.code(), lookup.message()))?;
    let actual = source.size();
    if actual != expected_shape {
        return Err((
            "shape_mismatch",
            format!("nn: {what} must have shape {expected_shape:?}, got {actual:?}"),
        ));
    }
    let detached = source.f_detach().map_err(|e| tch("nn", e))?;
    let mut copy = detached.f_zeros_like().map_err(|e| tch("nn", e))?;
    copy.f_copy_(&detached).map_err(|e| tch("nn", e))?;
    Ok(copy.set_requires_grad(true))
}

fn ones_param(shape: &[i64]) -> Result<Tensor, (&'static str, String)> {
    let t = Tensor::f_ones(shape, (Kind::Float, Device::Mps)).map_err(|e| tch("nn", e))?;
    Ok(t.set_requires_grad(true))
}

fn zeros_param(shape: &[i64]) -> Result<Tensor, (&'static str, String)> {
    let t = Tensor::f_zeros(shape, (Kind::Float, Device::Mps)).map_err(|e| tch("nn", e))?;
    Ok(t.set_requires_grad(true))
}

fn float_module_arg(
    args: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    default: f64,
) -> f64 {
    args.get(name).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Construct an optimizer over a module's parameters (issue 0009 exp 4).
fn build_optimizer(
    registry: &mut Registry,
    kind: &str,
    args: &serde_json::Map<String, serde_json::Value>,
) -> Result<crate::nn::Optimizer, (&'static str, String)> {
    use crate::nn::{OptimKind, Optimizer};
    let float_arg = |name: &str, default: f64| -> f64 {
        args.get(name).and_then(|v| v.as_f64()).unwrap_or(default)
    };
    let bool_arg =
        |name: &str| -> bool { args.get(name).and_then(|v| v.as_bool()).unwrap_or(false) };
    let module_handle = args
        .get("module")
        .and_then(|v| v.as_str())
        .ok_or(("bad_argument", format!("nn {kind}: missing module handle")))?;
    let module = registry
        .get_module(module_handle)
        .map_err(|l| (l.code(), l.message()))?;
    let params: Vec<Tensor> = module
        .parameters()
        .into_iter()
        .map(|t| t.shallow_clone())
        .collect();
    if params.is_empty() {
        return Err((
            "bad_argument",
            format!("nn {kind}: module has no parameters to optimize"),
        ));
    }
    let weight_decay = float_arg("weight_decay", if kind == "adamw" { 0.01 } else { 0.0 });
    let (optim_kind, default_lr) = match kind {
        "sgd" => {
            let momentum = float_arg("momentum", 0.0);
            let dampening = float_arg("dampening", 0.0);
            let nesterov = bool_arg("nesterov");
            if nesterov && (momentum <= 0.0 || dampening != 0.0) {
                return Err((
                    "bad_argument",
                    "nn sgd: nesterov requires momentum > 0 and dampening == 0".to_string(),
                ));
            }
            let lr = args
                .get("lr")
                .and_then(|v| v.as_f64())
                .ok_or(("bad_argument", "nn sgd: --lr is required".to_string()))?;
            return Ok(Optimizer::new(
                OptimKind::Sgd {
                    momentum,
                    dampening,
                    nesterov,
                },
                lr,
                weight_decay,
                params,
            ));
        }
        "adam" | "adamw" => (
            OptimKind::Adam {
                beta1: float_arg("beta1", 0.9),
                beta2: float_arg("beta2", 0.999),
                eps: float_arg("eps", 1e-8),
                decoupled: kind == "adamw",
            },
            0.001,
        ),
        "rmsprop" => (
            OptimKind::RmsProp {
                alpha: float_arg("alpha", 0.99),
                eps: float_arg("eps", 1e-8),
                momentum: float_arg("momentum", 0.0),
            },
            0.01,
        ),
        other => return Err(("bad_argument", format!("unknown optimizer kind: {other}"))),
    };
    let lr = float_arg("lr", default_lr);
    Ok(Optimizer::new(optim_kind, lr, weight_decay, params))
}

/// PyTorch nn.Linear default init: U(-1/sqrt(in), 1/sqrt(in)) for both
/// weight and bias (kaiming_uniform(a=sqrt(5)) reduces to exactly this).
/// Drawn on the seeded CPU generator (the randn convention), moved to
/// MPS, requires_grad set LAST (the issue-0008 non-leaf trap).
fn init_linear_param(shape: &[i64], in_features: i64) -> Result<Tensor, (&'static str, String)> {
    let bound = 1.0 / (in_features as f64).sqrt();
    let uniform = Tensor::f_rand(shape, (Kind::Float, Device::Cpu))
        .and_then(|t| t.f_mul_scalar(2.0 * bound))
        .and_then(|t| t.f_sub_scalar(bound))
        .and_then(|t| t.f_to_device(Device::Mps))
        .map_err(|e| tch("nn", e))?;
    Ok(uniform.set_requires_grad(true))
}

/// `--reduction mean|sum|none` (PyTorch's values; default mean).
fn parse_reduction(p: &Params) -> Result<tch::Reduction, OpError> {
    match p.str("reduction") {
        None | Some("mean") => Ok(tch::Reduction::Mean),
        Some("sum") => Ok(tch::Reduction::Sum),
        Some("none") => Ok(tch::Reduction::None),
        Some(other) => Err((
            "bad_argument",
            format!("invalid reduction: {other} (expected mean, sum, or none)"),
        )),
    }
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
                    ParamKind::HandleOrScalar => value.is_number() || value.is_string(),
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

    // Touch pass (issue 0006): mark operand and param-tensor handles as
    // used BEFORE the immutable resolution borrows begin. touch() is a
    // no-op on absent handles, so resolution still owns the error.
    for handle in tensor_handles {
        registry.touch(handle);
    }
    for param in spec.params {
        if param.kind == ParamKind::HandleOrScalar {
            if let Some(serde_json::Value::String(handle)) = params.map.get(param.name) {
                registry.touch(handle);
            }
        }
    }

    // Handles.
    let mut tensors: Vec<&Tensor> = Vec::with_capacity(tensor_handles.len());
    for handle in tensor_handles {
        match registry.get_tensor(handle) {
            Ok(t) => tensors.push(t),
            Err(lookup) => return Response::error(lookup.code(), lookup.message()),
        }
    }
    debug_assert!(
        tensors.iter().all(|t| t.device() == Device::Mps),
        "registry invariant violated: all tensors live on MPS"
    );

    // Resolve HandleOrScalar params whose value is a handle string into
    // tensor refs (issue 0005 exp 5). Plain &self borrows coexist with the
    // operand borrows; registry.insert only runs after apply returns.
    let mut param_tensors: std::collections::HashMap<&str, &Tensor> =
        std::collections::HashMap::new();
    for param in spec.params {
        if param.kind == ParamKind::HandleOrScalar {
            if let Some(serde_json::Value::String(handle)) = params.map.get(param.name) {
                match registry.get_tensor(handle) {
                    Ok(tensor) => {
                        param_tensors.insert(param.name, tensor);
                    }
                    Err(lookup) => return Response::error(lookup.code(), lookup.message()),
                }
            }
        }
    }

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

    match apply(spec, &tensors, &params, &param_tensors) {
        Ok(Applied::Tensors(outputs)) => {
            debug_assert!(match spec.results {
                ResultKind::Handles(n) => n == outputs.len(),
                // max/min/median: 1 or 2; split/chunk: N >= 1 (exp 4).
                ResultKind::VariableHandles => !outputs.is_empty(),
                _ => false,
            });
            Response::handles(
                outputs
                    .into_iter()
                    .map(|t| registry.insert_tensor(t))
                    .collect(),
            )
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

fn apply(
    spec: &OpSpec,
    t: &[&Tensor],
    p: &Params,
    pt: &std::collections::HashMap<&str, &Tensor>,
) -> Result<Applied, OpError> {
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
        "pow" => match pt.get("exponent") {
            Some(exponent) => one(op, t[0].f_pow(exponent)),
            None => match p.scalar("exponent").expect("required") {
                Scalar::Int(i) => one(op, t[0].f_pow_tensor_scalar(i)),
                Scalar::Float(f) => one(op, t[0].f_pow_tensor_scalar(f)),
            },
        },
        "clamp" => {
            let to_scalar = |s: Scalar| -> tch::Scalar {
                match s {
                    Scalar::Int(i) => i.into(),
                    Scalar::Float(f) => f.into(),
                }
            };
            // Tensor bounds (HandleOrScalar) take f_clamp_tensor; scalar
            // bounds keep the original single/double-bound calls.
            let (min_t, max_t) = (pt.get("min"), pt.get("max"));
            if min_t.is_some() || max_t.is_some() {
                return one(op, t[0].f_clamp_tensor(min_t.copied(), max_t.copied()));
            }
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
            let result = match p.scalar("value").expect("required") {
                Scalar::Int(i) => Tensor::f_full(&shape, i, (kind, Device::Mps)),
                Scalar::Float(f) => Tensor::f_full(&shape, f, (kind, Device::Mps)),
            };
            let tensor = result.map_err(|e| tch(op, e))?;
            let tensor = if p.bool("requires_grad") {
                mark_requires_grad(tensor)?
            } else {
                tensor
            };
            Ok(Applied::Tensors(vec![tensor]))
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
            let tensor = Tensor::f_randn(&shape, (kind, Device::Cpu))
                .and_then(|t| t.f_to_device(Device::Mps))
                .map_err(|e| tch(op, e))?;
            // requires_grad LAST, on the post-transfer tensor (the .to()
            // non-leaf trap, issue 0008): set before the move, the MPS
            // tensor is a non-leaf whose grad stays None forever.
            let tensor = if p.bool("requires_grad") {
                mark_requires_grad(tensor)?
            } else {
                tensor
            };
            Ok(Applied::Tensors(vec![tensor]))
        }
        // --- reductions sweep (issue 0005 exp 3) ---
        "prod" => match p.int("dim") {
            Some(dim) => one(
                op,
                t[0].f_prod_dim_int(dim, p.bool("keepdim"), None::<Kind>),
            ),
            None => one(op, t[0].f_prod(None::<Kind>)),
        },
        "amax" => match p.int("dim") {
            Some(dim) => one(op, t[0].f_amax(&[dim][..], p.bool("keepdim"))),
            None => one(op, t[0].f_amax(&[][..], p.bool("keepdim"))),
        },
        "amin" => match p.int("dim") {
            Some(dim) => one(op, t[0].f_amin(&[dim][..], p.bool("keepdim"))),
            None => one(op, t[0].f_amin(&[][..], p.bool("keepdim"))),
        },
        "max" | "min" | "median" => match p.int("dim") {
            Some(dim) => {
                let pair = match op {
                    "max" => t[0].f_max_dim(dim, p.bool("keepdim")),
                    "min" => t[0].f_min_dim(dim, p.bool("keepdim")),
                    _ => t[0].f_median_dim(dim, p.bool("keepdim")),
                };
                pair.map(|(values, indices)| Applied::Tensors(vec![values, indices]))
                    .map_err(|e| tch(op, e))
            }
            None => match op {
                "max" => one(op, t[0].f_max()),
                "min" => one(op, t[0].f_min()),
                _ => one(op, t[0].f_median()),
            },
        },
        "argmax" => one(op, t[0].f_argmax(p.int("dim"), p.bool("keepdim"))),
        "argmin" => one(op, t[0].f_argmin(p.int("dim"), p.bool("keepdim"))),
        "all" => match p.int("dim") {
            Some(dim) => one(op, t[0].f_all_dims(&[dim][..], p.bool("keepdim"))),
            None => one(op, t[0].f_all()),
        },
        "any" => match p.int("dim") {
            Some(dim) => one(op, t[0].f_any_dims(&[dim][..], p.bool("keepdim"))),
            None => one(op, t[0].f_any()),
        },
        "std" | "var" => {
            let correction: tch::Scalar = p.int("correction").unwrap_or(1).into();
            let dim_holder;
            let dim: Option<&[i64]> = match p.int("dim") {
                Some(d) => {
                    dim_holder = [d];
                    Some(&dim_holder[..])
                }
                None => None,
            };
            let result = if op == "std" {
                t[0].f_std_correction(dim, correction, p.bool("keepdim"))
            } else {
                t[0].f_var_correction(dim, correction, p.bool("keepdim"))
            };
            one(op, result)
        }
        "nansum" => {
            let dim_holder;
            let dim: Option<&[i64]> = match p.int("dim") {
                Some(d) => {
                    dim_holder = [d];
                    Some(&dim_holder[..])
                }
                None => None,
            };
            one(op, t[0].f_nansum(dim, p.bool("keepdim"), None::<Kind>))
        }
        "logsumexp" => one(
            op,
            t[0].f_logsumexp(&[p.int("dim").expect("required")][..], p.bool("keepdim")),
        ),
        "count_nonzero" => one(op, t[0].f_count_nonzero(p.int("dim"))),
        "cumsum" => one(
            op,
            t[0].f_cumsum(p.int("dim").expect("required"), None::<Kind>),
        ),
        "cumprod" => one(
            op,
            t[0].f_cumprod(p.int("dim").expect("required"), None::<Kind>),
        ),
        "norm" => {
            let pval = p.float("p").unwrap_or(2.0);
            match p.int("dim") {
                Some(dim) => one(
                    op,
                    t[0].f_norm_scalaropt_dim(pval, &[dim][..], p.bool("keepdim")),
                ),
                None => one(op, t[0].f_norm_scalaropt_dtype(pval, Kind::Float)),
            }
        }
        // --- comparison sweep ---
        "gt" => one(op, t[0].f_gt_tensor(t[1])),
        "lt" => one(op, t[0].f_lt_tensor(t[1])),
        "ge" => one(op, t[0].f_ge_tensor(t[1])),
        "le" => one(op, t[0].f_le_tensor(t[1])),
        "ne" => one(op, t[0].f_ne_tensor(t[1])),
        "logical_and" => one(op, t[0].f_logical_and(t[1])),
        "logical_or" => one(op, t[0].f_logical_or(t[1])),
        "logical_xor" => one(op, t[0].f_logical_xor(t[1])),
        "logical_not" => one(op, t[0].f_logical_not()),
        "isclose" => one(
            op,
            t[0].f_isclose(
                t[1],
                p.float("rtol").unwrap_or(1e-5),
                p.float("atol").unwrap_or(1e-8),
                false,
            ),
        ),
        "isnan" => one(op, t[0].f_isnan()),
        "isinf" => one(op, t[0].f_isinf()),
        "isfinite" => one(op, t[0].f_isfinite()),
        "isposinf" => one(op, t[0].f_isposinf()),
        "isneginf" => one(op, t[0].f_isneginf()),
        "equal" => t[0]
            .f_equal(t[1])
            .map(|b| Applied::Value(serde_json::Value::Bool(b)))
            .map_err(|e| tch(op, e)),
        "topk" => t[0]
            .f_topk(
                p.int("k").expect("required"),
                p.int("dim").unwrap_or(-1),
                !p.bool("smallest"),
                true,
            )
            .map(|(values, indices)| Applied::Tensors(vec![values, indices]))
            .map_err(|e| tch(op, e)),
        "argsort" => one(
            op,
            t[0].f_argsort(p.int("dim").unwrap_or(-1), p.bool("descending")),
        ),
        // --- linalg + shape sweep (issue 0005 exp 4) ---
        "matmul" => one(op, t[0].f_matmul(t[1])),
        "bmm" => one(op, t[0].f_bmm(t[1])),
        "dot" => one(op, t[0].f_dot(t[1])),
        "outer" => one(op, t[0].f_outer(t[1])),
        "einsum" => one(
            op,
            Tensor::f_einsum(p.str("equation").expect("required"), t, None::<&[i64]>),
        ),
        "tril" => one(op, t[0].f_tril(p.int("diagonal").unwrap_or(0))),
        "triu" => one(op, t[0].f_triu(p.int("diagonal").unwrap_or(0))),
        "diag" => one(op, t[0].f_diag(p.int("diagonal").unwrap_or(0))),
        "trace" => one(op, t[0].f_trace()),
        "det" => one(op, t[0].f_det()),
        "inverse" => one(op, t[0].f_inverse()),
        "svd" => t[0]
            .f_svd(false, true)
            .map(|(u, s, v)| Applied::Tensors(vec![u, s, v]))
            .map_err(|e| tch(op, e)),
        "solve" => one(op, Tensor::f_linalg_solve(t[0], t[1], true)),
        "reshape" => one(op, t[0].f_reshape(p.int_list("shape").expect("required"))),
        "permute" => one(op, t[0].f_permute(p.int_list("dims").expect("required"))),
        "transpose" => one(
            op,
            t[0].f_transpose(
                p.int("dim0").expect("required"),
                p.int("dim1").expect("required"),
            ),
        ),
        "t" => {
            let size = t[0].size();
            if size.len() != 2 {
                return Err((
                    "shape_mismatch",
                    format!("t: requires a 2-D tensor, got shape {size:?}"),
                ));
            }
            one(op, t[0].f_tr())
        }
        "squeeze" => match p.int("dim") {
            Some(dim) => one(op, t[0].f_squeeze_dim(dim)),
            None => one(op, t[0].f_squeeze()),
        },
        "unsqueeze" => one(op, t[0].f_unsqueeze(p.int("dim").expect("required"))),
        "flatten" => one(
            op,
            t[0].f_flatten(
                p.int("start_dim").unwrap_or(0),
                p.int("end_dim").unwrap_or(-1),
            ),
        ),
        "stack" => one(op, Tensor::f_stack(t, p.int("dim").unwrap_or(0))),
        "split" => t[0]
            .f_split(
                p.int("split_size").expect("required"),
                p.int("dim").unwrap_or(0),
            )
            .map(Applied::Tensors)
            .map_err(|e| tch(op, e)),
        "chunk" => t[0]
            .f_chunk(
                p.int("chunks").expect("required"),
                p.int("dim").unwrap_or(0),
            )
            .map(Applied::Tensors)
            .map_err(|e| tch(op, e)),
        "gather" => one(
            op,
            t[0].f_gather(p.int("dim").expect("required"), t[1], false),
        ),
        "index_select" => one(
            op,
            t[0].f_index_select(p.int("dim").expect("required"), t[1]),
        ),
        "masked_select" => {
            // Numeric mask cast via != 0 (documented nutorch-ism: no bool
            // input path exists yet).
            let mask = t[1].f_ne(0).map_err(|e| tch(op, e))?;
            one(op, t[0].f_masked_select(&mask))
        }
        "where" => {
            // Numeric cond cast via != 0 (documented nutorch-ism).
            let cond = t[0].f_ne(0).map_err(|e| tch(op, e))?;
            one(op, t[1].f_where_self(&cond, t[2]))
        }
        "narrow" => one(
            op,
            t[0].f_narrow(
                p.int("dim").expect("required"),
                p.int("start").expect("required"),
                p.int("length").expect("required"),
            ),
        ),
        "flip" => one(op, t[0].f_flip(p.int_list("dims").expect("required"))),
        "roll" => {
            let shifts = p.int_list("shifts").expect("required");
            let dims = p.int_list("dims").unwrap_or_default();
            one(op, t[0].f_roll(&shifts, &dims))
        }
        "repeat" => one(op, t[0].f_repeat(p.int_list("repeats").expect("required"))),
        "repeat_interleave" => one(
            op,
            t[0].f_repeat_interleave_self_int(
                p.int("repeats").expect("required"),
                p.int("dim"),
                None,
            ),
        ),
        "movedim" => one(
            op,
            t[0].f_movedim(
                p.int("source").expect("required"),
                p.int("destination").expect("required"),
            ),
        ),
        // --- creation + remainder sweep (issue 0005 exp 5) ---
        "zeros" | "ones" => {
            let shape = p.int_list("shape").expect("required");
            validate_shape(op, &shape)?;
            let kind = parse_table_kind(op, p.str("dtype"))?;
            let result = if op == "zeros" {
                Tensor::f_zeros(&shape, (kind, Device::Mps))
            } else {
                Tensor::f_ones(&shape, (kind, Device::Mps))
            };
            let tensor = result.map_err(|e| tch(op, e))?;
            let tensor = if p.bool("requires_grad") {
                mark_requires_grad(tensor)?
            } else {
                tensor
            };
            Ok(Applied::Tensors(vec![tensor]))
        }
        "eye" => {
            let n = p.int("n").expect("required");
            match p.int("m") {
                Some(m) => one(op, Tensor::f_eye_m(n, m, (Kind::Float, Device::Mps))),
                None => one(op, Tensor::f_eye(n, (Kind::Float, Device::Mps))),
            }
        }
        "arange" => {
            let to_scalar = |s: Scalar| -> tch::Scalar {
                match s {
                    Scalar::Int(i) => i.into(),
                    Scalar::Float(f) => f.into(),
                }
            };
            let end = to_scalar(p.scalar("end").expect("required"));
            let start = to_scalar(p.scalar("start").unwrap_or(Scalar::Int(0)));
            let step = to_scalar(p.scalar("step").unwrap_or(Scalar::Int(1)));
            one(
                op,
                Tensor::f_arange_start_step(start, end, step, (Kind::Float, Device::Mps)),
            )
        }
        "linspace" => {
            let to_scalar = |s: Scalar| -> tch::Scalar {
                match s {
                    Scalar::Int(i) => i.into(),
                    Scalar::Float(f) => f.into(),
                }
            };
            one(
                op,
                Tensor::f_linspace(
                    to_scalar(p.scalar("start").expect("required")),
                    to_scalar(p.scalar("end").expect("required")),
                    p.int("steps").expect("required"),
                    (Kind::Float, Device::Mps),
                ),
            )
        }
        "rand" => {
            let shape = p.int_list("shape").expect("required");
            validate_shape(op, &shape)?;
            // Seeded CPU generator -> MPS (the randn convention);
            // requires_grad LAST, post-transfer (the .to() non-leaf trap).
            let tensor = Tensor::f_rand(&shape, (Kind::Float, Device::Cpu))
                .and_then(|t| t.f_to_device(Device::Mps))
                .map_err(|e| tch(op, e))?;
            let tensor = if p.bool("requires_grad") {
                mark_requires_grad(tensor)?
            } else {
                tensor
            };
            Ok(Applied::Tensors(vec![tensor]))
        }
        "randint" => {
            let shape = p.int_list("shape").expect("required");
            validate_shape(op, &shape)?;
            let low = p.int("low").unwrap_or(0);
            let high = p.int("high").expect("required");
            one(
                op,
                Tensor::f_randint_low(low, high, &shape, (Kind::Int64, Device::Cpu))
                    .and_then(|t| t.f_to_device(Device::Mps)),
            )
        }
        "zeros_like" => one(op, t[0].f_zeros_like()),
        "ones_like" => one(op, t[0].f_ones_like()),
        "full_like" => match p.scalar("value").expect("required") {
            Scalar::Int(i) => one(op, t[0].f_full_like(i)),
            Scalar::Float(f) => one(op, t[0].f_full_like(f)),
        },
        "rand_like" | "randn_like" => {
            // By-shape on the seeded CPU generator -> MPS (golden parity).
            let shape = t[0].size();
            let result = if op == "rand_like" {
                Tensor::f_rand(&shape, (Kind::Float, Device::Cpu))
            } else {
                Tensor::f_randn(&shape, (Kind::Float, Device::Cpu))
            };
            one(op, result.and_then(|x| x.f_to_device(Device::Mps)))
        }
        "lerp" => match pt.get("weight") {
            Some(weight) => one(op, t[0].f_lerp_tensor(t[1], weight)),
            None => match p.scalar("weight").expect("required") {
                Scalar::Int(i) => one(op, t[0].f_lerp(t[1], i)),
                Scalar::Float(f) => one(op, t[0].f_lerp(t[1], f)),
            },
        },
        "addcmul" | "addcdiv" => {
            // tch 0.24 exposes no `value` parameter for addcmul/addcdiv, so
            // the scaled form is computed manually: a + value * (b ∘ c).
            let combined = if op == "addcmul" {
                t[1].f_mul(t[2])
            } else {
                t[1].f_div(t[2])
            }
            .map_err(|e| tch(op, e))?;
            let result = match p.scalar("value") {
                None => t[0].f_add(&combined),
                Some(value) => {
                    let scaled = match value {
                        Scalar::Int(i) => combined.f_mul_scalar(i),
                        Scalar::Float(f) => combined.f_mul_scalar(f),
                    }
                    .map_err(|e| tch(op, e))?;
                    t[0].f_add(&scaled)
                }
            };
            one(op, result)
        }
        "cross" => one(op, t[0].f_cross(t[1], p.int("dim"))),
        "kron" => one(op, t[0].f_kron(t[1])),
        "tensordot" => {
            let dims = p.int("dims").unwrap_or(2);
            let axes: Vec<i64> = (0..dims).collect();
            let a_axes: Vec<i64> =
                (t[0].size().len() as i64 - dims..t[0].size().len() as i64).collect();
            one(op, t[0].f_tensordot(t[1], &a_axes, &axes))
        }
        "take_along_dim" => one(
            op,
            t[0].f_take_along_dim(t[1], p.int("dim").expect("required")),
        ),
        // ATen's searchsorted self is the VALUES tensor; our spec order is
        // (sorted_sequence, values), matching torch.searchsorted.
        "searchsorted" => one(
            op,
            t[1].f_searchsorted(t[0], false, false, "left", None::<&Tensor>),
        ),
        "bucketize" => one(op, t[0].f_bucketize(t[1], false, false)),
        "msort" => one(op, t[0].f_msort()),
        "diff" => one(
            op,
            t[0].f_diff(
                1,
                p.int("dim").unwrap_or(-1),
                None::<&Tensor>,
                None::<&Tensor>,
            ),
        ),
        "scatter" => one(
            op,
            t[0].f_scatter(p.int("dim").expect("required"), t[1], t[2]),
        ),
        "bitwise_and" => one(op, t[0].f_bitwise_and_tensor(t[1])),
        "bitwise_or" => one(op, t[0].f_bitwise_or_tensor(t[1])),
        "bitwise_xor" => one(op, t[0].f_bitwise_xor_tensor(t[1])),
        "bitwise_not" => one(op, t[0].f_bitwise_not()),
        "bitwise_left_shift" => one(op, t[0].f_bitwise_left_shift(t[1])),
        "bitwise_right_shift" => one(op, t[0].f_bitwise_right_shift(t[1])),
        // torch.unique flattens first; tch exposes no flattened f_unique,
        // so flatten explicitly then unique along dim 0 (a rank-2 golden
        // pins this — f_unique_dim(-1) alone diverges for rank >= 2).
        "unique" => t[0]
            .f_flatten(0, -1)
            .and_then(|flat| flat.f_unique_dim(0, true, false, false))
            .map(|(values, _, _)| Applied::Tensors(vec![values]))
            .map_err(|e| tch(op, e)),
        // --- autograd surface (issue 0008) ---
        "backward" => {
            if !t[0].requires_grad() {
                return Err((
                    "bad_argument",
                    "backward: tensor does not require gradients (create with --requires_grad)"
                        .to_string(),
                ));
            }
            let numel = t[0].numel();
            if numel != 1 {
                return Err((
                    "bad_argument",
                    format!(
                        "backward: needs a scalar loss, got shape {:?} ({numel} elements) — reduce first (e.g. sum or mean)",
                        t[0].size()
                    ),
                ));
            }
            t[0].f_backward().map_err(|e| tch(op, e))?;
            Ok(Applied::Nothing)
        }
        "grad" => {
            let grad = t[0].f_grad().map_err(|e| tch(op, e))?;
            if !grad.defined() {
                return Err((
                    "bad_argument",
                    "no gradient: run backward first".to_string(),
                ));
            }
            // Snapshot, not a live view: .grad accumulates in place under
            // later backward calls, and a handle that silently changes is
            // action at a distance the shell cannot see. NOTE: tch's
            // f_clone is the clone-INTO-out variant (an aliasing trap the
            // unit test caught) — deep-copy explicitly instead.
            let detached = grad.f_detach().map_err(|e| tch(op, e))?;
            let mut snapshot = detached.f_zeros_like().map_err(|e| tch(op, e))?;
            snapshot.f_copy_(&detached).map_err(|e| tch(op, e))?;
            Ok(Applied::Tensors(vec![snapshot]))
        }
        "detach" => one(op, t[0].f_detach()),
        "zero_grad" => {
            // tch's own recipe (Tensor::zero_grad): detach_ then zero_ on
            // the grad alias, so the in-place zero is not tracked. The
            // grad stays DEFINED (zeros) — a later `grad` returns zeros.
            let mut grad = t[0].f_grad().map_err(|e| tch(op, e))?;
            if grad.defined() {
                let _ = grad.f_detach_().map_err(|e| tch(op, e))?;
                let _ = grad.f_zero_().map_err(|e| tch(op, e))?;
            }
            Ok(Applied::Nothing)
        }
        // --- losses (issue 0009 exp 3) ---
        "mse_loss" => one(op, t[0].f_mse_loss(t[1], parse_reduction(p)?)),
        "l1_loss" => one(op, t[0].f_l1_loss(t[1], parse_reduction(p)?)),
        "smooth_l1_loss" => one(
            op,
            t[0].f_smooth_l1_loss(t[1], parse_reduction(p)?, p.float("beta").unwrap_or(1.0)),
        ),
        "huber_loss" => one(
            op,
            t[0].f_huber_loss(t[1], parse_reduction(p)?, p.float("delta").unwrap_or(1.0)),
        ),
        "cross_entropy" => one(
            op,
            t[0].f_cross_entropy_loss(t[1], None::<&Tensor>, parse_reduction(p)?, -100, 0.0),
        ),
        "nll_loss" => one(
            op,
            t[0].f_nll_loss(t[1], None::<&Tensor>, parse_reduction(p)?, -100),
        ),
        "binary_cross_entropy" => one(
            op,
            t[0].f_binary_cross_entropy(t[1], None::<&Tensor>, parse_reduction(p)?),
        ),
        "binary_cross_entropy_with_logits" => one(
            op,
            t[0].f_binary_cross_entropy_with_logits(
                t[1],
                None::<&Tensor>,
                None::<&Tensor>,
                parse_reduction(p)?,
            ),
        ),
        "kl_div" => one(
            op,
            t[0].f_kl_div(t[1], parse_reduction(p)?, p.bool("log_target")),
        ),
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
        assert_eq!(registry.get_tensor(&h).unwrap().device(), Device::Mps);
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
        // Bare strings are MALFORMED handles post issue 0009 (no kind
        // prefix — the recorded clean break); well-formed-but-absent stays
        // unknown_handle; wrong prefix on a real object is wrong_kind.
        let (code, _) = expect_error(run(&mut registry, json!({"op":"sin","tensors":["nope"]})));
        assert_eq!(code, "bad_argument");
        let (code, _) = expect_error(run(
            &mut registry,
            json!({"op":"sin","tensors":["tensor://absent"]}),
        ));
        assert_eq!(code, "unknown_handle");
        let real = registry.insert_tensor(
            convert::json_to_tensor(&json!([1.0]), Kind::Float, Device::Mps).unwrap(),
        );
        let as_module = real.replace("tensor://", "nn://");
        let (code, message) = expect_error(run(
            &mut registry,
            json!({"op":"sin","tensors":[as_module]}),
        ));
        assert_eq!(code, "wrong_kind");
        assert!(message.contains("refers to a tensor, not a module"));
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
        let handle = registry.insert_tensor(non_finite);
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
            .get_tensor(&out)
            .unwrap()
            .f_to_device(Device::Cpu)
            .unwrap();
        assert_eq!(
            convert::tensor_to_json(&cpu).unwrap(),
            json!([0.5, 100.0, -100.0])
        );
    }
}

#[cfg(test)]
mod non_finite_predicate_semantics {
    use super::*;
    use serde_json::json;

    /// Golden inputs must be finite JSON, so the TRUE path of the predicate
    /// family is proven here with directly constructed non-finite values
    /// ([NaN, inf, -inf, 1.0] via 0-division on MPS float32).
    #[test]
    fn predicates_detect_non_finite_values() {
        let mut registry = Registry::new();
        let numerator =
            convert::json_to_tensor(&json!([0.0, 1.0, -1.0, 1.0]), Kind::Float, Device::Mps)
                .unwrap();
        let denominator =
            convert::json_to_tensor(&json!([0.0, 0.0, 0.0, 1.0]), Kind::Float, Device::Mps)
                .unwrap();
        let non_finite = numerator.f_div(&denominator).unwrap(); // [NaN, inf, -inf, 1.0]
        let handle = registry.insert_tensor(non_finite);

        let expectations = [
            ("isnan", json!([true, false, false, false])),
            ("isinf", json!([false, true, true, false])),
            ("isfinite", json!([false, false, false, true])),
            ("isposinf", json!([false, true, false, false])),
            ("isneginf", json!([false, false, true, false])),
        ];
        for (name, expected) in expectations {
            let spec = nutorch_ops::find(name).unwrap();
            let response = execute_table(
                &mut registry,
                spec,
                &[handle.clone()],
                &serde_json::Map::new(),
            );
            let out = match response {
                Response::Handles { handles, .. } => handles[0].clone(),
                other => panic!("{name}: expected handles, got {other:?}"),
            };
            let cpu = registry
                .get_tensor(&out)
                .unwrap()
                .f_to_device(Device::Cpu)
                .unwrap();
            assert_eq!(
                convert::tensor_to_json(&cpu).unwrap(),
                expected,
                "{name} true-path"
            );
        }
    }
}

#[cfg(test)]
mod free_semantics {
    use super::*;
    use crate::lifecycle::Lifecycle;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn run_free(registry: &mut Registry, request: serde_json::Value) -> Response {
        let parsed = parse_request(&request.to_string()).expect("parses");
        let lifecycle = Mutex::new(Lifecycle::new(None));
        let socket = PathBuf::from("/tmp/test.sock");
        let (response, shutdown) = handle_request(registry, &lifecycle, &socket, parsed);
        assert!(!shutdown);
        response
    }

    fn seed(registry: &mut Registry, n: usize) -> Vec<String> {
        (0..n)
            .map(|i| {
                let t =
                    convert::json_to_tensor(&json!([i as f64]), Kind::Float, Device::Mps).unwrap();
                registry.insert_tensor(t)
            })
            .collect()
    }

    #[test]
    fn free_removes_exactly_the_named_handles() {
        let mut registry = Registry::new();
        let h = seed(&mut registry, 3);
        let response = run_free(&mut registry, json!({"op":"free","handles":[h[0], h[2]]}));
        assert!(matches!(response, Response::Value { .. }));
        assert_eq!(registry.len(), 1);
        assert!(registry.check_ok(&h[1]));
    }

    #[test]
    fn free_is_atomic_known_unknown_known_frees_nothing() {
        let mut registry = Registry::new();
        let h = seed(&mut registry, 2);
        let response = run_free(
            &mut registry,
            json!({"op":"free","handles":[h[0], "nope", h[1]]}),
        );
        match response {
            // Bare string = malformed handle (issue 0009); atomicity must
            // hold under malformed middles too.
            Response::Error { code, .. } => assert_eq!(code, "bad_argument"),
            other => panic!("expected error, got {other:?}"),
        }
        // BOTH known handles survive — a remove-as-you-go bug fails here.
        assert!(registry.check_ok(&h[0]));
        assert!(registry.check_ok(&h[1]));
    }

    #[test]
    fn double_free_errors_visibly() {
        let mut registry = Registry::new();
        let h = seed(&mut registry, 1);
        let first = run_free(&mut registry, json!({"op":"free","handles":[h[0]]}));
        assert!(matches!(first, Response::Value { .. }));
        let second = run_free(&mut registry, json!({"op":"free","handles":[h[0]]}));
        match second {
            Response::Error { code, .. } => assert_eq!(code, "unknown_handle"),
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn free_all_empties_the_registry_and_reports_the_count() {
        let mut registry = Registry::new();
        seed(&mut registry, 4);
        let response = run_free(&mut registry, json!({"op":"free","all":true}));
        match response {
            Response::Value { value, .. } => assert_eq!(value["freed"], 4),
            other => panic!("expected value, got {other:?}"),
        }
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn free_rejects_bad_shapes() {
        let mut registry = Registry::new();
        let h = seed(&mut registry, 1);
        // Both present.
        let both = run_free(
            &mut registry,
            json!({"op":"free","all":true,"handles":[h[0]]}),
        );
        match both {
            Response::Error { code, .. } => assert_eq!(code, "bad_request"),
            other => panic!("expected error, got {other:?}"),
        }
        // Neither present (and all:false counts as not requested).
        for request in [json!({"op":"free"}), json!({"op":"free","all":false})] {
            let response = run_free(&mut registry, request);
            match response {
                Response::Error { code, .. } => assert_eq!(code, "bad_request"),
                other => panic!("expected error, got {other:?}"),
            }
        }
        // all:false WITH handles frees the handles.
        let response = run_free(
            &mut registry,
            json!({"op":"free","all":false,"handles":[h[0]]}),
        );
        assert!(matches!(response, Response::Value { .. }));
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn free_touches_the_idle_lease() {
        let mut registry = Registry::new();
        let h = seed(&mut registry, 1);
        let parsed =
            parse_request(&json!({"op":"free","handles":[h[0]]}).to_string()).expect("parses");
        let lifecycle = Mutex::new(Lifecycle::new(Some(std::time::Duration::from_secs(3600))));
        std::thread::sleep(std::time::Duration::from_millis(20));
        let idle_before = lifecycle.lock().unwrap().idle_secs();
        let socket = PathBuf::from("/tmp/test.sock");
        let _ = handle_request(&mut registry, &lifecycle, &socket, parsed);
        let idle_after = lifecycle.lock().unwrap().idle_secs();
        assert!(idle_after <= idle_before);
    }
}

#[cfg(test)]
mod tensors_listing_semantics {
    use super::*;
    use crate::lifecycle::Lifecycle;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::Duration;

    fn dispatch(
        registry: &mut Registry,
        lifecycle: &Mutex<Lifecycle>,
        request: serde_json::Value,
    ) -> Response {
        let parsed = parse_request(&request.to_string()).expect("parses");
        let socket = PathBuf::from("/tmp/test.sock");
        handle_request(registry, lifecycle, &socket, parsed).0
    }

    #[test]
    fn rows_match_registry_contents_including_bool() {
        let mut registry = Registry::new();
        let lifecycle = Mutex::new(Lifecycle::new(None));
        let a = registry.insert_tensor(
            convert::json_to_tensor(&json!([1.0, 2.0]), Kind::Float, Device::Mps).unwrap(),
        );
        let spec = nutorch_ops::find("eq").unwrap();
        let response = execute_table(
            &mut registry,
            spec,
            &[a.clone(), a.clone()],
            &serde_json::Map::new(),
        );
        assert!(matches!(response, Response::Handles { .. }));

        let listing = dispatch(&mut registry, &lifecycle, json!({"op":"tensors"}));
        let rows = match listing {
            Response::Value { value, .. } => value.as_array().unwrap().clone(),
            other => panic!("expected value, got {other:?}"),
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["handle"], a); // oldest first
        assert_eq!(rows[0]["dtype"], "float32");
        assert_eq!(rows[0]["bytes"], 8);
        assert_eq!(rows[1]["dtype"], "bool"); // the eq result
        assert_eq!(rows[1]["bytes"], 2); // 1 byte per element
    }

    #[test]
    fn ops_reset_operand_idle_and_listing_resets_nothing() {
        let mut registry = Registry::new();
        let lifecycle = Mutex::new(Lifecycle::new(Some(Duration::from_secs(3600))));
        let a = registry.insert_tensor(
            convert::json_to_tensor(&json!([1.0]), Kind::Float, Device::Mps).unwrap(),
        );
        let b = registry.insert_tensor(
            convert::json_to_tensor(&json!([2.0]), Kind::Float, Device::Mps).unwrap(),
        );
        std::thread::sleep(Duration::from_millis(1100));

        // An op touches its operand...
        let spec = nutorch_ops::find("sin").unwrap();
        let _ = execute_table(&mut registry, spec, &[a.clone()], &serde_json::Map::new());
        let idle_of = |registry: &Registry, handle: &str| {
            registry
                .list()
                .into_iter()
                .find(|row| row.handle == handle)
                .unwrap()
                .idle_secs
        };
        assert_eq!(idle_of(&registry, &a), 0);
        // ...but not the bystander.
        assert!(idle_of(&registry, &b) >= 1);

        // The listing itself touches neither tensors nor the lease.
        std::thread::sleep(Duration::from_millis(1100));
        let lease_idle_before = lifecycle.lock().unwrap().idle_secs();
        let _ = dispatch(&mut registry, &lifecycle, json!({"op":"tensors"}));
        assert!(idle_of(&registry, &a) >= 1);
        assert!(lifecycle.lock().unwrap().idle_secs() >= lease_idle_before);
    }
}

#[cfg(test)]
mod roundtrip_semantics {
    use super::*;
    use serde_json::json;

    fn cpu_json(registry: &Registry, handle: &str) -> serde_json::Value {
        let cpu = registry
            .get_tensor(handle)
            .unwrap()
            .f_to_device(Device::Cpu)
            .unwrap();
        convert::tensor_to_json(&cpu).unwrap()
    }

    #[test]
    fn bool_data_infers_bool_and_round_trips() {
        let tensor = build_input_tensor(&json!([true, false, true]), None, false).unwrap();
        assert_eq!(tensor.kind(), Kind::Bool);
        let mut registry = Registry::new();
        let h = registry.insert_tensor(tensor);
        assert_eq!(cpu_json(&registry, &h), json!([true, false, true]));
    }

    #[test]
    fn mixed_bool_and_number_without_dtype_errors() {
        let err = build_input_tensor(&json!([true, 1]), None, false).unwrap_err();
        assert_eq!(err.0, "bad_argument");
        assert!(err.1.contains("mixed booleans and numbers"));
    }

    #[test]
    fn explicit_dtype_casts_both_ways_like_pytorch() {
        // numbers -> bool via != 0 (the [2,0,-1] case proves != 0, not == 1)
        let t = build_input_tensor(&json!([0, 1, 2]), Some("bool"), false).unwrap();
        let mut registry = Registry::new();
        let h = registry.insert_tensor(t);
        assert_eq!(cpu_json(&registry, &h), json!([false, true, true]));
        let t = build_input_tensor(&json!([2, 0, -1]), Some("bool"), false).unwrap();
        let h = registry.insert_tensor(t);
        assert_eq!(cpu_json(&registry, &h), json!([true, false, true]));
        // bools -> float32
        let t = build_input_tensor(&json!([true, false]), Some("float32"), false).unwrap();
        assert_eq!(t.kind(), Kind::Float);
        let h = registry.insert_tensor(t);
        assert_eq!(cpu_json(&registry, &h), json!([1.0, 0.0]));
    }

    #[test]
    fn non_finite_tokens_round_trip_bit_exactly() {
        let t =
            build_input_tensor(&json!(["NaN", "Infinity", "-Infinity", 1.5]), None, false).unwrap();
        assert_eq!(t.kind(), Kind::Float);
        let mut registry = Registry::new();
        let h = registry.insert_tensor(t);
        // NaN -> NaN, ±inf -> ±inf, finite untouched — and NO null anywhere.
        assert_eq!(
            cpu_json(&registry, &h),
            json!(["NaN", "Infinity", "-Infinity", 1.5])
        );
        // Constructed non-finite values (0-division) export as tokens too.
        let spec = nutorch_ops::find("div").unwrap();
        let a = registry.insert_tensor(
            convert::json_to_tensor(&json!([0.0, 1.0, -1.0]), Kind::Float, Device::Mps).unwrap(),
        );
        let zero = registry.insert_tensor(
            convert::json_to_tensor(&json!([0.0, 0.0, 0.0]), Kind::Float, Device::Mps).unwrap(),
        );
        let response = execute_table(&mut registry, spec, &[a, zero], &serde_json::Map::new());
        let out = match response {
            Response::Handles { handles, .. } => handles[0].clone(),
            other => panic!("expected handles, got {other:?}"),
        };
        assert_eq!(
            cpu_json(&registry, &out),
            json!(["NaN", "Infinity", "-Infinity"])
        );
    }

    #[test]
    fn non_finite_tokens_reject_integer_dtypes() {
        let err = build_input_tensor(&json!(["NaN"]), Some("int64"), false).unwrap_err();
        assert_eq!(err.0, "bad_argument");
    }

    #[test]
    fn envelope_round_trip_preserves_dtype() {
        for (dtype, data) in [("int64", json!([1, 2, 3])), ("bool", json!([true, false]))] {
            let envelope = json!({"dtype": dtype, "data": data});
            let tensor = build_input_tensor(&envelope, None, false).unwrap();
            assert_eq!(convert::kind_name(tensor.kind()), dtype);
        }
    }

    #[test]
    fn envelope_conflicts_and_mismatches_error() {
        // Conflicting dtype flag.
        let envelope = json!({"dtype": "int64", "data": [1, 2]});
        let err = build_input_tensor(&envelope, Some("float32"), false).unwrap_err();
        assert_eq!(err.0, "bad_argument");
        assert!(err.1.contains("conflicts"));
        // Identical dtype flag is fine.
        assert!(build_input_tensor(&envelope, Some("int64"), false).is_ok());
        // Wrong shape.
        let envelope = json!({"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6]});
        let err = build_input_tensor(&envelope, None, false).unwrap_err();
        assert_eq!(err.0, "bad_argument");
        assert!(err.1.contains("does not match"));
        // Matching shape is fine.
        let envelope = json!({"shape": [3], "data": [1, 2, 3]});
        assert!(build_input_tensor(&envelope, None, false).is_ok());
        // Object without "data" is rejected.
        let err = build_input_tensor(&json!({"values": [1]}), None, false).unwrap_err();
        assert_eq!(err.0, "bad_argument");
    }
}

#[cfg(test)]
mod autograd_semantics {
    use super::*;
    use serde_json::json;

    fn run_op(registry: &mut Registry, name: &str, handles: &[String]) -> Response {
        let spec = nutorch_ops::find(name).unwrap();
        execute_table(registry, spec, handles, &serde_json::Map::new())
    }

    fn first_handle(response: Response) -> String {
        match response {
            Response::Handles { handles, .. } => handles[0].clone(),
            other => panic!("expected handles, got {other:?}"),
        }
    }

    fn value_of(registry: &Registry, handle: &str) -> serde_json::Value {
        let cpu = registry
            .get_tensor(handle)
            .unwrap()
            .f_to_device(Device::Cpu)
            .unwrap();
        convert::tensor_to_json(&cpu).unwrap()
    }

    fn tracked_leaf(registry: &mut Registry, data: serde_json::Value) -> String {
        let tensor = build_input_tensor(&data, None, true).unwrap();
        registry.insert_tensor(tensor)
    }

    /// x*x summed: grad = 2x. Builds the loss and runs backward.
    fn square_loss_backward(registry: &mut Registry, x: &str) -> String {
        let y = first_handle(run_op(registry, "mul", &[x.to_string(), x.to_string()]));
        let loss = first_handle(run_op(registry, "sum", &[y]));
        assert!(matches!(
            run_op(registry, "backward", &[loss.clone()]),
            Response::Value { .. } | Response::Handles { .. }
        ));
        loss
    }

    #[test]
    fn gradients_accumulate_and_zero_grad_resets_to_zeros() {
        let mut registry = Registry::new();
        let x = tracked_leaf(&mut registry, json!([1.0, 2.0]));
        square_loss_backward(&mut registry, &x);
        let g1 = first_handle(run_op(&mut registry, "grad", &[x.clone()]));
        assert_eq!(value_of(&registry, &g1), json!([2.0, 4.0]));
        // Second backward on a FRESH graph accumulates: grad doubles.
        square_loss_backward(&mut registry, &x);
        let g2 = first_handle(run_op(&mut registry, "grad", &[x.clone()]));
        assert_eq!(value_of(&registry, &g2), json!([4.0, 8.0]));
        // The first snapshot is immutable — untouched by the second pass.
        assert_eq!(value_of(&registry, &g1), json!([2.0, 4.0]));
        // zero_grad leaves a DEFINED zeros grad (pinned semantics).
        assert!(matches!(
            run_op(&mut registry, "zero_grad", &[x.clone()]),
            Response::Value { .. } | Response::Handles { .. }
        ));
        let g3 = first_handle(run_op(&mut registry, "grad", &[x]));
        assert_eq!(value_of(&registry, &g3), json!([0.0, 0.0]));
    }

    #[test]
    fn grad_before_backward_is_a_clean_error() {
        let mut registry = Registry::new();
        let x = tracked_leaf(&mut registry, json!([1.0]));
        match run_op(&mut registry, "grad", &[x]) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("run backward first"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn backward_on_non_scalar_names_the_shape() {
        let mut registry = Registry::new();
        let x = tracked_leaf(&mut registry, json!([1.0, 2.0, 3.0]));
        match run_op(&mut registry, "backward", &[x]) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(
                    error.contains("[3]"),
                    "error should name the shape: {error}"
                );
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn backward_on_untracked_tensor_errors() {
        let mut registry = Registry::new();
        let tensor = build_input_tensor(&json!(1.5), None, false).unwrap();
        let x = registry.insert_tensor(tensor);
        match run_op(&mut registry, "backward", &[x]) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("does not require gradients"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn detach_produces_an_untracked_handle() {
        let mut registry = Registry::new();
        let x = tracked_leaf(&mut registry, json!([1.0, 2.0]));
        let y = first_handle(run_op(&mut registry, "mul", &[x.clone(), x.clone()]));
        assert!(registry.get_tensor(&y).unwrap().requires_grad()); // tracked result
        let d = first_handle(run_op(&mut registry, "detach", &[y]));
        assert!(!registry.get_tensor(&d).unwrap().requires_grad());
    }

    #[test]
    fn freeing_an_intermediate_does_not_break_backward() {
        let mut registry = Registry::new();
        let x = tracked_leaf(&mut registry, json!([1.0, 3.0]));
        let y = first_handle(run_op(&mut registry, "mul", &[x.clone(), x.clone()]));
        let loss = first_handle(run_op(&mut registry, "sum", &[y.clone()]));
        // Free the intermediate's HANDLE; the graph still holds the tensor.
        registry.remove(&y);
        assert!(matches!(
            run_op(&mut registry, "backward", &[loss]),
            Response::Value { .. } | Response::Handles { .. }
        ));
        let g = first_handle(run_op(&mut registry, "grad", &[x]));
        assert_eq!(value_of(&registry, &g), json!([2.0, 6.0]));
    }

    #[test]
    fn requires_grad_rejects_int_dtypes() {
        let err = build_input_tensor(&json!([1, 2]), Some("int64"), true).unwrap_err();
        assert_eq!(err.0, "bad_dtype");
        assert!(err.1.contains("floating point"));
    }

    #[test]
    fn creation_ops_yield_tracked_mps_leaves() {
        // The .to() trap regression test: randn --requires_grad must be a
        // LEAF on MPS whose grad populates after backward.
        let mut registry = Registry::new();
        let spec = nutorch_ops::find("randn").unwrap();
        let mut params = serde_json::Map::new();
        params.insert("shape".into(), json!([2]));
        params.insert("requires_grad".into(), json!(true));
        let x = match execute_table(&mut registry, spec, &[], &params) {
            Response::Handles { handles, .. } => handles[0].clone(),
            other => panic!("expected handles, got {other:?}"),
        };
        assert!(registry.get_tensor(&x).unwrap().requires_grad());
        assert_eq!(registry.get_tensor(&x).unwrap().device(), Device::Mps);
        square_loss_backward(&mut registry, &x);
        let g = first_handle(run_op(&mut registry, "grad", &[x.clone()]));
        // grad = 2x, elementwise — verify against the tensor's own value.
        let x_cpu = registry
            .get_tensor(&x)
            .unwrap()
            .f_to_device(Device::Cpu)
            .unwrap();
        let doubled = x_cpu.f_mul_scalar(2).unwrap();
        let g_cpu = registry
            .get_tensor(&g)
            .unwrap()
            .f_to_device(Device::Cpu)
            .unwrap();
        assert_eq!(
            convert::tensor_to_json(&g_cpu).unwrap(),
            convert::tensor_to_json(&doubled).unwrap()
        );
    }
}

#[cfg(test)]
mod module_foundation_semantics {
    use super::*;
    use crate::lifecycle::Lifecycle;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn bespoke(registry: &mut Registry, request: serde_json::Value) -> Response {
        let parsed = parse_request(&request.to_string()).expect("parses");
        let lifecycle = Mutex::new(Lifecycle::new(None));
        let socket = PathBuf::from("/tmp/test.sock");
        handle_request(registry, &lifecycle, &socket, parsed).0
    }

    fn handle_of(response: Response) -> String {
        match response {
            Response::Handle { handle, .. } => handle,
            other => panic!("expected handle, got {other:?}"),
        }
    }

    fn make_tensor(registry: &mut Registry, data: serde_json::Value) -> String {
        let t = convert::json_to_tensor(&data, Kind::Float, Device::Mps).unwrap();
        registry.insert_tensor(t)
    }

    fn linear(registry: &mut Registry, args: serde_json::Value) -> Response {
        bespoke(registry, json!({"op":"nn","kind":"linear","args": args}))
    }

    #[test]
    fn construction_shapes_and_seeded_determinism() {
        let mut registry = Registry::new();
        let m = handle_of(linear(
            &mut registry,
            json!({"in_features":2,"out_features":3}),
        ));
        assert!(m.starts_with("nn://"));
        let module = registry.get_module(&m).unwrap();
        let params = module.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].size(), vec![3, 2]); // weight [out, in]
        assert_eq!(params[1].size(), vec![3]); // bias [out]
        assert!(params.iter().all(|p| p.requires_grad()));
        assert!(params.iter().all(|p| p.device() == Device::Mps));

        // Seeded determinism: same seed → identical weights.
        let weights_of = |registry: &mut Registry| -> serde_json::Value {
            tch::manual_seed(7);
            let m = handle_of(linear(registry, json!({"in_features":2,"out_features":3})));
            let module = registry.get_module(&m).unwrap();
            let cpu = module.parameters()[0].f_to_device(Device::Cpu).unwrap();
            convert::tensor_to_json(&cpu).unwrap()
        };
        assert_eq!(weights_of(&mut registry), weights_of(&mut registry));
    }

    #[test]
    fn explicit_weights_are_deep_copied_and_track_gradients() {
        let mut registry = Registry::new();
        let w = make_tensor(&mut registry, json!([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
        assert!(!registry.get_tensor(&w).unwrap().requires_grad());
        let m = handle_of(linear(
            &mut registry,
            json!({"in_features":2,"out_features":3,"weight": w, "no_bias": true}),
        ));
        let module = registry.get_module(&m).unwrap();
        // Module param tracks regardless of the source's setting…
        assert!(module.parameters()[0].requires_grad());
        // …and the SOURCE tensor is unchanged (deep copy, never aliased).
        assert!(!registry.get_tensor(&w).unwrap().requires_grad());
    }

    #[test]
    fn explicit_weight_shape_mismatch_errors() {
        let mut registry = Registry::new();
        let w = make_tensor(&mut registry, json!([[1.0, 2.0]])); // [1,2], not [3,2]
        match linear(
            &mut registry,
            json!({"in_features":2,"out_features":3,"weight": w}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "shape_mismatch");
                assert!(error.contains("[3, 2]"));
            }
            other => panic!("expected error, got {other:?}"),
        }
        // --bias-tensor with --no-bias contradicts.
        match linear(
            &mut registry,
            json!({"in_features":2,"out_features":3,"no_bias":true,"bias_tensor":"tensor://x"}),
        ) {
            Response::Error { code, .. } => assert_eq!(code, "bad_argument"),
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn sequential_consumes_children_atomically() {
        let mut registry = Registry::new();
        let a = handle_of(linear(
            &mut registry,
            json!({"in_features":2,"out_features":2}),
        ));
        let relu = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"relu","args":{}}),
        ));
        // Bad middle: registry unchanged (both children survive).
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sequential","args":{"children":[a, "nn://absent", relu]}}),
        ) {
            Response::Error { code, .. } => assert_eq!(code, "unknown_handle"),
            other => panic!("expected error, got {other:?}"),
        }
        assert!(registry.get_module(&a).is_ok());
        assert!(registry.get_module(&relu).is_ok());
        // Duplicate child: rejected, registry unchanged.
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sequential","args":{"children":[a, a]}}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("duplicate"));
            }
            other => panic!("expected error, got {other:?}"),
        }
        assert!(registry.get_module(&a).is_ok());
        // Happy path consumes: children gone, composite forward works.
        let m = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sequential","args":{"children":[a.clone(), relu.clone()]}}),
        ));
        assert!(registry.get_module(&a).is_err());
        assert!(registry.get_module(&relu).is_err());
        let x = make_tensor(&mut registry, json!([[1.0, -1.0]]));
        let y = bespoke(
            &mut registry,
            json!({"op":"forward","module": m, "tensor": x}),
        );
        assert!(matches!(y, Response::Handle { .. }));
    }

    #[test]
    fn forward_kind_validates_both_ways() {
        let mut registry = Registry::new();
        let m = handle_of(linear(
            &mut registry,
            json!({"in_features":2,"out_features":2}),
        ));
        let x = make_tensor(&mut registry, json!([[1.0, 2.0]]));
        // Swapped: tensor where module expected, module where tensor expected.
        match bespoke(
            &mut registry,
            json!({"op":"forward","module": x, "tensor": m}),
        ) {
            Response::Error { code, .. } => assert_eq!(code, "wrong_kind"),
            other => panic!("expected error, got {other:?}"),
        }
        // torch value on a module handle is wrong_kind too.
        match bespoke(&mut registry, json!({"op":"value","handle": m})) {
            Response::Error { code, .. } => assert_eq!(code, "wrong_kind"),
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn parameters_are_live_views_proven_by_grad_identity() {
        let mut registry = Registry::new();
        let m = handle_of(linear(
            &mut registry,
            json!({"in_features":2,"out_features":1}),
        ));
        let x = make_tensor(&mut registry, json!([[1.0, 2.0]]));
        let y = handle_of(bespoke(
            &mut registry,
            json!({"op":"forward","module": m, "tensor": x}),
        ));
        let run_table = |registry: &mut Registry, op: &str, handles: &[String]| -> Response {
            let spec = nutorch_ops::find(op).unwrap();
            execute_table(registry, spec, handles, &serde_json::Map::new())
        };
        let loss = match run_table(&mut registry, "sum", &[y]) {
            Response::Handles { handles, .. } => handles[0].clone(),
            other => panic!("{other:?}"),
        };
        let _ = run_table(&mut registry, "backward", &[loss]);
        // Gradients are readable through the parameters handles —
        // impossible unless they alias the module's tensors.
        let params = match bespoke(&mut registry, json!({"op":"nn_parameters","module": m})) {
            Response::Handles { handles, .. } => handles,
            other => panic!("{other:?}"),
        };
        let grad = run_table(&mut registry, "grad", &[params[0].clone()]);
        match grad {
            Response::Handles { handles, .. } => {
                let g = registry.get_tensor(&handles[0]).unwrap();
                assert_eq!(g.size(), vec![1, 2]);
                let cpu = g.f_to_device(Device::Cpu).unwrap();
                assert_eq!(
                    convert::tensor_to_json(&cpu).unwrap(),
                    json!([[1.0, 2.0]]) // d(sum(xW^T+b))/dW = x
                );
            }
            other => panic!("expected grad handles, got {other:?}"),
        }
    }

    #[test]
    fn activation_modules_match_table_ops() {
        let mut registry = Registry::new();
        let x = make_tensor(&mut registry, json!([-1.5, 0.0, 2.0]));
        for kind in ["relu", "sigmoid", "tanh"] {
            let m = handle_of(bespoke(
                &mut registry,
                json!({"op":"nn","kind": kind, "args": {}}),
            ));
            let via_module = handle_of(bespoke(
                &mut registry,
                json!({"op":"forward","module": m, "tensor": x.clone()}),
            ));
            let spec = nutorch_ops::find(kind).unwrap();
            let via_table =
                match execute_table(&mut registry, spec, &[x.clone()], &serde_json::Map::new()) {
                    Response::Handles { handles, .. } => handles[0].clone(),
                    other => panic!("{other:?}"),
                };
            let a = registry.get_tensor(&via_module).unwrap();
            let b = registry.get_tensor(&via_table).unwrap();
            assert!(a.f_equal(b).unwrap(), "{kind} module vs table op");
        }
    }

    #[test]
    fn unknown_module_kind_and_empty_sequential_error() {
        let mut registry = Registry::new();
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"transformer","args":{}}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("unknown module kind"));
            }
            other => panic!("expected error, got {other:?}"),
        }
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sequential","args":{"children":[]}}),
        ) {
            Response::Error { code, .. } => assert_eq!(code, "bad_argument"),
            other => panic!("expected error, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod loss_semantics {
    use super::*;
    use serde_json::json;

    #[test]
    fn bad_reduction_names_the_choices() {
        let mut registry = Registry::new();
        let t = convert::json_to_tensor(&json!([1.0]), Kind::Float, Device::Mps).unwrap();
        let a = registry.insert_tensor(t);
        let t = convert::json_to_tensor(&json!([2.0]), Kind::Float, Device::Mps).unwrap();
        let b = registry.insert_tensor(t);
        let spec = nutorch_ops::find("mse_loss").unwrap();
        let mut params = serde_json::Map::new();
        params.insert("reduction".into(), json!("median"));
        match execute_table(&mut registry, spec, &[a, b], &params) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("mean, sum, or none"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod optimizer_semantics {
    use super::*;
    use crate::lifecycle::Lifecycle;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn bespoke(registry: &mut Registry, request: serde_json::Value) -> Response {
        let parsed = parse_request(&request.to_string()).expect("parses");
        let lifecycle = Mutex::new(Lifecycle::new(None));
        let socket = PathBuf::from("/tmp/test.sock");
        handle_request(registry, &lifecycle, &socket, parsed).0
    }

    fn handle_of(response: Response) -> String {
        match response {
            Response::Handle { handle, .. } => handle,
            other => panic!("expected handle, got {other:?}"),
        }
    }

    fn weight_of(registry: &Registry, module: &str) -> serde_json::Value {
        let m = registry.get_module(module).unwrap();
        let cpu = m.parameters()[0].f_to_device(Device::Cpu).unwrap();
        convert::tensor_to_json(&cpu).unwrap()
    }

    /// linear(1,1) no bias, weight [[w]]; loss = (w*1 - 0)^2 → dL/dw = 2w.
    fn setup(registry: &mut Registry, w: f64) -> (String, String, String) {
        let wt = convert::json_to_tensor(&json!([[w]]), Kind::Float, Device::Mps).unwrap();
        let wh = registry.insert_tensor(wt);
        let module = handle_of(bespoke(
            registry,
            json!({"op":"nn","kind":"linear","args":{
                "in_features":1,"out_features":1,"weight": wh,"no_bias":true}}),
        ));
        let x = registry.insert_tensor(
            convert::json_to_tensor(&json!([[1.0]]), Kind::Float, Device::Mps).unwrap(),
        );
        let target = registry.insert_tensor(
            convert::json_to_tensor(&json!([[0.0]]), Kind::Float, Device::Mps).unwrap(),
        );
        (module, x, target)
    }

    fn train_step(registry: &mut Registry, module: &str, x: &str, target: &str, opt: &str) {
        let pred = handle_of(bespoke(
            registry,
            json!({"op":"forward","module": module, "tensor": x}),
        ));
        let spec = nutorch_ops::find("mse_loss").unwrap();
        let loss = match execute_table(
            registry,
            spec,
            &[pred, target.to_string()],
            &serde_json::Map::new(),
        ) {
            Response::Handles { handles, .. } => handles[0].clone(),
            other => panic!("{other:?}"),
        };
        let spec = nutorch_ops::find("backward").unwrap();
        let _ = execute_table(registry, spec, &[loss], &serde_json::Map::new());
        assert!(matches!(
            bespoke(registry, json!({"op":"step","optimizer": opt})),
            Response::Value { .. }
        ));
        let _ = bespoke(registry, json!({"op":"nn_zero_grad","handle": opt}));
    }

    #[test]
    fn hand_checked_sgd_step() {
        // w=1, x=1, target=0: loss=w², grad=2w=2; lr 0.1 → w' = 1 - 0.2 = 0.8.
        let mut registry = Registry::new();
        let (module, x, target) = setup(&mut registry, 1.0);
        let opt = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sgd","args":{"module": module, "lr": 0.1}}),
        ));
        train_step(&mut registry, &module, &x, &target, &opt);
        let w = weight_of(&registry, &module)[0][0].as_f64().unwrap();
        assert!((w - 0.8).abs() < 1e-6, "got {w}"); // f32 arithmetic
    }

    #[test]
    fn momentum_buffer_evolves_and_set_lr_changes_step_size() {
        let mut registry = Registry::new();
        let (module, x, target) = setup(&mut registry, 1.0);
        let opt = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sgd","args":{"module": module, "lr": 0.1, "momentum": 0.5}}),
        ));
        // Step 1: grad 2, buf = grad = 2 (FIRST-step clone), w = 1 - 0.2 = 0.8.
        train_step(&mut registry, &module, &x, &target, &opt);
        let w = weight_of(&registry, &module)[0][0].as_f64().unwrap();
        assert!((w - 0.8).abs() < 1e-6, "got {w}");
        // Step 2: grad = 1.6, buf = 0.5*2 + 1.6 = 2.6, w = 0.8 - 0.26 = 0.54.
        train_step(&mut registry, &module, &x, &target, &opt);
        let w = weight_of(&registry, &module);
        let value = w[0][0].as_f64().unwrap();
        assert!((value - 0.54).abs() < 1e-6, "got {value}");
        // set_lr to 0 → next step moves nothing.
        let _ = bespoke(
            &mut registry,
            json!({"op":"nn_set_lr","optimizer": opt, "lr": 1e-12}),
        );
        train_step(&mut registry, &module, &x, &target, &opt);
        let w2 = weight_of(&registry, &module);
        assert!((w2[0][0].as_f64().unwrap() - value).abs() < 1e-6);
    }

    #[test]
    fn step_skips_params_without_grad_and_rejects_empty_modules() {
        let mut registry = Registry::new();
        let (module, _, _) = setup(&mut registry, 1.0);
        let opt = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sgd","args":{"module": module, "lr": 0.1}}),
        ));
        // No backward has run: step is a no-op, not an error.
        assert!(matches!(
            bespoke(&mut registry, json!({"op":"step","optimizer": opt})),
            Response::Value { .. }
        ));
        assert_eq!(weight_of(&registry, &module), json!([[1.0]]));
        // Zero-parameter module rejected at construction.
        let relu = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"relu","args":{}}),
        ));
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sgd","args":{"module": relu, "lr": 0.1}}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("no parameters"));
            }
            other => panic!("expected error, got {other:?}"),
        }
        // Nesterov constraint.
        let (module2, _, _) = setup(&mut registry, 1.0);
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sgd","args":{"module": module2, "lr": 0.1, "nesterov": true}}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("nesterov requires"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn zero_grad_works_on_both_handle_kinds() {
        let mut registry = Registry::new();
        let (module, x, target) = setup(&mut registry, 1.0);
        let opt = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sgd","args":{"module": module, "lr": 0.1}}),
        ));
        train_step(&mut registry, &module, &x, &target, &opt); // leaves zeroed grads
                                                               // Re-backward then zero via the MODULE handle this time.
        let pred = handle_of(bespoke(
            &mut registry,
            json!({"op":"forward","module": module, "tensor": x}),
        ));
        let spec = nutorch_ops::find("sum").unwrap();
        let loss = match execute_table(&mut registry, spec, &[pred], &serde_json::Map::new()) {
            Response::Handles { handles, .. } => handles[0].clone(),
            other => panic!("{other:?}"),
        };
        let spec = nutorch_ops::find("backward").unwrap();
        let _ = execute_table(&mut registry, spec, &[loss], &serde_json::Map::new());
        assert!(matches!(
            bespoke(&mut registry, json!({"op":"nn_zero_grad","handle": module})),
            Response::Value { .. }
        ));
        // The module's param grad is now zeros.
        let m = registry.get_module(&module).unwrap();
        let grad = m.parameters()[0].f_grad().unwrap();
        assert!(grad.defined());
        let cpu = grad.f_detach().unwrap().f_to_device(Device::Cpu).unwrap();
        assert_eq!(convert::tensor_to_json(&cpu).unwrap(), json!([[0.0]]));
    }
}

#[cfg(test)]
mod module_sweep_semantics {
    use super::*;
    use crate::lifecycle::Lifecycle;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn bespoke(registry: &mut Registry, request: serde_json::Value) -> Response {
        let parsed = parse_request(&request.to_string()).expect("parses");
        let lifecycle = Mutex::new(Lifecycle::new(None));
        let socket = PathBuf::from("/tmp/test.sock");
        handle_request(registry, &lifecycle, &socket, parsed).0
    }

    fn handle_of(response: Response) -> String {
        match response {
            Response::Handle { handle, .. } => handle,
            other => panic!("expected handle, got {other:?}"),
        }
    }

    fn make(registry: &mut Registry, data: serde_json::Value) -> String {
        let t = convert::json_to_tensor(&data, Kind::Float, Device::Mps).unwrap();
        registry.insert_tensor(t)
    }

    fn forward(registry: &mut Registry, module: &str, x: &str) -> String {
        handle_of(bespoke(
            registry,
            json!({"op":"forward","module": module, "tensor": x}),
        ))
    }

    fn cpu_json(registry: &Registry, h: &str) -> serde_json::Value {
        let cpu = registry
            .get_tensor(h)
            .unwrap()
            .f_detach()
            .unwrap()
            .f_to_device(Device::Cpu)
            .unwrap();
        convert::tensor_to_json(&cpu).unwrap()
    }

    #[test]
    fn dropout_train_quartet_and_edges() {
        let mut registry = Registry::new();
        let x = make(&mut registry, serde_json::json!(vec![1.0; 1000]));
        let d = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"dropout","args":{"p": 0.25}}),
        ));
        // Determinism under manual_seed.
        tch::manual_seed(5);
        let __t = forward(&mut registry, &d, &x);
        let a = cpu_json(&registry, &__t);
        tch::manual_seed(5);
        let __t = forward(&mut registry, &d, &x);
        let b = cpu_json(&registry, &__t);
        assert_eq!(a, b);
        // Zero fraction ≈ p; kept elements scaled by 1/(1-p).
        let values = a.as_array().unwrap();
        let zeros = values.iter().filter(|v| v.as_f64().unwrap() == 0.0).count();
        let frac = zeros as f64 / values.len() as f64;
        assert!((frac - 0.25).abs() < 0.07, "zero fraction {frac}");
        let kept = values
            .iter()
            .map(|v| v.as_f64().unwrap())
            .find(|v| *v != 0.0)
            .unwrap();
        assert!((kept - 1.0 / 0.75).abs() < 1e-5, "scale {kept}");
        // Eval mode: identity.
        bespoke(
            &mut registry,
            json!({"op":"nn_mode","module": d, "train": false}),
        );
        let __t = forward(&mut registry, &d, &x);
        let e = cpu_json(&registry, &__t);
        assert_eq!(e, serde_json::json!(vec![1.0; 1000]));
        // p = 1 → all zeros, NO NaN.
        let d1 = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"dropout","args":{"p": 1.0}}),
        ));
        let __t = forward(&mut registry, &d1, &x);
        let z = cpu_json(&registry, &__t);
        assert!(z
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v.as_f64() == Some(0.0)));
        // p = 0 → identity even in train mode.
        let d0 = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"dropout","args":{"p": 0.0}}),
        ));
        let __t = forward(&mut registry, &d0, &x);
        let i = cpu_json(&registry, &__t);
        assert_eq!(i, serde_json::json!(vec![1.0; 1000]));
        // p out of range rejected.
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"dropout","args":{"p": 1.5}}),
        ) {
            Response::Error { code, .. } => assert_eq!(code, "bad_argument"),
            other => panic!("expected error, got {other:?}"),
        }
        // Gradient flows through the mask.
        let xt = make(&mut registry, json!([1.0, 1.0]));
        let tensor = registry.get_tensor(&xt).unwrap().set_requires_grad(true);
        let xg = registry.insert_tensor(tensor);
        let y = forward(&mut registry, &d, &xg);
        let spec = nutorch_ops::find("sum").unwrap();
        let loss = match execute_table(&mut registry, spec, &[y], &serde_json::Map::new()) {
            Response::Handles { handles, .. } => handles[0].clone(),
            other => panic!("{other:?}"),
        };
        let spec = nutorch_ops::find("backward").unwrap();
        let _ = execute_table(&mut registry, spec, &[loss], &serde_json::Map::new());
        let grad = registry.get_tensor(&xg).unwrap().f_grad().unwrap();
        assert!(grad.defined());
    }

    #[test]
    fn batch_norm_running_stats_evolve_and_eval_uses_them() {
        let mut registry = Registry::new();
        let bn = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"batch_norm","args":{"num_features": 2}}),
        ));
        let x = make(&mut registry, json!([[10.0, -10.0], [12.0, -8.0]]));
        // Eval BEFORE any training: running stats are (0,1) → output = x.
        bespoke(
            &mut registry,
            json!({"op":"nn_mode","module": bn, "train": false}),
        );
        let y = forward(&mut registry, &bn, &x);
        let before = cpu_json(&registry, &y);
        // Train-mode forward updates the running stats in place…
        bespoke(
            &mut registry,
            json!({"op":"nn_mode","module": bn, "train": true}),
        );
        let _ = forward(&mut registry, &bn, &x);
        // …so a later EVAL forward differs from the first one.
        bespoke(
            &mut registry,
            json!({"op":"nn_mode","module": bn, "train": false}),
        );
        let y = forward(&mut registry, &bn, &x);
        let after = cpu_json(&registry, &y);
        assert_ne!(before, after, "running stats did not move");
    }

    #[test]
    fn mode_propagates_through_sequential() {
        let mut registry = Registry::new();
        let d = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"dropout","args":{"p": 0.5}}),
        ));
        let seq = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sequential","args":{"children":[d]}}),
        ));
        bespoke(
            &mut registry,
            json!({"op":"nn_mode","module": seq, "train": false}),
        );
        let info = bespoke(&mut registry, json!({"op":"nn_info","module": seq}));
        match info {
            Response::Value { value, .. } => {
                let lines: Vec<String> = serde_json::from_value(value).unwrap();
                assert!(lines.iter().any(|l| l == "training: false"), "{lines:?}");
            }
            other => panic!("{other:?}"),
        }
        // Eval dropout inside the sequential is identity.
        let x = make(&mut registry, json!([1.0, 2.0]));
        let y = forward(&mut registry, &seq, &x);
        assert_eq!(cpu_json(&registry, &y), json!([1.0, 2.0]));
    }

    #[test]
    fn conv_groups_validation() {
        let mut registry = Registry::new();
        // groups = 0 must be a clean error, not a divide-by-zero panic.
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"conv2d","args":{
                "in_channels":4,"out_channels":4,"kernel_size":3,"groups":0}}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("positive integer"));
            }
            other => panic!("expected error, got {other:?}"),
        }
        // Non-divisible groups must error, not silently truncate.
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"conv2d","args":{
                "in_channels":4,"out_channels":4,"kernel_size":3,"groups":3}}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("divisible"));
            }
            other => panic!("expected error, got {other:?}"),
        }
        // conv_transpose validates OUT channels.
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"conv_transpose2d","args":{
                "in_channels":4,"out_channels":3,"kernel_size":2,"groups":2}}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("out_channels"));
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn conv_shape_validation_and_group_norm_consistency() {
        let mut registry = Registry::new();
        // Wrong conv weight shape errors.
        let w = make(&mut registry, json!([[1.0, 2.0]]));
        match bespoke(
            &mut registry,
            json!({"op":"nn","kind":"conv2d","args":{
                "in_channels":1,"out_channels":2,"kernel_size":2,"weight": w}}),
        ) {
            Response::Error { code, .. } => assert_eq!(code, "shape_mismatch"),
            other => panic!("expected error, got {other:?}"),
        }
        // group_norm: module forward equals tch's own f_group_norm (the
        // recorded golden exclusion — internal consistency is the pin).
        let gn = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"group_norm","args":{"num_groups":2,"num_channels":4}}),
        ));
        let x = make(
            &mut registry,
            json!([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]),
        );
        let y = forward(&mut registry, &gn, &x);
        let via_module = cpu_json(&registry, &y);
        let direct = registry
            .get_tensor(&x)
            .unwrap()
            .f_group_norm(2, None::<&Tensor>, None::<&Tensor>, 1e-5, true)
            .unwrap();
        let direct = convert::tensor_to_json(&direct.f_to_device(Device::Cpu).unwrap()).unwrap();
        assert_eq!(via_module, direct);
    }
}

#[cfg(test)]
mod save_load_semantics {
    use super::*;
    use crate::lifecycle::Lifecycle;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn bespoke(registry: &mut Registry, request: serde_json::Value) -> Response {
        let parsed = parse_request(&request.to_string()).expect("parses");
        let lifecycle = Mutex::new(Lifecycle::new(None));
        let socket = PathBuf::from("/tmp/test.sock");
        handle_request(registry, &lifecycle, &socket, parsed).0
    }

    fn handle_of(response: Response) -> String {
        match response {
            Response::Handle { handle, .. } => handle,
            other => panic!("expected handle, got {other:?}"),
        }
    }

    fn forward_json(registry: &mut Registry, m: &str, x: &str) -> serde_json::Value {
        let y = handle_of(bespoke(
            registry,
            json!({"op":"forward","module": m, "tensor": x}),
        ));
        let cpu = registry
            .get_tensor(&y)
            .unwrap()
            .f_detach()
            .unwrap()
            .f_to_device(Device::Cpu)
            .unwrap();
        convert::tensor_to_json(&cpu).unwrap()
    }

    fn model(registry: &mut Registry) -> String {
        let l = handle_of(bespoke(
            registry,
            json!({"op":"nn","kind":"linear","args":{"in_features":2,"out_features":3}}),
        ));
        let bn = handle_of(bespoke(
            registry,
            json!({"op":"nn","kind":"batch_norm","args":{"num_features":3}}),
        ));
        handle_of(bespoke(
            registry,
            json!({"op":"nn","kind":"sequential","args":{"children":[l, bn]}}),
        ))
    }

    #[test]
    fn round_trip_restores_forward_and_names_match_pytorch() {
        let dir = std::env::temp_dir().join(format!("nutorch-sl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let path_str = path.to_string_lossy().into_owned();

        let mut registry = Registry::new();
        tch::manual_seed(11);
        let original = model(&mut registry);
        // Eval mode so batch_norm uses (deterministic) running stats.
        bespoke(
            &mut registry,
            json!({"op":"nn_mode","module": original, "train": false}),
        );
        let x = registry.insert_tensor(
            convert::json_to_tensor(&json!([[1.0, 2.0]]), Kind::Float, Device::Mps).unwrap(),
        );
        let expected = forward_json(&mut registry, &original, &x);

        // Names match PyTorch's state_dict scheme.
        let module = registry.get_module(&original).unwrap();
        let names: Vec<String> = module
            .named_state()
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        assert_eq!(
            names,
            vec![
                "0.weight",
                "0.bias",
                "1.weight",
                "1.bias",
                "1.running_mean",
                "1.running_var",
                "1.num_batches_tracked"
            ]
        );

        assert!(matches!(
            bespoke(
                &mut registry,
                json!({"op":"nn_save","module": original, "path": path_str})
            ),
            Response::Value { .. }
        ));

        // Fresh same-arch model, different init.
        tch::manual_seed(99);
        let fresh = model(&mut registry);
        bespoke(
            &mut registry,
            json!({"op":"nn_mode","module": fresh, "train": false}),
        );
        let before = forward_json(&mut registry, &fresh, &x);
        assert_ne!(before, expected, "different init should differ");

        assert!(matches!(
            bespoke(
                &mut registry,
                json!({"op":"nn_load","module": fresh, "path": path_str})
            ),
            Response::Value { .. }
        ));
        let after = forward_json(&mut registry, &fresh, &x);
        assert_eq!(after, expected, "loaded model must reproduce the original");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_errors_leave_target_unchanged_and_preserve_aliasing() {
        let dir = std::env::temp_dir().join(format!("nutorch-sl2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("linear.safetensors");
        let path_str = path.to_string_lossy().into_owned();

        let mut registry = Registry::new();
        let small = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"linear","args":{"in_features":2,"out_features":3}}),
        ));
        bespoke(
            &mut registry,
            json!({"op":"nn_save","module": small, "path": path_str}),
        );

        // Wrong architecture: missing keys named, target unchanged.
        let other = model(&mut registry); // sequential, different keys
        let x = registry.insert_tensor(
            convert::json_to_tensor(&json!([[1.0, 2.0]]), Kind::Float, Device::Mps).unwrap(),
        );
        bespoke(
            &mut registry,
            json!({"op":"nn_mode","module": other, "train": false}),
        );
        let before = forward_json(&mut registry, &other, &x);
        match bespoke(
            &mut registry,
            json!({"op":"nn_load","module": other, "path": path_str}),
        ) {
            Response::Error { code, error, .. } => {
                assert_eq!(code, "bad_argument");
                assert!(error.contains("key"), "{error}");
            }
            other => panic!("expected error, got {other:?}"),
        }
        let after = forward_json(&mut registry, &other, &x);
        assert_eq!(before, after, "failed load must not mutate the target");

        // Missing file is a clean bad_argument.
        match bespoke(
            &mut registry,
            json!({"op":"nn_load","module": small, "path": "/nonexistent/x.safetensors"}),
        ) {
            Response::Error { code, .. } => assert_eq!(code, "bad_argument"),
            other => panic!("expected error, got {other:?}"),
        }

        // Optimizer aliasing survives a (successful) load: optimizer built
        // BEFORE load still steps the loaded weights.
        let target = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"linear","args":{"in_features":2,"out_features":3}}),
        ));
        let opt = handle_of(bespoke(
            &mut registry,
            json!({"op":"nn","kind":"sgd","args":{"module": target, "lr": 0.1}}),
        ));
        bespoke(
            &mut registry,
            json!({"op":"nn_load","module": target, "path": path_str}),
        );
        // Forward/sum/backward/step must move the LOADED weights.
        let y = handle_of(bespoke(
            &mut registry,
            json!({"op":"forward","module": target, "tensor": x}),
        ));
        let spec = nutorch_ops::find("sum").unwrap();
        let loss = match execute_table(&mut registry, spec, &[y], &serde_json::Map::new()) {
            Response::Handles { handles, .. } => handles[0].clone(),
            o => panic!("{o:?}"),
        };
        let spec = nutorch_ops::find("backward").unwrap();
        let _ = execute_table(&mut registry, spec, &[loss], &serde_json::Map::new());
        let w_before = forward_json(&mut registry, &target, &x);
        bespoke(&mut registry, json!({"op":"step","optimizer": opt}));
        let w_after = forward_json(&mut registry, &target, &x);
        assert_ne!(
            w_before, w_after,
            "optimizer must still alias loaded params"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
