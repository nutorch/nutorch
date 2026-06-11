//! torch: the Nutorch v2 thin client.
//!
//! One operation per invocation: build a one-line JSON request, send it to
//! nutorchd over the Unix socket, print the result. Handles print one per
//! line, so they compose in POSIX pipelines (multi-return ops emit several
//! lines). Deliberately has no tch dependency — the client stays thin.
//!
//! Argument grammar (issue 0005): an op's tensor slots fill
//! stdin-prefix/positional-suffix — with k positionals for arity n, the
//! first (n−k) slots are read from stdin, one handle per line; if k = n,
//! stdin is never read. Variadic ops take all stdin lines (when stdin is not
//! a TTY) plus positionals. Positional params follow the tensor slots; the
//! rest are flags.

use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{ExitCode, Stdio};
use std::time::Duration;

use nutorch_ops::{Arity, OpSpec, ParamKind};

fn default_socket_path() -> PathBuf {
    match std::env::var_os("TMPDIR") {
        Some(tmp) => PathBuf::from(tmp).join("nutorchd.sock"),
        None => PathBuf::from("/tmp/nutorchd.sock"),
    }
}

/// Liveness probe that DROPS the probe connection before returning.
///
/// The drop matters: holding a probe stream (e.g. as a `match` scrutinee,
/// whose temporary lives for the whole match arm) while sending a follow-up
/// request on a second connection deadlocks against the serial
/// one-connection-at-a-time daemon — it is still waiting on the open probe.
fn daemon_alive(socket: &Path) -> bool {
    UnixStream::connect(socket).is_ok()
}

/// The socket path with its extension replaced by `.log`
/// (nutorchd.sock -> nutorchd.log) — must agree with the daemon's convention.
fn log_path_for(socket: &Path) -> PathBuf {
    socket.with_extension("log")
}

/// Locate the daemon binary: `NUTORCHD_BIN` override, else next to `torch`.
fn nutorchd_binary() -> PathBuf {
    if let Some(bin) = std::env::var_os("NUTORCHD_BIN") {
        return PathBuf::from(bin);
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("nutorchd")))
        .unwrap_or_else(|| PathBuf::from("nutorchd"))
}

/// Auto-start (issue 0004): spawn nutorchd detached — stdin null,
/// stdout/stderr appended to the conventional log file — then poll the
/// socket until it answers.
fn ensure_daemon(socket: &Path) -> Result<(), String> {
    if daemon_alive(socket) {
        return Ok(());
    }
    let binary = nutorchd_binary();
    let log = log_path_for(socket);
    let open_log = || {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .map_err(|e| format!("cannot open daemon log {}: {e}", log.display()))
    };
    std::process::Command::new(&binary)
        .arg("--socket")
        .arg(socket)
        .stdin(Stdio::null())
        .stdout(open_log()?)
        .stderr(open_log()?)
        .spawn()
        .map_err(|e| format!("failed to start nutorchd ({}): {e}", binary.display()))?;
    for _ in 0..100 {
        if daemon_alive(socket) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Err(format!(
        "nutorchd did not come up within 5s; see {}",
        log.display()
    ))
}

fn round_trip(socket: &PathBuf, request: &serde_json::Value) -> Result<String, String> {
    let stream = UnixStream::connect(socket)
        .map_err(|e| format!("cannot connect to nutorchd at {}: {e}", socket.display()))?;
    let mut writer = stream
        .try_clone()
        .map_err(|e| format!("socket error: {e}"))?;
    let mut payload = request.to_string();
    payload.push('\n');
    writer
        .write_all(payload.as_bytes())
        .map_err(|e| format!("send failed: {e}"))?;
    writer.flush().map_err(|e| format!("send failed: {e}"))?;

    let mut response = String::new();
    BufReader::new(stream)
        .read_line(&mut response)
        .map_err(|e| format!("receive failed: {e}"))?;
    if response.trim().is_empty() {
        return Err("empty response from daemon".to_string());
    }
    Ok(response)
}

/// Send one request and parse the daemon's reply. Does not spawn.
fn exchange(socket: &PathBuf, request: &serde_json::Value) -> Result<serde_json::Value, String> {
    let response_text = round_trip(socket, request)?;
    let response: serde_json::Value =
        serde_json::from_str(&response_text).map_err(|e| format!("bad response: {e}"))?;
    if response["ok"] == serde_json::Value::Bool(true) {
        Ok(response)
    } else {
        Err(response["error"]
            .as_str()
            .unwrap_or("unknown daemon error")
            .to_string())
    }
}

/// Print a successful response: handles one per line; values as JSON.
fn print_response(response: &serde_json::Value) {
    if let Some(handles) = response["handles"].as_array() {
        for handle in handles {
            if let Some(h) = handle.as_str() {
                println!("{h}");
            }
        }
    } else if let Some(handle) = response["handle"].as_str() {
        println!("{handle}");
    } else if response.get("value").is_some() {
        println!("{}", response["value"]);
    }
}

// ---------- argument parsing ----------

/// Presence-only flags on bespoke (non-table) ops; all other bespoke
/// flags take a value.
const BESPOKE_PRESENCE_FLAGS: &[&str] = &[
    "all",
    "meta",
    "requires_grad",
    "no-bias",
    "no_bias",
    "nesterov",
];

struct RawArgs {
    op: String,
    positionals: Vec<String>,
    /// (name, value) — Bool flags carry None.
    flags: Vec<(String, Option<String>)>,
    socket: Option<String>,
    help: bool,
}

/// First pass: split argv into positionals and flags, with the op's spec
/// deciding which flags take values (Bool flags are presence-only).
fn parse_raw(spec: Option<&OpSpec>) -> Result<RawArgs, String> {
    let mut raw = std::env::args().skip(1);
    let op = raw.next().ok_or(GENERAL_USAGE)?;
    let mut positionals = Vec::new();
    let mut flags = Vec::new();
    let mut socket = None;
    let mut help = false;
    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--socket" => socket = Some(raw.next().ok_or("--socket needs a value")?),
            "--help" => help = true,
            "--device" => {
                return Err(
                    "the device option was removed; tensors always live on the GPU (mps)"
                        .to_string(),
                )
            }
            flag if flag.starts_with("--") => {
                let name = flag.trim_start_matches("--").to_string();
                let param = spec.and_then(|s| s.params.iter().find(|p| p.name == name));
                match param {
                    Some(p) if p.kind == ParamKind::Bool => flags.push((name, None)),
                    Some(_) => {
                        let value = raw.next().ok_or(format!("--{name} needs a value"))?;
                        flags.push((name, Some(value)));
                    }
                    // Bespoke ops validate their own flags below. Flags in
                    // BESPOKE_PRESENCE_FLAGS are presence-only — without
                    // this, `torch free --all $t` would silently swallow
                    // `$t` as the flag's value (design-review finding).
                    None if spec.is_none() && BESPOKE_PRESENCE_FLAGS.contains(&name.as_str()) => {
                        flags.push((name, None));
                    }
                    None if spec.is_none() => {
                        let value = raw.next().ok_or(format!("--{name} needs a value"))?;
                        flags.push((name, Some(value)));
                    }
                    None => return Err(format!("unknown flag: --{name} (see torch {op} --help)")),
                }
            }
            _ => positionals.push(arg),
        }
    }
    Ok(RawArgs {
        op,
        positionals,
        flags,
        socket,
        help,
    })
}

/// Read exactly `n` handles from stdin (one per line). Errors on a TTY (a
/// missing operand should be a usage error, not a hang) and on a count
/// mismatch.
fn stdin_handles(n: usize, op: &str) -> Result<Vec<String>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if std::io::stdin().is_terminal() {
        return Err(format!(
            "{op}: missing tensor operand(s) — pass handles as arguments or pipe them in (see torch {op} --help)"
        ));
    }
    let lines = read_stdin_lines()?;
    if lines.len() != n {
        return Err(format!(
            "{op}: expected {n} piped handle(s), got {}",
            lines.len()
        ));
    }
    Ok(lines)
}

fn read_stdin_lines() -> Result<Vec<String>, String> {
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .map_err(|e| format!("failed to read stdin: {e}"))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

/// Parse a flag/positional value according to its spec kind.
fn parse_param_value(
    op: &str,
    param: &nutorch_ops::ParamSpec,
    text: &str,
) -> Result<serde_json::Value, String> {
    let bad = || {
        format!(
            "{op}: parameter {} must be {:?}, got: {text}",
            param.name, param.kind
        )
    };
    match param.kind {
        ParamKind::Int => text
            .parse::<i64>()
            .map(serde_json::Value::from)
            .map_err(|_| bad()),
        ParamKind::Float => text
            .parse::<f64>()
            .map(serde_json::Value::from)
            .map_err(|_| bad()),
        ParamKind::Scalar => {
            if let Ok(i) = text.parse::<i64>() {
                Ok(serde_json::Value::from(i))
            } else {
                text.parse::<f64>()
                    .map(serde_json::Value::from)
                    .map_err(|_| bad())
            }
        }
        ParamKind::IntList => {
            let value: serde_json::Value = serde_json::from_str(text).map_err(|_| bad())?;
            if value.is_array() && value.as_array().unwrap().iter().all(|v| v.is_i64()) {
                Ok(value)
            } else {
                Err(bad())
            }
        }
        ParamKind::Bool => Ok(serde_json::Value::Bool(true)),
        ParamKind::Str => Ok(serde_json::Value::from(text)),
        // Number → scalar; anything else is assumed to be a tensor handle
        // (a handle can never parse as a number). A typo'd scalar surfaces
        // from the daemon as a malformed-handle error (issue 0009) — honest
        // either way.
        ParamKind::HandleOrScalar => {
            if let Ok(i) = text.parse::<i64>() {
                Ok(serde_json::Value::from(i))
            } else if let Ok(f) = text.parse::<f64>() {
                Ok(serde_json::Value::from(f))
            } else {
                Ok(serde_json::Value::from(text))
            }
        }
    }
}

/// Build the generic table-op request from the grammar.
fn build_table_request(spec: &OpSpec, args: &RawArgs) -> Result<serde_json::Value, String> {
    let op = spec.name;
    let positional_params: Vec<_> = spec.params.iter().filter(|p| p.positional).collect();

    // Split positionals into tensor handles and trailing positional params.
    let (tensor_positionals, param_positionals): (Vec<&String>, Vec<&String>) = {
        let m = args.positionals.len();
        let p = positional_params.len();
        if m < p {
            let missing = positional_params[m];
            return Err(format!(
                "{op}: missing required parameter <{}> ({})",
                missing.name,
                spec.usage()
            ));
        }
        let split = m - p;
        (
            args.positionals[..split].iter().collect(),
            args.positionals[split..].iter().collect(),
        )
    };

    // Tensor slots.
    let tensors: Vec<String> = match spec.tensors {
        Arity::Exactly(n) => {
            if tensor_positionals.len() > n {
                return Err(format!("{op}: too many arguments ({})", spec.usage()));
            }
            let from_stdin = stdin_handles(n - tensor_positionals.len(), op)?;
            from_stdin
                .into_iter()
                .chain(tensor_positionals.iter().map(|s| s.to_string()))
                .collect()
        }
        Arity::AtLeast(n) => {
            let mut tensors: Vec<String> = if std::io::stdin().is_terminal() {
                Vec::new()
            } else {
                read_stdin_lines()?
            };
            tensors.extend(tensor_positionals.iter().map(|s| s.to_string()));
            if tensors.len() < n {
                return Err(format!(
                    "{op}: needs at least {n} tensors, got {} ({})",
                    tensors.len(),
                    spec.usage()
                ));
            }
            tensors
        }
    };

    // Params: positionals (in spec order), then flags.
    let mut params = serde_json::Map::new();
    for (param, text) in positional_params.iter().zip(param_positionals.iter()) {
        params.insert(param.name.to_string(), parse_param_value(op, param, text)?);
    }
    for (name, value) in &args.flags {
        let param = spec
            .params
            .iter()
            .find(|p| p.name == *name)
            .expect("validated in parse_raw");
        let parsed = match value {
            None => serde_json::Value::Bool(true),
            Some(text) => parse_param_value(op, param, text)?,
        };
        params.insert(name.clone(), parsed);
    }

    Ok(serde_json::json!({ "op": op, "tensors": tensors, "params": params }))
}

// ---------- bespoke ops ----------

fn positional_or_stdin(args: &RawArgs, index: usize, what: &str) -> Result<String, String> {
    if let Some(value) = args.positionals.get(index) {
        return Ok(value.clone());
    }
    if std::io::stdin().is_terminal() {
        return Err(format!(
            "missing {what}: pass it as an argument or pipe it in"
        ));
    }
    let lines = read_stdin_lines()?;
    lines
        .into_iter()
        .next()
        .ok_or_else(|| format!("missing {what}: pass it as an argument or pipe it in"))
}

fn build_bespoke_request(args: &RawArgs) -> Result<serde_json::Value, String> {
    match args.op.as_str() {
        "tensor" => {
            let mut dtype = None;
            let mut requires_grad = false;
            for (name, value) in &args.flags {
                match name.as_str() {
                    "dtype" => dtype = value.clone(),
                    "requires_grad" => requires_grad = true,
                    other => return Err(format!("unknown flag: --{other}")),
                }
            }
            let data_text = positional_or_stdin(args, 0, "tensor data")?;
            let data: serde_json::Value = serde_json::from_str(&data_text)
                .map_err(|e| format!("tensor data is not valid JSON: {e}"))?;
            Ok(serde_json::json!({
                "op": "tensor", "data": data, "dtype": dtype,
                "requires_grad": requires_grad,
            }))
        }
        "value" => {
            let mut meta = false;
            for (name, _) in &args.flags {
                match name.as_str() {
                    "meta" => meta = true,
                    other => return Err(format!("unknown flag: --{other}")),
                }
            }
            let handle = positional_or_stdin(args, 0, "tensor handle")?;
            Ok(serde_json::json!({ "op": "value", "handle": handle, "meta": meta }))
        }
        "free" => {
            let mut all = false;
            for (name, _) in &args.flags {
                match name.as_str() {
                    "all" => all = true,
                    other => return Err(format!("unknown flag: --{other}")),
                }
            }
            let mut handles = args.positionals.clone();
            if !all && handles.is_empty() && !std::io::stdin().is_terminal() {
                handles = read_stdin_lines()?;
            }
            if all && !handles.is_empty() {
                return Err("free: --all and handles are mutually exclusive".to_string());
            }
            if all {
                return Ok(serde_json::json!({ "op": "free", "all": true }));
            }
            if handles.is_empty() {
                return Err(
                    "free: pass handles as arguments, pipe them in, or use --all".to_string(),
                );
            }
            Ok(serde_json::json!({ "op": "free", "handles": handles }))
        }
        "step" => {
            if let Some((name, _)) = args.flags.first() {
                return Err(format!("unknown flag: --{name}"));
            }
            let optimizer = positional_or_stdin(args, 0, "optimizer handle")?;
            Ok(serde_json::json!({ "op": "step", "optimizer": optimizer }))
        }
        "forward" => {
            if let Some((name, _)) = args.flags.first() {
                return Err(format!("unknown flag: --{name}"));
            }
            // Dual input: `torch forward $m $x` or `$x | torch forward $m`.
            let module = args
                .positionals
                .first()
                .cloned()
                .ok_or("forward: usage: torch forward <nn://module> [tensor]")?;
            let tensor = positional_or_stdin(args, 1, "input tensor handle")?;
            Ok(serde_json::json!({ "op": "forward", "module": module, "tensor": tensor }))
        }
        other => Err(format!("unknown op: {other} (see `torch ops`)")),
    }
}

/// Generic kind builder: positional Ints by name, then flags (presence
/// flags stay bool; handle-bearing flags stay strings; numbers parse).
fn build_kind_request(
    kind: &str,
    args: &RawArgs,
    positionals: &[&str],
    _reserved: &[&str],
) -> Result<serde_json::Value, String> {
    let mut nn_args = serde_json::Map::new();
    for (index, name) in positionals.iter().enumerate() {
        let value = args
            .positionals
            .get(index + 1)
            .ok_or(format!("nn {kind}: missing <{name}>"))?
            .parse::<i64>()
            .map_err(|_| format!("nn {kind}: {name} must be an integer"))?;
        nn_args.insert(name.to_string(), value.into());
    }
    collect_kind_flags(kind, args, &mut nn_args)?;
    Ok(serde_json::json!({ "op": "nn", "kind": kind, "args": nn_args }))
}

fn collect_kind_flags(
    kind: &str,
    args: &RawArgs,
    nn_args: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<(), String> {
    for (name, value) in &args.flags {
        let key = name.replace('-', "_");
        match (key.as_str(), value) {
            ("no_bias", _) => {
                nn_args.insert("no_bias".into(), true.into());
            }
            ("weight" | "bias_tensor", Some(handle)) => {
                nn_args.insert(key, handle.clone().into());
            }
            (_, Some(text)) => {
                // Numbers go as numbers (ints stay ints for Int args).
                if let Ok(i) = text.parse::<i64>() {
                    nn_args.insert(key, i.into());
                } else if let Ok(f) = text.parse::<f64>() {
                    nn_args.insert(key, serde_json::json!(f));
                } else {
                    return Err(format!("nn {kind}: --{name} must be a number"));
                }
            }
            (_, None) => return Err(format!("nn {kind}: --{name} needs a value")),
        }
    }
    Ok(())
}

/// `torch nn <kind> [args…]`: the module construction subcommand
/// (issue 0009). Kind specs are a small client-side match until the
/// module sweep needs a declarative table.
fn build_nn_request(args: &RawArgs) -> Result<serde_json::Value, String> {
    let kind = args
        .positionals
        .first()
        .ok_or("usage: torch nn <linear|relu|sigmoid|tanh|gelu|sequential|parameters|info> …")?;
    let mut nn_args = serde_json::Map::new();
    match kind.as_str() {
        "linear" => {
            let parse_feat = |index: usize, name: &str| -> Result<i64, String> {
                args.positionals
                    .get(index)
                    .ok_or(format!("nn linear: usage: torch nn linear <in> <out>"))?
                    .parse::<i64>()
                    .map_err(|_| format!("nn linear: {name} must be an integer"))
            };
            nn_args.insert("in_features".into(), parse_feat(1, "in_features")?.into());
            nn_args.insert("out_features".into(), parse_feat(2, "out_features")?.into());
            for (name, value) in &args.flags {
                match (name.as_str(), value) {
                    ("no-bias" | "no_bias", _) => {
                        nn_args.insert("no_bias".into(), true.into());
                    }
                    ("weight", Some(handle)) => {
                        nn_args.insert("weight".into(), handle.clone().into());
                    }
                    ("bias-tensor" | "bias_tensor", Some(handle)) => {
                        nn_args.insert("bias_tensor".into(), handle.clone().into());
                    }
                    (other, _) => return Err(format!("nn linear: unknown flag --{other}")),
                }
            }
            Ok(serde_json::json!({ "op": "nn", "kind": "linear", "args": nn_args }))
        }
        "relu" | "sigmoid" | "tanh" | "gelu" => {
            Ok(serde_json::json!({ "op": "nn", "kind": kind, "args": {} }))
        }
        // Data-driven kinds: (positional int args…, then flags parse by
        // name with hyphen→underscore tolerance; string flags carry handles).
        "conv1d" | "conv2d" | "conv_transpose2d" => build_kind_request(
            kind,
            args,
            &["in_channels", "out_channels", "kernel_size"],
            &[],
        ),
        "embedding" => build_kind_request(kind, args, &["num_embeddings", "embedding_dim"], &[]),
        "batch_norm" => build_kind_request(kind, args, &["num_features"], &[]),
        "group_norm" => build_kind_request(kind, args, &["num_groups", "num_channels"], &[]),
        "softmax" => build_kind_request(kind, args, &["dim"], &[]),
        "max_pool2d" | "avg_pool2d" => build_kind_request(kind, args, &["kernel_size"], &[]),
        "dropout" | "leaky_relu" | "flatten" => build_kind_request(kind, args, &[], &[]),
        "layer_norm" => {
            let shape_text = args
                .positionals
                .get(1)
                .ok_or("usage: torch nn layer_norm '[shape]' [--eps E]")?;
            let shape: serde_json::Value = serde_json::from_str(shape_text)
                .map_err(|_| "nn layer_norm: shape must be a JSON int list".to_string())?;
            let mut nn_args = serde_json::Map::new();
            nn_args.insert("normalized_shape".into(), shape);
            collect_kind_flags(kind, args, &mut nn_args)?;
            Ok(serde_json::json!({ "op": "nn", "kind": "layer_norm", "args": nn_args }))
        }
        "save" | "load" => {
            let module = args
                .positionals
                .get(1)
                .cloned()
                .ok_or(format!("usage: torch nn {kind} <nn://module> <path>"))?;
            let raw_path = args
                .positionals
                .get(2)
                .cloned()
                .ok_or(format!("usage: torch nn {kind} <nn://module> <path>"))?;
            // Resolve relative paths against the USER's cwd, not the
            // daemon's (the daemon was spawned who-knows-where).
            let path = std::path::Path::new(&raw_path);
            let absolute = if path.is_absolute() {
                raw_path
            } else {
                std::env::current_dir()
                    .map_err(|e| format!("cannot resolve cwd: {e}"))?
                    .join(path)
                    .to_string_lossy()
                    .into_owned()
            };
            Ok(serde_json::json!({
                "op": if kind == "save" { "nn_save" } else { "nn_load" },
                "module": module, "path": absolute,
            }))
        }
        "train" | "eval" => {
            let module = args
                .positionals
                .get(1)
                .cloned()
                .ok_or(format!("usage: torch nn {kind} <nn://module>"))?;
            Ok(serde_json::json!({
                "op": "nn_mode", "module": module, "train": kind == "train",
            }))
        }
        "sequential" => {
            let mut children: Vec<String> = args.positionals[1..].to_vec();
            if children.is_empty() && !std::io::stdin().is_terminal() {
                children = read_stdin_lines()?;
            }
            nn_args.insert("children".into(), serde_json::json!(children));
            Ok(serde_json::json!({ "op": "nn", "kind": "sequential", "args": nn_args }))
        }
        "sgd" | "adam" | "adamw" | "rmsprop" => {
            let module = args
                .positionals
                .get(1)
                .cloned()
                .ok_or(format!("usage: torch nn {kind} <nn://module> [--lr …]"))?;
            nn_args.insert("module".into(), module.into());
            for (name, value) in &args.flags {
                match (name.as_str(), value) {
                    ("nesterov", _) => {
                        nn_args.insert("nesterov".into(), true.into());
                    }
                    (flag, Some(text)) => {
                        let key = flag.replace('-', "_");
                        let number: f64 = text
                            .parse()
                            .map_err(|_| format!("nn {kind}: --{flag} must be a number"))?;
                        nn_args.insert(key, serde_json::json!(number));
                    }
                    (flag, None) => return Err(format!("nn {kind}: --{flag} needs a value")),
                }
            }
            Ok(serde_json::json!({ "op": "nn", "kind": kind, "args": nn_args }))
        }
        "zero_grad" => {
            let handle = args
                .positionals
                .get(1)
                .cloned()
                .ok_or("usage: torch nn zero_grad <optim://…|nn://…>")?;
            Ok(serde_json::json!({ "op": "nn_zero_grad", "handle": handle }))
        }
        "set_lr" => {
            let optimizer = args
                .positionals
                .get(1)
                .cloned()
                .ok_or("usage: torch nn set_lr <optim://…> <lr>")?;
            let lr: f64 = args
                .positionals
                .get(2)
                .ok_or("usage: torch nn set_lr <optim://…> <lr>")?
                .parse()
                .map_err(|_| "nn set_lr: lr must be a number".to_string())?;
            Ok(serde_json::json!({ "op": "nn_set_lr", "optimizer": optimizer, "lr": lr }))
        }
        "parameters" => {
            let module = args
                .positionals
                .get(1)
                .cloned()
                .ok_or("usage: torch nn parameters <nn://module>")?;
            Ok(serde_json::json!({ "op": "nn_parameters", "module": module }))
        }
        "info" => {
            let module = args
                .positionals
                .get(1)
                .cloned()
                .ok_or("usage: torch nn info <nn://module>")?;
            Ok(serde_json::json!({ "op": "nn_info", "module": module }))
        }
        other => Err(format!(
            "unknown nn kind: {other} (expected linear, relu, sigmoid, tanh, gelu, sequential, parameters, or info)"
        )),
    }
}

/// `torch tensors`: list the registry, oldest first. Never spawns the
/// daemon — a dead daemon truthfully holds no tensors, so daemon-down
/// prints nothing and exits 0 (`torch tensors | torch free` no-ops).
fn run_tensors(args: &RawArgs, socket: &PathBuf) -> Result<(), String> {
    if !args.positionals.is_empty() || !args.flags.is_empty() {
        return Err("usage: torch tensors".to_string());
    }
    if !daemon_alive(socket) {
        return Ok(());
    }
    let response = exchange(socket, &serde_json::json!({"op":"tensors"}))?;
    let rows = match response["value"].as_array() {
        Some(rows) => rows.clone(),
        None => return Err("tensors: malformed response".to_string()),
    };
    // Column widths for eyes; single-space minimum for awk.
    let shapes: Vec<String> = rows
        .iter()
        .map(|r| {
            let dims: Vec<String> = r["shape"]
                .as_array()
                .map(|a| a.iter().map(|d| d.to_string()).collect())
                .unwrap_or_default();
            format!("[{}]", dims.join(","))
        })
        .collect();
    let shape_w = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let dtype_w = rows
        .iter()
        .filter_map(|r| r["dtype"].as_str().map(str::len))
        .max()
        .unwrap_or(0);
    for (row, shape) in rows.iter().zip(&shapes) {
        println!(
            "{}  {:<shape_w$}  {:<dtype_w$}  {}  {}s  {}s",
            row["handle"].as_str().unwrap_or("?"),
            shape,
            row["dtype"].as_str().unwrap_or("?"),
            row["bytes"],
            row["age_secs"],
            row["idle_secs"],
        );
    }
    Ok(())
}

// ---------- daemon verbs ----------

fn daemon_pid(socket: &PathBuf) -> Result<u64, String> {
    let status = exchange(socket, &serde_json::json!({"op":"status"}))?;
    status["value"]["pid"]
        .as_u64()
        .ok_or_else(|| "status response missing pid".to_string())
}

fn print_status(status: &serde_json::Value) {
    let v = &status["value"];
    let opt = |field: &serde_json::Value, suffix: &str| -> String {
        match field.as_u64() {
            Some(n) => format!("{n}{suffix}"),
            None => "none".to_string(),
        }
    };
    println!("pid: {}", v["pid"]);
    println!("uptime: {}s", v["uptime_secs"]);
    println!("device: {}", v["device"].as_str().unwrap_or("?"));
    println!("ttl: {}", opt(&v["ttl_secs"], "s"));
    println!("idle: {}s", v["idle_secs"]);
    println!("remaining: {}", opt(&v["remaining_secs"], "s"));
    println!("tensors: {}", v["tensors"]);
    println!("memory: ~{} bytes", v["approx_bytes"]);
    println!("socket: {}", v["socket"].as_str().unwrap_or("?"));
    println!("log: {}", v["log"].as_str().unwrap_or("?"));
}

/// `torch daemon <verb>`: positionals only, never stdin (these verbs have no
/// pipeline form). status/ttl/stop never spawn; start/restart do.
fn run_daemon_verb(args: &RawArgs, socket: &PathBuf) -> Result<(), String> {
    let verb = args
        .positionals
        .first()
        .ok_or("usage: torch daemon <status|ttl|stop|restart|start>")?;
    match verb.as_str() {
        "status" => {
            if !daemon_alive(socket) {
                return Err(format!("daemon not running (socket: {})", socket.display()));
            }
            let status = exchange(socket, &serde_json::json!({"op":"status"}))?;
            print_status(&status);
            Ok(())
        }
        "ttl" => {
            let duration = args
                .positionals
                .get(1)
                .ok_or("usage: torch daemon ttl <duration>  (e.g. 30m, 2h, none)")?;
            if !daemon_alive(socket) {
                return Err(format!("daemon not running (socket: {})", socket.display()));
            }
            let reply = exchange(socket, &serde_json::json!({"op":"set_ttl","ttl":duration}))?;
            match reply["value"]["ttl_secs"].as_u64() {
                Some(secs) => println!("ttl: {secs}s"),
                None => println!("ttl: none"),
            }
            Ok(())
        }
        "stop" => {
            if !daemon_alive(socket) {
                println!("daemon not running (nothing to stop)");
                return Ok(());
            }
            exchange(socket, &serde_json::json!({"op":"shutdown"}))?;
            println!("daemon stopped");
            Ok(())
        }
        "restart" => {
            if daemon_alive(socket) {
                exchange(socket, &serde_json::json!({"op":"shutdown"}))?;
                // The old daemon flushes the shutdown reply BEFORE unlinking;
                // wait for the socket to actually die so the new daemon
                // cannot probe the dying one, yield, and leave zero daemons.
                for _ in 0..60 {
                    if !daemon_alive(socket) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            ensure_daemon(socket)?;
            println!("daemon restarted (pid {})", daemon_pid(socket)?);
            Ok(())
        }
        "start" => {
            if daemon_alive(socket) {
                println!("already running (pid {})", daemon_pid(socket)?);
            } else {
                ensure_daemon(socket)?;
                println!("started (pid {})", daemon_pid(socket)?);
            }
            Ok(())
        }
        other => Err(format!(
            "unknown daemon verb: {other} (expected status, ttl, stop, restart, or start)"
        )),
    }
}

// ---------- help & ops listing ----------

const GENERAL_USAGE: &str = "usage: torch <op> [tensors...] [params...] [--flags]\n       torch ops              list available operations\n       torch <op> --help      usage for one operation\n       torch daemon <verb>    status | ttl | stop | restart | start";

fn print_ops() {
    for category in nutorch_ops::categories() {
        println!("{category}:");
        for spec in nutorch_ops::OPS.iter().filter(|s| s.category == category) {
            println!("  {:<14} {}", spec.name, spec.summary);
        }
    }
    println!("\nbespoke:");
    println!(
        "  {:<14} create a tensor from JSON data (--dtype)",
        "tensor"
    );
    println!("  {:<14} read a tensor back as JSON", "value");
}

fn print_op_help(op: &str) {
    match op {
        "tensor" => println!("usage: torch tensor <json-data> [--dtype <kind>]"),
        "value" => println!("usage: torch value [handle]   (or pipe the handle in)"),
        "daemon" => println!("usage: torch daemon <status|ttl|stop|restart|start>"),
        name => {
            if let Some(spec) = nutorch_ops::find(name) {
                println!("{}", spec.usage());
                println!("  {}", spec.summary);
            } else {
                println!("unknown op: {name}");
            }
        }
    }
}

// ---------- main ----------

fn run() -> Result<(), String> {
    let op_name = std::env::args().nth(1).ok_or(GENERAL_USAGE)?;

    if op_name == "ops" {
        print_ops();
        return Ok(());
    }
    if op_name == "--help" || op_name == "help" {
        println!("{GENERAL_USAGE}");
        return Ok(());
    }

    let spec = nutorch_ops::find(&op_name);
    let args = parse_raw(spec)?;

    if args.help {
        print_op_help(&args.op);
        return Ok(());
    }

    let socket = args
        .socket
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path);

    if args.op == "daemon" {
        return run_daemon_verb(&args, &socket);
    }

    if args.op == "tensors" {
        return run_tensors(&args, &socket);
    }

    if args.op == "nn" {
        let request = build_nn_request(&args)?;
        if !daemon_alive(&socket) {
            ensure_daemon(&socket)?;
        }
        let response = exchange(&socket, &request)?;
        // zero_grad/set_lr are loop verbs: quiet on success.
        if matches!(
            args.positionals.first().map(String::as_str),
            Some("zero_grad") | Some("set_lr") | Some("train") | Some("eval")
        ) {
            return Ok(());
        }
        // nn info's value is a list of human lines, printed as lines.
        if args.positionals.first().map(String::as_str) == Some("info") {
            if let Some(lines) = response["value"].as_array() {
                for line in lines {
                    if let Some(text) = line.as_str() {
                        println!("{text}");
                    }
                }
                return Ok(());
            }
        }
        print_response(&response);
        return Ok(());
    }

    let request = match spec {
        Some(spec) => build_table_request(spec, &args)?,
        None => build_bespoke_request(&args)?,
    };

    // Auto-start (issue 0004): tensor work spawns the daemon when it is down.
    if !daemon_alive(&socket) {
        ensure_daemon(&socket)?;
    }
    let response = exchange(&socket, &request)?;
    // The rm convention: free prints nothing on success (the daemon's
    // {"freed":N} stays on the wire for tooling); step likewise (it runs
    // inside training loops where stdout noise is cost).
    if args.op != "free" && args.op != "step" {
        print_response(&response);
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("torch: {message}");
            ExitCode::FAILURE
        }
    }
}
