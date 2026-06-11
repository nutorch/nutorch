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
const BESPOKE_PRESENCE_FLAGS: &[&str] = &["all"];

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
        // (a UUID can never parse as a number). A typo'd scalar surfaces as
        // unknown_handle from the daemon — an honest error.
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
            for (name, value) in &args.flags {
                match name.as_str() {
                    "dtype" => dtype = value.clone(),
                    other => return Err(format!("unknown flag: --{other}")),
                }
            }
            let data_text = positional_or_stdin(args, 0, "tensor data")?;
            let data: serde_json::Value = serde_json::from_str(&data_text)
                .map_err(|e| format!("tensor data is not valid JSON: {e}"))?;
            Ok(serde_json::json!({ "op": "tensor", "data": data, "dtype": dtype }))
        }
        "value" => {
            if let Some((name, _)) = args.flags.first() {
                return Err(format!("unknown flag: --{name}"));
            }
            let handle = positional_or_stdin(args, 0, "tensor handle")?;
            Ok(serde_json::json!({ "op": "value", "handle": handle }))
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
        other => Err(format!("unknown op: {other} (see `torch ops`)")),
    }
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
    // {"freed":N} stays on the wire for tooling).
    if args.op != "free" {
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
