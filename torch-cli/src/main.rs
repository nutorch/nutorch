//! torch: the Nutorch v2 thin client (PoC, issue 0002).
//!
//! One operation per invocation: build a one-line JSON request, send it to
//! nutorchd over the Unix socket, print the result. Handles print as bare
//! strings so they compose in POSIX pipelines; when an op needs a tensor
//! handle and none is given positionally, one line is read from stdin (the
//! dual input pattern's pipeline form). Deliberately has no tch dependency —
//! the client stays thin.

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::ExitCode;

struct Args {
    op: String,
    positional: Vec<String>,
    dtype: Option<String>,
    socket: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args().skip(1);
    let op = raw.next().ok_or("usage: torch <op> [args...]")?;
    let mut positional = Vec::new();
    let mut dtype = None;
    let mut socket = None;
    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--dtype" => dtype = Some(raw.next().ok_or("--dtype needs a value")?),
            "--socket" => socket = Some(raw.next().ok_or("--socket needs a value")?),
            "--device" => {
                return Err(
                    "the device option was removed; tensors always live on the GPU (mps)"
                        .to_string(),
                )
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag: {flag}")),
            _ => positional.push(arg),
        }
    }
    Ok(Args {
        op,
        positional,
        dtype,
        socket,
    })
}

fn default_socket_path() -> PathBuf {
    match std::env::var_os("TMPDIR") {
        Some(tmp) => PathBuf::from(tmp).join("nutorchd.sock"),
        None => PathBuf::from("/tmp/nutorchd.sock"),
    }
}

/// Positional argument if present, else one line from stdin (pipeline form).
fn positional_or_stdin(args: &Args, index: usize, what: &str) -> Result<String, String> {
    if let Some(value) = args.positional.get(index) {
        return Ok(value.clone());
    }
    let mut line = String::new();
    std::io::stdin()
        .read_to_string(&mut line)
        .map_err(|e| format!("failed to read {what} from stdin: {e}"))?;
    let line = line.trim();
    if line.is_empty() {
        return Err(format!(
            "missing {what}: pass it as an argument or pipe it in"
        ));
    }
    Ok(line.to_string())
}

/// Two handles for a binary op. Two positionals → (a, b). One positional →
/// a comes from stdin and the positional is b (pipeline form, matching v1's
/// pipeline-is-first-operand convention). Zero positionals → error.
fn binary_handles(args: &Args, op: &str) -> Result<(String, String), String> {
    match args.positional.len() {
        2 => Ok((args.positional[0].clone(), args.positional[1].clone())),
        1 => {
            let a = positional_or_stdin(args, 1, "left-hand tensor handle")?;
            Ok((a, args.positional[0].clone()))
        }
        0 => Err(format!(
            "{op} needs two tensor handles (two arguments, or pipe one in and pass one)"
        )),
        n => Err(format!("{op} takes two tensor handles, got {n} arguments")),
    }
}

fn build_request(args: &Args) -> Result<serde_json::Value, String> {
    match args.op.as_str() {
        "tensor" => {
            let data_text = positional_or_stdin(args, 0, "tensor data")?;
            let data: serde_json::Value = serde_json::from_str(&data_text)
                .map_err(|e| format!("tensor data is not valid JSON: {e}"))?;
            Ok(serde_json::json!({
                "op": "tensor",
                "data": data,
                "dtype": args.dtype,
            }))
        }
        "full" => {
            let shape_text = args
                .positional
                .first()
                .ok_or("full needs a shape, e.g. torch full '[2,2]' 1")?;
            let shape: serde_json::Value = serde_json::from_str(shape_text)
                .map_err(|e| format!("shape is not valid JSON: {e}"))?;
            let value_text = args
                .positional
                .get(1)
                .ok_or("full needs a fill value, e.g. torch full '[2,2]' 1")?;
            let value: serde_json::Value = serde_json::from_str(value_text)
                .map_err(|e| format!("fill value is not a number: {e}"))?;
            Ok(serde_json::json!({
                "op": "full",
                "shape": shape,
                "value": value,
                "dtype": args.dtype,
            }))
        }
        "add" => {
            let (a, b) = binary_handles(args, "add")?;
            Ok(serde_json::json!({ "op": "add", "a": a, "b": b }))
        }
        "mm" => {
            let (a, b) = binary_handles(args, "mm")?;
            Ok(serde_json::json!({ "op": "mm", "a": a, "b": b }))
        }
        "mean" => {
            let handle = positional_or_stdin(args, 0, "tensor handle")?;
            Ok(serde_json::json!({ "op": "mean", "handle": handle }))
        }
        "value" => {
            let handle = positional_or_stdin(args, 0, "tensor handle")?;
            Ok(serde_json::json!({ "op": "value", "handle": handle }))
        }
        other => Err(format!("unknown op: {other}")),
    }
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

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let socket = args
        .socket
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path);
    let request = build_request(&args)?;
    let response_text = round_trip(&socket, &request)?;
    let response: serde_json::Value =
        serde_json::from_str(&response_text).map_err(|e| format!("bad response: {e}"))?;

    if response["ok"] == serde_json::Value::Bool(true) {
        if let Some(handle) = response["handle"].as_str() {
            println!("{handle}");
        } else {
            println!("{}", response["value"]);
        }
        Ok(())
    } else {
        Err(response["error"]
            .as_str()
            .unwrap_or("unknown daemon error")
            .to_string())
    }
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
