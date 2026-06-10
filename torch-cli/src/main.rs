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
    device: Option<String>,
    dtype: Option<String>,
    socket: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args().skip(1);
    let op = raw.next().ok_or("usage: torch <op> [args...]")?;
    let mut positional = Vec::new();
    let mut device = None;
    let mut dtype = None;
    let mut socket = None;
    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--device" => device = Some(raw.next().ok_or("--device needs a value")?),
            "--dtype" => dtype = Some(raw.next().ok_or("--dtype needs a value")?),
            "--socket" => socket = Some(raw.next().ok_or("--socket needs a value")?),
            _ => positional.push(arg),
        }
    }
    Ok(Args {
        op,
        positional,
        device,
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

fn build_request(args: &Args) -> Result<serde_json::Value, String> {
    match args.op.as_str() {
        "tensor" => {
            let data_text = positional_or_stdin(args, 0, "tensor data")?;
            let data: serde_json::Value = serde_json::from_str(&data_text)
                .map_err(|e| format!("tensor data is not valid JSON: {e}"))?;
            Ok(serde_json::json!({
                "op": "tensor",
                "data": data,
                "device": args.device,
                "dtype": args.dtype,
            }))
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
