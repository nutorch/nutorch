//! nutorchd: the Nutorch v2 daemon (PoC, issue 0002).
//!
//! Owns the tensor registry and the LibTorch context; serves newline-delimited
//! JSON requests over a Unix socket. One connection at a time (PoC).
//!
//! Known PoC simplification: stale-socket removal on startup is unconditional;
//! a second daemon on the same path steals it from a live first daemon (see
//! issues/0002-nutorchd-poc/02-daemon-spine.md).

mod convert;
mod protocol;
mod registry;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;

use protocol::{Request, Response};
use registry::Registry;
use tch::Device;

fn default_socket_path() -> PathBuf {
    match std::env::var_os("TMPDIR") {
        Some(tmp) => PathBuf::from(tmp).join("nutorchd.sock"),
        None => PathBuf::from("/tmp/nutorchd.sock"),
    }
}

fn socket_path_from_args() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--socket" {
            if let Some(path) = args.next() {
                return PathBuf::from(path);
            }
        }
    }
    default_socket_path()
}

fn handle_request(registry: &mut Registry, request: Request) -> Response {
    match request {
        Request::Tensor {
            data,
            device,
            dtype,
        } => {
            let device = match convert::parse_device(device.as_deref()) {
                Ok(d) => d,
                Err(e) => return Response::error(e),
            };
            let kind = match convert::parse_kind(dtype.as_deref()) {
                Ok(k) => k,
                Err(e) => return Response::error(e),
            };
            match convert::json_to_tensor(&data, kind, device) {
                Ok(tensor) => Response::handle(registry.insert(tensor)),
                Err(e) => Response::error(e),
            }
        }
        Request::Value { handle } => match registry.get(&handle) {
            Some(tensor) => {
                let cpu = match tensor.f_to_device(Device::Cpu) {
                    Ok(t) => t,
                    Err(e) => return Response::error(convert::tch_error(e)),
                };
                match convert::tensor_to_json(&cpu) {
                    Ok(value) => Response::value(value),
                    Err(e) => Response::error(e),
                }
            }
            None => Response::error(format!("unknown handle: {handle}")),
        },
    }
}

fn serve_connection(registry: &mut Registry, stream: UnixStream) -> std::io::Result<()> {
    let mut writer = stream.try_clone()?;
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => handle_request(registry, request),
            Err(e) => Response::error(format!("bad request: {e}")),
        };
        let mut payload = serde_json::to_string(&response).expect("response serializes");
        payload.push('\n');
        writer.write_all(payload.as_bytes())?;
        writer.flush()?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let socket_path = socket_path_from_args();
    // PoC: unconditional stale-socket removal (see module doc).
    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)?;

    println!("nutorchd (PoC, issue 0002)");
    println!("socket: {}", socket_path.display());
    println!("MPS available: {}", tch::utils::has_mps());

    let mut registry = Registry::new();
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(e) = serve_connection(&mut registry, stream) {
                    eprintln!("connection error: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}
