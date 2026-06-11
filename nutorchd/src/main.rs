//! nutorchd: the Nutorch v2 daemon.
//!
//! Owns the tensor registry and the LibTorch context; serves newline-delimited
//! JSON requests over a Unix socket. One connection at a time (PoC).
//!
//! Socket handling (issue 0004): probe-before-bind — a live daemon is never
//! displaced; a newcomer finding a live daemon exits 0 quietly (see
//! issues/0004-daemon-lifecycle/01-daemon-side-lifecycle.md).

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nutorchd::dispatch::{handle_request, parse_request, require_mps};
use nutorchd::lifecycle::{self, Lifecycle};
use nutorchd::registry::Registry;

const DEFAULT_TTL: Duration = Duration::from_secs(3600);

fn default_socket_path() -> PathBuf {
    match std::env::var_os("TMPDIR") {
        Some(tmp) => PathBuf::from(tmp).join("nutorchd.sock"),
        None => PathBuf::from("/tmp/nutorchd.sock"),
    }
}

struct DaemonArgs {
    socket: PathBuf,
    ttl: Option<Duration>,
}

/// `--ttl` flag wins over `NUTORCHD_TTL` env, which wins over the 1h default.
fn parse_daemon_args() -> Result<DaemonArgs, String> {
    let mut socket = None;
    let mut ttl_text: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--socket" => socket = Some(args.next().ok_or("--socket needs a value")?),
            "--ttl" => ttl_text = Some(args.next().ok_or("--ttl needs a value")?),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let ttl = match ttl_text.or_else(|| std::env::var("NUTORCHD_TTL").ok()) {
        Some(text) => lifecycle::parse_ttl(&text)?,
        None => Some(DEFAULT_TTL),
    };
    Ok(DaemonArgs {
        socket: socket
            .map(PathBuf::from)
            .unwrap_or_else(default_socket_path),
        ttl,
    })
}

/// Probe-before-bind (issue 0004): never steal a live daemon's socket.
/// Returns Ok(None) when another daemon is already serving the path (the
/// caller should exit 0 quietly). Known residual: a simultaneous-start
/// TOCTOU window remains (both probe before either binds); see the
/// experiment doc.
fn probe_and_bind(socket: &Path) -> std::io::Result<Option<UnixListener>> {
    if UnixStream::connect(socket).is_ok() {
        return Ok(None);
    }
    // Refused or missing: any file present is stale — remove and bind.
    let _ = std::fs::remove_file(socket);
    UnixListener::bind(socket).map(Some)
}

/// Serve one connection on its own thread (issue 0007). The registry lock
/// is the execution queue: each request dispatches under it, so ops stay
/// strictly serialized while connections are concurrent — a parked or
/// stuck peer holds nothing but its own thread.
fn serve_connection(
    registry: &Mutex<Registry>,
    lifecycle: &Mutex<Lifecycle>,
    socket: &Path,
    stream: UnixStream,
) -> std::io::Result<()> {
    let mut writer = stream.try_clone()?;
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match parse_request(&line) {
            Ok(request) => {
                // Lock ordering everywhere: registry, then lifecycle
                // (handle_request takes the lifecycle lock inside).
                let mut guard = registry.lock().unwrap();
                let (response, shutdown) = handle_request(&mut guard, lifecycle, socket, request);
                if shutdown {
                    // Hold the registry lock straight through exit: no op
                    // can start between the shutdown response and process
                    // death (concurrent peers see EOF — same as a crash,
                    // which clients already handle).
                    write_response(&mut writer, &response)?;
                    println!("shutdown requested; exiting");
                    let _ = std::fs::remove_file(socket);
                    std::process::exit(0);
                }
                response
            }
            Err(error_response) => error_response,
        };
        // Normal responses write OUTSIDE the lock; streams are
        // per-connection, so there is nothing to interleave with.
        write_response(&mut writer, &response)?;
    }
    Ok(())
}

fn write_response(
    writer: &mut UnixStream,
    response: &nutorchd::protocol::Response,
) -> std::io::Result<()> {
    let mut payload = serde_json::to_string(response).expect("response serializes");
    payload.push('\n');
    writer.write_all(payload.as_bytes())?;
    writer.flush()
}

fn main() -> std::io::Result<()> {
    if let Err(message) = require_mps() {
        eprintln!("{message}");
        std::process::exit(1);
    }

    let args = match parse_daemon_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("nutorchd: {message}");
            std::process::exit(2);
        }
    };

    let listener = match probe_and_bind(&args.socket)? {
        Some(listener) => listener,
        None => {
            println!(
                "nutorchd already running on {}; exiting",
                args.socket.display()
            );
            return Ok(());
        }
    };

    println!("nutorchd");
    println!("pid: {}", std::process::id());
    println!("socket: {}", args.socket.display());
    println!("device: mps");
    match args.ttl {
        Some(ttl) => println!("ttl: {}s (idle)", ttl.as_secs()),
        None => println!("ttl: none"),
    }

    let lifecycle = Arc::new(Mutex::new(Lifecycle::new(args.ttl)));

    let registry = Arc::new(Mutex::new(Registry::new()));

    // Expiry watcher: wakes ~1x/second. It takes the REGISTRY lock first,
    // then checks the lease (same registry→lifecycle order as the request
    // path — no deadlock), so expiry can never fire while an op is
    // mid-execution or has touched-but-not-started: any request that won
    // the lock already touched. Exits holding the lock.
    {
        let lifecycle = Arc::clone(&lifecycle);
        let registry = Arc::clone(&registry);
        let socket = args.socket.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(1));
            let guard = registry.lock().unwrap();
            if lifecycle.lock().unwrap().expired() {
                println!("idle ttl expired; exiting");
                let _ = std::fs::remove_file(&socket);
                std::process::exit(0);
            }
            drop(guard);
        });
    }

    // Signal handler: SIGTERM/SIGINT unlink the socket and exit cleanly.
    {
        let socket = args.socket.clone();
        let mut signals = signal_hook::iterator::Signals::new([
            signal_hook::consts::SIGTERM,
            signal_hook::consts::SIGINT,
        ])?;
        std::thread::spawn(move || {
            if signals.forever().next().is_some() {
                println!("signal received; exiting");
                let _ = std::fs::remove_file(&socket);
                std::process::exit(0);
            }
        });
    }

    // Thread-per-connection (issue 0007): a slow or stuck client parks
    // its own thread and blocks nobody.
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let registry = Arc::clone(&registry);
                let lifecycle = Arc::clone(&lifecycle);
                let socket = args.socket.clone();
                std::thread::spawn(move || {
                    if let Err(e) = serve_connection(&registry, &lifecycle, &socket, stream) {
                        eprintln!("connection error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}
