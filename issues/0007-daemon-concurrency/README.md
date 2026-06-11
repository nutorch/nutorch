+++
status = "closed"
opened = "2026-06-11"
closed = "2026-06-11"
+++

# Issue 7: Concurrent connections, serialized execution

## Goal

Multiple shell clients can talk to nutorchd at the same time: a slow, idle, or
stuck client never blocks anyone else. Computations stay strictly serialized —
concurrency in connection handling, no parallelism anywhere.

## Background

The daemon's accept loop (issue 0002, unchanged since) serves exactly one
connection from accept to EOF before looking at the next. Any client that
connects and stalls — a forgotten `nc -U $socket`, a script holding its
connection open, a wedged process — blocks every other client indefinitely. This
is not hypothetical: issue 0004's probe deadlock was exactly this class (a
liveness probe holding the serial daemon's one connection while a second
connection waited forever), fixed client-side by dropping the probe stream. The
daemon-side hole remains: today the only recovery from a stuck peer is
`torch daemon restart`, which — post issue 0006 — costs the whole registry
unless everything was exported first.

## The Contract

Settled before this issue was opened:

- **Concurrent connections.** N clients connected simultaneously; each gets its
  own daemon thread; an idle or hung connection parks its thread and costs
  nobody anything.
- **Serialized execution.** Ops run one at a time behind a shared lock. Two
  clients submitting heavy `mm`s queue; the wait is bounded by real work, never
  by a peer that is merely sitting on the socket. The lock IS the execution
  queue.
- **No parallelism promised, anywhere.** Not in compute, not in the contract,
  not implied by the docs. Overlapping GPU work (MPS streams) is a separate
  future decision this issue explicitly does not take.

## Analysis

The shape is small and the building blocks already exist:

- **State sharing.** `Lifecycle` is already `Arc<Mutex<…>>` (issue 0004).
  `Registry` moves to `Arc<Mutex<Registry>>`; `serve_connection` takes clones;
  each request locks the registry for the duration of its dispatch.
  `tch::Tensor` is `Send`, so entries may cross thread boundaries inside the
  registry; references never leave the lock's critical section (the existing
  `execute_table` borrow discipline is unchanged — it all happens under one lock
  acquisition).
- **Thread-per-connection.** The accept loop spawns
  `std::thread::spawn(serve_connection(…))`. No tokio, no async runtime — a
  local Unix socket with a handful of clients does not justify one. No thread
  pool either, unless review argues otherwise: shell usage produces few
  concurrent connections, and threads are cheap at this scale.
- **Design corners (each gets decided in an experiment, not here):**
  1. **Shutdown coordination.** `shutdown` and idle-TTL expiry currently
     `std::process::exit(0)` after unlinking the socket. With in-flight requests
     on other threads, exit must not corrupt a peer's mid-response write. The
     likely answer: shutdown takes the registry lock (so no op is
     mid-execution), responses are line-buffered and flushed per request anyway,
     and abrupt socket death on exit is already a condition clients handle
     (daemon death mid-connection is indistinguishable from a crash today). The
     experiment decides how much grace is owed.
  2. **Idle-TTL coherence.** The lease must not expire while a request is
     executing. Today that cannot happen (the serial loop touches before
     dispatch); with threads, a long op plus an aggressive TTL could race.
     Likely answer: touch under the same registry lock as execution, or hold a
     lease guard for the request duration.
  3. **Error isolation.** One client's protocol garbage produces an error
     response on ITS stream and affects no other connection. (Mostly free with
     thread-per-connection; the experiment proves it.)
  4. **Accounting.** `status` reporting (tensors, bytes, idle) reads under the
     same lock; `tensors` likewise. Free/list semantics from issue 0006 are
     unchanged — they just serialize like everything else.
- **What stays untouched:** the wire protocol (NDJSON, one line per request),
  the op table, the grammar, the client. A concurrency change that forces
  protocol or client changes has exceeded this issue's mandate.

## Verification Sketch

The issue closes when a stuck client demonstrably costs nothing: open a raw
connection and send nothing; concurrently run real ops from other shells;
everything proceeds; the idle connection eventually costs only its own thread.
Plus: parallel-client stress (N shells looping ops against one daemon, all
results exact), shutdown-under-load behaving per the decided semantics, and the
full existing suite (unit, golden, lifecycle) green unchanged.

## Experiments

- [Experiment 1: Thread-per-connection, one lock as the execution queue](01-thread-per-connection.md)
  — **Pass** (one file changed; the parked-client hang demonstrated before and
  dead after; 600-tensor stress exact; shutdown under load with zero
  user-visible disruption via auto-spawn)

## Conclusion

**Solved**, in one experiment, at the declared price: the whole change is one
file (`nutorchd/src/main.rs`) — `Arc<Mutex<Registry>>`, thread-per-connection,
the lock as the execution queue. The wire protocol, op table, grammar, client,
and every existing test are untouched.

The contract delivered exactly as settled before the issue opened: concurrent
connections (a parked client — demonstrated to wedge the pre-change daemon — now
costs only its own thread), serialized execution (parallel `mm`s queue behind
the lock and come out whole-tensor-equal to solo runs; an 8-shell stress landed
exactly its 600 tensors with zero errors), and no parallelism anywhere.

All four design corners resolved:

1. **Shutdown** holds the registry lock from dispatch straight through `exit()`
   — no op can be mid-execution at death; peers see EOF, which clients already
   handle. Under load the observed behavior was stronger than designed:
   post-stop clients transparently auto-spawn a successor (issue 0004's
   contract), zero user-visible disruption.
2. **TTL coherence**: the expiry watcher checks the lease under the registry
   lock (same registry→lifecycle order as the request path), so expiry can never
   fire mid-op — closing a race the SERIAL daemon already had — while an idle
   open connection correctly fails to pin the lease.
3. **Error isolation**: per-connection streams; garbage answers on its own
   stream only.
4. **Accounting**: `status`/`tensors` snapshot under the same lock, exercised
   under concurrent load.

The stuck-client defect class — the reason this issue existed — is dead.
