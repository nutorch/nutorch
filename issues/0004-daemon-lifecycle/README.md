+++
status = "open"
opened = "2026-06-10"
+++

# Issue 4: Daemon lifecycle — auto-start, a 1-hour idle TTL, and `torch daemon` commands

## Goal

Make the daemon invisible plumbing with a bounded memory horizon: any `torch`
command **auto-starts** the daemon if it is not running; the daemon **shuts
itself down after 1 hour of inactivity** (configurable at spawn and at runtime);
and a `torch daemon` command family makes the lifecycle analyzable and
controllable (`status`, `ttl`, `stop`, `restart`, `start`).

## Background

The PoC (issues [0002](../0002-nutorchd-poc/README.md),
[0003](../0003-mps-only/README.md)) has no lifecycle story: the daemon is
started by hand, runs until killed, handles no signals (a kill strands the
socket file), and a second daemon started on the same path silently steals it
from a live first daemon — both recorded as PoC debts. The client has no
tooling; a missing daemon is just "cannot connect".

The product decision: **you should not have to think about the daemon.** It
starts itself when needed. Its memory lasts long enough to be useful — an hour
of inactivity by default — but not far longer than you need, so a forgotten
daemon does not hold GPU memory overnight. Both halves are load-bearing:
auto-start makes expiry harmless (the next command gets a fresh daemon, never an
error), and expiry makes auto-start safe to forget.

## Analysis

### Start: client auto-spawn

- On connect failure, the client spawns `nutorchd` detached (binary located next
  to the `torch` binary; overridable), passing the resolved socket path
  explicitly, then polls the socket until it answers (bounded wait, clear error
  on startup failure), and retries the request. `torch daemon start` invokes the
  same path explicitly (pre-warming); all other commands do it implicitly.
- **Spawn races resolve at the bind**: before binding, the daemon probes the
  socket for liveness; if a live daemon answers, the newcomer exits quietly and
  the client proceeds against the winner. This replaces the unconditional
  stale-socket removal and fixes the socket-steal debt — a live daemon can never
  lose its socket.
- Daemon stdout/stderr go to a log file beside the socket
  (`$TMPDIR/nutorchd.log`); `torch daemon status` reports the path.

### Lifetime: sliding idle TTL, default 1 hour

- **Sliding, not a fixed fuse**: every tensor operation resets the idle clock.
  Active use never loses tensors; abandonment costs at most one idle TTL of GPU
  memory. (A fixed-from-start lifetime would kill the daemon mid- work and
  silently hand the next command an empty registry — rejected.)
- **Read-only daemon commands (`status`) do not reset the clock** — observing
  the daemon must not immortalize it. Tensor ops do.
- Configurable three ways: a `--ttl <duration>` daemon flag, a `NUTORCHD_TTL`
  env var (inherited by auto-spawned daemons), and at runtime via
  `torch daemon ttl <duration>` (a protocol op — no restart). Human-friendly
  durations (`90s`, `30m`, `2h`); `none` = run forever.
- On expiry: unlink the socket, log the shutdown, exit 0. Lost tensors are the
  documented contract of the memory horizon; auto-spawn turns the consequence
  into a clean empty restart.

### End: three exits, all clean

1. **Idle expiry** (the normal case) — above.
2. **`torch daemon stop`** — a graceful `shutdown` protocol op: finish the
   in-flight request, unlink the socket, exit 0.
3. **Signals** (SIGTERM/SIGINT) — caught; socket unlinked; clean exit. Fixes the
   stranded-socket debt. (std has no signal API; this is the moment to take a
   small signal-handling dependency.)

### Commands

```
torch daemon status      # running? pid, socket path, uptime, device,
                         #   ttl, idle time, time remaining before expiry,
                         #   tensor count, approx memory held, log path
torch daemon ttl <dur>   # set TTL on the live daemon (e.g. 4h, none)
torch daemon stop        # graceful shutdown now
torch daemon restart     # stop + start (fresh registry, explicit)
torch daemon start       # pre-warm explicitly (otherwise automatic)
```

Backed by three protocol additions: `status`, `set_ttl`, `shutdown`. `status` is
what makes the policy trustworthy: it always shows how long the memory has left
and how much it is holding. All verbs honor `--socket`.

### Socket path (confirmed, unchanged)

The default stays `$TMPDIR/nutorchd.sock` (per-user `0700` Darwin temp dir →
private by OS permissions, one daemon per user for free, ephemeral in a way that
is harmless under a 1-hour TTL), `/tmp/nutorchd.sock` fallback. The auto-spawn
path passes the socket explicitly to the daemon it spawns, which removes
client/daemon default-divergence for spawned daemons; `status` shows the
resolved path for the rest.

### Implementation note

The current accept loop blocks forever; self-termination needs either a timer
thread watching the idle deadline or a polling accept. Small, but it is the
structural change that makes expiry possible — settled in the experiment.

### Out of scope (separate issues)

- Concurrency (the one-connection-at-a-time loop stays).
- Tensor lifecycle within a live daemon (`free`, per-tensor TTLs, named handles)
  — tensor lifecycle ≠ daemon lifecycle.
- launchd integration (an install/distribution concern; auto-spawn makes it
  unnecessary for now) and any auto-spawn-on-login behavior.
- Persistence / save-load (the real long-term answer to "I left and my tensors
  died").

## Experiments

- [Experiment 1: Daemon-side lifecycle — idle TTL, clean exits, probe-bind, and the three protocol ops](01-daemon-side-lifecycle.md)
  — **Pass** (expiry/renewal/status/set_ttl/shutdown/signals all behave; both
  socket debts closed; 32 tests green)
- [Experiment 2: Client auto-spawn and the `torch daemon` command family](02-client-autospawn-and-verbs.md)
  — **Designed**
