+++
[implementer]
agent = "claude-code"
model = "claude-fable-5"
+++

# Experiment 2: Client auto-spawn and the `torch daemon` command family

## Description

The user-facing half: any tensor command **auto-starts** the daemon when the
socket is dead, and `torch daemon status|ttl|stop|restart|start` exposes the
Experiment-1 machinery. After this experiment the issue's goal statement holds
end to end: you never think about the daemon, and you can always ask it how long
your memory has left.

Experiment 1 deliberately made this layer thin: every verb is one wire op plus
formatting; auto-spawn is "spawn what `probe_and_bind` already knows how to
welcome".

## Changes

1. **Auto-spawn** (`torch-cli/src/main.rs`):
   - On connect failure for a **tensor op**, the client: locates `nutorchd`
     (same directory as the `torch` binary via `current_exe()`; overridable with
     `NUTORCHD_BIN`), spawns it detached with `--socket <resolved path>` and
     stdout/stderr appended to the conventional log file (socket path with
     `.log`), then polls the socket (~50 ms interval, ~5 s bound) until it
     answers, and retries the request once. Startup failure → a clear error
     naming the log file. The spawned daemon gets `Stdio::null()` for stdin and
     is fully detached, so it survives the client's exit. (The daemon's
     probe-bind makes spawn races harmless — losers exit 0 — except for the
     simultaneous-start TOCTOU window already recorded and deferred in
     Experiment 1.)
   - **Which commands auto-spawn**: tensor ops, `daemon start`, and
     `daemon restart` do. `daemon status`, `daemon ttl`, and `daemon stop` do
     **not** — observing, configuring, or stopping a daemon must not create one.
2. **`torch daemon <verb>`** (new op family in the client; no protocol changes —
   Experiment 1 shipped the ops):
   - `status` → `status` op, printed as human-readable lines (pid, uptime, ttl,
     idle, remaining, tensors, memory, socket, log). Not running → "not running
     (socket: …)" on stderr, **exit 1** (scriptable liveness check), and no
     spawn.
   - `ttl <duration>` → `set_ttl` op; prints the new ttl. Not running → error,
     exit 1.
   - `stop` → `shutdown` op; prints confirmation. Not running → "not running",
     **exit 0** (stopping nothing is success — idempotent).
   - `restart` → `stop` semantics (ignore not-running), **then poll until the
     old socket is gone (or refuses) before spawning** — the old daemon flushes
     its shutdown response before unlinking, so a new daemon spawned too eagerly
     could probe the dying one, yield, and leave zero daemons. Then the
     auto-spawn path; prints the new pid via a follow-up `status`.
   - `start` → the auto-spawn path; prints "started (pid N)" or "already running
     (pid N)".
   - **The `daemon` verb family parses its subcommand and arguments from
     positionals only and never falls back to stdin** (unlike `value`/`mean`,
     these verbs have no pipeline form; a missing `ttl <duration>` is a plain
     usage error, not a blocking stdin read).
3. **Docs**:
   - root `README.md`: a short "The daemon" paragraph — starts automatically,
     idles out after 1 hour (configurable), `torch daemon status` to inspect,
     tensors live only as long as the daemon (the memory-horizon contract);
   - root `AGENTS.md`: the Vision bullet on tensor lifetime is updated — the
     daemon lifecycle (auto-start, sliding idle TTL) is now real, while
     tensor-level lifecycle (named handles, `free`, per-tensor TTLs) remains
     future work; Directory Structure gains `src/lifecycle.rs`.
4. **Tests**: the client remains I/O-bound glue, so this experiment's
   correctness lives in the behavioral checks below (the daemon-side logic they
   exercise is already unit-tested). No new unit tests beyond what refactoring
   requires; the no-warnings gate still applies.

## Verification

From the repo root; dedicated `--socket` paths under `/tmp`; everything cleaned
up by the checks. `T=./target/debug/torch`.

1. **Hygiene**: `cargo build` 0 warnings; `cargo test` green (32 from Exp 1);
   `cargo fmt --all -- --check` clean; `dprint check` clean on touched files;
   `git status --porcelain v1/` empty.
2. **Cold start (the headline)**: with no daemon and no socket file,
   `$T tensor '[1,2,3]' --socket S | $T value --socket S` → `[1.0,2.0,3.0]` —
   the daemon was spawned transparently; `$T daemon status --socket S` shows it
   running; the log file `S.log` exists and contains the banner.
3. **status semantics**: after `daemon stop`, `daemon status` prints "not
   running", exits 1, and does NOT spawn a daemon (no socket file appears).
4. **Expiry → transparent respawn (the full invisible loop)**: `daemon ttl 2s`,
   wait ~4s (daemon expires), then a plain `$T tensor '[9]' … | $T value …`
   succeeds → `[9.0]` with a NEW daemon (different pid in `status` than before
   expiry).
5. **stop**: `daemon stop` → confirmation, socket gone; `daemon stop` again →
   "not running", exit 0.
6. **restart**: with a live daemon holding a tensor, `daemon restart` → new pid,
   and the old handle is now `unknown handle` (fresh registry — explicit and
   expected).
7. **start**: `daemon start` on a dead socket → "started (pid N)"; again →
   "already running" with the same pid.
8. **ttl verb**: `daemon ttl 30m` → reports 1800s; `status` agrees (`remaining`
   ≤ 1800).
9. **Default socket end-to-end, safely isolated**: one run with NO `--socket`
   anywhere, but under a **private `TMPDIR`** (`env TMPDIR=$(mktemp -d)`), so
   the default-path code (`$TMPDIR/nutorchd.sock`) is exercised for real while a
   genuine user daemon on the actual default socket can never be touched:
   `$T tensor '[1]' | $T value` → `[1.0]`, then `$T daemon stop`, then the
   private dir is removed. (Stopping a real in-use daemon would destroy live GPU
   tensors — the check must be structurally incapable of that.)
10. **Docs**: README contains the auto-start + 1-hour story; AGENTS.md Vision
    reflects the implemented daemon lifecycle.

**Pass** = all ten. **Partial** = auto-spawn works but a verb misbehaves in a
recorded, non-destructive way. **Fail** = cold start does not work, expiry does
not respawn transparently, or a non-spawning verb spawns a daemon.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**First pass: CHANGES REQUIRED** — 1 Required, 3 Optional, 1 Nit:

- [Required] Verification check 9 ran tensor ops and `daemon stop` against the
  REAL default socket with no guard — on a machine with a genuine daemon in use
  (the expected steady state once this issue ships), the check would have
  stopped it and destroyed live GPU tensors. **Fixed:** check 9 now exercises
  the default-path code under a private `TMPDIR` (`env TMPDIR=$(mktemp -d)`),
  making the hazard structurally impossible.
- [Optional] `restart`'s stop-then-spawn had a latent race (the old daemon
  flushes its shutdown response before unlinking, so an eager respawn could
  probe the dying daemon, yield, and leave zero daemons). **Fixed:** restart
  polls until the old socket is gone before spawning.
- [Optional] The daemon verbs' stdin behavior was unspecified against the
  client's blocking stdin-fallback helper. **Fixed:** positionals only, never
  stdin.
- [Optional] Spawned daemon's stdin unspecified. **Fixed:** `Stdio::null()`
  - detached note.
- [Nit] "spawn races harmless" overstated. **Fixed:** now qualified by the
  deferred Exp-1 TOCTOU window.

**Re-review (fresh context): APPROVED** — all five confirmed resolved; the
reviewer verified the check-9 fix against both binaries' `default_socket_path()`
implementations (private TMPDIR genuinely isolates both the spawned daemon and
the stopping client). No new findings.
