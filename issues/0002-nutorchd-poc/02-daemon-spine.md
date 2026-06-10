+++
[implementer]
agent = "claude-code"
model = "claude-fable-5"

[review.design]
agent = "claude-code"
subagent = "adversarial-reviewer"
model = "claude-opus"

[review.result]
agent = "claude-code"
subagent = "adversarial-reviewer"
model = "claude-opus"
+++

# Experiment 2: The daemon spine — nutorchd + `torch` client, `tensor`→handle→`value`

## Description

Build the smallest version of the v2 architecture that is real: `nutorchd` owns
a tensor registry behind a Unix socket; a separate thin `torch` CLI sends one
operation per invocation; and a tensor created by **one** client process is
retrieved by a **different** client process — the cross-process persistence
property that justifies the daemon's existence.

Scope is deliberately two operations only — `tensor` (data upload) and `value`
(data download) — because they exercise every structural piece (socket,
protocol, dispatch, registry, JSON↔tensor conversion both ways, handle
lifecycle, stdin piping) without any math. The remaining four ops are nearly
pure dispatcher entries once this spine stands, and follow in the next
experiment.

Per the issue: the wire protocol is **deliberately throwaway** —
newline-delimited JSON over a Unix socket, chosen for debuggability
(`nc -U`-able), not merit.

## Changes

1. **Workspace**: add member `torch-cli` to the root `Cargo.toml`.

2. **`nutorchd` crate** (replaces the diagnostic stub):
   - `src/main.rs`: bind a `std::os::unix::net::UnixListener` on the socket path
     (`--socket <path>` flag, default `$TMPDIR/nutorchd.sock`, fallback
     `/tmp/nutorchd.sock`; remove a stale socket file on startup). **Known PoC
     simplification**: the stale-socket removal is unconditional — a second
     daemon started on the same path silently steals it from a live first
     daemon, stranding that daemon's handles. Accepted for the PoC; the
     lifecycle/naming issue that follows this one must not inherit it silently.
     Accept connections in a loop; one connection at a time (PoC). For each
     connection, read newline-delimited JSON requests, dispatch, write one JSON
     response line per request. On startup print the socket path and the MPS
     availability line (keeping Experiment 1's diagnostic value).
   - `src/registry.rs`: `HashMap<String, tch::Tensor>` + UUID-v4 handle
     generation (dependency: `uuid` with `v4` feature).
   - `src/convert.rs`: JSON↔tensor conversion ported from v1
     (`v1/cargo/src/lib.rs` `value_to_tensor`/`tensor_to_value`), with
     `serde_json::Value` in place of Nushell `Value`:
     - JSON number/array-of-numbers/nested arrays → tensor (recursive), then
       cast to the requested dtype — **default `float32`, matching v1's
       `--dtype` default** (fidelity note: Python `torch.tensor([1,2,3])` infers
       int64; v1 chose float32 as the shell default and v1 is the reference —
       the PoC pipelines' expected `[5.0, 7.0, 9.0]` encodes this);
     - tensor → JSON (0-D → number, N-D → nested arrays), via CPU copy.
   - `src/protocol.rs`: request/response types with serde:
     - request
       `{"op":"tensor","data":<json>,"device":"cpu|mps","dtype":"float32|float64|int32|int64"}`
       → response `{"ok":true,"handle":"<uuid>"}`;
     - request `{"op":"value","handle":"<uuid>"}` →
       `{"ok":true,"value":<json>}`;
     - any error → `{"ok":false,"error":"<message>"}` (unknown op, missing
       handle, bad data, bad device/dtype — Rust-side validation per the
       carried-forward principles).
   - dependencies added: `serde` (derive), `serde_json`, `uuid`.
   - unit tests (in-process, no socket): JSON→tensor→JSON round trip for a flat
     list, a nested 2-D list, and a scalar; default-dtype check (ints in →
     floats out); error cases (unknown handle, ragged nested array).
3. **`torch-cli` crate** (new): binary named `torch`.
   - `src/main.rs`: parse
     `<op> [args...] [--device d] [--dtype t]
     [--socket path]`; ops this
     experiment: `tensor <json-array>` and `value [handle]`. **Stdin piping**:
     when an op needs a tensor argument and none is given positionally, read one
     line from stdin (the dual input pattern's pipeline form). Connect to the
     socket, send the one-line JSON request, print the response payload: handles
     as bare strings, values as compact JSON, errors to stderr with exit code 1.
   - No `tch` dependency — the client must stay thin (serde_json + std only).
4. **Root `AGENTS.md`**: update the Directory Structure section to record the
   now-real v2 source tree (`nutorchd/`, `torch-cli/`, `Cargo.toml`,
   `.cargo/config.toml`, `.venv-torch`/`.libtorch` convention) — discharging the
   "recorded here when the scaffolding lands" note.

## Verification

From the repo root. Hygiene gates first, then the behavioral proof.

1. **Hygiene**: `cargo build` exit 0, no warnings from workspace crates;
   `cargo test` (workspace) green — conversion unit tests plus Experiment 1's
   `mps_smoke` still passing; `cargo fmt --all -- --check` clean; `dprint check`
   clean on files created/edited by this experiment;
   `git status --porcelain v1/` empty.
2. **Spine round trip (the core proof), as separate processes**:
   ```bash
   ./target/debug/nutorchd &            # daemon process
   h=$(./target/debug/torch tensor '[1,2,3]')        # client process #1
   ./target/debug/torch value "$h"                   # client process #2
   # → [1.0,2.0,3.0]  (float32 default), printed as JSON
   ```
   Pass requires: `$h` is a UUID-shaped string; client #2 (a different process
   from client #1) returns `[1.0,2.0,3.0]`; the daemon stays up throughout.
3. **Stdin piping**:
   `./target/debug/torch tensor '[1,2,3]' | ./target/debug/torch value` prints
   `[1.0,2.0,3.0]`.
4. **MPS placement**: `h=$(torch tensor '[1,2,3]' --device mps)` then
   `torch value $h` returns `[1.0,2.0,3.0]` (created on MPS, read back via CPU
   copy).
5. **Error paths, with explicit survival probes**: `torch value not-a-handle`
   exits 1 with an error on stderr, then a successful
   `torch tensor '[9]' | torch value` immediately after asserts the daemon
   survived (a positive liveness assertion, not an inference); likewise a ragged
   array (`[[1,2],[3]]`) is rejected with an error, followed by the same
   liveness probe.
6. **2-D round trip**: `torch tensor '[[1,2],[3,4]]' | torch value` returns
   `[[1.0,2.0],[3.0,4.0]]`.
7. **Teardown**: checks 2–6 reuse the single daemon started in check 2; at the
   end, kill the daemon and confirm the socket file is gone, so the run is
   reproducible and self-cleaning.

**Pass** = all of the above. **Partial** = spine works on CPU but MPS placement
fails (recorded; MPS compute was proven in Experiment 1, so a failure here would
be placement-specific). **Fail** = the cross-process round trip (check 2) does
not work.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**Verdict: APPROVED** — no Required findings. The reviewer verified the
load-bearing fidelity claim against v1 source (`v1/cargo/src/lib.rs:197`
confirms `Kind::Float` is v1's no-`--dtype` default, so float32-not-int64 is
faithful to the reference); confirmed check 2 genuinely proves cross-process
persistence; confirmed the one-connection-at-a-time loop cannot deadlock the
stdin-piping check (the second client reads stdin before connecting); and
confirmed the AGENTS.md Directory Structure fold-in discharges an explicit
deferral rather than being scope creep. Two Optional findings and one Nit, all
folded in: (1) the unconditional stale-socket removal is now recorded as a known
PoC simplification the lifecycle issue must not silently inherit; (2) error-path
checks gained explicit post-error liveness probes; (3) verification gained a
teardown step and states checks 2–6 share one daemon.

## Result

**Result:** Pass

The spine stands. All behavioral checks pass with direct binary execution (plain
shell, no cargo environment):

```
handle: 3c7bf6d1-eaa5-4b99-a54e-0b6f36a8ee98     # client process #1
[1.0,2.0,3.0]                                    # read by client process #2
[1.0,2.0,3.0]                                    # stdin piping
[1.0,2.0,3.0]                                    # created on MPS, read back
torch: unknown handle: not-a-handle  (exit 1) → liveness probe: [9.0]
torch: ragged or mismatched nested list: ...  (exit 1) → liveness probe: [9.0]
[[1.0,2.0],[3.0,4.0]]                            # 2-D round trip
```

The daemon startup log reports `MPS available: true`. Hygiene: `cargo build` 0
warnings; `cargo test` green (9 conversion/registry unit tests + the 3
Experiment-1 smoke tests); `cargo fmt --all -- --check` clean; `dprint check`
clean; `git status --porcelain v1/` empty.

**Discovery (the load-bearing one): torch-sys 0.24 bakes no rpath when
`LIBTORCH` is set.** The first verification run failed wholesale — the daemon
died at exec with
`dyld: Library not loaded: @rpath/libtorch_cpu.dylib …
no LC_RPATH's found`.
Experiment 1 never saw this because `cargo run` / `cargo test` inherit the
`.cargo/config.toml` `[env]` DYLD path; **direct execution from a shell — the
PoC's whole point — exercised dylib resolution for the first time** (and
corrects Experiment 1's recorded belief that torch-sys bakes an rpath; it does
not in this configuration). Fix, added to `.cargo/config.toml` beyond the
design's change list: `[build] rustflags` baking two repo-relative rpaths
(`@loader_path/../../.libtorch/lib` for `target/<profile>/<bin>`, `…/../../../`
for test binaries under `deps/`). Verified via `otool -l`: both `LC_RPATH`
entries present. PoC limitation, recorded: binaries resolve dylibs only while
inside the repo; the install story belongs to a later issue.

**Two smaller deviations, recorded:**

1. **tch error hygiene**: raw tch errors stringify with a full C++ backtrace,
   which the first ragged-array check dumped to the user. Added
   `convert::tch_error` (first line only) on every fallible tch call — the
   carried-forward good-error-messages principle applied to the daemon.
2. **Teardown expectation corrected**: the design said "kill the daemon and
   confirm the socket is gone", but a Unix socket file outlives its process —
   `kill` leaves it behind. The teardown now kills the daemon and removes the
   socket file explicitly; daemon-side cleanup on signal is deferred to the
   lifecycle issue (alongside the already-recorded unconditional stale-socket
   removal).

Also fixed during implementation: two `Registry` methods (`len`/`is_empty`)
written speculatively tripped the no-warnings gate as dead code and were removed
— the gate works.

## Conclusion

The v2 architecture is no longer hypothetical: a tensor uploaded by one shell
process lives in the daemon and is read back by another process, on CPU and MPS,
over a `nc`-debuggable NDJSON socket, with errors that return instead of killing
the daemon (fallible `f_*` tch calls — a deliberate departure from v1's
panicking plugin idiom, which a daemon cannot afford).

For the next experiment: the four compute ops (`full`, `add`, `mm`, `mean`) are
dispatcher entries plus argument plumbing on this spine — `full` takes
shape+fill, the binary ops take two handles (client: positional XOR stdin),
`mean` takes one. Port validation patterns from `v1/cargo/src/command_*.rs`,
keep every tch call fallible, and finish with the issue's two PoC pipelines
end-to-end (exact expected values `[5.0,7.0,9.0]` and `1000.0`), plus the
cross-device mismatch error case (`cpu` tensor ⊕ `mps` tensor must error
cleanly, not crash).

## Result Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only),
reviewing the pre-commit working tree. **First pass: CHANGES REQUIRED** — one
Required finding: the AGENTS.md Directory Structure update added the v2 tree but
left the trailing "the v2 source tree does not exist yet" paragraph in place,
contradicting the tree above it and failing change-list item 4's promised
discharge. **Fixed** (paragraph rewritten to describe the tree as the issue-0002
PoC scaffolding). **Re-review (fresh context): APPROVED** — finding confirmed
resolved, no new findings.

Beyond the finding, the first-pass reviewer independently reproduced everything:
all hygiene gates green (12 tests, 0 warnings, fmt/dprint clean, v1 untouched);
started the daemon **directly with the dylib env vars stripped**, proving the
rpath fix (both `LC_RPATH` entries confirmed via `otool`); ran the full
behavioral suite including the one-line ragged error and both liveness probes;
verified all four recorded deviations against the code and confirmed the
"corrects Experiment 1's rpath belief" claim is fair; confirmed conversion
fidelity against `v1/cargo/src/lib.rs` with no real divergence; and confirmed
the client crate has no tch dependency.
