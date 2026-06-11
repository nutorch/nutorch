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

# Experiment 1: Remove the device option; require MPS

## Description

One experiment for the whole issue: the device concept leaves the API surface
(client flag, wire field, daemon parsing), the daemon requires MPS at startup,
all tensors are created on MPS unconditionally, and the docs are updated. The
change is small and tightly coupled — code and docs must flip together or the
docs lie at the result commit.

This also settles the issue's three open questions:

1. **Wire `device` field: reject, with a helpful error.** serde's
   `deny_unknown_fields` does not work on internally tagged enums (a known serde
   limitation), so the rejection is explicit: the daemon parses each request
   line to `serde_json::Value` first and rejects any request object containing a
   `"device"` key with an error that explains the removal, before deserializing
   into `Request`. (Other unknown fields remain ignored, as today — the
   special-case is only the removed option, where silent ignoring would invert
   the meaning of old scripts.)
2. **`binary_operands` device check: demote to `debug_assert`.** With every
   tensor created on MPS the user-facing mismatch error is unreachable; a
   `debug_assert_eq!` documents and enforces the registry invariant in dev
   builds at zero release cost.
3. **Startup MPS requirement: a unit-testable guard.** `require_mps()` returns
   `Result<(), String>`; `main` exits non-zero with its message. On this machine
   only the `Ok` path is integration-testable (MPS is present); the `Err` path
   is trivial by inspection and unit-covered via the guard's structure.

## Changes

1. **Protocol** (`nutorchd/src/protocol.rs`): remove the `device` field from
   `Tensor` and `Full` variants.

2. **Daemon** (`nutorchd/src/main.rs`):
   - `require_mps()` guard; `main` prints the error and exits non-zero when MPS
     is unavailable ("nutorchd requires an Apple-silicon Mac with MPS"); the
     startup banner becomes "device: mps" (replacing the availability line —
     availability is now a precondition, not a report);
   - `serve_connection`: the explicit `"device"`-key rejection described above,
     error text pointing at this issue ("the device option was removed: tensors
     always live on the GPU (mps)");
   - `Tensor`/`Full` dispatch: create on `Device::Mps` unconditionally;
   - `binary_operands`: user-facing device-mismatch error → `debug_assert_eq!`
     on the two devices;
   - `value` keeps its internal CPU copy for serialization (implementation
     detail, per the issue analysis).

3. **Conversion** (`nutorchd/src/convert.rs`): delete `parse_device` and its
   test; `json_to_tensor` keeps its `device` parameter (it is generic over
   device; callers now always pass `Device::Mps`, unit tests may pass CPU — they
   test conversion logic, not placement policy).

4. **Client** (`torch-cli/src/main.rs`): remove `--device` parsing, and stop
   sending the field. Additionally, unknown `--flags` become a clean error
   instead of being swallowed as positionals (today `--device mps` would degrade
   into two confusing positional args), with a dedicated hint when the unknown
   flag is exactly `--device`: "the device option was removed; tensors always
   live on the GPU (mps)".

5. **Tests**:
   - removed: `parse_device` tests (with the function);
   - updated: none of the existing dispatch tests mention devices (they already
     run device-less requests — they now implicitly create on MPS, which this
     machine supports);
   - added: a dispatch-level test that a request carrying `"device"` is rejected
     with the removal message (exercised via the same path `serve_connection`
     uses — the rejection helper is factored so it is unit-testable without a
     socket); a test asserting a created tensor's `device()` is `Device::Mps`; a
     `require_mps()` Ok-path test.

6. **Docs**:
   - root `AGENTS.md`: Vision — the stack diagram's device line becomes
     Metal/MPS, the "one per GPU device" daemon note becomes a future-platforms
     caveat (Mac/MPS-only for now; platform expansion is a daemon-level
     decision, never a per-tensor option); Carried-Forward Principles —
     principle 4 rewritten to record that issue 0003 retired device placement
     (surviving content: no auto-casting, no broadcasting surprises);
   - root `README.md`: teaser pipeline loses `--device mps`; a prominent
     requirement line: Apple-silicon Mac (MPS) only, GPU-only by design;
   - issue 0002 docs untouched (immutable history), `v1/` untouched.

## Verification

From the repo root; behavioral checks against a directly-executed daemon on a
dedicated `--socket`, torn down (kill + rm socket) at the end.

1. **Hygiene**: `cargo build` 0 warnings; `cargo test` green;
   `cargo fmt --all -- --check` clean; `dprint check` clean on touched files;
   `git status --porcelain v1/` empty; `git status --porcelain issues/0002*`
   empty (immutability).
2. **Device-less pipelines, exact** (the issue's flagship simplification):
   ```bash
   a=$(torch tensor '[1,2,3]')
   b=$(torch tensor '[4,5,6]')
   torch add $a $b | torch value     # → [5.0,7.0,9.0]
   torch full '[1000,1000]' 1 | torch mm "$(torch full '[1000,1000]' 1)" \
     | torch mean | torch value      # → 1000.0
   ```
   (These run on MPS by construction; placement is asserted by the new unit
   test, since no op exposes a device to observe externally.)
3. **`--device` is gone from the client**:
   `torch tensor '[1,2,3]' --device
   mps` → exit 1 with the dedicated removal
   hint on stderr; AND a generic unknown flag
   (`torch tensor '[1]' --frobnicate x`) also errors cleanly (the new
   unknown-flag rejection's general path, not just the `--device` special case).
4. **`device` is gone from the wire**:
   `printf '{"op":"tensor","data":[1],"device":"cpu"}\n' | nc -U -w 1
   <socket>`
   → `{"ok":false,...}` with the removal message; daemon survives (liveness
   probe). (`-w 1` because the daemon holds connections open until client EOF;
   the timeout makes the probe terminate deterministically.)
5. **Errors and liveness unchanged**: the mm shape-mismatch check still errors
   cleanly and the daemon survives.
6. **No stale `--device` in the live docs**:
   `rg -n -- "--device" README.md AGENTS.md` → zero matches (issue folders
   0002/0003 and `v1/` are history and exempt).
7. **Startup banner** reports `device: mps` (and the daemon starts — this
   machine has MPS; the refusal path is unit/inspection-covered per open
   question 3).

**Pass** = all seven. **Partial** = code lands but a doc check (6) or the wire
rejection (4) needs follow-up recorded. **Fail** = pipelines break or a device
option remains reachable anywhere in the v2 surface.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**Verdict: APPROVED** — no Required findings. The reviewer confirmed the
load-bearing claims: serde's `deny_unknown_fields` is genuinely unsupported on
internally tagged enums, so the parse-to-Value-first rejection is the right
mechanism and slots minimally into `serve_connection`; the `debug_assert`
demotion is **provably safe** (it traced all five `registry.insert` sites —
`tensor`/`full` create on MPS, `add`/`mm`/`mean` inherit operand device, and
`value`'s CPU copy is serialized but never inserted — so no CPU tensor can enter
the registry); the doc plan correctly leaves v1's historical "CPU/CUDA/MPS"
mentions alone and the `--device` grep cannot false-positive on them; and the
unknown-flag rejection is a scoped fix (today unknown flags silently corrupt
requests as positionals), not creep. Two Optional findings, both folded in: (1)
`nc -U` gets `-w 1` so the wire probe terminates deterministically; (2)
verification #3 now also exercises the generic unknown-flag path, not just the
`--device` special case.

## Result

**Result:** Pass

All seven checks pass:

```
=== startup banner ===
device: mps
=== device-less pipelines (the flagship simplification) ===
[5.0,7.0,9.0]
1000.0
=== --device gone from the client ===
torch: the device option was removed; tensors always live on the GPU (mps)   (exit 1)
torch: unknown flag: --frobnicate                                            (exit 1)
=== device gone from the wire ===
{"ok":false,"error":"the device option was removed (issue 0003): tensors always live on the GPU (mps)"}
→ liveness probe: [9.0]
=== mm shape error + liveness unchanged ===
torch: mm shape mismatch: inner dimensions must match, got [2, 3] and [2, 3] (exit 1) → [9.0]
=== no stale --device in live docs ===
rg -n -- "--device" README.md AGENTS.md → zero matches
```

**Hygiene:** `cargo build` 0 warnings; `cargo test` green — 22 tests (8
conversion + 1 registry + 10 dispatch, including the three new ones:
`require_mps` Ok-path, created-tensors-live-on-MPS, device-field rejection with
the removal message; plus the 3 MPS smoke tests); `cargo fmt` and `dprint check`
clean; `v1/` and `issues/0002*` untouched (immutability gates both empty).

Implementation matched the approved design with **one small recorded
deviation**: the root README's `## Status` section was also rewritten (from
"design phase" to "proof of concept working") — not in the Changes list, but
necessary because a device-less working teaser cannot sit under a "not yet
implemented" label, and the section was already stale relative to issue 0002's
close. Everything else as designed: `device` field removed from `Tensor`/`Full`
protocol variants; `parse_request` does the explicit pre-parse `"device"`-key
rejection (factored and unit-tested without a socket, as designed); creation ops
hardcode `Device::Mps`; `binary_operands`' user-facing mismatch error became a
`debug_assert_eq!` registry invariant; `require_mps()` guards startup and the
banner reports `device: mps`; the client rejects `--device` with the dedicated
hint and all other unknown flags generically (previously they were silently
swallowed as positionals); docs updated (AGENTS.md vision diagram now ends at
"Apple-silicon GPU", principle 4's device-placement clause is recorded as
retired by this issue, README states GPU-only/Mac-only prominently and the
teaser pipeline is device-less).

## Conclusion

The device option no longer exists anywhere in the v2 surface: not as a client
flag, not as a wire field, not in the daemon API. Tensors are born on the GPU
because there is nowhere else to be born; the daemon will not start on a machine
that cannot honor that. The UX got _shorter_ — the teaser pipeline lost its only
flag — which is the point: the flagship use case is now the only case.

Carried forward for future issues: platform expansion (CUDA on Linux) is a
daemon-level decision recorded in AGENTS.md; the wire-level `device` rejection
should eventually age out with the throwaway protocol itself (protocol-design
issue); the registry MPS invariant is debug-asserted, so a future op that
accidentally inserts a CPU tensor fails loudly in dev builds.

## Result Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only),
reviewing the pre-commit working tree. **Verdict: APPROVED** — no Required
findings. The reviewer independently reran everything on its own daemon
instance: all hygiene gates (forced rebuild, 22 tests, fmt/dprint clean, v1 and
issue-0002 immutability gates empty), the device-less pipelines
(`[5.0,7.0,9.0]`, `1000.0`), both client flag rejections, the raw wire rejection
via `nc -U -w 1` with liveness, and the shape-error path. It audited every
remaining `device` occurrence in the v2 sources and confirmed each is messaging,
internal implementation, the debug assertion, comments, or tests — no
user-supplied device is parsed anywhere — and re-proved the debug_assert
demotion safe by tracing all five `registry.insert` sites. One Optional finding,
folded in: the Result's "no deviations" claim overstated — the README
Status-section rewrite was a (defensible, now-recorded) deviation from the
enumerated Changes. One Nit (teaser output spacing now matches real daemon
output) — intentional, no action.
