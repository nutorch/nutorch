+++
status = "open"
opened = "2026-06-10"
+++

# Issue 3: GPU-only — drop the device option, require MPS

## Goal

Make nutorch GPU-only: every tensor lives on the GPU, with **no `device` option
anywhere** — not in the client CLI, not in the wire protocol, not in the daemon
API. Mac-only for now, which means the one and only device is MPS. Update the
code and the docs to match.

## Background

The PoC (issue [0002](../0002-nutorchd-poc/README.md)) carried v1's device
model: a `--device` flag defaulting to CPU, with explicit placement and a
Rust-side device-mismatch check.

The product decision that supersedes it: **the reason this library exists is GPU
compute from the shell.** If CPU tensors were the goal, any scripting language
already does that better. A CPU default — or even a CPU _option_ — is a footgun
that makes the flagship use case opt-in. So:

- **GPU-only.** There is no device choice. Creation ops place tensors on the
  GPU, period.
- **Mac-only for now**, so the GPU is MPS. The daemon requires MPS and refuses
  to start without it — a clear startup error beats per-op failures on
  unsupported machines.
- **No `device` option at all.** Not a flag with a new default — the concept
  leaves the API surface. (Future platform expansion, e.g. CUDA on Linux, would
  be a per-platform "the GPU" decision at daemon level — still not a per-tensor
  device option.)

This deliberately retires part of carried-forward principle 4 ("explicit device
placement") as v1 stated it: with exactly one device there is nothing to place.
The principle's surviving content — no silent auto-casting, no broadcasting
surprises — is unaffected.

## Analysis

### Code changes (v2 workspace only; `v1/` stays frozen)

- **Protocol** (`nutorchd/src/protocol.rs`): remove the `device` field from
  `tensor` and `full` requests. A request that still sends `device` is rejected
  (serde `deny_unknown_fields`) rather than silently ignored — the honest
  contract during the transition.
- **Daemon** (`nutorchd/src/main.rs`, `convert.rs`):
  - on startup: if `tch::utils::has_mps()` is false, print a clear error and
    exit non-zero (Mac/MPS required);
  - `parse_device` is deleted; creation ops place tensors on `Device::Mps`
    unconditionally;
  - the Rust-side device-mismatch check in `binary_operands` becomes
    structurally unreachable (every registry tensor is MPS) — remove it and its
    tests, or keep it as a debug assertion; decided in the experiment;
  - **internal CPU copies remain an implementation detail**: `value` still
    copies to CPU to serialize, and conversion may stage through CPU buffers.
    "GPU-only" is an API statement about where tensors live, not a ban on the
    CPU existing.
- **Client** (`torch-cli/src/main.rs`): remove `--device` parsing entirely;
  passing `--device` is an unknown-flag error.
- **Tests**: update unit tests (device-parsing tests and the cross-device
  mismatch test go away with the feature); `mps_smoke` keeps the MPS-required
  assertion (now also the daemon's startup contract); the CPU matmul test stays
  as an internal diagnostic (it tests the stack, not the API).

### Doc changes

- **Root `AGENTS.md`**: Vision section (the diagram's "native CUDA/MPS/CPU"
  framing and "one per GPU device" note become MPS/Mac-only with a
  future-platforms caveat); Carried-Forward Principles (rewrite principle 4 as
  above, recording that this issue retired device placement); anything else
  mentioning `--device`.
- **Root `README.md`**: the teaser pipeline loses `--device mps` (it's implicit
  now — that's the whole point); state the Mac/MPS-only requirement prominently.
- **Issue 0002's docs are immutable** — the PoC pipelines there keep their
  `--device mps` flags as historical record; this issue's experiments are the
  record of the change.
- **`v1/` untouched** (frozen; v1 had a device option and that stays true as
  history).

### What does NOT change

- `--dtype` survives (dtype is orthogonal to device; float32 default, float64
  rejected at the MPS boundary by tch as today).
- The `value`/serialization path, handles, socket, protocol shape otherwise.
- The dual input pattern and all six ops' semantics.

### Open questions for the experiment(s) to settle

1. Reject-vs-ignore for a `device` field arriving on the wire (analysis above
   says reject; confirm serde mechanics and error quality).
2. Whether `binary_operands` keeps a debug-only device assertion as
   belt-and-braces against future registry bugs.
3. Whether the daemon's startup MPS requirement needs a test hook (it cannot be
   integration-tested on this machine, where MPS is always present — likely a
   unit-testable guard function).
