+++
status = "open"
opened = "2026-06-10"
+++

# Issue 2: nutorchd proof of concept — tensors in a daemon, used from any shell

## Goal

Build a working proof of concept of the v2 architecture: a `nutorchd` daemon
process that owns tensor memory, and a thin client that can create a tensor or
two, perform an operation on them **on the GPU (MPS)**, and get the result back
— built against the latest PyTorch/libtorch and the latest tch-rs that pairs
with it.

## Background

[Issue 1](../0001-archive-v1/README.md) archived v1 (the Nushell plugin) into
`v1/` and reoriented the repository toward v2. The vision (root
[AGENTS.md](../../AGENTS.md)) is a standalone daemon owning the tensor registry,
with string handles passed over a Unix socket so any shell is a client. This
issue builds the smallest thing that proves that spine end-to-end.

Two constraints carried in from issue 1's conclusion:

1. **The toolchain must be proven first.** v1 no longer builds: Xcode 26.4's
   clang rejects libtorch 2.x headers (`std::is_arithmetic` specialization in
   `c10/util/strong_type.h`) when compiling `torch-sys v0.20.0` against the
   Homebrew Python torch install. v2 builds against the same tch-rs/libtorch
   stack, so the first experiment of this issue must prove the chosen
   tch-rs/libtorch pairing compiles and sees MPS on this machine **before any
   design depends on it**. We do not need to fix the v1 build; v2 must build.
2. **v1 is the reference implementation.** The conversion logic, shape
   validation, and tch-rs call patterns in `v1/cargo/src/` (especially `lib.rs`,
   `command_tensor.rs`, `command_full.rs`, `command_mm.rs`, `command_mean.rs`,
   `command_value.rs`, `command_add.rs`) port into the daemon's dispatcher. Port
   from v1; never edit it.

## Analysis

### Versions: latest, exactly paired

As of June 2026, the latest tch-rs is **0.24.0**, which requires **libtorch
v2.11.0** (the pairing is strict and exact-version; tch 0.23.0 ↔ libtorch
2.10.0). The experiment that pins versions must re-verify the latest pairing at
design time rather than trusting this paragraph.

Strategy: do **not** point `LIBTORCH` at the Homebrew Python torch install (that
header/toolchain mismatch is what killed the v1 build). Use either tch's
`download-libtorch` build feature or the official libtorch macOS arm64 download,
pinned to the exact version tch expects, and confirm MPS is available through it
(`tch::Device::Mps`, `utils::has_mps()`).

### The operation set (six commands)

Each operation proves a distinct part of the architecture; nothing redundant:

| Op       | What it proves                                                                                                        |
| -------- | --------------------------------------------------------------------------------------------------------------------- |
| `tensor` | Client→daemon **data upload**: a real array in, MPS placement, registry insert, handle out                            |
| `full`   | Creation at scale without shipping data; exact-value verification (all-ones)                                          |
| `add`    | Simplest binary op; completes the small round-trip pipeline                                                           |
| `mm`     | **Two-handle lookup across separate client invocations** — the property that justifies the daemon; visible MPS payoff |
| `mean`   | Unary reduction, so the big pipeline returns a scalar, not 1M elements over the socket                                |
| `value`  | Daemon→client **data download**, completing the round trip                                                            |

The two pipelines the PoC must run, from a plain POSIX shell:

```bash
# Round trip (small, exact):
a=$(torch tensor '[1,2,3]' --device mps)
b=$(torch tensor '[4,5,6]' --device mps)
torch add $a $b | torch value        # → [5.0, 7.0, 9.0]

# GPU showcase (big, still exact):
torch full '[1000,1000]' 1 --device mps \
  | torch mm "$(torch full '[1000,1000]' 1 --device mps)" \
  | torch mean | torch value          # → 1000.0, exactly
```

All-ones inputs make every expected value exact (a `[N,N]` ones-matmul yields
every element = N; mean = N), so verification needs no float tolerances and no
random seeds.

### Architecture sketch (PoC-grade)

- **`nutorchd`**: a daemon process owning a `HashMap<String, tch::Tensor>`
  registry (ported from v1's `TENSOR_REGISTRY` concept), the LibTorch context,
  and GPU memory. Listens on a Unix socket.
- **`torch`** (thin client): one operation per invocation — connect, send
  request, print the resulting handle (or value) to stdout, exit. Reads a handle
  from stdin when no positional tensor argument is given, so POSIX pipelines
  compose (the dual input pattern's pipeline form, nearly free).
- **Wire protocol: deliberately throwaway.** Something trivially debuggable
  (e.g. newline-delimited JSON over the Unix socket) is fine. "PoC" means the
  protocol is allowed to be embarrassing; protocol design gets its own issue
  once the spine is proven.
- **Crate layout**: a fresh v2 Rust workspace at the repo root (location and
  naming decided in the experiments; `v1/` stays untouched). The root
  `AGENTS.md` Directory Structure section gets updated when the scaffolding
  lands.

### What the PoC proves

1. The latest tch-rs/libtorch pairing **builds on this machine** and exposes MPS
   (the issue-1 mandate).
2. Tensor memory **outlives a client process**: a handle created by one `torch`
   invocation is usable by a later, separate invocation — the core daemon
   property v1 could not have.
3. GPU compute works end-to-end from a plain shell (bash), with exact,
   assertable results.
4. The v1 command logic ports cleanly into a daemon dispatcher.

### Explicitly out of scope

- Autograd (`backward`, `grad`, `zero_grad`, `sgd_step`)
- Tensor lifecycle management (`free`, TTLs, sessions, named handles)
- The full dual-input surface beyond stdin handle piping
- `manual_seed` / random tensors (deterministic inputs make them unnecessary)
- Protocol design quality, auth, multi-daemon, Nushell premium client — each
  deferred to later issues
- Fixing the v1 build (explicitly not a goal)

## Experiments

- [Experiment 1: Toolchain proof — tch 0.24.0 + libtorch 2.11.0 builds and sees MPS](01-toolchain-proof.md)
  — **Pass** (libtorch 2.11 headers fixed the v1-killing clang error; MPS
  exact-value matmul green; libtorch pinned via repo-local venv +
  `.cargo/config.toml`, since no arm64 libtorch zip exists)
- [Experiment 2: The daemon spine — nutorchd + `torch` client, `tensor`→handle→`value`](02-daemon-spine.md)
  — **Pass** (cross-process handle persistence proven on CPU and MPS; found and
  fixed the no-rpath-from-torch-sys gap that broke direct shell execution)
