+++
status = "open"
opened = "2026-06-10"
+++

# Issue 5: The full tensor-op surface — a high-quality CLI for every PyTorch tensor operation

## Goal

Implement the complete PyTorch tensor-operation surface — if it operates on one
or more tensors, it belongs here — as a high-quality CLI: PyTorch semantics
(including broadcasting), intelligent one-line errors, a single deterministic
argument grammar (positionals + multi-line stdin, mixable), discoverable help,
and fidelity verified mechanically against real PyTorch.

## Background

After issues 0002–0004 the spine is solid: daemon, handles, MPS-only, invisible
lifecycle. But v2 speaks six ops; v1 spoke forty; PyTorch speaks hundreds. The
product's binding constraint is now breadth — you cannot even subtract two
tensors. This issue closes that gap **without** lowering quality: the same
review gates, exact-value verification, and never-dies daemon discipline,
applied to ~200+ ops via architecture rather than heroism.

## Analysis

### What "all tensor operations" means (the scope definition)

The documented `torch.*` operations that take and return tensors — PyTorch's own
taxonomy, roughly 200–250 ops:

| Category              | ~Count | Examples                                                          |
| --------------------- | ------ | ----------------------------------------------------------------- |
| Pointwise math        | ~100   | add, sub, mul, div, sin, exp, log, sqrt, abs, pow, clamp, sigmoid |
| Reductions            | ~30    | sum, mean, max, min, argmax, prod, std, var, norm, all, any       |
| Comparison            | ~20    | eq, ne, lt, gt, le, ge, isnan, allclose, sort, topk               |
| Linear algebra / BLAS | ~30    | mm, matmul, bmm, dot, outer, einsum, linalg.{svd,inv,det,solve}   |
| Shape & indexing      | ~40    | reshape, permute, transpose, squeeze, cat, stack, split, gather,  |
|                       |        | index_select, masked_select, where, narrow, flip, roll, tril      |
| Creation              | ~20    | zeros, ones, full, rand, randn, randint, arange, linspace, eye    |

Plus `manual_seed` (random ops need it for deterministic verification).

Notable in-scope subtleties:

- **Comparison ops return Bool tensors** — serialization gains a bool path.
- **Multi-return ops** (`sort` → values+indices, `svd` → U/S/V) — the protocol
  gains multi-handle responses.
- **Slicing** has no CLI bracket syntax; the functional forms (`narrow`,
  `select`, `index_select`, `slice`) are the equivalent — stated explicitly so
  nobody expects `t[1:3]`.

### Exclusions (beyond mere tensors)

- **`torch.nn`** (layers, Conv2d, attention, loss modules) — compositions of
  tensor ops with learned state; module-land.
- **`torch.optim`** (Adam, schedulers) — optimizer state machines.
- **DataLoader/datasets, `torch.distributed`, `torch.compile`/JIT, torch.hub** —
  infrastructure.
- **Autograd** (`backward`, `grad`, `requires_grad`) — deferred to its own
  issue; it is a separate system with real daemon-world design questions.
- **save/load** — its own issue.
- **Sparse, quantized, complex dtypes** — deferred (MPS support partial).
- **In-place variants** (`add_`) and **`out=` variants** — excluded permanently,
  not deferred: every op output is a new handle; that is the handle model (v1
  reached the same conclusion).

### Architecture: a declarative op table, not 250 hand-written arms

The PoC pattern (protocol variant + dispatch arm + client parser per op) does
not scale to this issue. Experiment 1 must establish a **single declarative op
table** — one row per op: name, tensor arity (fixed or variadic), scalar and
flag parameters, result kind (handle / handles / value) — from which flow:

- protocol handling (a generic op request: name + tensor slots + params);
- daemon dispatch (table-driven lookup → fallible tch call);
- client argument parsing and usage errors;
- `torch ops` (list by category) and per-op usage/help;
- test scaffolding.

The marginal op then costs one table row, the tch call mapping, and tests.
Whether the table is a Rust const, a macro, or generated source is settled in
Experiment 1.

### Error handling: PyTorch semantics, one-line errors, machine-readable codes

- **Broadcasting is in.** `[2,3] + [3]` broadcasts (that is PyTorch's semantics,
  not a convenience); `[2,3] + [4]` errors. This **amends carried-forward
  principle 4** ("no automatic broadcasting surprises") the way issue 0003
  amended its device clause: a recorded retirement — broadcasting is the
  documented semantics of the API we pledge fidelity to, and an `add` that
  disagrees with every PyTorch doc is the real surprise.
- Every error: **one line**, naming the **op**, the **offending
  shapes/dtypes/values**, and **what was expected** — e.g.
  `add: shapes [2,3] and [4] are not broadcastable`. Rust-side pre-validation
  wherever it beats tch's message; fallible `f_*` everywhere (the daemon never
  dies); client exit 1 with the message on stderr.
- The protocol error response gains a machine-readable **`code`** field
  (`unknown_handle`, `shape_mismatch`, `bad_dtype`, `bad_argument`, …) alongside
  the prose, so scripts can branch without parsing sentences.
- dtype edges handled deliberately: float64 errors **cleanly** at the MPS
  boundary (no leaked tch internals); integer division semantics follow PyTorch
  (`div` vs `floor_divide`).

### Argument grammar: stdin-prefix, positional-suffix, mixable

An op's tensor parameters are ordered slots:
`torch op [t1] [t2] … [scalars]
[--flags]`.

- **stdin fills the leftmost missing tensor slots, one handle per line**:

  ```bash
  torch add $a $b                            # all positional
  torch tensor '[1]' | torch add $b          # stdin → slot 1
  printf "$a\n$b\n" | torch add              # two-line stdin → both slots
  printf "$a\n$b\n$c\n" | torch cat --dim 0  # variadic: N lines
  ```

- **Mix and match: yes** — stdin is always the prefix, positionals the suffix;
  one deterministic rule, no guessing.
- **If no slots are missing, stdin is never read.** This **amends
  carried-forward principle 2's "XOR enforcement" clause** (a recorded
  retirement, with reasoning): v1 errored on pipeline+args conflicts, but a
  POSIX CLI that reads stdin to detect a "conflict" blocks on terminals, steals
  input from enclosing `while read` loops, and behaves differently inside
  pipelines. Positionals win; stdin untouched; documented. (v1's behavior, for
  the record: pipeline value = first operand, XOR enforced per command via its
  Some/Some "Conflicting input" check.)
- Scalars and tensors never compete: the op table types each position (handle
  slot vs number/JSON slot) — no heuristic sniffing.

### Fidelity verification: golden tests generated from real PyTorch

`.venv-torch` contains the exact PyTorch this daemon links. A generator script
runs each op on fixed inputs **in Python torch** and emits
inputs/expected-outputs (and expected _errors_) as JSON; the test suite replays
them through the daemon and asserts byte-faithful agreement. Hundreds of ops
verified mechanically against the same libtorch — including the error cases — is
what keeps "high quality" true at op #200. The generator and harness are part of
Experiment 1.

### Discoverability

With 200+ ops, help is a feature, not a nicety — generated from the op table:
`torch ops` (list, by category), `torch <op> --help` (usage, slots, flags), and
a README ops section that points at `torch ops` rather than duplicating it.

### Experiment shape (settled as they come, per the workflow)

Experiment 1 carries the architecture: the op table, the generic protocol op,
the grammar (incl. multi-line stdin and the principle-2 amendment), the error
contract (incl. broadcasting and the principle-4 amendment), the golden-test
generator, and ~10 representative ops — one of each shape: binary broadcasting,
unary, variadic, multi-return, scalar-param, random (+seed), comparison (bool
result), reduction with `--dim`. Subsequent experiments are category sweeps
riding the table, refining the pattern between batches. Never listed upfront
beyond that — each batch's result informs the next.

### Out of scope

Autograd, save/load, sparse/quantized/complex dtypes (all above); concurrency
(separately motivated and deferred); the Nushell premium client; protocol
redesign (the NDJSON protocol remains throwaway — the generic op request lives
inside it).

## Experiments

- [Experiment 1: The op table — architecture, grammar, errors, golden tests, and 15 representative ops](01-op-table-architecture.md)
  — **Pass** (the loom works: 29/29 goldens vs real PyTorch; found the MPS RNG
  gap and a serde float-precision bug in the first hour)
- [Experiment 2: The first pointwise sweep (~55 ops)](02-pointwise-sweep.md) —
  **Pass** (57 ops landed, zero structural changes; goldens 29→90; one MPS
  exclusion recorded: heaviside)
