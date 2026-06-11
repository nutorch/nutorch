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

# Experiment 2: The first pointwise sweep (~55 ops)

## Description

The first category sweep on the Experiment-1 loom: the pointwise math surface.
Each op is one table row, one apply mapping (typically one line), and golden
coverage. The sweep also discharges two small owed items: `--alpha` on
`add`/`sub` (PyTorch fidelity; v1 had `--alpha` on add) and the
`softmax`/`log_softmax` pair (PyTorch makes dim required; v1 defaulted it to the
last dim — we make `--dim` required for PyTorch fidelity; reduction in float32
matches both).

**MPS support is determined empirically by the golden generator**: it runs every
op on MPS in Python first; any op the linked PyTorch cannot run on MPS is
excluded from the table (this is an MPS-only product — shipping an op that
always errors is worse than honest absence) and recorded in the Result.

## Changes

1. **`ops/src/lib.rs`**: compact row constructors (`unary(...)`,
   `binary_broadcasting(...)`) so each new op is one readable line; then the
   rows. The planned list (subject to the MPS-support oracle):
   - **unary (~42)**:
     `abs acos acosh asin asinh atan atanh ceil cos cosh
     deg2rad erf erfc exp exp2 expm1 floor frac log log10 log1p log2 logit
     neg rad2deg reciprocal relu round rsqrt sgn sigmoid sign sinc sinh sqrt
     square tan tanh trunc`
     plus `softmax` / `log_softmax` (required `--dim`; float32 reduction) and
     `nan_to_num` (optional `--nan/--posinf/--neginf` Float flags).
   - **binary, broadcasting (~13)**:
     `mul div maximum minimum atan2 fmod remainder floor_divide hypot
     copysign xlogy logaddexp heaviside`.
     `pow` gains nothing (stays scalar-exponent; tensor-exponent pow is recorded
     as a later spec point alongside clamp's tensor bounds). `positive` is
     skipped (identity). `digamma lgamma i0` ride the oracle: included if MPS
     supports them, excluded-and-recorded otherwise — same as everything else.
   - **`add`/`sub` gain `--alpha`** (Scalar flag, default 1), with the two
     formulas stated separately because the signs differ: **`add`:
     `a + alpha·b`; `sub`: `a − alpha·b`** (PyTorch semantics). Golden cases for
     BOTH `add --alpha` and `sub --alpha` so the harness would catch a sign
     error.
2. **`nutorchd/src/dispatch.rs`**: apply mappings (one line per unary; `--alpha`
   routed via `f_add` / scaled forms; `softmax`/`log_softmax` via their tch
   calls with `Kind::Float`).
3. **`scripts/gen-golden.py`**: a data-driven sweep section — domain-aware
   sample inputs per op (e.g. `acos` on [-1,1], `log` on positives, `acosh` on
   [1,∞)); one golden case per unary, two per binary (broadcast + exact),
   `--alpha` cases, `softmax` cases; regenerated goldens stay dprint-clean and
   byte-stable (the Experiment-1 lesson).
4. **No grammar, protocol, client, or doc changes** — that is the point of the
   loom. (The README op count lives behind `torch ops`, which is generated.)
5. The ops-crate `table_has_fifteen_ops` count test becomes count-agnostic
   (uniqueness + invariants stay; a hardcoded census per sweep would be a stale
   literal every time).

## Verification

1. **Hygiene**: `cargo build` 0 warnings; `cargo test` green;
   `cargo fmt --all -- --check` clean; `dprint check` clean on touched files
   (including the regenerated golden.json); `git status --porcelain v1/` empty.
2. **Golden suite green** with one case minimum per new op (the count grows from
   29 to ~85+; the harness floor assertion is raised accordingly).
3. **Generator regeneration is byte-stable** (run twice, identical file).
4. **Live spot-checks** (a handful, not exhaustive — the goldens are the
   exhaustive layer): `mul`/`div` pipelines; `relu` of a mixed-sign tensor;
   `add --alpha 2`; `softmax --dim 0` sums to ~1; an excluded-op name (if any)
   returns `unknown_op`.
5. **`torch ops` count** equals the new table size + 2 bespoke (programmatic).
6. **MPS-support exclusions recorded**: every op from the planned list that the
   oracle rejected is named in the Result with the Python error line.

**Pass** = all six. **Partial** = sweep lands minus recorded exclusions plus any
op whose golden disagrees (recorded, excluded, follow-up). **Fail** = the loom
required structural surgery (that would mean Experiment 1's architecture claim
was wrong).

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**First pass: CHANGES REQUIRED** — 1 Required (the `--alpha` formula was stated
once for both ops; `sub` is `a − alpha·b`, the opposite sign — exactly the
fidelity class this experiment exists to prevent; fixed with separate formulas
plus mandatory golden cases for both), 2 Optional (softmax's required-`--dim`
was misattributed to v1, which defaults to the last dim — reworded as a
PyTorch-fidelity choice; thirteen unhomed pointwise ops folded in or explicitly
dispositioned: `positive` skipped as identity, `digamma/lgamma/i0` ride the MPS
oracle), 1 Nit (the hardcoded table-count test becomes count-agnostic). **Second
pass: CHANGES REQUIRED** — the README index link still carried the stale pre-fix
title/count; fixed. **Approved** with the index in sync (per-finding
confirmations and the 55-op arithmetic verified by the reviewer).

## Result

**Result:** Pass

The loom held: 57 new ops landed as table rows + one-line apply mappings +
golden cases, with **zero structural changes** to the grammar, protocol, client,
or docs. The table grew 15 → 72 ops (42 sweep unaries + softmax + log_softmax +
nan_to_num + 12 broadcasting binaries; `--alpha` added to add/sub).

- **Golden suite: 90/90** (was 29), regenerated byte-stable (SHA `59be0f65…`)
  and dprint-clean; harness floor raised to 85.
- **Hygiene**: build 0 warnings; 31 daemon unit tests + 3 ops + 3 smoke + the
  golden harness all green; fmt/dprint clean; `v1/` untouched.
- **Live spot checks**: mul/div pipelines exact; relu; `add --alpha 2` →
  `[10,16]` and `sub --alpha 2` → `[2,0]` (the sign semantics the design review
  caught, now golden-guarded both ways); softmax sums to ~1 (f32); `torch ops` =
  74 = 72 table + 2 bespoke (programmatic).
- **MPS oracle exclusions (recorded per the design)**: exactly one —
  **`heaviside`**:
  `The operator 'aten::heaviside.out' is not currently
  implemented for the MPS device`
  (PyTorch 2.11/MPS). Not in the table; `torch heaviside` → `unknown_op`.
  `digamma`, `lgamma`, and `i0` all passed the oracle and shipped. `positive`
  skipped as identity (recorded).
- **One generator find**: Python's `json.dumps` emits bare `NaN`/`Infinity`
  (invalid JSON) — golden inputs must be finite. The `nan_to_num` golden covers
  param plumbing on finite input; the real NaN/inf replacement semantics moved
  to a Rust dispatch unit test that constructs non-finite tensors directly (0/0
  division). A constraint all future sweeps inherit: golden case INPUTS must be
  finite; non-finite semantics get Rust tests.

## Conclusion

The sweep pattern works at production tempo: ~57 ops in one pass, every one
golden-verified against the linked PyTorch, with the oracle catching the one op
MPS cannot run. The marginal-cost claim from Experiment 1 held precisely (row +
apply line + sample). Next sweeps: reductions + comparison (including the
Bool-returning family: gt/lt/ge/le/ne/isnan/isinf/logical_*), then linalg +
shape/indexing, then creation + remainder — same recipe, plus the deferred spec
extensions (tensor-exponent pow, tensor-bound clamp) when their categories come
up.

## Result Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only),
reviewing the pre-commit working tree. **Verdict: APPROVED — no Required
findings; one Optional, folded in** (the Result's recorded golden hash was stale
— taken before the nan_to_num case rename; replaced with the real sha256). The
reviewer independently verified everything: 0-warning builds of both touched
crates (forced rebuild), all tests green, byte-stable golden regeneration,
golden expectations spot-checked against fresh independent torch computations
(xlogy, logaddexp, both alpha cases), **both alpha signs live** ([10,16] /
[2,0]), softmax's required `--dim` failing cleanly rather than hanging, the
heaviside MPS exclusion reproduced verbatim in Python, the non-finite nan_to_num
Rust test passing, `torch ops` = 74 = 72 + 2, and the workflow state (plan
commit contains no result; result commit unmade).
