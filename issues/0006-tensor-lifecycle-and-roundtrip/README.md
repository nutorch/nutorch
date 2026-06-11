+++
status = "open"
opened = "2026-06-11"
+++

# Issue 6: Tensor lifecycle (free + list) and lossless JSON round-trip

## Goal

Make tensor memory manageable and tensor data durable using shell-native
mechanisms: explicit `torch free`, a `torch tensors` listing, and a JSON
import/export path that round-trips every tensor the daemon can hold — bool
dtypes, non-finite values, and dtype metadata included. No save/load feature; no
garbage collector.

## Background

Issue 0004 established the memory-horizon contract: tensors live exactly as long
as the daemon. Issue 0005 filled the registry with 172 ops' worth of ways to
create tensors — and zero ways to remove one. Every op output accumulates until
the daemon dies. `torch daemon status` shows `tensors`/`approx_bytes` climbing;
nothing reclaims them. A long working session grows GPU memory without bound,
and the only relief valve is killing the daemon and losing everything.

**Why there is no garbage collector.** A GC traces reachability from roots. Our
handles are opaque strings living in shell variables, scripts, files, and
command history — invisible to the daemon by construction. A handle untouched
for an hour may be `$WEIGHTS` in a script that runs tomorrow. Reachability-based
collection is impossible; anything automatic would be policy-based (age, LRU,
memory pressure), and policy-based _destruction_ silently invalidates handles a
user still holds, converting a memory problem into a correctness problem.
Rejected.

**Why there is no save/load.** Import/export already exists — `torch
tensor`
(JSON in) and `torch value` (JSON out) — so persistence is shell redirection:
`torch value $w > w.json`, `w=$(torch tensor "$(cat
w.json)")`. It is naturally
_selective_: the user saves the two tensors they care about, never the
intermediates a daemon-side "save all" would dump. Restarting the daemon then
becomes the documented coarse-grained reclaim valve (the Jupyter restart-kernel
workflow) instead of data loss.

**Why the round-trip path needs fixing first.** Three recorded gaps keep
import/export from carrying that persistence weight today:

1. **Non-finite values export as `null`.** JSON has no NaN/Infinity; serde maps
   them to `null`, so a tensor containing NaN round-trips corrupted. (This is
   the same wall behind the golden suite's finite-inputs constraint, issue 0005
   experiment 2.)
2. **Output carries no dtype.** `torch value` prints bare data; re-import of an
   int64 tensor silently yields float32 unless the user remembers
   `--dtype int64`.
3. **No bool input path.** Comparison results (`eq`, `isnan`, …) print as
   `true`/`false` but `json_to_tensor` cannot read booleans back — bool tensors
   do not round-trip at all. The same gap forced the `!= 0` cast nutorch-ism on
   `where`/`masked_select` (issue 0005 experiment 4).

## Proposed Solution

Four strands, sized for one experiment each (final shape decided per experiment,
as always):

1. **`torch free`** — `torch free $t1 $t2 …`, the stdin form (`… | torch free`
   ends a pipeline by reclaiming it), and `torch free --all`. Frees are
   idempotent-friendly: freeing an unknown handle is an error
   (`unknown_handle`), freeing twice is therefore visible, and `--all` empties
   the registry. Wire: bespoke or table op — decided in the experiment (it is
   registry-mutating, like nothing in the op table).
2. **`torch tensors`** — list what the registry holds: handle, shape, dtype,
   approximate bytes, age, seconds since last touch. Plain text columns
   (shell-friendly), one tensor per line, handle first so
   `torch tensors | awk '{print $1}' | torch free` composes. The daemon `status`
   op stays summary-level; this is the detail view.
3. **Lossless round-trip** — the three fixes:
   - bool input path in `json_to_tensor` (accept `true`/`false`, infer
     `Kind::Bool`; retire the `where`/`masked_select` cast note or keep the cast
     for numeric convenience — decided in the experiment);
   - a non-finite policy for `tensor_to_json`: emit `"NaN"`, `"Infinity"`,
     `"-Infinity"` string tokens and accept them on input (lossless, documented
     dialect), or error loudly on non-finite export — decided in the experiment,
     with golden-style tests either way;
   - a dtype envelope: `torch value --meta` emits
     `{"dtype": "...", "shape": [...], "data": [...]}` and `torch tensor`
     accepts that envelope as input, so a round-trip can carry its own dtype
     without the user remembering `--dtype`.
4. **Document the relief valve** — README workflow section: export what you
   keep, `torch daemon restart` to reclaim everything, re-import. This is the
   supported coarse-grained reclaim path, stated plainly.

Out of scope, recorded for later if ever needed: memory budgets with LRU
spill-to-disk (the only sound _automatic_ mechanism — eviction, not destruction
— and its on-disk format would be an internal detail); opt-in per-tensor TTLs
(cheap atop the issue-0004 lease machinery, but opt-in only); binary transfer
formats for very large tensors.

## Carried Constraints

- Handles never die silently: nothing but `free`, `free --all`, or daemon exit
  removes a tensor.
- The dual input pattern and stdin-prefix grammar apply to `free` like any other
  handle-consuming command.
- PyTorch fidelity is not at stake here (these are nutorch-native verbs), but
  the JSON dialect for non-finite values must be documented wherever `value`
  output is described.
- Standard hygiene gates (build clean, fmt, dprint on touched files, tests
  green, `v1/` untouched) and the AI review gates apply to every experiment.
