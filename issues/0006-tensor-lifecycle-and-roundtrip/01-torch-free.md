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

# Experiment 1: `torch free` — the first verb that removes

## Description

Give the registry its first reclaim path: `torch free` destroys tensors by
handle. Three forms, per the dual input pattern:

```bash
torch free $t1 $t2            # positional handles, variadic (>= 1)
$t | torch free               # stdin form; ends a pipeline by reclaiming it
some_pipeline | torch free    # any number of handles, one per line
torch free --all              # empty the registry
```

**Wire shape: bespoke, not table.** Table ops resolve handles to borrowed
`&Tensor` refs and insert _new_ results; `free` needs `&mut` registry access to
_remove_ entries — a different contract from every `apply()` arm. A bespoke op
keeps the table's invariants intact: `{"op":"free","handles":["…"]}` or
`{"op":"free","all":true}`, response `{"ok":true,"value":{"freed":N}}`.

**Semantics, decided here:**

- **Atomic**: all requested handles are validated first; if any is unknown,
  nothing is freed and the response is `unknown_handle` naming the first missing
  one. (Freeing twice therefore errors visibly, per the issue.)
- **`--all` and positional/stdin handles are mutually exclusive** — mixing them
  is a usage error in the client.
- **No handles, no stdin, no `--all`** → usage error (a bare `torch free` at a
  terminal must not hang; same `IsTerminal` rule as the grammar).
- **stdout prints nothing on success** (the `rm` convention; exit 0). The
  daemon's `freed` count exists on the wire for tooling, but the CLI stays
  quiet.
- **Frees touch the idle lease** (they are tensor work; a long manual cleanup
  session should not race the daemon's TTL).

## Changes

1. **`nutorchd/src/registry.rs`**:
   `remove(&mut self, handle) ->
   Option<Tensor>` and
   `clear(&mut self) -> usize`.
2. **`nutorchd/src/protocol.rs`**:
   `Bespoke::Free { handles:
   Option<Vec<String>>, all: Option<bool> }`
   (serde-friendly; exactly one must be present — validated in dispatch, with
   `bad_request` otherwise).
3. **`nutorchd/src/dispatch.rs`**: the `free` arm — validate-then-remove
   (atomic; ALL handles checked before ANY removal), `--all` via `clear()`;
   `"free"` joins the bespoke-name list in `parse_request`.
   `all:
   Some(false)` is treated as "not requested" (so `{"all": false}`
   alone is the missing-input `bad_request`, and
   `{"all": false, "handles": […]}` frees the handles). Unit tests: free one,
   free several, **free `[known_a, unknown, known_b]` errors AND both known
   handles survive** (a remove-as-you-go bug fails this, not just a count
   check), double free errors, `free all` empties, both-present and
   neither-present shapes are `bad_request`.
4. **`torch-cli/src/main.rs`**: `free` joins the bespoke builder — variadic
   positionals; stdin lines when not a TTY and no positionals; `--all` flag; the
   mutual-exclusion and empty-input usage errors; on success print nothing.
   **`parse_raw` change required** (design-review finding): with `spec = None`
   the current code treats EVERY `--flag` as value-taking, so `torch free --all`
   would error ("--all needs a value") and `torch free --all $t` would silently
   swallow `$t` as the flag's value. Fix: a presence-only-flag set for bespoke
   ops, consulted in the `spec.is_none()` branch (`--all` pushes
   `(name, None)`), so mixing `--all` with handles surfaces as the
   mutual-exclusion error, never as a swallowed argument.
5. **No goldens** — `free` has no PyTorch counterpart; Rust unit tests are the
   verification layer (the golden suite is untouched).

## Verification

1. **Hygiene**: build 0 warnings; fmt clean; dprint clean on touched files; all
   existing tests still green (207 goldens untouched); `v1/` untouched.
2. **Unit tests** (dispatch): the six cases in Changes item 3.
3. **Live**:
   - create 3 tensors → `torch daemon status` shows `tensors: 3`;
     `torch free $a` → status shows 2; `$b | torch free` → 1; `torch free --all`
     → 0;
   - `torch free $a` again → `unknown handle` on stderr, exit 1;
   - `torch free --all` with no trailing token succeeds;
     `torch free --all
     $t` is rejected as mutual exclusion (not a swallowed
     argument);
   - `torch free $known $unknown` → error, AND the known handle still works
     afterward (atomicity, observed end-to-end);
   - `torch add $a $b | torch free` — a pipeline that reclaims its own result;
   - the registry accounting drops by the freed tensor's size: create a large
     tensor (`randn '[4096,4096]'`, ~67 MB), record `approx_bytes`, free it,
     confirm the delta. (This observes the registry's own accounting —
     `Σ numel × elt_size` — not the MPS allocator, which may cache freed pages;
     allocator-level reclamation is explicitly NOT claimed.)
4. **Lease**: `torch free` resets the idle clock (observable via
   `torch daemon status` idle seconds).

**Pass** = all four. **Fail** = the bespoke shape couldn't express the semantics
without touching the table machinery.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**First pass: CHANGES REQUIRED** — 2 Required: (1) the plan said "--all flag"
without prescribing the `parse_raw` change it needs — with `spec = None` every
flag is value-taking, so `torch free --all` would error and `--all $t` would
silently swallow `$t`; fixed with a presence-only-flag set for bespoke ops plus
live checks for both shapes. (2) The "freed memory is actually reclaimed" live
check overclaimed: `approx_bytes` is registry accounting (`Σ numel × elt_size`),
not allocator observation — MPS may cache freed pages; reworded to claim exactly
what it proves. 1 Optional folded in: the atomicity unit test now pins ordering
with `[known_a, unknown, known_b]` so a remove-as-you-go bug fails it. 1 Nit
folded in: `all: false` semantics defined ("not requested"). The reviewer
confirmed the bespoke-vs-table rationale against execute_table's borrow
contract, the serde tag mapping (`Free` → `"free"` needs no rename), the
Response::Value shape, and the lease-touch convention. Approved with the fixes
in place (the fixes are the reviewer's prescriptions verbatim).

## Result

**Result:** Pass

The registry has its first reclaim path, in exactly the bespoke shape the design
called for — the op table and its machinery are untouched.

- **Unit tests**: all six designed cases green (38 daemon unit tests total, up
  from 32), including the ordering-pinned atomicity case
  (`[known_a, unknown, known_b]` errors and BOTH known handles survive) and the
  `all: false` semantics (alone → `bad_request`; with handles → frees them).
- **Live**: create 3 → free by arg → 2 → free by stdin → 1 → `--all` → 0;
  success prints nothing (exit 0); double free → `unknown handle`, exit 1; the
  atomic mix errors AND the known handle still reads back `[4.0]`; `--all $t` is
  the mutual-exclusion error (not a swallowed argument — the design-review fix,
  observed); bare `torch free` at EOF-stdin is a usage error, no hang;
  `torch add $e $f | torch free` reclaims a pipeline's own result; registry
  accounting walks 16 → 67,108,880 → 16 bytes around a freed
  `randn '[4096,4096]'` (the exact 64 MiB + the 16-byte survivor); `free` resets
  the idle clock (2s → 0s).
- **Hygiene**: build 0 warnings; fmt clean; 207 goldens untouched and green;
  `v1/` untouched.
- **One implementation addition beyond the design text**: the client suppresses
  `print_response` for `free` (the daemon's `{"freed":N}` would otherwise have
  printed, violating the rm convention). One line, in the declared client site.

## Conclusion

The bespoke shape was right: `free` needed `&mut` registry access nothing in the
table contract offers, and it cost zero table changes. The presence-only-flag
set (`BESPOKE_PRESENCE_FLAGS`) the design review forced into existence is
reusable for any future bespoke flag. Next: `torch
tensors` — the listing that
makes `free` targetable (`torch tensors | awk '{print $1}' | torch free` is the
composition the issue promises).

## Result Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only),
reviewing the pre-commit working tree. **First pass: CHANGES REQUIRED** — one
Required finding, purely workflow: the issue README's index line still said
**Designed** while this file recorded **Pass** (the status update had silently
failed against dprint's line wrapping — the third silent-replace escape this
project has logged; fixed with an asserting replace). Everything substantive was
verified and held: all three design-review mandates confirmed real in code and
live (`--all` alone succeeds, `--all
$t` is mutual-exclusion not a swallowed
argument; the accounting claim correctly scoped to the registry, not the
allocator; the ordering-pinned atomicity test asserts both survivors);
validate-before-remove confirmed at the code level; the full live matrix
reproduced (double free, EOF usage error with no hang, silent success, pipeline
self-reclaim, `{"freed":N}` on the wire via nc, `all:false` both shapes, lease
reset, the byte-exact 67,108,864 accounting walk); hygiene gates all green with
the 207 goldens untouched.
