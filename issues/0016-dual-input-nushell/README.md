+++
status = "closed"
opened = "2026-06-12"
closed = "2026-06-12"
+++

# Issue 16: The Dual Input Pattern reaches Nushell

## Goal

The generated Nushell module honors carried-forward principle #2 — the Dual
Input Pattern — exactly as the CLI does: every tensor operation accepts its
leading tensor(s) from the pipeline OR as arguments, with the same slot-filling
rule. `$a | nutorch add $b` and `nutorch add $a $b` both work, and the website's
dual-input documentation shows both forms for both shells.

## Background

The user caught the inconsistency reading the issue-0015 docs: the bash panel
shows two forms (argument and pipeline), the nu panel shows one. That is not a
documentation gap — the issue-0015 review forced the prose to be honest about a
real limitation: the issue-0010 module generator emits pipeline-first wrappers
whose leading tensor comes ONLY from `$in`
(`export def "nutorch add" [t2: string, …]: string -> string`). With the module
loaded, `nutorch add $a $b` parses `$a` as `t2` and errors.

This contradicts the project's own contract. AGENTS.md, Carried-Forward
Principles, #2: "Every operation supports both pipeline form
(`$t1 | torch add $t2`) and argument form (`torch add $t1 $t2`). This is not
optional." The CLI implements it via the stdin-prefix grammar — stdin fills the
leftmost MISSING tensor slots, and is never read when nothing is missing. The
module quietly holds Nushell — the self-described premium client — to a weaker
contract than bash.

## Analysis

- **The change lives in the module GENERATOR** (`torch nu-module`, in
  torch-cli's Rust source), not in hand-edits to `nutorch.nu` (generated,
  staleness-tested). Wrappers for ops with leading tensor parameters gain
  optional positionals with the CLI's slot-shifting rule:
  - `$in` present → `$in` fills the leftmost tensor slot; positionals shift
    right (first positional is the SECOND tensor).
  - no `$in` → positionals fill slots left to right.
  - `$in` present with NO slots missing → the pipe is SILENTLY IGNORED,
    mirroring the CLI exactly (review correction: the stdin-prefix grammar "is
    never read when nothing is missing" — the retired XOR clause exists
    precisely because conflict-detection reads block on terminals). The arity
    error that DOES exist is too many positionals, with or without a pipe.
- **Type signatures loosen where needed**: optional leading positionals mean
  `[t1?: string, t2?: string, --alpha: number]` shapes and an input type of
  `any` (string handle or nothing) — the generator owns the disambiguation logic
  in the wrapper body. Multi-tensor ops (e.g. `cat`/`stack` taking lists,
  `where_` taking three) and zero-tensor ops (creation) are emitted per their
  existing table metadata, which already knows each op's tensor arity.
- **Blast radius**, all mechanical and gated: regenerate `nutorch.nu`
  (committed; the torch-cli `include_str!` staleness test must keep passing, so
  the generator change and the regenerated module land together); the Rust
  suite; the website docs — the dual-input sections in getting-started and
  tensors get their argument form back in the nu panel and the bent prose
  straightens; the issue-0015 twins still hold (they use pipeline form, which
  remains valid); `check:content` and `check:tabs` re-run.
- **Verification shape**: golden parity in nu — for a sample op set, both forms
  produce identical values; the arity-error path errors clearly; the full nu
  training script still passes; the website gates stay green.

## Experiments

- [Experiment 1: Generator delegation — one grammar, two shells](01-generator-delegation.md)
  — **Pass** (173 wrappers regenerated incl. the prelude's forward; 11/11 parity
  checks; CLI arity errors surface through the module; variadic and creation
  wrappers byte-untouched)
- [Experiment 2: The docs straighten — both forms, both shells](02-the-docs-straighten.md)
  — **Pass** (all five pipeline-first hedges retired; both dual-input nu panels
  show both forms, verified live; count map unchanged)

## Conclusion

**The goal is met.** `nutorch add $a $b` and `$a | nutorch add $b` are equals —
carried-forward principle #2 now holds in both shells, implemented ONCE: the
wrappers forward positionals verbatim and pipe `$in` when present, and the CLI's
stdin-prefix grammar fills the leftmost missing slots and owns all validation.
173 wrappers regenerated (172 delegated table ops plus the prelude's `forward`,
which the design review caught as its own violation); variadic and creation
wrappers untouched; the 11/11 parity harness (`scripts/test-dual-input.nu`) is
now a standing acceptance script. The docs straightened: both dual-input panels
show both forms, verified live, and every "pipeline-first" hedge — five in the
docs plus the module banner the result reviewer caught — is gone.

Two contract subtleties the reviews pinned down for the record: an unneeded pipe
is SILENTLY IGNORED (the grammar never reads stdin when no slots are missing —
the retired XOR clause exists because conflict-detection reads block on
terminals), and the CLI's under-supply message is context-dependent (TTY:
"missing tensor operand(s)…"; non-TTY: "expected N piped handle(s), got 0"). The
published keg module picks all of this up at the next release, as usual.

## Scope

In: the `nu-module` generator change (Rust, torch-cli); regenerated
`nutorch.nu`; module-level tests of both forms (the discriminating
explicit-`use` harness); website docs updates (dual-input sections, any prose
that says "pipeline-first by design"); the existing gates re-run.

Out (recorded): changing the CLI's own grammar (already correct); renaming
module commands; the homepage hero/demo content (already pipeline-form, stays
valid); website deployment.
