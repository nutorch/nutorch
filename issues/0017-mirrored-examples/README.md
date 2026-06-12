+++
status = "open"
opened = "2026-06-12"
+++

# Issue 17: Mirrored examples, everywhere — including the reference

## Goal

Two related example-quality problems, fixed together:

1. **Every paired example mirrors line-for-line** (user decision, overriding the
   each-shell-at-its-most-idiomatic choice): the bash and Nushell panels show
   the SAME structure — same number of lines, same forms (argument form mirrors
   argument form, pipe mirrors pipe), same comments — so the toggle reads as a
   translation, not two different programs.
2. **The reference pages join the site**: their per-op example lines are
   currently plain unhighlighted fences with no shell toggle — they become
   syntax-highlighted bash/nu PAIRS, rendered as tab groups by the existing
   issue-0015 plugin, on all nine generated pages.

## Background

The user, reading the hero pair, asked why the bash side creates tensors by
argument (`torch tensor '[1,2,3]'`) while the nu side pipes
(`[1 2 3] | nutorch tensor`). The answer was idiom, not capability — the CLI
fully supports stdin for data AND handles (verified live:
`echo '[1,2,3]' | torch tensor` works; the nu module's own wrappers depend on
it). The user's call: idiom loses, mirroring wins.

The reference pages (issue 0012 experiment 3) emit each op's `usage:` line in a
deliberately language-less fence — no highlighting, exempt from the shell tabs
by design. With 185 ops across nine pages, that is most of the site's "examples"
rendered as flat text while everything else got the two-shell treatment.

## Analysis

- **Mirroring exposes two more prelude dual-input gaps** (the issue-0016
  pattern, one layer down): `nutorch tensor` accepts data ONLY from `$in` (the
  CLI takes a JSON argument: `torch tensor '[1,2,3]'`), and `nutorch value`
  accepts its handle ONLY from `$in` (the CLI accepts `torch value $h`). True
  line-for-line mirroring of the canonical examples needs these prelude verbs to
  accept the argument form too — same delegation shape issue 0016 used (forward
  args; pipe `$in` when present; for `tensor`, a positional data value encodes
  exactly as `$in` does today). Without this, "mirroring" would force every bash
  example into pipe style instead of letting each pair pick the clearest shared
  shape.
- **Which shared shape per example** is an editorial decision made per-example
  at design time (e.g. the hero likely keeps capture-then-use: bash
  `a=$(torch tensor '[1,2,3]')` mirrored by nu
  `let a = (nutorch tensor [1 2 3])`); the dual-input sections keep showing BOTH
  forms in BOTH panels (that is their topic).
- **The reference generator** (`gen-ops-reference.ts`) currently emits one plain
  fence per op. It changes to emit a PAIR per op — the usage shape in bash
  (`torch add <t1> <t2> [--alpha <Scalar>]`) and its nu mirror
  (`nutorch add <t1> <t2> [--alpha <number>]`) — which the rehype plugin pairs
  into a tab group automatically (185 new groups; the `check:tabs` count map
  updates from 0 to per-category counts on the nine reference pages).
  Highlighting placeholders under the bash/nu grammars is acceptable visual win;
  the honesty checker's verb scan covers the new fences automatically (op names
  are real verbs).
- **Verification shape**: the canonical mirrored examples run live in both
  shells (the discriminating harness for nu; the CLI directly for bash); the
  prelude dual-input additions get parity entries in
  `scripts/test-dual-input.nu`; the count map becomes the executable form of
  "reference pages have tabs now"; fence-level baseline diff re-proves the
  plugin only wraps.

## Experiments

- [Experiment 1: Prelude dual input — tensor and value learn both hands](01-prelude-dual-input.md)
  — **Pass** (both prelude verbs dual; 14/14 parity incl. the nuon-compared
  non-finite case; the capture-$in-first nu gotcha caught by the harness)
- [Experiment 2: The mirror — every pair, line for line](02-the-mirror.md) —
  **Pass** (15/15 pairs mirrored and gate-enforced by check:mirror; the vacuous
  `where bytes > 1mb` filter caught and fixed in three places)
- [Experiment 3: Reference pairs — 185 ops join the two-shell site](03-reference-pairs.md)
  — **Designed**

## Scope

In: prelude dual input for `tensor` and `value` (generator/prelude Rust +
regenerated module + parity tests); mirrored rewrites of all paired examples on
the landing page and docs pages; the reference generator emitting highlighted
bash/nu pairs; count-map and gate updates; both-mode screenshots.

Out (recorded): concrete runnable per-op examples on reference pages (the pairs
show usage SHAPES; live-verified concrete examples for 185 ops is its own future
effort); changing the CLI; website deployment.
