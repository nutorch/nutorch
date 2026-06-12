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

# Experiment 3: Reference pairs — 185 ops join the two-shell site

## Description

The generated reference pages stop being flat text: each op's language-less
usage fence becomes a highlighted bash/nu PAIR that the issue-0015 rehype plugin
renders as a tab group — 185 new groups across the nine category pages, produced
entirely by the generator. After this the issue closes.

**Decisions, made here:**

1. **Per-op emission changes** in `gen-ops-reference.ts`: the captured `usage:`
   line drops its `usage:` prefix (it is now an example shape, not a help dump)
   and becomes a ```bash fence; the nu twin is the same line with the leading
   `torch` replaced by `nutorch` — a one-token mirror, valid because issue
   0016/0017-exp-1 made the module accept the CLI's exact positional grammar.
   Placeholders (`<t1>`, `<shape>`, `[--alpha <Scalar>]`) stay verbatim in both
   fences; the summary line stays above the pair.
2. **`check:tabs` count map** updates the nine reference pages from 0 to their
   category sizes (creation 14, pointwise 71, comparison 25, reduction 21,
   linalg 17, shape 23, loss 9, autograd 4, utility 1 — the executable form of
   "the reference has tabs now").
3. **`check:mirror` learns to recurse** into the reference subdir — 185
   generated pairs join the gate (all trivially 1 = 1, but the gate then covers
   regressions in the GENERATOR too).
4. **`check-content`'s docs scan also recurses** (it was non-recursive — a
   recorded limitation since issue 0012): the new reference fences' verbs get
   validated against `torch ops --json` like all other fences. This passes ONLY
   after a regex fix the recursion forces into scope (design-review catch): the
   verb extractor's character class `[a-z_-]+` has no digits, so six real ops
   truncate to non-ops (`deg2rad`→`deg`, `expm1`→`expm`, `i0`→`i`,
   `rad2deg`→`rad`, `l1_loss`→`l`, `smooth_l1_loss`→`smooth_l`) and would fail
   the scan. The class widens to `[a-z0-9_-]+` as part of this experiment — in
   BOTH of its occurrences (the docs scan and the `index.astro` scan), so the
   hero scan does not retain the latent vacuous-pass behavior either.
5. **Recursive walks key by path relative to the docs root** (design-review
   catch): `autograd.md` exists at both the top level and under `reference/`, so
   basename keys would collapse two distinct files into one diagnostic / one
   `EXCEPTIONS` namespace in `check-mirror.ts` and `check-content.ts`. Pair ids
   and error messages use `reference/autograd.md`-style relative paths.
6. **Generator invariants hold**: dprint fixed point (bash/nu fences are
   untouched by dprint), byte-stable regeneration, `check:ops-ref` staleness
   intact, orphan detection unchanged.

## Changes

1. **`website/scripts/gen-ops-reference.ts`**: the pair emission.
2. **`website/src/content/docs/reference/*.md`**: regenerated (9 files).
3. **`website/scripts/check-shell-tabs.ts`**: reference count map.
4. **`website/scripts/check-mirror.ts`** + **`check-content.ts`**: recursive
   docs walks keyed by docs-root-relative path; the verb-extraction character
   class gains digits (`[a-z0-9_-]+`) in both occurrences.
5. **Nothing else** — no Rust, no module, no plugin, no `v1/`.

## Verification

1. **Build + structure**: 20 pages; the nine reference pages emit their
   category-count tab groups (count map green); ids unique on the biggest page
   (pointwise, 71 groups).
2. **Coverage preserved**: 185 `###` headings, heading set equals the op table
   (the issue-0012 assertion re-run); every op now has exactly one bash and one
   nu fence.
3. **Gates**: `check:ops-ref` (staleness + byte-stable double regen),
   `check:mirror` (now 200 pairs: 15 editorial + 185 generated), `check:content`
   (recursive), `check:tabs`, `check:links`, `check:theme`, dprint fixed point
   on regenerated files; zero `.rs` diffs.
4. **Spot honesty**: for 3 ops across categories, the bash fence equals the live
   `torch <op> --help` usage line minus the prefix. The systematic guarantee
   that every shown `nutorch <op>` is a real module verb is the recursive
   `check:content` membership scan (after the regex fix); module acceptance of
   the CLI's positional grammar — including the underscore flag spellings
   (`--requires_grad`, `--start_dim`, `--log_target`), which the module source
   declares verbatim — was gated by issue 0016 and experiment 1's parity
   harness.
5. **By eye**: one reference page (pointwise), nu tab, both modes.

**Pass** = all five. **Fail** = any reference page off its category count, any
generated pair unequal, or staleness/fixed-point drift.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**First pass: CHANGES REQUIRED** — 1 Required: the recursive `check:content`
claim ("must pass by construction") was false — the verb extractor's character
class `[a-z_-]+` has no digits, so six real ops (`deg2rad`, `expm1`, `i0`,
`rad2deg`, `l1_loss`, `smooth_l1_loss`) truncate to non-ops and would fail the
scan (folded as the explicit regex widening in decision 4 and Changes item 4).
The reviewer independently confirmed the load-bearing claims: all nine count-map
values match live `torch ops --json` exactly, and the one-token mirror is
genuinely valid (all 185 ops have wrappers, argument form works, and every
underscore flag in a usage line is declared with the same spelling in the module
and accepted by nu 0.113 live). 2 Optionals folded: recursive walks key by
docs-root-relative path because `autograd.md` exists at two levels (decision 5),
and verification step 4 now names the systematic guarantees instead of a
tautological 3-op eyeball. **Second pass (fresh context): APPROVED** — all folds
verified against the actual scripts (the six failing truncations and the five
vacuously-passing ones reproduced; the two-level `autograd.md` collision
confirmed; the existing `EXCEPTIONS` key survives relative-path keying; the
underscore flag declarations located in `nutorch.nu` by line). One Optional
folded: the regex widens in BOTH its occurrences (docs scan and `index.astro`
scan) so the hero scan sheds the same latent vacuous-pass behavior; one Nit
folded: the record cites the module source and the exp-1 harness rather than an
unreproducible reviewer action.
