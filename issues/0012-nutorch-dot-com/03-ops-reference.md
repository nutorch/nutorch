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

# Experiment 3: The ops reference — generated from the table

## Description

The per-op reference, generated from the binaries themselves so it cannot drift:
`torch ops --json` (name, category, summary — 185 ops, 9 categories) plus each
op's real `torch <op> --help` usage line. Hand-writing 185 entries is how docs
rot; this page set regenerates in seconds and a staleness gate keeps it true.

**Decisions, made here:**

1. **Shape: one page per category** (issue design question 2 settled): 9 pages
   under `src/content/docs/reference/` — creation, pointwise, comparison,
   reduction, linalg, shape, loss, autograd, utility, in that logical order. 185
   single-op pages would drown the sidebar; categories match how `torch ops`
   already groups, and every op gets a stable `#anchor` from its h3 heading.
2. **Entry format**: `### <name>` + the one-line summary (from the table) + ONLY
   the `usage:` line from `torch <op> --help` (line 1 — line 2 duplicates the
   table summary; review catch), verbatim, in a plain (no-language) fence —
   usage grammar is not valid bash, so it gets no Shiki lang. The honesty
   checker doesn't see these pages at all (it reads the docs dir
   non-recursively, and the generated pages carry no bash/nu fences anyway).
3. **The generator**: `website/scripts/gen-ops-reference.ts` (bun) runs the real
   binary (private TMPDIR; `--help` is client-side and fast), emits the 9
   markdown files WITH frontmatter (`section: "Reference"`, `order: 20 + i` so
   the section lands after the hand-written pages, `title`, `description` with
   the op count), and stops the daemon it may have spawned. The sidebar picks
   the section up by construction — that was Experiment 2's point.
4. **Generated output is committed AND gated for staleness** (issue design
   question 2's second half settled): `gen-ops-reference.ts --check` regenerates
   to a temp dir and byte-compares against the committed files, exiting non-zero
   on drift — same philosophy as the `nutorch.nu` staleness test. A
   `check:ops-ref` package script wires it; verification runs it. The
   byte-compare invariant (review catch): the generator's raw output must be a
   dprint FIXED POINT — gate 6 proves it with `dprint check`, and if dprint ever
   disagrees the generator is the bug, never the committed files; the staleness
   gate then can't be tripped by a formatter delta.
5. **dprint-stable by construction**: the generator emits lines that dprint
   would not change (short prose lines, fences untouched), PROVEN by the gate
   `dprint check` on the generated files — if dprint ever disagrees, the
   generator is fixed, not the output hand-patched.
6. **`ops.md` placeholder retired**: the "coming in the next stage" sentence
   from Experiment 2 is replaced with real links to the category pages.
7. **Determinism**: ops come out of the table in its own stable order; the
   generator does not sort, timestamp, or randomize anything, so regeneration is
   byte-identical run to run (this is what makes the staleness gate meaningful).

## Changes

1. **`website/scripts/gen-ops-reference.ts`** (NEW): generate + `--check` modes.
2. **`website/src/content/docs/reference/*.md`** (NEW, 9 files, GENERATED —
   committed).
3. **`website/src/content/docs/ops.md`**: placeholder paragraph → links to the
   reference pages.
4. **`website/package.json`**: `gen:ops-ref` and `check:ops-ref` scripts.
5. **No Rust changes; `v1/` untouched.**

## Verification

1. **Build gate**: `bun run build` exits 0; 9 new `/docs/reference/*` routes in
   `dist/`; the sidebar shows the Reference section after Install, with all 9
   categories, on every docs page (collection-driven — asserted in built HTML).
2. **Coverage**: the 9 pages together contain exactly 185 `###` op headings, and
   the set of heading names equals the set of names in `torch ops --json`
   (asserted by script, not eyeball).
3. **Staleness gate**: `bun run check:ops-ref` green on the committed files;
   then adversarially: corrupt a temp COPY and confirm the checker fails (a gate
   that cannot fail is no gate).
4. **Honesty by construction + spot-check**: usage lines are captured from the
   real binary, but spot-check 5 ops across different categories by running
   `torch <op> --help` manually and comparing against the page.
5. **Both modes, by screenshot**: one reference page (pointwise — the longest)
   captured light and dark; legible, anchors work, on-brand.
6. **Hygiene**: dprint check clean on ALL generated markdown + touched files;
   `check:content` still green; frozen-lockfile green; `v1/` and the Rust tree
   untouched.

**Pass** = all six. **Fail** = heading set ≠ op set, staleness gate inert, or
generated pages need hand-editing to satisfy dprint.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only +
harmless live runs). **Verdict: APPROVED (first pass).** The reviewer verified
every load-bearing premise against the live binary and installed site code: 185
ops / 9 categories, byte-identical `torch ops --json` across runs (determinism);
`--help` exits 0 for ALL 185 ops, always starts with a `usage:` line, and spawns
no daemon; Astro auto-ids on headings (`### add` → `#add`) with all op names
anchor-safe and globally unique; nested collection ids (`reference/pointwise`)
route correctly by construction; `order: 20+i` lands Reference last; plain
fences are ignored by the honesty checker and left untouched by dprint even past
80 columns (while prose IS rewrapped — the generator must pre-wrap); the 21
summaries containing markdown-special characters render literally with no
escaping hazard. One Optional folded: the staleness byte-compare is only valid
if generator output is a dprint fixed point — now pinned as the invariant
(dprint disagreement = generator bug). Two Nits folded: capture only the
`usage:` line (line 2 of `--help` duplicates the summary), and the honesty
checker's non-recursive read is the real reason the generated pages escape it.
