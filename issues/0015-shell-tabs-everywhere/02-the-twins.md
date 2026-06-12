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

# Experiment 2: The twins — Nushell pairs for the remaining examples

## Description

Pure content on top of Experiment 1's mechanism: every remaining applicable bash
example gets its Nushell twin, judged fence by fence against the exemption rules
and verified live before display. After this, the issue's goal is met and it
closes.

**The inventory, judged at design time** (module coverage verified: `nutorch nn`
and `nutorch daemon` are `--wrapped` passthroughs; `nutorch
add` carries
`--alpha`; `nutorch ops`/`tensors`/`free` return native data):

| Page                | Fence                             | Verdict                                                             |
| ------------------- | --------------------------------- | ------------------------------------------------------------------- |
| getting-started     | install block                     | EXEMPT (brew, shell-agnostic)                                       |
| getting-started     | dual-input forms                  | PAIR (nu idiom: pipeline-first `$a \| nutorch add $b`)              |
| daemon              | control block (status/ttl/stop/…) | PAIR (`nutorch daemon …` passthrough)                               |
| daemon              | status report (plain fence)       | EXEMPT (not bash — output, not commands)                            |
| tensors             | dual input pattern                | PAIR                                                                |
| tensors             | creating tensors                  | PAIR (`nutorch full`/`randn`/`arange` wrappers)                     |
| tensors             | export/import (`value --meta >`)  | EXEMPT (the bash form IS the example; the module's `value` returns  |
|                     |                                   | native data and has no `--meta` — recorded per the spine)           |
| tensors             | census/free block                 | PAIR                                                                |
| ops                 | `add --alpha 2`                   | PAIR                                                                |
| ops                 | discoverability (`ops`, `--help`) | PAIR (`nutorch ops` is a native TABLE — the nu twin shows the       |
|                     |                                   | structured win; `torch mean --help` stays as the external call)     |
| autograd            | main block                        | PAIR (forms proven in exp 1's second twin + `zero_grad`/`detach`)   |
| autograd            | losses one-liner                  | PAIR                                                                |
| neural-networks     | building modules                  | PAIR (`nutorch nn …` passthrough; handles are raw strings)          |
| neural-networks     | training loop                     | PAIR (the proven train-regression.nu forms)                         |
| neural-networks     | save/load                         | PAIR (`nutorch nn save/load` passthrough)                           |
| nushell page        | everything                        | EXEMPT (nu-only by nature; the `--json` block is deliberately bash) |
| install-from-source | both blocks                       | EXEMPT (build commands / CLI install verification)                  |
| reference pages     | usage fences                      | EXEMPT (generated CLI usage, plain fences)                          |

Exactly TWELVE new pairs across six pages (review count — the reviewer walked
every fence and matched the table).

**Decisions, made here:**

1. **Every twin runs live before display** via the discriminating explicit-`use`
   form, exactly as Experiment 1 did; where the twin shows output, the run must
   produce it (seeded via `manual_seed` where determinism is needed; outputs
   omitted where the bash twin omits them).
2. **Idiom over transliteration — under a SAME-OBSERVABLE-EFFECT constraint**
   (review catch, a constraint not a suggestion): any twin that displays output
   must produce the IDENTICAL displayed value (seeded — e.g. the training loop's
   `2.46e-7` final loss), and where the forms genuinely diverge (the dual-input
   fences exist to show argument AND pipeline forms, but the module is
   pipeline-first by design), the nu panel shows the idiom only and the
   surrounding prose is adjusted so it stays accurate for BOTH panels (no "both
   of these work" sentence hovering over a panel that shows one form). `print`
   surfaces mid-script values (the exp-1 result-review lesson).
3. **`check:tabs` learns the new counts**: the gate's getting-started assertion
   updates from "two groups" to the new exact count, and gains a page-sweep
   assertion — every docs page's group count equals the inventory above, and
   every page NOT in the inventory (reference subpages, the docs index, the 404)
   asserts ZERO groups (review nit — the count map must be complete to be an
   executable exemption table).
4. **The fence-level baseline diff re-runs** (both builds cache-clean via the
   now-self-cleaning build script, one with `SHELL_TABS_DISABLE=1`) — all pages'
   `<pre>` lists identical between disabled and enabled builds.
5. **The issue closes after this experiment** (the conclusion summarizes both).

## Changes

1. **`website/src/content/docs/{getting-started,daemon,tensors,ops,autograd,neural-networks}.md`**:
   the nu twins per the inventory.
2. **`website/scripts/check-shell-tabs.ts`**: updated counts + the page-sweep
   assertion.
3. **Nothing else** — no plugin/script/CSS changes (the mechanism is frozen), no
   Rust, no `v1/`.

## Verification

1. **Every twin reproduces live** (the stop-everything gate), with seeded
   determinism where output is displayed.
2. **Build + structure**: group counts per page match the inventory exactly (the
   page-sweep assertion in `check:tabs`); exempt pages gain zero wrappers.
3. **The fence-level baseline diff**: identical `<pre>` lists across all pages,
   disabled vs enabled builds.
4. **All gates**: `check:tabs` (updated), `check:content` (all new nu fences
   scanned), `check:links`, `check:ops-ref`, `check:theme`, brand gate; dprint
   clean; zero `.rs` diffs.
5. **By eye**: one content-heavy page (tensors) screenshotted in both shells ×
   both modes.

**Pass** = all five. **Fail** = any twin the module rejects, any exempt fence
gaining a wrapper, or any group count off the inventory.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**Verdict: APPROVED (first pass).** The reviewer exhaustively enumerated every
fence on every docs page and confirmed the inventory COMPLETE — twelve PAIRs,
six exemption classes, no missed fence, no wrong verdict — and every PAIR
feasible against the actual module (passthroughs, `add --alpha`, native-table
registry verbs, `mse_loss`/`zero_grad`/`detach`, and the `full`/`randn`/`arange`
signatures all verified by line number in nutorch.nu). The exemptions held up,
including `value --meta` (the module's `value` genuinely has no such flag) and
the unlabeled daemon-status fence. Two Optionals folded: the prose summary
corrected to twelve pairs / six pages, and decision 2 hardened into a
same-observable-effect CONSTRAINT (seeded identical displayed values; where
forms diverge, the nu panel shows the idiom only and the prose is adjusted to
stay accurate over both panels — the dual-input fences and the training loop's
`2.46e-7` named explicitly). Nit folded: the page-sweep count map asserts zero
for every non-inventoried page.
