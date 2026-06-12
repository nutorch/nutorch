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

# Experiment 1: Generator delegation — one grammar, two shells

## Description

The module generator stops re-deriving argument shapes and DELEGATES the Dual
Input Pattern to its single source of truth: the CLI's stdin-prefix grammar.
Wrappers for tensor-taking ops accept positionals as a rest parameter and
forward them verbatim; `$in`, when present, is piped to the CLI exactly as
today. The CLI then fills the leftmost missing tensor slots from stdin — the
same rule, the same validation, the same error messages in both shells.

**Current generator behavior** (`torch-cli/src/main.rs`, `generate_nu_module`):
`Arity::Exactly(n)` emits the FIRST tensor as `$in`-only and tensors 2..n as
required named positionals — which is precisely the asymmetry the issue names.
`Arity::AtLeast(_)` ops (cat, stack) already accept both forms (rest
positionals + piped-list join), and `Exactly(0)` creation ops take no pipe; both
stay as they are.

**Decisions, made here:**

1. **For `Exactly(n ≥ 1)` ops, the wrapper signature becomes
   `[...args: any, --flags…]: any -> <result>`** — tensor handles AND any
   non-tensor positionals (dims, scalars, shapes) travel in one rest parameter,
   forwarded in order. The body converts each rest arg generically: lists →
   `to json -r` (the CLI's compact IntList form), everything else →
   `into string`. Flags keep their typed declarations (the part of the signature
   that genuinely aids completion).
2. **`$in` handling**: `let __in = $in` then branch — null → bare invocation
   (`^torch <op> …args`); non-null → `$__in | ^torch <op> …args`. No slot
   arithmetic in nu AT ALL: the CLI decides which slots stdin fills and errors
   with its existing messages — under-supply: "missing tensor operand(s) — pass
   handles as arguments or pipe them in"; over-supply (too many POSITIONALS):
   "too many arguments". A pipe alongside fully supplied positionals is SILENTLY
   IGNORED, by contract (review correction — the grammar never reads stdin when
   nothing is missing; the retired XOR clause exists because conflict-detection
   reads block on terminals). One grammar, owned in one place — if the CLI's
   grammar ever changes, the module follows for free.
3. **Lost named positionals are compensated in the generated comment**: each
   wrapper's `#` line gains the op's CLI usage shape (the same `usage:` string
   the reference pages show), so `which`/source readers and the docs still see
   the arity. (Typed per-slot signatures under optional shifting would make the
   names LIE when `$in` shifts everything — a rest parameter is the honest
   signature.)
4. **The generator change and the regenerated `nutorch.nu` land together** (the
   `include_str!` staleness test forces this), plus the brew keg copy is NOT
   touched (next release, as always). The autoloaded module on this machine is
   the keg's — module-level verification uses the REPO module via the
   explicit-`use` harness, as established.
5. **A committed parity harness**: `scripts/test-dual-input.nu` — for a sample
   spanning the shapes (`add` w/ flag, `mm`, `mse_loss`, `zero_grad`
   single-tensor, `gather` tensor+tensor with `--dim` (review correction — dim
   is a FLAG in the table, not a positional), `reshape` tensor+list, `cat`
   variadic both ways), assert pipeline form and argument form produce IDENTICAL
   values — for `zero_grad`, whose result is `nothing`, parity is asserted on
   the post-call `grad` read (review nit) — and assert the REAL arity errors:
   under-supply ("missing tensor operand(s) …") and too many positionals without
   a pipe ("too many arguments"). Runs in verification and stays as a repo
   acceptance script beside `train-regression.nu`.
6. **The PRELUDE's `nutorch forward` joins the fix** (review catch — it is
   pipeline-only for the tensor while the CLI's `forward` is dual; the issue's
   goal says every tensor operation, and `forward` lives in `NU_PRELUDE`, not
   the generated table). It adopts the same delegation shape: rest args
   forwarded, `$in` piped when present. `nutorch step` is already dual and
   stays. Other prelude verbs (`tensor`, `value`, `free`) take data/handle
   lists, not leading-tensor slots — unchanged.
7. **Docs are Experiment 2** — this experiment is the Rust change, the
   regenerated module, and the parity proof. (The issue-0015 website twins keep
   working untouched: pipeline form remains valid.)

## Changes

1. **`torch-cli/src/main.rs`**: `generate_nu_module` — the `Exactly(n ≥ 1)` arm
   rewritten per decisions 1–3; helper for the generic rest-arg conversion;
   usage line in the comment.
2. **`nutorch.nu`**: regenerated (committed).
3. **`scripts/test-dual-input.nu`** (NEW): the parity harness.
4. **No ops-table changes; no daemon changes; no website changes (exp 2); no
   `v1/`.**

## Verification

1. **Hygiene**: `cargo fmt -- --check`, build 0 warnings, full Rust suite green
   — including the regenerated-module staleness test.
2. **Parity harness green** (explicit-`use` of the REPO module, private TMPDIR):
   both forms identical values across the sample; arity errors named.
3. **Both existing nu acceptance scripts still pass**:
   `scripts/train-regression.nu` (pipeline forms throughout) and the issue-0015
   docs twins' forms (spot-run two).
4. **The module diff is reviewable**: every `Exactly(n ≥ 1)` wrapper changed
   shape; `AtLeast`/`Exactly(0)` wrappers byte-identical (asserted by diff
   inspection, since the generator must not disturb them).
5. **Website gates untouched and green**: `check:content` (fences unchanged),
   `check:tabs` (twins unchanged) — run to prove no regression from the module
   regen.

**Pass** = all five. **Fail** = any parity mismatch, any `AtLeast`/`Exactly(0)`
wrapper drift, or a worse error message than the CLI's own.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**First pass: CHANGES REQUIRED** — 2 Required. First: the design claimed
pipe-plus-full-positionals is an arity error; the CLI silently ignores an
unneeded pipe BY CONTRACT (`stdin_handles(0)` never reads; the retired XOR
clause exists because conflict-detection reads block on terminals) — the spine
and decision 2 corrected, and the harness now asserts the two REAL errors with
their verbatim texts ("missing tensor operand(s) — pass handles as arguments or
pipe them in"; "too many arguments"). Second: the prelude's `nutorch forward`
was an unaddressed principle-#2 violation the generator change would never touch
— decision 6 brings it into scope with the same delegation shape (CLI-side
`forward` confirmed dual; `nutorch step` already dual; remaining prelude verbs
excluded with reasons). Optional folded: `gather` is tensor+tensor with `--dim`
as a FLAG (the table says so), not tensor+int+tensor. Nits folded: exact error
strings; `zero_grad` parity asserted on the post-call `grad` read since the op
returns nothing. **Second pass: APPROVED** — all folds verified against the
source by line number; the reviewer confirmed the silent-ignore path
(`n - positionals == 0` requests zero stdin handles), the dual CLI `forward`,
the `usage()` string availability for the generated comments, and that the
harness sample covers the list-conversion, typed-flag, and untouched-`AtLeast`
paths.
