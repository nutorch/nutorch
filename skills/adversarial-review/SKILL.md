---
name: adversarial-review
description:
  "Run an in-session adversarial review of Nutorch work using a fresh-context
  subagent (Claude: `adversarial-reviewer`; Codex: multi_agent_v1.spawn_agent).
  Use at experiment design/result gates, or whenever the user asks for
  adversarial / skeptical / red-team review without an external reviewer CLI."
---

# Adversarial Review

Run a fresh-context, read-only adversarial review **inside the current agent
session** by delegating to an adversarial subagent. No external reviewer CLI, no
session id, no logs to manage — spawn a subagent and it returns its verdict and
findings.

Runtime-specific invocation:

- **Claude:** use the `Agent` tool with `subagent_type: "adversarial-reviewer"`,
  defined in `.claude/agents/adversarial-reviewer.md`.
- **Codex:** use `multi_agent_v1.spawn_agent`. Pass the adversarial reviewer
  instructions in the spawn prompt, plus concrete artifact paths. Do **not** try
  to use Claude's `Agent` tool.

This is the in-session counterpart of the `claude-review` skill, which shells
out to a separate `claude -p` process. Use this skill when you want the review
to run in the same session; use a cross-model reviewer when you specifically
want a **different model's** independent read (see "Self-review caveat" below).
For now, Nutorch's reviewers are Claude reviewing Claude; cross-model reviewers
(Codex and others) will be added later.

## When this skill applies

- The user asks for an "adversarial review", "skeptical review", "red team",
  "try to break this", or similar.
- An experiment reaches its **design gate** (after the design is written, before
  implementation) or its **result gate** (after implementation + result
  recording, before the result commit). These are the two required AI review
  gates in `AGENTS.md`'s experiment flow.
- A change is large, risky, or touches the tensor registry, autograd, device
  placement, the dual-input contract, data conversion (`value_to_tensor` /
  `tensor_to_value`), or the plugin protocol boundary.
- Before closing an issue after a complex series of experiments.

## Reviewer posture

The subagent runs in **its own fresh context window** — it does **not** see this
conversation unless you explicitly fork context. It receives only what you put
in the spawn prompt plus whatever it reads itself with its available tools. It
must be instructed to try to reject the work on evidence, verify claimed gate
results independently where feasible, and return a structured verdict.

The Claude agent file already contains the reviewer mandate. For Codex spawn
prompts (when cross-model review arrives), use:

```text
You are the adversarial reviewer for Nutorch. You are separate from whoever
produced the work under review. Your default posture is skepticism. Try to
reject the work, but every objection must be grounded in evidence you can point
to.

Read-only discipline: do not edit, write, create, move, delete, stage, commit,
push, or run mutating commands. Use shell commands only for inspection and
read-only verification such as git diff/log/show/status, rg, cargo build,
cargo fmt -- --check, dprint check, and the Nushell test suite.
If a check would modify files, do not run it; state that you could not verify
it.

Return:
VERDICT: APPROVED | CHANGES REQUIRED
Then findings, most severe first:
[Required] file:line — issue · Evidence: ... · Fix: ...
[Optional] file:line — issue · Evidence: ... · Fix: ...
[Nit] file:line — issue · Fix: ...

Approve only when zero Required findings remain. Do not invent findings.
```

Because it starts blind, **you must hand it the artifacts** — point it at the
files; do not paraphrase them. Give it:

- the experiment file (`issues/<n>/NN-*.md`);
- the relevant diff (tell it the exact `git diff` / `git diff --staged` /
  `git show <ref>` command to run, or the changed file paths);
- the source files it should scrutinize;
- the PyTorch behavior to compare against, for fidelity checks (e.g. the
  relevant PyTorch doc semantics, or reference Python implementations under
  `v1/npm/nn.nu/*.py`);
- `AGENTS.md` and the issue `README.md` as the workflow contract;
- any command output whose truth matters (test counts, build logs).

## Invocation

### Claude invocation

Spawn the subagent with Claude's `Agent` tool,
`subagent_type: "adversarial-reviewer"`. Put the review task and artifact
pointers in the prompt. Example:

> Use the **adversarial-reviewer** subagent to review the Experiment 3 design.
> Read `issues/0002-nutorchd-architecture/03-*.md`, `AGENTS.md`, and
> `v1/cargo/src/lib.rs`. Try to reject the design; return your verdict and
> findings.

The subagent's final message — its `VERDICT` plus findings — comes back to you
as the tool result. It is not shown to the user automatically; relay the
high-signal parts.

### Codex invocation (future)

Use `multi_agent_v1.spawn_agent` with `fork_context: false` unless the review
explicitly needs the current conversation. Put the reviewer mandate, review
task, and artifact pointers in the prompt. Wait for the agent only when its
verdict gates your next step. Close the agent after recording or acting on its
result.

### Design-gate prompt template

```text
Review this Nutorch experiment DESIGN with fresh context. Do not edit anything.

Read:
- the experiment file: issues/<n>/NN-<slug>.md
- the workflow contract: AGENTS.md and issues/<n>/README.md
- the relevant source files (for v1 reference: v1/cargo/src/<files>)

Try to reject this design. Check:
- the issue README links this experiment with status Designed;
- the experiment has Description, Changes, and Verification;
- scope is narrow enough for one experiment, and matches exactly what was asked;
- the technical plan is correct and faithful to PyTorch semantics;
- new/changed commands honor the dual input pattern;
- verification has concrete pass/fail criteria that would actually prove the goal;
- required hygiene checks are present (cargo build, cargo fmt -- --check, the
  Nushell test suite, dprint check, git diff --check).

Return VERDICT (APPROVED | CHANGES REQUIRED) then findings (Required/Optional/Nit)
with file:line, evidence, and a concrete fix. Approve only if no Required remain.
```

### Result-gate prompt template

```text
Review this COMPLETED Nutorch experiment with fresh context. Do not edit anything.

Read:
- the experiment file (Description, Changes, Verification, Result): issues/<n>/NN-<slug>.md
- the implementation diff: run `git diff <plan-commit>..HEAD -- <paths>` (or the
  working tree if not yet committed)
- the changed source files
- the workflow contract: AGENTS.md

Try to reject this result. Check:
- the implementation matches the approved scope — no unrequested changes;
- it is correct and faithful to PyTorch semantics; find the specific divergence
  if any;
- new/changed commands honor the dual input pattern with XOR enforcement;
- the tests actually prove the claim (not vacuous, cover the interesting cases);
- independently verify the claimed gate results where feasible: run the
  experiment's stated verification commands (for work in the active v2 crate,
  once it exists: `cargo build --release`, `cargo fmt -- --check`, its test
  suite; the archived v1 reference builds from `v1/cargo`); report any
  mismatch with the stated numbers;
- the experiment file has Result and Conclusion, and the README status matches;
- the result commit has NOT been made before this review.

Return VERDICT then findings (Required/Optional/Nit) with file:line, evidence, and
a concrete fix. Approve only if no Required remain.
```

### Re-review prompt template

```text
Re-review ONLY the fixes for your prior findings, with fresh context. Do not edit.
For each prior finding, confirm whether it is now resolved, citing the new
file:line. Report any new Required finding the fix introduced. Approve only if no
Required remain.
```

## After the review: lead-agent judgment

You (the implementing agent) stay responsible for the outcome. The review is
input, not a verdict you must obey blindly.

1. **Accept** findings that are real correctness, fidelity, verification, scope,
   or workflow issues. Fix them before proceeding.
2. **Reject** false positives explicitly, with a one-line reason — do not
   silently ignore a finding.
3. **Re-review** after non-trivial fixes (use the re-review template) until no
   Required findings remain.
4. **Record** the review in the experiment file: that it was an adversarial
   subagent with fresh context, which agent/model performed it, the findings,
   the fixes, and the final verdict.
5. Respect the commit gates: do not implement after a design review until the
   plan commit exists; do not design the next experiment after a result review
   until the result commit exists.

## Self-review caveat (read this)

This subagent is usually the **same model family** as the implementer (Claude
reviewing Claude). That is convenient and fast, but a same-model reviewer shares
blind spots and can drift toward agreement. The subagent's design fights this
with fresh context, a hard "try to reject on evidence" mandate, read-only
discipline, independent re-verification of claimed results, and a
no-approval-with-Required-findings gate — but it does not fully replace a
genuinely different model.

Therefore:

- For routine gates, the in-session adversarial subagent is a reasonable
  default.
- For **high-risk** work (registry/autograd semantics, FFI behavior, anything
  that already failed once), prefer an external check via `claude-review` (a
  separate `claude -p` process), and — once cross-model reviewers are set up — a
  genuinely different model, or run both and reconcile.
- You can raise rigor by spawning the subagent **two or three times in
  parallel** with different emphases (e.g. one on correctness, one on PyTorch
  fidelity, one on verification quality) and treating any Required finding from
  any pass as blocking. This breaks single-perspective blind spots without
  leaving the session.

## Notes

- The subagent is **read-only by discipline**, not necessarily by hard sandbox.
  It may have shell access so it can run `git diff` and read-only builds/tests.
  The prompt must forbid mutating commands.
- The Claude named agent's `model` is set in
  `.claude/agents/adversarial-reviewer.md`.
- Claude named subagents are loaded at session start. Codex-native use does not
  depend on the Claude agent registry; it relies on `multi_agent_v1.spawn_agent`
  plus this skill's reviewer mandate.
