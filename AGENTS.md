# Nutorch

**Nutorch v2 (nutorchd) is a shell-agnostic tensor daemon**: GPU-accelerated
PyTorch tensor operations from any shell, built on tch-rs (Rust bindings for
LibTorch, PyTorch's C++ backend).

[Agent development guide](https://agents.md/). `CLAUDE.md` is a symlink to this
file — Claude, Codex, and any other agent read the same contract.

## Rules

Do exactly what your user says. No more, no less. NEVER assume they want
something they didn't ask for. NEVER change code unless explicitly asked.

When editing Rust code, always run `cargo fmt`. Accept the formatter output as
the source of truth. Do not manually undo, minimize, or selectively revert
`cargo fmt` formatting changes, including import ordering or wrapping changes.

Markdown, TOML, and JSON files are formatted with dprint (`dprint fmt`, config
in `dprint.json`). Accept the formatter output as the source of truth.

`v1/` is the archived v1 implementation — a frozen historical reference. Do not
develop in it. The only allowed class of change there is mechanical (e.g. a path
fix required by repo tooling). v2 work ports from v1; it never edits it.

## Vision

v1 proved the core idea: tensors live in a Rust-owned registry, and the shell
passes **string handles** (UUIDs) through pipelines — actual tensor data never
crosses the process boundary. But v1 was a Nushell plugin, so the registry's
lifetime was at the mercy of Nushell's plugin garbage collector, and the
audience was limited to Nushell users.

**v2 decouples the registry from the shell:**

```
bash / zsh / fish / nushell / python / anything
    ↓ thin `torch` CLI client
    ↓ Unix socket (request/response protocol)
nutorchd            ← owns the tensor registry, LibTorch context,
    ↓ tch-rs           GPU memory, and autograd graphs
LibTorch (C++)
    ↓ Metal (MPS)
Apple-silicon GPU
```

- **nutorchd** is a standalone daemon (one copy, or a configurable number) that
  maintains the tensor database. Tensors are referenced by string identifiers.
- **GPU-only, Mac-only for now** (issue 0003): every tensor lives on the GPU —
  on Apple silicon, MPS — and there is **no device option anywhere** in the API.
  The daemon requires MPS and refuses to start without it. Future platform
  expansion (e.g. CUDA on Linux) is a daemon-level "the GPU" decision per
  platform, never a per-tensor option.
- A **thin CLI client** (`torch`) sends one operation per invocation over a Unix
  socket and prints the resulting handle to stdout. Because handles are plain
  text on stdout/stdin, real POSIX pipelines compose — the dual input pattern
  survives in bash almost untouched.
- **Any shell works out of the box.** Nushell remains the premium client
  (structured data, native serialization), but bash, zsh, fish, and scripts in
  any language are first-class citizens.
- **Tensor lifetime becomes a first-class concept** — named handles, sessions,
  TTLs — instead of a side effect of plugin process management.

The v2 architecture, wire protocol, lifecycle model, and client surface are
designed through issues in `issues/` — the design record lives there, not here.
This section stays a stable summary.

## Carried-Forward Principles (from v1)

v1's architecture record is `v1/AGENTS.md`; its code is the reference
implementation that v2 ports from. These v1 principles remain binding for v2:

1. **String handles are the interface.** Tensor data never leaves Rust; clients
   hold and pass opaque string identifiers.
2. **Dual Input Pattern.** Every operation supports both pipeline form
   (`$t1 | torch add $t2`) and argument form (`torch add $t1 $t2`), with XOR
   enforcement (reject both-or-neither). This is not optional — it is how the
   tool feels native to both PyTorch users and shell users.
3. **PyTorch API fidelity.** Command names, argument order, defaults, and
   semantics match PyTorch wherever possible.
4. **Explicit over implicit.** No auto-casting, no automatic broadcasting
   surprises. (v1's "manual device placement" clause was retired by issue 0003:
   there is exactly one device — the GPU — so there is nothing to place.)
5. **Validate in Rust, not C++.** Pre-validate shapes, dims, and dtypes before
   tch-rs calls — LibTorch errors are opaque and crash-prone; Rust-side
   validation gives good error messages.

## Directory Structure

```
nutorch/
├── README.md                    # Project overview (v2 direction, status)
├── AGENTS.md                    # This file (agent contract)
├── CLAUDE.md                    # Symlink to AGENTS.md
├── LICENSE                      # Apache 2.0
├── dprint.json                  # Formatter config (md/toml/json)
│
├── Cargo.toml                   # ⭐ v2 Rust workspace (members below)
├── nutorchd/                    # The daemon: registry, socket, dispatch
│   ├── src/main.rs              #   socket loop + request dispatch
│   ├── src/registry.rs          #   handle → tch::Tensor map
│   ├── src/convert.rs           #   JSON ↔ tensor (ported from v1)
│   ├── src/protocol.rs          #   NDJSON wire types (PoC, throwaway)
│   └── tests/mps_smoke.rs       #   toolchain/MPS proof (issue 0002 exp 1)
├── torch-cli/                   # Thin client; binary is named `torch`
│   └── src/main.rs              #   args/stdin → one request → stdout
├── .cargo/config.toml           # Force-pins LIBTORCH to .libtorch (venv)
│                                #   (.venv-torch + .libtorch symlink are
│                                #    gitignored; see Cargo.toml header)
│
├── issues/                      # ⭐ Issues and Experiments (the workflow)
│   ├── README.md                # Generated index (scripts/build-issues-index.sh)
│   └── {NNNN}-{slug}/           # One folder per issue
│       ├── README.md            # Issue spine: frontmatter, goal, experiments index
│       └── NN-{slug}.md         # One file per experiment
│
├── skills/                      # Agent skills (symlinked from .claude/skills)
│   ├── adversarial-review/      # In-session fresh-context review subagent
│   ├── claude-review/           # External claude -p reviewer with session log
│   ├── commit/                  # GitPoet commit messages
│   └── create-skill/            # Meta-skill for authoring new skills
│
├── scripts/
│   └── build-issues-index.sh    # Regenerate issues/README.md
│
├── .claude/
│   ├── skills -> ../skills      # Symlink
│   └── agents/
│       └── adversarial-reviewer.md  # Named reviewer subagent definition
│
├── v1/                          # ⭐ Archived v1 (frozen reference implementation):
│                                #   v1/README.md (user docs), v1/AGENTS.md
│                                #   (architecture record), v1/TODO.md (quality
│                                #   tracker), v1/cargo (the Nushell plugin),
│                                #   v1/npm (helper packages), v1/raw-images
│
├── docs/
│   └── archive/                 # Historical chat-session records (pre-workflow)
│
└── logs/                        # Review logs and scratch output (gitignored)
```

The v2 source tree above is the PoC scaffolding from issue 0002; its layout
evolves with the issues that build on it (protocol design, lifecycle, the
Nushell client), and this section is updated as that happens.

## Issues and Experiments

Every significant piece of work gets an issue in `issues/`. Issues describe the
problem, provide background, and propose solutions. Experiments are the
incremental steps that solve the problem.

### Issue Structure

Each issue is a **folder**. The `README.md` is the issue **spine** (frontmatter,
goal, background, analysis, an ordered index of experiments, and the final
conclusion). **Every experiment is its own numbered file** in the same folder —
the README never contains experiment bodies, only links to them.

```
issues/0002-nutorchd-architecture/
├── README.md                     ← spine: frontmatter, goal, background,
│                                    the ordered Experiments index, conclusion
├── 01-stand-up-daemon.md         ← Experiment 1 (full body in its own file)
├── 02-wire-first-op.md           ← Experiment 2
└── 03-...                        ← one file per experiment, in sequence
```

The folder name is `{NNNN}-{slug}`. The number is zero-padded to 4 digits and
globally sequential across the whole project. The slug is lowercase, hyphenated,
and describes the topic.

**Why one file per experiment:** it keeps experiments ordered and easy to read,
access, and organize (up to ~100 per issue with clean `NN-` filenames), and —
critically — it makes experiments easy to **automate**: each experiment is a
discrete file created and tracked from the README, rather than ever-growing
edits to one monolithic document.

The full index of all issues is at `issues/README.md`. Regenerate it with:

```bash
scripts/build-issues-index.sh
```

#### Frontmatter

Every `README.md` starts with TOML frontmatter:

```
+++
status = "open"
opened = "2026-06-10"
+++
```

Or for closed issues:

```
+++
status = "closed"
opened = "2026-06-10"
closed = "2026-06-10"
+++
```

Issues may add their own TOML frontmatter keys — to `README.md`, experiment
files, or other issue docs — for issue-specific metadata such as per-experiment
agent provenance, as long as:

- the reserved workflow keys are preserved: `README.md` always carries `status`
  and `opened` (plus `closed` when closed), unchanged in name and meaning;
- additive keys are valid TOML between the `+++` delimiters and do not
  contradict the reserved keys or the index tooling —
  `scripts/build-issues-index.sh` reads only the reserved README keys and
  ignores the rest;
- the issue documents its own added schema in its `README.md`.

#### README.md structure

After the frontmatter, a new issue's `README.md` has these sections:

1. **Title** (H1) — `# Issue {N}: {descriptive title}`
2. **Goal** — One or two sentences describing the desired outcome.
3. **Background** — Context, prior work, constraints.
4. **Architecture** / **Analysis** / **Proposed Solutions** — Technical details.

A new issue's README has **no experiments listed yet**.

As experiments are created, the README grows an **`## Experiments`** section: an
ordered list linking to each experiment file, one per line, with a one-line
status. The README holds the links and statuses only — never the experiment
bodies. Example:

```markdown
## Experiments

- [Experiment 1: Stand up the daemon](01-stand-up-daemon.md) — **Pass**
- [Experiment 2: Wire the first op](02-wire-first-op.md) — **Partial** (needs a
  length-prefixed framing fix)
- [Experiment 3: …](03-….md) — **Designed**
```

Keep each status to one of: `Designed`, `In progress`, `Pass`, `Partial`,
`Fail`. Update the line when the experiment's result is recorded, so the README
doubles as an at-a-glance progress tracker.

When the issue is solved or abandoned, add the **`## Conclusion`** section to
the README (see "Closing an Issue").

#### Experiment files

Each experiment lives in its **own file** `NN-{slug}.md` in the issue folder,
where `NN` is a zero-padded two-digit number in creation order (`01`, `02`, …,
up to `99`). The slug is lowercase-hyphenated and describes the experiment.

An experiment file may begin with an optional TOML frontmatter block
(`+++ … +++`) before its H1 title — for issue-specific metadata such as agent
provenance. Experiment frontmatter is optional and must not replace the required
H1 title and H2 sections below it.

Each experiment file contains:

1. **Title** (H1) — `# Experiment {N}: {descriptive title}`
2. **Description** — What and why.
3. **Changes** — Specific code changes, listed by file.
4. **Verification** — How to test. Concrete steps and pass/fail criteria.
5. **Result** and **Conclusion** — added after the experiment runs (see
   "Recording results").

Keep each file focused; if one grows past ~1000 lines, that is a sign the
experiment is too big and should be split into the next numbered experiment.

### Multiple Open Issues

Multiple issues can be open at the same time. This allows interleaving work — a
large issue like the nutorchd daemon can stay open while smaller issues are
opened and closed alongside it.

### Experiments

#### When to create an experiment

Only after the issue's requirements are clear. Each experiment is designed,
implemented, and concluded before the next one is designed.

**Never list experiments upfront.** The outcome of each experiment informs what
comes next.

#### Experiment structure

Each experiment is its own file `NN-{slug}.md` (see "Experiment files" above),
and is added as a new link in the README's `## Experiments` index the moment it
is created. Inside the file, use an H1 title and H2 sections:

1. **Title** (H1) — `# Experiment {N}: {descriptive title}`
2. **Description** (H2) — What and why.
3. **Changes** (H2) — Specific code changes, listed by file.
4. **Verification** (H2) — How to test. Concrete steps and pass/fail criteria.
5. **Result** / **Conclusion** (H2) — added after it runs.

#### Verification gates

The standard hygiene gates apply to the **active v2 code once it exists**: build
clean (no new warnings), `cargo fmt -- --check` clean, tests green,
`dprint check` clean for touched markdown/TOML/JSON, and conformance to the
carried-forward principles (dual input pattern, PyTorch API fidelity).

Until the v2 scaffolding lands, each experiment defines its concrete
verification commands in its own Verification section. `v1/` is frozen and not
edited except for mechanical path fixes; the archived v1 build/test gates are
recorded in `v1/AGENTS.md`.

#### One at a time

Design and implement one experiment at a time. The result of Experiment 1
directly informs what Experiment 2 should be.

#### AI review gate

Every experiment must be reviewed by another AI agent before moving to the next
stage. For now the reviewer is **Claude reviewing Claude** — the in-session
`adversarial-reviewer` subagent (see the `adversarial-review` skill) or an
external `claude -p` process (see the `claude-review` skill). Cross-model
reviewers (Codex and others) will be added later; the gate itself does not
change.

1. **Design review before implementation**
   - After writing the experiment design, ask another AI agent to review it.
   - Fix all real issues found by the review.
   - Record the review result in the experiment file.
   - Do not implement the experiment until the reviewing agent approves the
     design.

2. **Result review before the next experiment**
   - After implementation, verification, and result recording, ask another AI
     agent to review the completed experiment and result.
   - Fix all real issues found by the review.
   - Record the completion-review result in the experiment file.
   - Do not design or implement the next experiment until the reviewing agent
     approves the completed output.

The reviewing agent must be separate from the implementation pass — at minimum a
fresh-context subagent; ideally (later) a different model entirely.

#### Experiment commits

Every experiment has two required commit points:

1. **Plan commit** — after the experiment design is written, reviewed, fixed,
   approved, and linked from the issue README, commit the experiment plan before
   implementation begins.
2. **Result commit** — after implementation, verification, result recording,
   completion review, and any required fixes, commit the experiment result
   before designing the next experiment.

These commits must be separate. Do not combine an experiment plan and its result
in the same commit, and do not start the next experiment before the previous
experiment's result commit exists.

#### Recording results

After testing, append the result **inside the experiment's own file**, below
Verification:

```markdown
## Result

**Result:** Pass / Partial / Fail

{description}

## Conclusion

{what we learned, what the next experiment should be}
```

Then update that experiment's status on its line in the README's
`## Experiments` index (`Designed` → `Pass`/`Partial`/`Fail`). All three
outcomes are valuable — failed experiments eliminate dead ends.

### Closing an Issue

When the issue is solved or abandoned, add a `## Conclusion` section to the
**`README.md`** (after the `## Experiments` index), summarizing what was learned
and the outcome. Update the frontmatter to `status = "closed"` with a `closed`
date. Regenerate the index:

```bash
scripts/build-issues-index.sh
```

### Immutability

Closed issues are historical records. They are **immutable** and must NEVER be
modified. History stays as it was written.

### Process Summary

1. **Create the issue** — `issues/{NNNN}-{slug}/README.md` with frontmatter,
   goal, background, analysis. No experiments yet.
2. **Design Experiment 1** — Create `01-{slug}.md` with the experiment body, and
   add a link to it under `## Experiments` in the README (status `Designed`).
3. **Review and commit the plan** — Get another AI agent to approve the design,
   fix real findings, record the review result, and commit the experiment plan.
4. **Implement Experiment 1** — Write the code.
5. **Record the result** — Append `## Result` / `## Conclusion` inside
   `01-{slug}.md`, and update its status on the README index line.
6. **Review and commit the result** — Get another AI agent to approve the
   completed output, fix real findings, record the completion review, and commit
   the experiment result.
7. **Repeat** — Create `02-{slug}.md` for the next experiment (the prior result
   informs it), link it from the README, and continue until the goal is met.
8. **Close the issue** — Write the `## Conclusion` in the README, update
   frontmatter, rebuild the index.

## Remember

NEVER change code unless explicitly asked. NEVER make unrequested changes.
Always do EXACTLY what your user asks — no more, no less.
