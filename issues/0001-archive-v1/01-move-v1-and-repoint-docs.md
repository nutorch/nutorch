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

# Experiment 1: Move v1 into `v1/` and re-point the documentation

## Description

Perform the archive in one coherent step: tag the pre-move state, `git mv` the
v1 implementation into `v1/`, split the v1 architecture documentation out of the
root `AGENTS.md`, and rewrite the root `AGENTS.md` and `README.md` to describe
v2 (nutorchd). One experiment rather than two because the move and the doc
rewrite are coupled: moving the files without re-pointing the docs would leave
the root documentation describing paths that no longer exist at the result
commit.

## Changes

1. **Tag**: `git tag v1-final` on the current HEAD (`2233da9`) before any move.

2. **Moves** (all via `git mv`):
   - `cargo/` → `v1/cargo/`
   - `npm/` → `v1/npm/`
   - `TODO.md` → `v1/TODO.md`
   - `raw-images/` → `v1/raw-images/`
   - `README.md` → `v1/README.md`

3. **`v1/README.md`**: prepend an archive notice (v1 is the frozen
   proof-of-concept Nushell plugin; v2 lives at the repo root; paths in this
   document are relative to `v1/`). Body otherwise unchanged.

4. **`v1/AGENTS.md`** (new): the v1-specific sections lifted verbatim from the
   root `AGENTS.md` — Project Overview, The Wrapping Layers, Counter-Intuitive
   Facts, File Structure, Key File Patterns, Architecture Deep Dive, Development
   Workflow (build/test/install), Command Categories, Common Flags, Known
   Limitations, Design Philosophy — under an archive-notice header. The v1
   "Future Directions" section is dropped (superseded by v2; the v1 README
   retains the user-facing status text).

5. **Root `AGENTS.md`** (rewritten): keeps the title, Rules, Issues and
   Experiments workflow, and Remember sections unchanged in substance; replaces
   the v1 architecture content with:
   - a v2 **Vision** section: nutorchd owns the tensor registry (LibTorch
     context, GPU memory, autograd graphs); a thin `torch` CLI client passes
     string handles over a Unix socket; any shell is a client; Nushell is the
     premium client; design issues live in `issues/`;
   - **Carried-forward principles** from v1: string handles as the interface
     (tensor data never crosses the process boundary), the dual input pattern,
     PyTorch API fidelity, explicit device placement, Rust-side validation for
     good error messages;
   - a **Directory Structure** section reflecting the new layout, with `v1/`
     described as the frozen reference implementation that v2 ports from;
   - **Verification gates** reworded: the gates (cargo build, cargo fmt, dprint,
     tests) apply to the active v2 code once it exists; concrete commands are
     defined per-issue until the v2 scaffolding lands; `v1/` is frozen and not
     edited except for mechanical path fixes.

6. **Root `README.md`** (new): short — what Nutorch v2 is, status (design phase,
   see `issues/README.md`), what v1 was and where it lives (`v1/`, including the
   original README and demo screenshots).

7. **Path-reference fixes** in workflow tooling (mechanical, concrete):
   - `skills/adversarial-review/SKILL.md`:
     - `npm/nn.nu/*.py` → `v1/npm/nn.nu/*.py`;
     - the Claude invocation example's `cargo/src/lib.rs` →
       `v1/cargo/src/lib.rs`;
     - the design-gate template's "the relevant source: `cargo/src/<files>`" →
       "the relevant source files (for v1 reference: `v1/cargo/src/<files>`)";
     - the result-gate template's gate-command bullet becomes: "independently
       verify the claimed gate results where feasible: run the experiment's
       stated verification commands (for work in the active v2 crate, once it
       exists: `cargo build --release`, `cargo fmt -- --check`, its test suite;
       the archived v1 reference builds from `v1/cargo`); report any mismatch
       with the stated numbers".
   - `skills/claude-review/SKILL.md`: no path changes needed (verified: it
     references `issues/` and the repo root only).
   - `.claude/agents/adversarial-reviewer.md`: the read-only verification
     parenthetical becomes: "read-only builds and tests (`cargo build` and
     `cargo fmt -- --check` from the active crate directory — for the archived
     v1 reference, `v1/cargo` — `dprint check`, and the Nushell test suite via
     `nu -c \"use node_modules/test.nu; test run-tests\"` from the active test
     directory — `v1/cargo/test` for v1)".
   - Root `AGENTS.md` "Verification gates" subsection: the concrete
     `cargo build --release` (from `cargo/`) / `cargo fmt` / `cargo/test` /
     TODO.md bullets are replaced with: gates apply to the **active v2 code once
     it exists** (build clean, fmt clean, tests green, `dprint check` clean,
     dual-input and PyTorch-fidelity conformance); until the v2 scaffolding
     lands, each experiment defines its concrete commands in its Verification
     section; `v1/` is frozen and not edited except for mechanical path fixes
     (the archived v1 gates are recorded in `v1/AGENTS.md`).
   - All other `cargo/` / `npm/` / `TODO.md` references in root `AGENTS.md`
     (Counter-Intuitive Facts, Development Workflow, File Structure, etc.) move
     wholesale into `v1/AGENTS.md`, where they are correct relative to `v1/`'s
     content (build/test instructions there are rewritten as `v1/cargo` paths
     where they give absolute-from-repo-root commands).

8. **Formatting**: `dprint fmt` scoped to the files created or edited in this
   experiment (not repo-wide; pre-existing files under `v1/` keep their
   formatting).

## Verification

Concrete checks, all from the repo root:

1. **History follows**:
   `git log --follow --oneline v1/cargo/src/lib.rs | tail -1` shows the original
   (pre-move) first commit of the file, AND
   `git log --follow --oneline v1/README.md | wc -l` is greater than 1 (the
   moved v1 README must trace back past the move into the old root `README.md`
   history — this guards against the new root README masking a
   delete-and-recreate that loses history).
2. **Tag**: `git tag --points-at 2233da9` lists `v1-final`.
3. **Layout**: root no longer contains `cargo/`, `npm/`, `TODO.md`,
   `raw-images/`; `v1/` contains exactly `cargo/`, `npm/`, `raw-images/`,
   `TODO.md`, `README.md`, `AGENTS.md`.
4. **No stale references (mechanical, zero-tolerance)**:
   `rg -nP "(?<!v1/)(?<!\.)\b(cargo|npm|raw-images)/" README.md AGENTS.md skills/ .claude/agents/`
   returns **zero matches** — every surviving reference in those files must be
   the `v1/`-prefixed form (`v1/cargo/`, `v1/npm/`, `v1/raw-images/`) or
   `~/.cargo/...` (excluded by the lookbehinds), and
   `rg -n "TODO\.md" README.md AGENTS.md skills/ .claude/agents/` returns only
   `v1/TODO.md`-prefixed references (or none). A bare match is a Fail for this
   check; there is no "intentional bare reference" allowance. (The
   `issues/0001-archive-v1/` files themselves discuss the old paths by design
   and are excluded from this check.)
5. **Build still works at the new path**: `cargo build --release` from
   `v1/cargo` succeeds (the environment has `LIBTORCH` set and libtorch
   present). This is a Pass criterion since the environment supports it.
6. **Formatting**: `dprint check` clean for the explicit list of files created
   or edited by this experiment: root `README.md`, root `AGENTS.md`,
   `v1/README.md`, `v1/AGENTS.md`, `skills/adversarial-review/SKILL.md`,
   `.claude/agents/adversarial-reviewer.md`, and `issues/0001-archive-v1/*.md`.
   Repo-wide `dprint check` is **not** expected clean — some pre-existing v1
   files (e.g. `v1/TODO.md`, `v1/npm/nn.nu/pyproject.toml`) predate dprint and
   move verbatim; they are frozen and intentionally not reformatted.
7. **Symlinks intact**: `CLAUDE.md` still resolves to `AGENTS.md`;
   `.claude/skills` still resolves to `skills/`.

**Pass** = all seven checks hold.

**Partial** = moves and docs land but the v1 build fails for an environmental
reason unrelated to the move (recorded, with evidence that the failure pre-dates
the move).

**Fail** = history does not follow the moves, or root docs still describe the v1
layout.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**First pass: CHANGES REQUIRED** — 2 Required, 2 Optional:

- [Required] Verification #4's rg pattern could not distinguish repointed `v1/`
  references from wrongly-left bare ones, and its acceptance language ("only
  matches that intentionally point into v1/") was subjective — the check could
  pass while stale paths remained. **Fixed:** zero-tolerance mechanical pattern
  with `(?<!v1/)(?<!\.)` lookbehinds; no allowance.
- [Required] Changes item 7 said the stranded gate commands
  (`.claude/agents/adversarial-reviewer.md:28-30`, root `AGENTS.md`
  verification-gate bullets) would be "generalized" without stating to what.
  **Fixed:** concrete replacement wording spelled out in item 7.
- [Optional] History check covered only `v1/cargo/src/lib.rs`; the riskiest move
  is `README.md` (a new root README is created in the same operation).
  **Fixed:** added a `--follow` check on `v1/README.md`.
- [Optional] dprint scope ambiguous; repo-wide check would scan moved v1 files
  that predate dprint. **Fixed:** explicit file list; repo-wide cleanliness
  explicitly not expected.

**Re-review (fresh context): APPROVED** — all four findings confirmed resolved
(reviewer independently ran the tightened pattern against the pre-move tree and
verified its lookbehind behavior); no new findings.

## Result

**Result:** Partial

All verification checks pass except #5 (the v1 build), which fails for an
environmental reason proven to pre-date the move.

1. **History follows — Pass** (with a sequencing discovery, below). Pre-commit
   form: the staged move commit contains **111 renames and 0 non-rename
   entries** (`git diff --staged -M --name-status`), including
   `R100 cargo/src/lib.rs → v1/cargo/src/lib.rs` and
   `R095 README.md → v1/README.md` (the similarity delta is the archive notice).
   Post-commit `--follow` confirmation appended below after the move commit
   landed.
2. **Tag — Pass.** `git tag --points-at 2233da9` → `v1-final`.
3. **Layout — Pass.** Root: no `cargo/`, `npm/`, `TODO.md`, `raw-images/`. `v1/`
   contains exactly `cargo/`, `npm/`, `raw-images/`, `TODO.md`, `README.md`,
   `AGENTS.md`.
4. **No stale references — Pass.** The zero-tolerance pattern returns zero
   matches; all `TODO.md` references are `v1/TODO.md`-prefixed. Notably, this
   check **failed on its first run** against the freshly written root
   `AGENTS.md`: the new Directory Structure tree listed `cargo/`, `npm/`,
   `raw-images/`, `TODO.md` as bare labels inside the `v1/` subtree. Fixed by
   rewriting that tree entry with `v1/`-prefixed labels. The mechanical check
   caught a real instance of the exact defect class it was designed for.
5. **Build — Fail (environmental, pre-existing).** `cargo build --release` from
   `v1/cargo` exits 101: clang rejects **libtorch's own headers** while
   compiling the `torch-sys v0.20.0` dependency
   (`/opt/homebrew/.../torch/include/c10/util/strong_type.h:1608: error:
   'is_arithmetic' cannot be specialized`
   under `-Winvalid-specialization`, `-mmacosx-version-min=26.4`). No
   repo-relative path appears in the failing compile command except the
   `target/` output directory. **Control:** a git worktree at the pre-move
   `v1-final` tag fails identically — exit 101, same error signature (4
   occurrences) — proving the failure pre-dates the move and is a
   toolchain↔libtorch incompatibility, not a move artifact. (The
   `v1/cargo/target/release/nu_plugin_torch` binary on disk is from an earlier
   successful build with an older toolchain.)
6. **Formatting — Pass.** `dprint check` clean on the full explicit file list.
7. **Symlinks — Pass.** `CLAUDE.md → AGENTS.md`; `.claude/skills → ../skills`.

**Sequencing discovery (affects the commit structure):** the result lands as
**two commits** — first the pure move, then the new root docs. Reason: git
rename detection only pairs an added path with a **deleted** path in the same
commit. With the new root `README.md` present, the old path is never deleted, so
`git diff -M` paired old-`README.md` ↔ new-root-`README.md` (`M`) and classified
`v1/README.md` as `A` — which would have silently broken
`git log --follow v1/README.md`. This is precisely the failure mode the design
reviewer's Optional finding #3 anticipated; the split-commit structure is the
fix. The workflow's "result commit" gate is read as a gate point (all result
commits exist before the next experiment), not a count of one.

**Post-commit confirmation** (run between the move commit `2aba2e9` and the docs
commit): `git log --follow --oneline v1/cargo/src/lib.rs | tail -1` →
`1566d7d make rust folder. move nu code to nu folder.` (the file's original
commit), and `git log --follow --oneline v1/README.md | wc -l` → `35` — both
trace through the move. Check 1 confirmed in full.

## Conclusion

The archive is done: v1 lives frozen in `v1/` with its history intact and its
own architecture record; the root documentation now describes v2 (nutorchd) and
the carried-forward principles; the workflow tooling points at the right paths.
The original layout is one checkout away at `v1-final`.

What we learned:

1. **v1 does not build on the current toolchain** (Xcode 26.4 clang rejects
   libtorch 2.x headers). This is a pre-existing environmental fact, recorded
   here rather than fixed — `v1/` is frozen, and a workaround (e.g.
   `CXXFLAGS=-Wno-error=invalid-specialization` or an older toolchain) can be
   applied by whoever needs a v1 reference build. **Consequence for v2:**
   nutorchd will build against the same torch-sys/libtorch stack, so the v2
   architecture issue must verify toolchain↔libtorch compatibility in its very
   first build experiment, before any design depends on it.
2. **Mechanical verification gates earn their keep** — the zero-tolerance
   reference check caught a defect in a file written minutes earlier, and the
   design review's history-follow concern materialized exactly as predicted.

The next step is **issue 0002: the nutorchd architecture** — wire protocol,
daemon lifecycle, client surface, and a first build experiment that proves the
tch-rs stack compiles on this toolchain.

## Result Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only),
reviewing the pre-commit state (staged move + working-tree docs), per the
workflow's review-before-result-commit gate. **Verdict: APPROVED — no Required,
Optional, or Nit findings.** The reviewer independently reproduced every
verification claim: 111/111 staged entries are pure renames (confirmed
`R100 cargo/src/lib.rs` and `R095 README.md` directly); the tag, layout,
zero-tolerance reference scan, dprint, and symlink checks all reproduce; both
build logs end in the identical libtorch-header error (4 occurrences each), with
the control log compiling from the pre-move `v1-final` worktree — confirming the
Partial classification is honest. It also concretely confirmed the two-commit
rationale (the index simultaneously holds `D README.md` from the rename and
`?? README.md` for the new root file — a combined commit would have re-paired
them and broken `--follow`) and endorsed reading the result-commit gate as a
gate point rather than a count of one.
