+++
status = "open"
opened = "2026-06-10"
+++

# Issue 1: Archive v1 (the Nushell plugin) to make room for v2 (nutorchd)

## Goal

Move the complete v1 implementation — the Nushell plugin, its tests, the
Nushell-based helper packages, and the v1 documentation — into a frozen `v1/`
folder, and rewrite the top-level documentation (`AGENTS.md`, `README.md`) to
describe the v2 direction, so the repository root is ready for nutorchd
development.

## Background

Nutorch v1 is a working proof of concept: a Nushell plugin (`cargo/`, 40 PyTorch
methods) that keeps tensors in an in-process registry and passes UUID string
handles through Nushell pipelines, plus Nushell helper packages (`npm/nn.nu`,
`npm/beautiful.nu`).

v1 proved the core idea but has a structural ceiling: the tensor registry lives
inside a plugin process whose lifetime is controlled by Nushell's plugin garbage
collector, and the plugin protocol limits the audience to Nushell users.

**v2 (nutorchd)** decouples the registry from the shell: a standalone daemon
owns the tensors (and the LibTorch context, GPU memory, and autograd graphs),
and a thin CLI client passes string handles over a Unix socket. Any shell —
bash, zsh, fish, Nushell — becomes a client; Nushell remains the best one. The
v1 insight (string handles as the interface, tensors never leaving Rust) is the
load-bearing idea that carries forward; the Nushell plugin protocol was just its
first transport.

v1's code is not dead weight — it is the **reference implementation** for v2.
The 40 `command_*.rs` files contain the validation logic, dual-input parsing,
and tch-rs call patterns that will be ported into the daemon's dispatcher. This
is why v1 is archived **in-tree** (like a vendored upstream) rather than on a
branch: the port works best with the source sitting next to the destination.

## Analysis

### What moves into `v1/`

| Current path     | New path         | Notes                                |
| ---------------- | ---------------- | ------------------------------------ |
| `cargo/`         | `v1/cargo/`      | Plugin crate, tests, chat archives   |
| `npm/`           | `v1/npm/`        | nn.nu, beautiful.nu                  |
| `TODO.md`        | `v1/TODO.md`     | v1 quality/completeness tracker      |
| `raw-images/`    | `v1/raw-images/` | Screenshots of the v1 plugin UX      |
| `README.md`      | `v1/README.md`   | v1 user docs, with an archive notice |
| (from AGENTS.md) | `v1/AGENTS.md`   | v1 architecture sections, lifted out |

All moves use `git mv` so history follows the files (`git log --follow`).

### What stays at the root

Everything version-independent: `AGENTS.md`/`CLAUDE.md` (symlink), `issues/`,
`skills/`, `scripts/`, `docs/` (historical chat archives), `LICENSE`,
`dprint.json`, `.claude/`, `.gitignore`.

### Frozen, not immutable-by-tooling

`v1/` is a historical reference in the same spirit as closed issues: do not
develop in it. The one allowed class of change is mechanical (e.g. a path fix
required by repo tooling). v2 work never edits v1 code — it ports from it.

### Documentation restructuring

- **Root `AGENTS.md`**: keeps Rules, the Issues and Experiments workflow, and
  Remember; gains a v2 Vision section (nutorchd architecture) and a
  carried-forward design principles section; the v1-specific architecture
  content (wrapping layers, counter-intuitive facts, command pattern, command
  categories, plugin build/test instructions) moves to `v1/AGENTS.md`.
- **Root `README.md`**: new, brief — v2 direction, status (design phase, see
  `issues/`), pointer to `v1/` for the working proof of concept.
- **Skills and the reviewer agent**: literal paths that would dangle
  (`cargo/test`, `npm/nn.nu/*.py`, etc.) are updated to their `v1/` locations or
  generalized to "the active crate" where they describe v2 gates that do not
  exist yet.

### Tagging

The last pre-move commit is tagged `v1-final` so the original layout is one
checkout away.

## Experiments

- [Experiment 1: Move v1 into `v1/` and re-point the documentation](01-move-v1-and-repoint-docs.md)
  — **Partial** (archive, docs, and history all land; the v1 build fails on the
  current toolchain — proven pre-existing via a control build at `v1-final`)
