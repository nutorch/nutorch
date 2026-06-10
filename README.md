# Nutorch

**GPU-accelerated PyTorch tensor operations from any shell.**

Nutorch v2 is **nutorchd**: a standalone daemon that owns a tensor database
(backed by [tch-rs](https://github.com/LaurentMazare/tch-rs) / LibTorch, the C++
engine behind [PyTorch](https://pytorch.org/)) and serves thin CLI clients over
a Unix socket. Tensors are referenced by string identifiers, so handles flow
through ordinary pipelines — in bash, zsh, fish, or any other shell.
[Nushell](https://www.nushell.sh/) remains the premium client, with structured
data and native serialization.

```bash
# The shape of things to come (design phase — not yet implemented):
a=$(torch tensor '[1,2,3]' --device mps)
b=$(torch tensor '[4,5,6]' --device mps)
torch add $a $b | torch value
# → [5.0, 7.0, 9.0]
```

## Status

**v2 is in the design phase.** The architecture is being worked out in the open
— see the issue tracker at [issues/README.md](issues/README.md) and the agent
contract / vision in [AGENTS.md](AGENTS.md).

## v1: the proof of concept

Nutorch began as a Nushell plugin — 40 PyTorch operations, GPU acceleration
(CPU/CUDA/MPS), autograd, and neural-network training, all from the Nushell
command line. It proved the core idea that carries v2: tensors stay in a
Rust-owned registry, and the shell passes string handles through pipelines.

v1 is archived, frozen, and fully working in [`v1/`](v1/):

- [v1/README.md](v1/README.md) — user documentation, installation, demos
  (including screenshots in `v1/raw-images/`)
- [v1/AGENTS.md](v1/AGENTS.md) — the v1 architecture record
- [v1/TODO.md](v1/TODO.md) — v1 implementation status and quality tracking

The original pre-archive layout is available at the git tag `v1-final`.

## Copyright

Copyright (C) 2025-2026 Identellica LLC
