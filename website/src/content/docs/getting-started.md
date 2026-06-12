---
title: Getting started
description: Install NuTorch with Homebrew and run your first GPU tensor pipeline from any shell.
order: 1
section: Start
---

NuTorch puts GPU tensors in your shell. A daemon (`nutorchd`) owns the tensors
and the GPU; the `torch` CLI sends it one operation per invocation and prints a
**handle** — a plain string — to stdout. Handles flow through ordinary
pipelines, so tensor programs compose the way shell programs always have.

## Install

```bash
brew tap nutorch/nutorch
brew trust nutorch/nutorch   # brew 6.0+ requires trusting third-party taps
brew install nutorch
```

A prebuilt bottle pours in seconds. Requires an Apple-silicon Mac — every tensor
lives on the GPU via Metal (MPS), and that is the point of the library. (No
Homebrew? See [installing from source](/docs/install-from-source/).)

## First tensors

```bash
a=$(torch tensor '[1,2,3]')
b=$(torch tensor '[4,5,6]')
torch add $a $b | torch value
# [5.0,7.0,9.0]   computed on the GPU
```

Three things just happened:

1. **The daemon started itself.** Any `torch` command auto-starts `nutorchd` if
   it isn't running. You never manage it (but you can —
   [the daemon](/docs/daemon/)).
2. **You got handles, not data.** `$a` is a string like `tensor://6c0e3f…`. The
   tensor itself never left the GPU.
3. **The pipeline composed.** `torch add $a $b` printed a new handle;
   `torch value` read it from stdin and printed the data as JSON.

Every operation accepts its leftmost tensor from stdin or as an argument — both
of these work:

```bash
torch add $a $b           # argument form
echo $a | torch add $b    # pipeline form: stdin fills the leftmost slot
```

## A taste of more

```bash
m=$(torch randn '[3,3]')
torch mm $m $m | torch mean | torch value     # matrix product, then mean

w=$(torch randn '[3]' --requires_grad)        # autograd is built in
loss=$(torch mul $w $w | torch sum)
torch backward $loss
torch grad $w | torch value
```

Run `torch ops` to list every operation (185 of them) and `torch <op> --help`
for any one.

## Where to go next

- [The daemon](/docs/daemon/) — lifecycle, idle TTL, status.
- [Tensors and handles](/docs/tensors/) — the dual input pattern, export/import,
  freeing memory.
- [Autograd](/docs/autograd/) and [neural networks](/docs/neural-networks/) —
  training, from the shell.
- [Nushell](/docs/nushell/) — structured data in and out.
