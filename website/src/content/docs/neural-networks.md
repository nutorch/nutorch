---
title: Neural networks
description: Modules and optimizers are daemon-resident objects with their own handles. Compose, train, and save PyTorch-compatible models from the shell.
order: 6
section: Deep learning
---

Modules are daemon-resident objects with `nn://` handles; optimizers get
`optim://`. You compose them, run them, and train them from the shell — and the
result interchanges with PyTorch via safetensors.

## Building modules

```bash
l=$(torch nn linear 2 3)                         # PyTorch-default init, seeded
m=$(torch nn sequential $l "$(torch nn relu)")   # consumes the child handles
y=$(torch forward $m $x)                         # or: $x | torch forward $m
torch nn parameters $m                           # tensor:// handles — LIVE views
torch nn info $m                                 # architecture, param counts
torch nn info $m --json                          # the same, as JSON
```

Module kinds (19): `linear`, `conv1d`, `conv2d`, `conv_transpose2d`,
`embedding`, `layer_norm`, `batch_norm`, `group_norm`, `dropout`, `relu`,
`sigmoid`, `tanh`, `gelu`, `leaky_relu`, `softmax`, `max_pool2d`, `avg_pool2d`,
`flatten`, `sequential`.

`torch nn train $m` / `torch nn eval $m` switch dropout and batch-norm behavior.
Pretrained weights load at construction —
`torch nn linear 2 3 --weight $w --bias-tensor $b` (deep-copied; your tensors
are never mutated).

## Parameters are live views

The handles from `torch nn parameters` alias the module's actual weights.
Gradients populate through them after `backward`; optimizer steps are visible
through them. That is the live-view contract — no copying, no syncing.

## Training

A full regression — fit y = 2x + 1 — in a handful of lines:

```bash
torch manual_seed 42
x=$(torch tensor '[[0.0],[1.0],[2.0],[3.0]]')
y=$(torch tensor '[[1.0],[3.0],[5.0],[7.0]]')
model=$(torch nn linear 1 1)
opt=$(torch nn sgd $model --lr 0.05)

for i in $(seq 200); do
  pred=$(torch forward $model $x)
  loss=$(torch mse_loss $pred $y)
  torch backward $loss
  torch step $opt
  torch nn zero_grad $opt
done
torch value $loss     # 6.0012 → 2.46e-7 on the reference run
```

Optimizers: `sgd`, `adam`, `adamw`, `rmsprop` — with the PyTorch defaults and
flags (momentum, weight decay, betas, …); usage errors show each one's
parameters.

## Saving and loading

```bash
torch nn save $m model.safetensors        # state_dict, buffers included
torch nn load $fresh model.safetensors    # into a same-architecture module
```

The format is safetensors with PyTorch's state-dict naming (including
`num_batches_tracked` for batch norm), so checkpoints move between NuTorch and
PyTorch in both directions.

Complete runnable scripts live in the repo: `scripts/train-regression.sh`,
`scripts/train-classify.sh`, and the Nushell twin `scripts/train-regression.nu`.
