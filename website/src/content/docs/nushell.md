---
title: Nushell
description: A generated module gives Nushell native structured data over the same daemon — tables in, tables out.
order: 7
section: Clients
---

Bash gets pipelines; [Nushell](https://www.nushell.sh/) gets pipelines _and_
structure. A generated module wraps every operation so tensors come from and
return to native Nushell values — no JSON wrangling.

## Setup

**If you installed NuTorch with Homebrew, there is nothing to set up.** The
formula ships a vendor-autoload stub, Nushell sources it at startup, and
`nutorch` commands are simply there in every new session — no `use` line, no
`config.nu` edit. (Mechanism: Nushell autoloads every `.nu` file in
`$nu.vendor-autoload-dirs`, and brew-built Nushell scans
`$(brew --prefix)/share/nushell/vendor/autoload`.)

The manual fallback — for from-source installs, or a Nushell that was not
installed by Homebrew (cargo, MacPorts, nightly builds) and so may not scan the
brew prefix:

```nu
# one-time: autoload in every session
mkdir ~/.config/nushell/autoload
'use "/opt/homebrew/share/nutorch/nutorch.nu" *'
  | save ~/.config/nushell/autoload/nutorch.nu

# or just this session
use /opt/homebrew/share/nutorch/nutorch.nu *
```

A current copy of `nutorch.nu` also ships in the repo root; regenerate it any
time with:

```nu
torch nu-module | save -f nutorch.nu
```

## Structured data in and out

```nu
let t = ([[1 2] [3 4]] | nutorch tensor)
$t | nutorch mm $t | nutorch value            # a native table
nutorch tensors | where bytes > 1mb | get handle | each {|h| nutorch free $h }
```

Wrappers are pipeline-first — the first tensor slot is `$in` — and every one of
the 185 ops is wrapped, plus the registry and daemon verbs. Non-finite values
cross the boundary as REAL Nushell NaN/infinity floats; the JSON dialect
(`"NaN"`, `"Infinity"`, `"-Infinity"`) is handled for you in both directions.

## Training, natively

The regression from [neural networks](/docs/neural-networks/), as Nushell:

```nu
use nutorch.nu *

nutorch manual_seed 42 | ignore
let x = ([[0.0] [1.0] [2.0] [3.0]] | nutorch tensor)
let y = ([[1.0] [3.0] [5.0] [7.0]] | nutorch tensor)
let model = (nutorch nn linear 1 1)
let opt = (nutorch nn sgd $model --lr 0.05)

for i in 1..200 {
  let loss = ($x | nutorch forward $model | nutorch mse_loss $y)
  $loss | nutorch backward
  $opt | nutorch step
  nutorch nn zero_grad $opt | ignore
}
```

The full script (with assertions) is `scripts/train-regression.nu` in the repo —
it converges to the same losses as its zsh twin.

## Plain JSON everywhere else

The structured verbs also serve any other tool via `--json`:

```bash
torch tensors --json
torch ops --json
torch daemon status --json
torch nn info $m --json
```
