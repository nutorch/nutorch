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

# Experiment 6: Save and load — the state_dict for nested modules

## Description

The issue's last strand, and the one place shell redirection genuinely cannot
substitute: a nested module's state. Format: **safetensors** (tch's
`Tensor::write_safetensors` / `Tensor::read_safetensors` — the actual exported
names, corrected by review) — the modern interchange standard, readable by
Python's `safetensors` library (cross-verification in this experiment) and the
HF ecosystem; the exp-2 decision record already established that rejecting
VarStore did not foreclose this.

```bash
torch nn save $model weights.safetensors
fresh=$( …same architecture, different init… )
torch nn load $fresh weights.safetensors
# forward($fresh, x) now equals forward($model, x)
```

**Decisions, made here:**

1. **State naming is PyTorch's state_dict scheme**: a leaf module's entries are
   `weight`/`bias` (+ buffer names); Sequential children prefix their index —
   `0.weight`, `2.running_mean`, nesting recursively. A new
   `named_state() -> Vec<(String, &Tensor)>` walks the tree; it includes
   **buffers** — state_dict does, `parameters()` does not. **BatchNorm gains the
   `num_batches_tracked` buffer (int64 scalar, incremented per train-mode
   forward) — design-review finding**: PyTorch's BatchNorm state_dict always
   carries it, so without it BOTH interchange directions break (Python
   `load_state_dict(strict=True)` errors on the missing key; a PyTorch-written
   file hits our all-or-nothing unexpected- key rejection). True parity is the
   point of the feature, so the buffer is added rather than the deviation
   recorded.
2. **`load` is `load_state_dict` into an EXISTING module**: same architecture
   required; every file key must match a module key with an identical shape
   (missing/unexpected/mismatched → `bad_argument` naming the key,
   all-or-nothing — no partial loads); values copy in place under `no_grad` (so
   optimizer views and autograd leaves survive — the live-view contract extends
   through load).
3. **Tensors save from CPU** (safetensors prefers host memory; copied off-device
   at save, moved to MPS at load). The file is portable to Python:
   `safetensors.torch.load_file` reads it — verified in this experiment (the
   `safetensors` pip package is installed into the gitignored `.venv-torch` for
   verification; recorded).
4. **Paths are the user's business**: the daemon writes/reads exactly the path
   given (it is a local, single-user daemon — same trust model as the shell
   itself). Relative paths resolve against the DAEMON's cwd, which is wherever
   it was spawned — the docs say "use absolute paths" and the CLI resolves
   relative paths to absolute before sending (client-side, so the user's cwd
   wins — the only sane semantic).
5. **Wire**: `Bespoke::NnSave { module, path }`,
   `Bespoke::NnLoad
   { module, path }`. CLI: `torch nn save <nn://m> <path>`,
   `torch nn load <nn://m> <path>`.
6. **Error mapping**: a missing/unreadable file at load and an unwritable path
   at save are caller mistakes → `bad_argument` naming the path; other tch I/O
   failures pass through as `torch_error` (the established split).

## Changes

1. **`nutorchd/src/nn.rs`**: `named_state()` (params + buffers, state_dict
   naming); `load_state(&mut self, entries)` (validate-all then copy-in-place
   under no_grad).
2. **`nutorchd/src/protocol.rs`** + **dispatch**: the two variants/arms; unit
   tests: round-trip equality (save → fresh module → load → forward identical);
   name scheme (sequential with batch_norm shows `0.weight`, `1.running_mean`,
   …); shape-mismatch and missing-key errors leave the target UNCHANGED; load
   preserves requires_grad and optimizer-view aliasing (an optimizer built
   before load still steps the loaded weights).
3. **`torch-cli/src/main.rs`**: `nn save`/`nn load` (path resolved to absolute
   client-side).
4. **Python cross-check** (Verification, not a committed artifact):
   `safetensors.torch.load_file` on a daemon-written file yields bitwise-equal
   tensors.
5. **`README.md`**: the save/load lines in the nn section.

## Verification

1. **Hygiene**: standard (goldens untouched — file I/O is unit/live tested, not
   goldened).
2. **Unit tests**: the five in Changes item 2.
3. **Live**: train the regression model (the committed script's shape), save,
   construct fresh, load, verify forward equality via `equal`; load with a
   wrong-architecture target errors naming the key; the Python cross-check reads
   the file and matches values.
4. **The training scripts still pass** (regression guard).

**Pass** = all four. **Fail** = the format required VarStore after all, or load
broke the live-view/optimizer aliasing contract.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**First pass: CHANGES REQUIRED** — 1 Required: our BatchNorm lacked the
`num_batches_tracked` buffer that PyTorch's state_dict always carries, so BOTH
interchange directions would break for any batch_norm-bearing model (Python
strict load errors on the missing key; PyTorch-written files hit our
all-or-nothing unexpected-key rejection — the reviewer verified the exact key
list in Python). The buffer is now added (int64 scalar, train-forward
incremented) — true parity over a recorded deviation, because interchange IS the
feature. Optionals folded: the tch API names corrected to the actual exports
(`Tensor::write_safetensors`/ `read_safetensors` — `write_n`/`read_n` are
internal), and the error mapping pinned (missing file/unwritable path →
`bad_argument` naming the path; other I/O → `torch_error`). The reviewer
confirmed sound: CPU-side save matches the API's host-memory reconstruction;
in-place `copy_` under no_grad preserves the optimizer-aliasing contract;
validate-all-before- copy; client-side path resolution; the silent f64→f32
`copy_` cast matching PyTorch's own load behavior.

## Result

**Result:** Pass

The state_dict travels. Save/load landed exactly as designed, with the
num_batches_tracked buffer making the files genuine PyTorch interchange.

- **Unit tests** (79 daemon tests): the full round trip (seeded model → save →
  fresh different-init model → load → forward EXACTLY reproduces the original,
  batch_norm running stats included); the name scheme asserted verbatim
  (`0.weight … 1.running_mean, 1.running_var,
  1.num_batches_tracked` —
  PyTorch's keys); failed loads (wrong architecture, missing keys) leave the
  target byte-unchanged; missing file → `bad_argument`; **optimizer aliasing
  survives load** (an optimizer built before load still steps the loaded weights
  — the live-view contract through `copy_` under no_grad).
- **Live**: round trip exact (orig == after ≠ before); wrong-architecture load
  errors naming the key (`shape mismatch for bias: module [4],
  file [1]`);
  **the Python cross-check reads the daemon's file** via
  `safetensors.torch.load_file` with matching keys and values (`safetensors`
  pip-installed into the gitignored venv, as recorded); `train-regression.sh`
  still passes.
- **Hygiene**: build 0 warnings; fmt/dprint clean; 255 goldens untouched and
  green; `v1/` untouched.

## Conclusion

The issue's last strand is delivered. Nothing in the object model resisted:
named_state walks the same tree parameters() does, load is the free/ sequential
atomicity invariant applied to files, and safetensors gives interoperability for
free. The issue can close.

## Result Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only),
reviewing the pre-commit working tree. **Verdict: APPROVED — no Required,
Optional, or Nit findings.** The reviewer reproduced everything and went further
than the design asked: the buffer verified live (exactly 3 after 3 train
forwards, read back by Python as int64), the key list verbatim, the round trip
bit-exact, **interchange verified in BOTH directions** (a hand-written PyTorch
state_dict from Python loaded into a daemon module, forward matching PyTorch
exactly), atomicity and error mapping exercised across
missing/unexpected/mismatched keys and missing files, optimizer aliasing
surviving load, client-side relative paths resolving, both training scripts
passing, and registry hygiene (no intermediate-tensor leakage from save/load).
It also noted the fixed-momentum BatchNorm not consuming the counter is itself
PyTorch parity (PyTorch reads it only under momentum=None, a mode we don't
expose). **Close readiness: READY** — the issue's full scope audited as
delivered or honestly excluded, all four recorded decisions honored, all Out
items absent.
