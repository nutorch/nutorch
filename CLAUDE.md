# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

**Nutorch is a proof-of-concept Nushell plugin** that demonstrates how to bring
PyTorch tensor operations to the command line by wrapping tch-rs (Rust bindings
for LibTorch, PyTorch's C++ backend).

**This is NOT production-ready software.** It is an experimental project
exploring the intersection of shell scripting and deep learning, with the goal
of making GPU-accelerated tensor operations available directly in the terminal.

### Project Status & Quality Tracking

**See [TODO.md](TODO.md) for complete implementation status and quality
metrics.**

The TODO.md file tracks:

- **Implementation completeness**: Which PyTorch methods are implemented
  (39/200+)
- **Quality criteria**: Test coverage, helper usage, validation, etc. for each
  method
- **Progress to 1.0**: What needs to be done for each method to be
  production-ready
- **Future roadmap**: PyTorch methods not yet implemented

When working on this project, **always consult TODO.md** to:

1. Check which methods need quality improvements
2. See what tests are missing
3. Understand which patterns need to be applied consistently
4. Track overall project completeness

### The Wrapping Layers

```
Nushell (Shell)
    ↓ nu-plugin protocol (MsgPack)
Nutorch Plugin (Rust)
    ↓ tch-rs bindings
LibTorch (C++)
    ↓ native CUDA/MPS/CPU
Hardware (GPU/CPU)
```

Each layer adds abstraction:

- **Nushell**: Shell with structured data (tables, lists, records)
- **Nutorch**: Plugin converting between Nushell values and tensors
- **tch-rs**: Safe Rust API over unsafe C++ FFI
- **LibTorch**: PyTorch's production C++ library
- **Hardware**: Actual compute devices

## Counter-Intuitive Facts

### 1. **Tensors Never Leave Rust**

Despite Nushell being a data-processing shell, **actual tensor data never
crosses the plugin boundary**. Commands pass UUID strings (like
`"a3f2e8c1-..."`) through pipelines, not tensors.

```nushell
# This looks like it passes a tensor through the pipe
let $x = [1 2 3] | torch tensor

# But $x is actually just a string UUID
$x | describe  # → "string"
```

**Why?** Nushell can only serialize simple types (strings, numbers, lists,
records) via MsgPack. Multi-dimensional arrays with GPU backing don't fit this
model.

### 2. **The Registry is a Memory Leak Risk**

The `TENSOR_REGISTRY` is a global HashMap that never automatically clears
(except via garbage collection). Every tensor operation creates a new UUID
entry:

```nushell
# Each intermediate result creates a registry entry
torch full [1000 1000] 1 | torch mm (torch full [1000 1000] 1) | torch mean
# → 3 tensors stored in registry, only the final ID returned
```

**Mitigation**: Nushell's plugin garbage collection (default: 10 seconds) kills
the plugin process, clearing the registry. Users should configure longer GC
intervals:

```nushell
$env.config.plugin_gc = {
  plugins: {
    nutorch: {
      stop_after: 10min  # Prevent premature tensor deletion
    }
  }
}
```

### 3. **In-Place Operations Aren't Really In-Place**

PyTorch has in-place operations (e.g., `tensor.add_(1)`), but Nutorch can't
expose them idiomatically because:

- Nushell is immutable-by-default
- UUID strings are the interface, not tensor references
- `torch sgd_step` appears to mutate, but actually does internal in-place
  updates while returning the same UUID

```nushell
let $w = torch full [2 2] 1 --requires_grad true
# This returns the same UUID, but modifies the underlying tensor
[$w] | torch sgd_step --lr 0.1
```

**Why it works**: The UUID still points to the same PyTorch tensor object, which
was mutated via `f_sub_()` in Rust.

### 4. **Device Placement is Critical and Manual**

Unlike PyTorch which can auto-transfer tensors, Nutorch **requires explicit
device matching**:

```nushell
# This FAILS - device mismatch
let $cpu_tensor = torch full [2 2] 1
let $gpu_tensor = torch full [2 2] 1 --device mps
$cpu_tensor | torch add $gpu_tensor  # ❌ Error from LibTorch

# Must manually ensure both tensors on same device
let $a = torch full [2 2] 1 --device mps
let $b = torch full [2 2] 1 --device mps
$a | torch add $b  # ✅ Works
```

### 5. **Gradients are Stored in PyTorch's Graph, Not Registry**

The `torch grad` command returns a _new_ UUID pointing to the gradient tensor,
but the gradient itself lives in PyTorch's autograd graph attached to the
original tensor.

```nushell
let $w = torch full [1] 2 --requires_grad true
let $loss = $w | torch mean
$loss | torch backward

# This creates a NEW registry entry for the gradient
let $grad_id = $w | torch grad
# The gradient tensor is cloned from PyTorch's graph into registry
```

### 6. **Binary Name ≠ Package Name**

The Cargo.toml specifies:

```toml
[package]
name = "nutorch" # Crate name

[[bin]]
name = "nu_plugin_torch" # Binary name (Nushell convention)
```

This means:

- `cargo install nutorch` installs the package
- But the binary is `~/.cargo/bin/nu_plugin_torch`
- And Nushell commands use `torch` namespace (not `nutorch`)

### 7. **Tests Use NPM, Not Cargo**

Despite being a Rust project, the test suite uses Nushell's testing framework
via NPM:

```bash
cd cargo/test
pnpm install  # Installs test.nu framework
nu -c "use node_modules/test.nu; test run-tests"
```

This is because tests verify **Nushell command behavior**, not Rust internals.

### 8. **Shape Validation Happens in Rust, Not PyTorch**

Many commands pre-validate tensor shapes before calling tch-rs:

```rust
// command_gather.rs validates dimensions before tch call
if dim < 0 || dim >= tensor_rank {
    return Err(LabeledError::new("Invalid dimension")...);
}
```

**Why?** tch-rs errors from C++ are opaque and crash-prone. Rust-side validation
provides better error messages.

## File Structure

```
nutorch/
├── README.md                    # User-facing documentation
├── LICENSE                      # Apache 2.0
├── NOTICE                       # Copyright notice
├── CLAUDE.md                    # This file (root level)
├── dprint.json                  # Code formatter config
│
├── cargo/                       # ⭐ Main Rust plugin implementation
│   ├── Cargo.toml               # Package: "nutorch", Binary: "nu_plugin_torch"
│   ├── CLAUDE.md                # Old version (now superseded by root)
│   ├── README.md                # Build/development instructions
│   │
│   ├── src/
│   │   ├── main.rs              # Plugin entry point (3 lines)
│   │   ├── lib.rs               # Core: registry, conversions, utilities
│   │   ├── command_*.rs         # 37 command implementations (one per file)
│   │   └── ...                  # Each command is ~100-200 lines
│   │
│   ├── test/
│   │   ├── package.json         # NPM config for test.nu framework
│   │   ├── test_*.nu            # 26 Nushell test files
│   │   └── node_modules/        # test.nu testing framework
│   │
│   ├── chat.md                  # Development history (current)
│   └── chat.archive.*.md        # Historical development discussions
│
├── npm/                         # Nushell-based utilities (higher-level)
│   │
│   ├── nn.nu/                   # Neural network helpers
│   │   ├── package.json         # NPM package metadata
│   │   ├── pyproject.toml       # Python reference implementation metadata
│   │   ├── README.md            # Usage instructions
│   │   ├── mod.nu               # Main module file
│   │   ├── *.nu                 # Training loop helpers, loss functions
│   │   ├── *.py                 # Reference Python implementations
│   │   └── chat.*.md            # Development discussions
│   │
│   └── beautiful.nu/            # Data visualization utilities
│       ├── package.json
│       ├── README.md
│       └── *.nu
│
├── demo/                        # Example usage scripts (untracked)
│   └── *.json                   # Test data files
│
├── raw-images/                  # Screenshots for README
│   └── *.png
│
└── chat.*.md                    # Root-level development discussions (4 archives)
```

### Key File Patterns

#### **Command Implementation Pattern** (`cargo/src/command_*.rs`)

Every command follows this exact structure:

```rust
use nu_plugin::PluginCommand;
use nu_protocol::{Category, Example, LabeledError, PipelineData, Signature, ...};
use crate::{NutorchPlugin, TENSOR_REGISTRY, ...};

pub struct CommandXxx;

impl PluginCommand for CommandXxx {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str { "torch xxx" }
    fn description(&self) -> &str { "..." }
    fn signature(&self) -> Signature {
        Signature::build("torch xxx")
            .input_output_types(vec![...])
            .optional/required/named(...)
            .category(Category::Custom("torch".into()))
    }
    fn examples(&self) -> Vec<Example> { vec![...] }
    fn run(&self, ..., input: PipelineData) -> Result<PipelineData, LabeledError> {
        // 1. Extract input (pipeline XOR argument)
        // 2. Fetch tensor(s) from TENSOR_REGISTRY
        // 3. Perform operation via tch-rs
        // 4. Store result in TENSOR_REGISTRY with new UUID
        // 5. Return UUID as string in PipelineData
    }
}
```

**Naming convention**:

- Struct: `CommandXxx` (PascalCase)
- Command: `torch xxx` (lowercase with spaces)
- File: `command_xxx.rs` (snake_case)

#### **Test Pattern** (`cargo/test/test_*.nu`)

```nushell
use std assert
use std/testing *

@test
def "Test xxx scenario 1" [] {
  let $t = torch xxx [args]
  # Perform operations
  let $result = $t | torch value
  assert ($result == expected)
}

@test
def "Test xxx scenario 2" [] {
  # Another test case
}
```

**Pattern**: Multiple `@test` functions per file, each testing a specific
scenario.

#### **Chat History Pattern**

- `chat.md`: Current work-in-progress discussions
- `chat.archive.N.md`: Archived discussions (numbered sequentially)
- Each subdirectory (`cargo/`, `npm/nn.nu/`) has its own chat files

**Size**: Chat files are 200-500KB, documenting every implementation decision.

## Architecture Deep Dive

### The Tensor Registry (`lib.rs:88-91`)

```rust
lazy_static! {
    pub static ref TENSOR_REGISTRY: Mutex<HashMap<String, Tensor>>
        = Mutex::new(HashMap::new());
}
```

**Critical properties**:

- **Global**: Single process-wide registry
- **Thread-safe**: `Mutex` guards concurrent access (though Nushell plugins are
  single-threaded)
- **String keys**: UUIDs from `uuid::Uuid::new_v4()`
- **Tensor values**: `tch::Tensor` (reference-counted internally by LibTorch)

**Lifecycle**:

1. Command creates tensor via tch-rs
2. UUID generated, tensor inserted: `registry.insert(uuid.clone(), tensor)`
3. UUID returned to Nushell as string
4. Subsequent commands look up tensor: `registry.get(&uuid)`
5. Plugin exit clears entire registry (Nushell GC kills process)

### Data Conversion (`lib.rs:198-336`)

#### **Nushell → Tensor** (`value_to_tensor`)

Handles recursive conversion:

```rust
Value::List → Tensor
  - Flat list [1,2,3] → 1D tensor
  - Nested [[1,2],[3,4]] → 2D tensor (recursive stacking)
  - Deep nesting [[[...]]] → ND tensor
Value::Int/Float → 0D (scalar) tensor
```

**Auto type inference**: All-integer lists → `Int64`, any float → `Float`

#### **Tensor → Nushell** (`tensor_to_value`)

Inverse operation:

```rust
0D tensor → Value::Int or Value::Float
1D tensor → Value::List (flat)
ND tensor → Value::List (nested, recursive)
```

**Element extraction**: Uses `tensor.get(i)` in loop (no direct `to_vec()` in
tch-rs)

### Dual Input Pattern

**Every tensor operation supports both**:

```nushell
# Pipeline input
$tensor | torch xxx

# Argument input
torch xxx $tensor
```

**Implementation** (standard across all commands):

```rust
let piped = match input {
    PipelineData::Value(v, _) => Some(v),
    PipelineData::Empty => None,
    _ => return Err(...),
};
let arg0 = call.nth(0);

match (&piped, &arg0) {
    (None, None) => Err("Missing input"),
    (Some(_), Some(_)) => Err("Conflicting input"),  // XOR enforcement
    _ => Ok(...)
}
```

### Autograd Implementation

#### **Forward Pass**

```nushell
let $x = torch randn [5 3] --requires_grad true
let $w = torch randn [3 2] --requires_grad true
let $y = $x | torch mm $w
```

**Behind the scenes**:

- `--requires_grad true` → `tensor.set_requires_grad(true)` in Rust
- PyTorch builds computation graph automatically
- Each operation creates new tensor with gradient function attached

#### **Backward Pass** (`command_backward.rs`)

```nushell
$loss | torch backward
```

**Implementation**:

1. Validates `loss.numel() == 1` (scalar only)
2. Calls `loss.backward()` (PyTorch computes all gradients)
3. Gradients stored in autograd graph, not registry

#### **Gradient Access** (`command_grad.rs`)

```nushell
let $grad = $w | torch grad
```

**Implementation**:

1. Fetch tensor from registry
2. Call `tensor.grad()` (returns `Tensor`, possibly undefined)
3. Check `grad.defined()` → return `null` if false
4. **Clone gradient into registry** with new UUID
5. Return gradient UUID

**Quirk**: Gradients are cloned/copied into registry, not referenced.

#### **Optimizer** (`command_sgd_step.rs`)

```nushell
[$w1 $w2] | torch sgd_step --lr 0.01
```

**Implementation**:

1. Accepts **list** of parameter UUIDs
2. Fetches all tensors from registry
3. **In-place update** within `tch::no_grad()`:
   ```rust
   p.f_sub_(&(p.grad() * lr))  // p -= lr * grad
   ```
4. Returns **same UUIDs** (tensor objects mutated, not replaced)

## Development Workflow

### Building

```bash
cd cargo
cargo build --release
# Binary: cargo/target/release/nu_plugin_torch
```

### Testing

```bash
cd cargo/test
pnpm install  # First time only
nu -c "use node_modules/test.nu; test run-tests"
```

### Installing Locally

```bash
# From cargo/ directory
cargo build --release
plugin add target/release/nu_plugin_torch
plugin use torch
```

### Environment Variables (macOS with Homebrew Python)

```nushell
$env.LIBTORCH = "/opt/homebrew/lib/python3.11/site-packages/torch"
$env.LD_LIBRARY_PATH = ($env.LIBTORCH | path join "lib")
$env.DYLD_LIBRARY_PATH = ($env.LIBTORCH | path join "lib")
```

**Why needed?** LibTorch is a dynamic library; Rust needs to find it at compile
and runtime.

## Command Categories (37 total)

### Tensor Creation (6)

- `torch tensor` - From Nushell lists
- `torch full` - Filled with value
- `torch randn` - Random normal
- `torch linspace` - Linear spacing
- `torch arange` - Range with step
- _(zeros/ones via `torch full 0/1`)_

### Element-wise Binary Ops (5)

- `torch add` (with `--alpha` for scaled addition)
- `torch sub`
- `torch mul`
- `torch div`
- `torch maximum` - Element-wise max of two tensors

### Element-wise Unary Ops (7)

- `torch neg`
- `torch sin`
- `torch exp`
- `torch softmax`
- `torch log_softmax`
- `torch detach` - Detach from autograd graph
- _(see lib.rs for complete list)_

### Reduction Ops (3)

- `torch mean`
- `torch max` - Return max value
- `torch argmax` - Return index of max

### Matrix Ops (2)

- `torch mm` - Matrix multiplication
- `torch t` - Transpose (2D only)

### Shape Manipulation (7)

- `torch shape` - Get shape as list
- `torch squeeze` - Remove size-1 dims
- `torch unsqueeze` - Add size-1 dim
- `torch reshape` - Change shape
- `torch cat` - Concatenate along existing dim
- `torch stack` - Stack along new dim
- `torch repeat` - Repeat tensor
- `torch repeat_interleave` - Repeat elements

### Indexing (1)

- `torch gather` - Advanced indexing (for loss functions)

### Autograd (4)

- `torch backward` - Backpropagation
- `torch grad` - Access gradients
- `torch zero_grad` - Clear gradients
- `torch detach` - Stop gradient tracking

### Optimization (1)

- `torch sgd_step` - Vanilla SGD update

### Utilities (4)

- `torch value` - Tensor → Nushell
- `torch free` - Remove from registry
- `torch devices` - List available devices
- `torch manual_seed` - Set random seed

## Common Flags

**All tensor creation commands**:

- `--device <cpu|cuda|cuda:N|mps>` (default: cpu)
- `--dtype <float32|float64|int32|int64>` (default: float32)
- `--requires_grad <bool>` (default: false)

**Many operations**:

- `--dim <int>` - Dimension to operate along

## Known Limitations

1. **No automatic broadcasting** - Must manually match shapes
2. **No operator overloading** - Can't use `$a + $b`, must use `torch add`
3. **No Python-style slicing** - Must use specific commands
4. **No model serialization** - Can't save/load trained models yet
5. **macOS-only tested** - Windows/Linux may need customization
6. **Limited activation functions** - Only sin, softmax, log_softmax
7. **No optimization state** - SGD only, no Adam/momentum
8. **No conv/pooling layers** - Linear operations only

## Future Directions (from README TODO)

- [ ] Implement remaining PyTorch tensor operations
- [ ] Add `tch::nn` module wrapper (layers, optimizers)
- [ ] Model serialization (save/load)
- [ ] More optimizers (Adam, RMSprop)
- [ ] Convolutional and pooling operations
- [ ] Better dimension validation across all commands
- [ ] Windows/Linux compatibility testing

## Design Philosophy

1. **PyTorch API compatibility** - Command names and argument order match
   PyTorch where possible
2. **Nushell idioms** - Pipeline-friendly, structured data output
3. **Explicit over implicit** - Manual device placement, no auto-casting
4. **Proof-of-concept first** - Focus on demonstrating feasibility, not
   completeness
5. **Shell-native deep learning** - Make GPU programming accessible from
   terminal
