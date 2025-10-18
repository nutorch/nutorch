# Nutorch

**PyTorch tensor operations for Nushell** - GPU-accelerated machine learning
from the command line.

Nutorch is a [Nushell](https://github.com/nushell/nushell) plugin that wraps
[tch-rs](https://github.com/LaurentMazare/tch-rs), which itself is a wrapper for
libtorch, the C++ backend of [PyTorch](https://pytorch.org/).

**In other words: Nutorch is to Nushell what PyTorch is to Python.**

---

**Current Status:** 40 PyTorch methods implemented with 100% quality coverage
**Project Stage:** Alpha - Proof of concept, functional for neural network
training

---

## Documentation

- **[README.md](README.md)** - You are here (Getting Started & Installation)
- **[CLAUDE.md](CLAUDE.md)** - Architecture, development guide,
  counter-intuitive facts
- **[TODO.md](TODO.md)** - Implementation status, quality tracking, roadmap
- **[cargo/README.md](cargo/README.md)** - Additional build notes

## Demo

Compare CPU and GPU time for a large matrix multiplication on macOS:

```nu
timeit {torch full [20000, 20000] 1 | torch mm (torch full [20000, 20000] 1) | torch mean | torch value}
timeit {torch full [20000, 20000] 1 --device mps | torch mm (torch full [20000, 20000] 1 --device mps) | torch mean | torch value}
```

If you have an NVIDIA GPU, substitute `mps` with `cuda`:

```nu
timeit {torch full [20000, 20000] 1 --device cuda | torch mm (torch full [20000, 20000] 1 --device cuda) | torch mean | torch value}
```

![Matmul Demo](./raw-images/screenshot-matmul.png)

## Why?

Because Nushell is a shell, not just a programming language, this makes it
possible to operate on tensors on your GPU directly from the command line,
making Nutorch one of the most convenient ways to do data analysis if you spend
a lot of time in the terminal.

## Features

- **Dual Input Pattern**: Commands work both PyTorch-style and Nushell-style
  ([learn more in CLAUDE.md](CLAUDE.md#dual-input-pattern-core-design-principle))
- **40 PyTorch Operations**: Tensor creation, math ops, shape manipulation,
  autograd
- **GPU Acceleration**: Full support for CPU, CUDA, and MPS (Apple Silicon)
- **Autograd Support**: Automatic differentiation and gradient-based
  optimization
- **Shell Integration**: Combine with standard Nushell commands for powerful
  workflows

See [TODO.md](TODO.md) for the complete list of implemented methods and roadmap.

## Installation

### Prerequisites

- You must have [Nushell](https://www.nushell.sh/) installed to use Nutorch.
- You must have [libtorch](https://pytorch.org/get-started/locally/) (PyTorch)
  installed on your system.
- You must have the [Rust toolchain](https://www.rust-lang.org/tools/install)
  installed on your system.

**Nutorch is only tested with macOS at this time.** While it should work on any
platform in principle, it will most likely not work out of the box with Windows
or Linux, and may require customization. **I will happily accept pull requests
if you can make it work on Windows or Linux!**

### Install via Nushell and Cargo

I assume you are using macOS and have installed all the prerequisites.

You can install Nutorch globally or locally. Either way, you will need to know
the absolute path to the `nu_plugin_torch` binary to be able to run it. You will
also need to know the absolute path to your libtorch installation.

First, identify the absolute path to your libtorch installation.

You will need to set three environment variables to use Nutorch:

- `LIBTORCH`: The absolute path to your libtorch installation.
- `LD_LIBRARY_PATH`: The path to the `lib` directory inside your libtorch
  installation.
- `DYLD_LIBRARY_PATH`: The same as `LD_LIBRARY_PATH`, but for macOS.

If you installed Python and PyTorch via Homebrew, the path to your libtorch
installation is likely:

```nu
$env.LIBTORCH = "/opt/homebrew/lib/python3.11/site-packages/torch"
$env.LD_LIBRARY_PATH = ($env.LIBTORCH | path join "lib")
$env.DYLD_LIBRARY_PATH = ($env.LIBTORCH | path join "lib")
```

You can add that code to your Nushell configuration file, or source them in a
local environment.

Next, you will need to install the plugin. Let us assume you want to install it
globally. You can do so by running the following command:

```nu
cargo install nutorch
```

After install, you will need to know the absolute path to the `nu_plugin_torch`
binary. You can find it in the Cargo bin directory, which is:

```
~/.cargo/bin/nu_plugin_torch
```

You can then add the plugin by using the `plugin add` command in Nushell:

```nu
plugin add ~/.cargo/bin/nu_plugin_torch
```

Next, you will need to actually use the plugin, which can be done by running:

```nu
plugin use torch
```

If all is successful, you now have the plugin installed and ready to go, and you
can run test commands, like this:

```nu
[1 2 3] | torch tensor --device mps | torch add (torch tensor [4 5 6] --device mps) | torch value
```

Output:

```
╭───┬──────╮
│ 0 │ 5.00 │
│ 1 │ 7.00 │
│ 2 │ 9.00 │
╰───┴──────╯
```

### Garbage Collection Configuration

After installing the plugin, you may want to lengthen the garbage collection
interval in your Nushell settings:

```nu
$env.config.plugin_gc = {
  plugins: {
    nutorch: {
      stop_after: 10min
    }
  }
}
```

By default, all tensors are deleted (garbage collected by Nushell) after 10
seconds. By increasing this to 10 minutes or longer, this gives you time to
perform other functions before your tensors are deleted from memory.

### Recommended Supplementary Tools

To use Nutorch effectively, it is also recommended to install and use
[Termplot](https://github.com/termplot/termplot), which is a plotting tool
specifically designed to work with Nutorch.

## Usage

Nutorch is a plugin for Nushell that provides a set of commands for working with
tensors. The basic idea is that you load tensors into memory, either on your CPU
or GPU, and perform operations on them, and then read them back to Nushell for
further processing or visualization.

### Basic Usage

```nu
# Load a tensor into memory
let $tensor = torch tensor [1 2 3] --device mps
# Perform operations on the tensor
let $result = $tensor | torch add (torch tensor [4 5 6] --device mps)
# Read the result back to Nushell
$result | torch value
```

### Pipeline-Style Operations

It is common to use pipes with Nutorch commands, which is one of the primary
benefits of Nushell. You can chain commands together to perform complex
operations on tensors all on one line, directly in your terminal.

For instance:

```nu
torch full [1000, 1000] 1 --device mps | torch mm (torch full [1000, 1000] 1 --device mps) | torch mean | torch value
```

That command performs a large matrix multiplication on your GPU and then prints
the mean of the result.

### Dual Input Pattern

Nutorch commands support both PyTorch-style and Nushell-style syntax. This means
you can write commands in a way that feels natural whether you're coming from
PyTorch or Nushell:

```nu
# Pipeline style (Nushell-like)
$tensor1 | torch add $tensor2

# Argument style (PyTorch-like)
torch add $tensor1 $tensor2
```

Both styles work for most operations! For a deep dive on this design principle,
see the
[Dual Input Pattern section in CLAUDE.md](CLAUDE.md#dual-input-pattern-core-design-principle).

### Available Commands

You can see what commands are available by running:

```nu
torch --help
```

The commands are designed to be similar to the PyTorch API, so that wherever
possible you can use the same command names with the same arguments in the same
order as PyTorch. Furthermore, the commands are also designed to be as Nushelly
as possible, meaning you can pipe in input tensors for most commands where
appropriate, making powerful one-liners possible.

## Project Status

**Current Implementation:** 40 PyTorch methods with 100% quality coverage

See [TODO.md](TODO.md) for:

- Full list of implemented methods by category
- Quality metrics and test coverage for each method
- Missing PyTorch methods and roadmap
- Progress toward v1.0 release

## Development

See [CLAUDE.md](CLAUDE.md) for:

- Architecture deep dive and implementation details
- Development workflow and testing procedures
- Counter-intuitive facts about the implementation
- File structure and patterns

## Copyright

Copyright (C) 2025 Identellica LLC
