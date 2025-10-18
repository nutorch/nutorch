# Nutorch

Nutorch is a [Nushell](https://github.com/nushell/nushell) plugin that wraps
[tch-rs](https://github.com/LaurentMazare/tch-rs), which itself is a wrapper for
libtorch, the C++ backend of [PyTorch](https://pytorch.org/).

In other words, **Nutorch is to Nushell what PyTorch is to Python.**

## Why?

Because Nushell is a shell, not just a programming language, this makes it
possible to operate on tensors on your GPU directly from the command line,
making Nutorch one of the most convenient ways to do data analysis if you spend
a lot of time in the terminal.

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

## Installation

### Prequisites

- You must have [Nushell](https://www.nushell.sh/) installed to use Nutorch.
- You must have [libtorch](https://pytorch.org/get-started/locally/) (PyTorch)
  installed on your system.
- You must have the [Rust toolchain](https://www.rust-lang.org/tools/install)
  installed on your system.

**Nurtorch is only tested with macOS at this time.** While it should work on any
platform in principle, it will most likely not work out of the box with Windows
or Linux, and may require customization. **I will happily accept pull requests
if you can make it work on Windows or Linux!**

### Install via Nushell and Cargo

I assume you are using macOS have have installed all the prerequisites.

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

You can add that code to your Nushell configuration file, or souce them in a
local enviornment.

Next, you will need to install the plugin. Let us assume you want to install it
globally. You can do so by running the following command:

```nu
cargo install nutorch
```

After install, you should will need to know the absolute path to the
`nu_plugin_torch` binary. You can find it in the Cargo bin directory, which is:

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

### Recommended Supplementary Tools

To use Nutorch, it is also recommended to install and use
[Termplot](https://github.com/termplot/termplot), which is a plotting tool
specifically designed to work with Nutorch.

### Garbage Collection

After installing the plugin, you may want to lengthen the garbage collection
interval in your nushell settings:

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

## Usage

Nutorch is a plugin for Nushell that provides a set of commands for working with
tensors. The basic idea is that you load tensors into memory, either on your CPU
or GPU, and perform operations on them, and then read them back to Nushell for
further processing or visualization.

The basic usage is as follows:

```nu
# Load a tensor into memory
let $tensor = torch tensor [1 2 3] --device mps
# Perform operations on the tensor
let $result = $tensor | torch add (torch tensor [4 5 6] --device mps)
# Read the result back to Nushell
$result | torch value
```

It is common to use pipes with Nutorch commands, which is one of the primary
benefits of Nushell. You can chain commands together to perform complex
operations on tensors all on one line, directly in your terminal.

For instance:

```nu
torch full [1000, 1000] 1 --device mps | torch mm (torch full [1000, 1000] 1 --device mps) | torch mean | torch value
```

That command performs a large matrix multiplication on your GPU and then prints
the mean of the result.

You can see what commands are available by running:

```nu
torch --help
```

The commands are designed to be similar to the PyTorch API, so that wherever
possible you can insert the same commands with the same names in the same order
as PyTorch. Furthermore, the commands are also designed to be as Nushelly as
possible, meaning you can pipe in input tensors for most commands where
appropriate, make powerful one-liners possible.

## TODO

Nutorch is an alpha-quality project. Currently, the existing set of commands are
technically adequate to train neural networks. However, the vast majority of the
PyTorch API is not yet implemented. The following is a list of commands that are
currently implemented, and those that are planned for future implementation.

### Command Attributes

For MVP:

- [x] First pass: Make neural network work
- [ ] Second pass: Update all tensor methods and test them
- [ ] Third pass: Look for any issues with tensor methods.

For manipulating tensors:

- [ ] Always check dimensionality makes sense if necessary
- [ ] Always take a tensor as input if possible
- [ ] Always take a list of tensors as input if PyTorch takes a list or var args
      for arguments
- [ ] Always take a tensor as argument if possible
- [ ] Always take a list of tensors as argument if PyTorch takes a list or var
      args for arguments

For creating tensors:

- [ ] Always take `dtype` as an argument if possible
- [ ] Always take `device` as an argument if possible
- [ ] Always take `requires_grad` as an argument if possible

### Commands

- [x] manual_seed
- [x] linspace
- [x] randn
- [x] mm
- [x] full
- [x] tensor
- [x] mul
- [x] add
- [x] sub
- [x] div
- [x] neg
- [x] gather
- [x] squeeze
- [x] unsqueeze
- [x] detach
- [x] arange
- [x] stack
- [x] repeat
- [x] repeat_interleave
- [ ] all other tch tensor operations
- [ ] tch nn module
- [x] add autograd setting to torch.tensor
- [x] add autograd setting to torch.randn
- [x] add autograd setting to torch.full
- [x] add autograd setting to torch.mm
- [x] add autograd setting to torch.linspace

## Copyright

Copyright (C) 2025 Identellica LLC
