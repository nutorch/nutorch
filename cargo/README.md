# Nutorch

Nutorch is a plugin for Nushell which wraps
[tch-rs](https://crates.io/crates/tch), which itself is a wrapper for libtorch,
the C++ backend of [PyTorch](https://pytorch.org/). In other words, **Nutorch is
like PyTorch but for Nushell instead of Python**.

Please find more information about Nutorch at the
[GitHub repository](https://github.com/nutorch/nutorch).

## Testing

Tests are run in nushell.

First, find the test directory from the main project directory: `cd cargo/test`.

Then install the test package in the test directory with `pnpm install`.

Then run `use node_modules/test.nu`.

Then run `test run-tests`.

The nushell tests should then execute, and all should pass.

## Information

Copyright (C) 2025 Identellica LLC
