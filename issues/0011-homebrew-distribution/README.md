+++
status = "open"
opened = "2026-06-11"
+++

# Issue 11: Homebrew distribution — nutorch installs like a normal package

## Goal

`brew tap ryanxcharles/nutorch && brew install nutorch` puts working `torch` and
`nutorchd` binaries (and `nutorch.nu`) on a Mac in seconds, with libtorch
vendored invisibly — no Python venv dance, no Rust toolchain, no checkout. The
first real version of nutorch exists: tagged, stamped (`torch --version`),
reproducible.

## Background — the channel decision

Three channels were weighed (recorded from design discussion):

1. **GitHub clone-and-build** — the substrate; nearly works today but the
   binaries are non-relocatable (rpaths point into the gitignored `.libtorch`
   venv symlink) and the bootstrap is folklore in a Cargo.toml comment. Made
   excellent as part of this issue, since every other channel builds on it.
2. **crates.io — REJECTED as an install path, recorded.** `cargo install` cannot
   work out of the box on the only platform nutorch supports: tch's
   `download-libtorch` is broken on Apple Silicon (the issue-0002 discovery), so
   users would hit a build error unless they had already done the libtorch dance
   manually — at which point cloning the repo is strictly better. Publishing
   source crates for discoverability is a recorded possible follow-up, never the
   install story.
3. **Homebrew — the primary channel (user decision).** Best UX/DX by far for the
   actual audience (Mac users; nutorch is Mac/MPS-only by design). A **formula**
   in a **personal tap** — not a cask (casks are prebuilt GUI bundles), not
   Homebrew core (notability gates; maybe later).

## Decisions Already Made

1. **libtorch is VENDORED** (user decision: install like a normal package). The
   version pairing is strict — tch 0.24.0 ↔ torch 2.11.0 — and
   `depends_on "pytorch"` was rejected because brew's pytorch version drifts and
   one upgrade would break the pairing.
2. **No big files in any git repo.** A brew formula carries only `url` +
   `sha256` pairs; brew downloads and verifies at install time:
   - **libtorch bytes come from PyPI directly** — the formula declares the torch
     2.11.0 macOS-arm64 wheel as a resource (wheels are zips; the formula
     extracts `torch/lib/*.dylib` into nutorch's own keg). PyPI artifacts are
     immutable, so the URL+hash pin is permanent — the same trick the
     `.venv-torch` bootstrap does, executed by brew.
   - **Prebuilt nutorch bytes are BOTTLES on GitHub Release assets** (tap repo
     releases; assets free, 2GB/each) — seconds-long binary installs with no
     Rust toolchain. The source-build path remains the auditable fallback (brew
     auto-installs rust as a build dep).
   - **Git LFS explicitly rejected**: quotas, cost, and brew has no affordance
     for it.
3. **Bottles start as locally-built** ("Ryan's laptop builds it") — honest for a
   personal tap; CI bottling is a recorded follow-up.

## Scope

In: relocatable binaries (rpaths to a stable install location, not the
checkout); a bootstrap/install script replacing the Cargo.toml folklore;
`torch --version` (and a version story — everything is 0.0.1 today; the first
tag is presumably v0.1.0); the wheel-extraction logic shared between bootstrap
and formula; the tap repository with the formula (source-build proven end to
end); bottling to a GitHub Release; `nutorch.nu` installed somewhere `use`-able;
docs (README install section).

Out (recorded): crates.io publication; Homebrew core submission; CI bottling;
Linux/anything-not-Mac (issue 0003's contract); auto-update/upgrade machinery
beyond what brew gives for free.

## Design Questions (settled per-experiment)

1. **The rpath/install layout**: where do installed binaries expect libtorch
   (`@loader_path/../libtorch/lib`? a keg-relative path?), and does ONE layout
   serve both the brew keg and a manual `scripts/install.sh` destination — or do
   we re-rpath at install time (`install_name_tool`)?
2. **The wheel slimming question**: the formula can copy all of `torch/lib` or
   just the dylibs nutorch actually links — measure what the binaries dlopen and
   decide (smaller keg vs. fragility to tch's internal linking).
3. **Version stamping**: compile-time env (`CARGO_PKG_VERSION` + git sha?) and
   whether `nutorchd` reports it via `daemon status` too.
4. **The tap repo's relationship to this repo**: separate `homebrew-nutorch`
   repository (the brew convention) — how its formula updates are driven from
   nutorch releases (manually first; recorded).
5. **codesigning/quarantine**: do downloaded bottles hit Gatekeeper on Apple
   Silicon (ad-hoc signatures may suffice for CLI binaries — verify, decide,
   record).
