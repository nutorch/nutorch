+++
status = "open"
opened = "2026-06-12"
+++

# Issue 15: Shell tabs on every example

## Goal

Every applicable code example on the website toggles between **bash / zsh** and
**Nushell** — the "bun vs. node.js" docs pattern — with ONE site-wide
preference: toggle anywhere, remembered everywhere (localStorage), default
**bash**, all tab groups on a page flipping together. The control looks exactly
like the homepage hero tabs, and the logic lives in ONE place — no per-example
reimplementation.

## Background

Issue 0013 experiment 5 built the prototype: the hero's two-tab control
(`bash / zsh` | `Nushell`), localStorage persistence (`hero-shell` key), panels
swapped by a few lines of vanilla JS, active styling keyed on `aria-selected`.
This issue generalizes it site-wide.

Decisions taken with the user before opening (2026-06-12):

1. **Authoring is PAIRED FENCES** (chosen over MDX): in docs markdown, a
   `` ```bash `` fence immediately followed by a `` ```nu `` fence renders as
   one tab group. Authors keep writing plain markdown; the honesty checker keeps
   seeing ordinary fences; no MDX dependency, no content-format migration.
2. **The hero unifies** into the same system: one storage key (`shell`, values
   `posix`/`nu`) shared by the hero tabs and every docs tab group. The
   `hero-shell` key from experiment 0013/5 is retired (migrated if present).
3. **Live sync**: clicking any toggle flips every tab group on the page in the
   same instant, and the saved preference applies everywhere else on next load.

## Analysis

- **Mechanism**: Astro's markdown pipeline runs remark → rehype with Shiki
  highlighting in between; a small REHYPE plugin (registered in
  `astro.config.mjs`) walks the docs HTML tree for adjacent
  `<pre data-language="bash">` + `<pre data-language="nu">` siblings and wraps
  each pair in tab-group markup (`role="tablist"` buttons + two panels). The
  buttons/panels reuse the exact classes the hero control established; ONE
  shared script (in the docs layout / Base) drives every group via event
  delegation — the abstract-component ask, realized at the pipeline level since
  plain `.md` cannot embed components.
- **The hero refactors** to emit the same markup (or adopts the shared script
  with its existing markup) so there is one implementation; its storage key
  migrates.
- **Content work**: the bash examples across getting-started, daemon, tensors,
  autograd, neural-networks, and ops get Nushell twins — each twin RUN LIVE
  before display via the discriminating explicit-`use` form (the issue-0014
  lesson: `nu -c` skips autoload, and pipeline-form module calls cannot be
  satisfied by the CLI symlink).
- **Exemptions** (no toggle): the brew install block (shell-agnostic); the
  homepage "See it run" side-by-side (user: fine as-is); the Nushell docs page's
  nu-only examples; the generated reference's usage fences (CLI output, not
  shell examples); `value --meta` style export/import where the bash form IS the
  example — judged per example, recorded in the experiment.
- **Gates extend**: the CDP harness grows a tab-sync matrix (default bash
  everywhere; click one group → all groups flip; preference persists across
  pages; hero and docs share the key); the honesty checker already scans `nu`
  fences (the new twins are covered automatically); the brand gate and fence
  byte-integrity arguments are unaffected because the plugin works on the BUILT
  tree, not the markdown source.

## Experiments

- [Experiment 1: The mechanism — paired fences become synced tabs](01-the-mechanism.md)
  — **Pass** (plugin + one shared script + unified key; fence-level identity
  across all 20 pages; CDP matrix incl. both-groups-one-click, hero cross-page,
  legacy migration; the node_modules/.astro cache surprise recorded)
- [Experiment 2: The twins — Nushell pairs for the remaining examples](02-the-twins.md)
  — **Designed**

## Scope

In: the rehype tab-group plugin; the shared toggle script + unified storage key
(hero migration included); Nushell twins for the applicable docs examples,
live-verified; the extended CDP gate; both-mode screenshots.

Out (recorded): MDX (rejected — paired fences chosen); the homepage side-by-side
section (stays); the reference pages' usage fences; website deployment (its own
future issue).
