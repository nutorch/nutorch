+++
status = "closed"
opened = "2026-06-12"
closed = "2026-06-12"
+++

# Issue 14: `nutorch` in Nushell, with nothing to type

## Goal

`nutorch` commands work in Nushell out of the box — no `use nutorch.nu *` at the
start of every session. Open a new Nushell, type `nutorch tensor '[1,2]'`, and
it works.

No adversarial review for this issue (user decision, 2026-06-12) — experiments
run design → plan commit → implement → result commit, with verification carrying
the weight.

## Background

The v2 Nushell story is the generated module `nutorch.nu`: rich wrappers
(`nutorch tensor`, `nutorch mm`, …) that take and return native Nushell values
and call `^torch` underneath. But a module only exists in scope after
`use nutorch.nu *` — which today must be typed (or put in `config.nu`) by hand.
The user's actual request was zero-setup availability in Nushell.

(A first version of this issue misread the request as "make `nutorch` a CLI name
in every shell" and shipped a `nutorch → torch` symlink through `install.sh` and
the formula before being corrected. That experiment record was removed at user
direction; the symlink changes themselves remain in the tree and in history —
harmless, and unrelated to this goal.)

## Analysis

Two mechanisms, not mutually exclusive:

1. **User-level**: one line in `config.nu` —
   `use /opt/homebrew/share/nutorch/nutorch.nu *`. Works today, but it is
   per-user setup: exactly the thing the goal wants to eliminate.
2. **Package-level (the real fix): Nushell vendor autoload.** Nushell sources
   every `.nu` file in `$nu.vendor-autoload-dirs` at startup —
   `$(brew --prefix)/share/nushell/vendor/autoload` is on that list for
   brew-installed Nushell. This is how starship, zoxide, and carapace ship
   zero-config Nushell integration. The nutorch formula installs a one-line stub
   there (`use ".../share/nutorch/nutorch.nu" *`), and `brew install nutorch`
   makes `nutorch` work in every new Nushell session with no config edit.

Open questions for the experiment design:

- **The contract the design rests on is prefix-relative, not machine-specific**:
  Homebrew builds Nushell with its vendor-autoload directory pinned to the same
  `HOMEBREW_PREFIX` every formula installs into, so the stub's location and
  Nushell's search path are derived from one variable that brew resolves per
  machine (`/opt/homebrew` on Apple silicon, `/usr/local` on Intel). The
  agreement holds on every brew-installed pairing BY CONSTRUCTION. (Verified
  once locally as a premise smoke test — `$nu.vendor-autoload-dirs` includes the
  brew prefix path and an autoloaded `use … *` exports into session scope; the
  local check can falsify the claim, not prove it.)
- **The fallback story for machines outside that contract**: Nushell installed
  NOT via brew (cargo, MacPorts, nightly) may never scan the brew prefix, and
  versions predating vendor autoload lack the mechanism entirely. The design
  must say what those users do (the documented one-line `use` in `config.nu`, or
  a file in `$nu.user-autoload-dirs`) and the docs must say which mechanism
  applies when.
- `install.sh` parity: from-source installs have no brew-built Nushell guarantee
  at all — likely reduces to the documented fallback above.
- The published tap and bottle pick the stub up with the next release (same
  precedent as the MIT metadata and the CLI symlink).

## Experiments

- [Experiment 1: The vendor-autoload stub](01-vendor-autoload.md) — **Pass**
  (formula-written stub, brew-linked; fresh unconfigured nu session ran the
  module; fallback documented and printed by install.sh)

## Conclusion

**The goal is met — with a corrected verification record** (see the experiment's
Correction section). A new INTERACTIVE Nushell session has `nutorch` commands
with zero user configuration: the formula ships a one-line vendor-autoload stub
that Nushell sources at config-loading startup, and the brew-built `nu` bakes
the brew-prefix autoload dir in at compile time — a contract even sturdier than
the spine's original framing, holding regardless of environment. The originally
recorded `nu -c` proof was a FALSE POSITIVE (the output came from the same-named
CLI symlink; `-c` skips config and therefore autoload entirely) — the corrected
evidence is module-command existence in interactive sessions, confirmed by the
user in his REPL. Scripts and `-c` one-liners load the module explicitly with
`use`, the right practice for scripts anyway. Users outside the contract
(non-brew Nushell, from-source installs) get a documented one-line fallback,
printed by `install.sh` and spelled out on the Nushell docs page. The published
tap picks the stub up with the next tagged release. (The issue's first draft — a
CLI symlink built on a misread request — was removed at user direction; its code
remains harmlessly in the tree, now brew-managed after the rebuild.)

## Scope

In: the vendor-autoload stub in `dist/nutorch.rb`; whatever `install.sh` parity
is reasonable; docs (Nushell page: replace the manual `use` framing with "it's
just there" + the manual line as fallback); local hand-applied stub so the
user's machine gets the behavior now.

Out (recorded): republishing the tap / re-bottling before the next release;
reverting the issue's earlier CLI-symlink side effect (separate concern, the
user has not asked); any plugin mechanism (v1's dead end).
