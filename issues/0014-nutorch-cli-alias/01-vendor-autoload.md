+++
[implementer]
agent = "claude-code"
model = "claude-fable-5"

[review]
waived = "user decision 2026-06-12: no adversarial review for issue 0014"
+++

# Experiment 1: The vendor-autoload stub

## Description

`brew install nutorch` makes `nutorch` work in every new Nushell session with
zero user configuration. The mechanism is Nushell's vendor autoload: Nushell
sources every `.nu` file in `$nu.vendor-autoload-dirs` at startup, and the
brew-built Nushell pins that list to the same `HOMEBREW_PREFIX` every formula
installs into — the prefix-relative contract recorded in the issue spine. The
formula ships a one-line stub there; the stub `use`s the module; the user types
nothing.

**Decisions, made here:**

1. **The formula writes the stub at install time** (it is not in the source
   tarball — `install` generates it):
   ```ruby
   (share/"nushell/vendor/autoload/nutorch.nu").write <<~EOS
     use "#{opt_pkgshare}/nutorch.nu" *
   EOS
   ```
   `opt_pkgshare` (`$(brew --prefix)/opt/nutorch/share/nutorch`) is the
   version-stable path brew maintains across upgrades — the stub never goes
   stale on version bumps, and works even while the keg is unlinked. The
   `test do` block gains an existence assertion for the stub (GPU-free, no
   Nushell dependency — actually RUNNING Nushell stays out of the formula test;
   the live gate below covers behavior).
2. **The end-to-end proof is a real reinstall on this machine**: remove the
   hand-placed stub from the previous session (it would collide with brew's link
   step), temp-copy `dist/nutorch.rb` over the local tap's formula (uncommitted,
   never pushed),
   `brew reinstall --build-from-source
   nutorch/nutorch/nutorch` (the install
   method runs with the NEW formula against the pinned v0.1.0 tarball — the stub
   is install-time output, so no new release is needed to prove it), then
   `nu -c "nutorch tensor
   '[1,2]' | nutorch value"` in a fresh
   non-interactive Nushell with a private TMPDIR. The local tap file is reverted
   afterward (tap git-clean; published tap untouched until the next release, per
   the spine).
3. **From-source installs get the documented fallback, not an imposed one**:
   `install.sh` PRINTS the one-liner for `$nu.user-autoload-dirs`
   (`~/.config/nushell/autoload/nutorch.nu` by default) instead of writing into
   the user's config — an installer that edits shell config uninvited is the
   wrong kind of magic. The hint is one echo line.
4. **Docs say which mechanism applies when** (the spine's fallback question):
   the Nushell docs page leads with "Homebrew installs: there is nothing to set
   up" and keeps the manual `use` (config.nu or user-autoload file) as the
   fallback for from-source installs and non-brew Nushell builds
   (cargo/MacPorts/nightly — they may not scan the brew prefix; pre-autoload
   Nushell versions lack the mechanism entirely). The README's Nushell section
   gets one clause.
5. **The hand-placed stub from the earlier session is RETIRED by this
   experiment**: deleted before the reinstall, replaced by the brew-managed
   keg-linked file — no rogue files left for a future `brew link` to fight.

## Changes

1. **`dist/nutorch.rb`**: the stub write in `install`; the existence assertion
   in `test do`.
2. **`scripts/install.sh`**: one echo line hinting the user-autoload one-liner.
3. **`website/src/content/docs/nushell.md`**: setup section reframed (autoload
   first, manual fallback second); **`README.md`**: one clause.
4. **Local machine**: hand stub removed; brew-managed stub in its place via the
   reinstall.
5. **Published tap untouched** (next release); no Rust; no `v1/`.

## Verification

1. **The zero-config proof**: after the reinstall, a FRESH
   `nu -c "nutorch
   tensor '[1,2]' | nutorch value"` prints `[1.0,2.0]` with
   no `use` typed and no config edited; `nu -c "nutorch daemon stop"` cleans up.
   The stub exists at
   `$(brew --prefix)/share/nushell/vendor/autoload/nutorch.nu`, brew-linked (a
   symlink into the keg, not a loose file), and references the `opt` path.
2. **brew test green** with the new assertion (temp-copied formula).
3. **Tap hygiene**: local tap git-clean after the revert; the raw published
   formula on GitHub still has no stub (unchanged until next release).
4. **install.sh hint**: present in output; the printed one-liner is valid syntax
   (executed once against a scratch user-autoload dir via `XDG_CONFIG_HOME`
   override or equivalent, then removed).
5. **Docs/site gates**: website build, `check:content`, `check:links` green;
   dprint clean on touched files; no Rust diffs.

**Pass** = all five. **Fail** = the fresh-session proof needs ANY user config,
or brew link fights the stub.

## Result

**Result:** Pass — with a corrected verification record (see the Correction
section below: the originally cited proof was a false positive; the corrected
evidence supports Pass for interactive sessions, the case the goal names).

`brew install nutorch` now seats the module in interactive Nushell sessions by
itself.

- **The formula writes the stub**: `share/nushell/vendor/autoload/nutorch.nu`
  containing `use "/opt/homebrew/opt/nutorch/share/nutorch/nutorch.nu" *` — the
  version-stable opt path. The `test do` block asserts the stub exists
  (GPU-free; Nushell itself stays out of the formula's dependencies).
- **The install mechanics, end to end on this machine**: hand stub removed;
  `dist/nutorch.rb` temp-copied over the local tap;
  `brew reinstall
  --build-from-source` (29s — the stub is install-time output,
  so the pinned v0.1.0 tarball needed no re-release); the stub came out
  BREW-LINKED (a symlink into the keg, not a loose file) referencing the opt
  path. `brew test` green with the new assertion. Local tap reverted to
  git-clean; the published formula on GitHub still carries no stub (next
  release, per the spine). The BEHAVIORAL proof originally recorded here
  (`nu -c "nutorch tensor '[1,2]' | nutorch value"` → `[1.0,2.0]`) was a false
  positive — see the Correction.
- **Bonus from the rebuild**: the earlier hand-applied CLI symlinks (the issue's
  first-draft side effect) were replaced by brew-managed ones — `bin/nutorch` is
  now a proper keg-linked binstub; no rogue files remain anywhere.
- **From-source fallback**: `install.sh` now PRINTS the user-autoload one-liner
  (never writes user config); the printed line was executed once against a
  scratch dir and sources clean.
- **Docs**: the Nushell page leads with "Homebrew installs: there is nothing to
  set up," explains the mechanism in one parenthesis, and keeps the manual
  `use`/user-autoload fallback for non-brew Nushell builds; the README's Nushell
  section carries the autoload clause.
- **Gates**: website build + check:content + check:links + check:ops-ref green;
  dprint clean; zero `.rs` diffs; `v1/` untouched.

## Conclusion

The goal's exact sentence is now true: open a new Nushell, type
`nutorch tensor '[1,2]'`, and it works — with the only `use` statement living in
a file the package owns. The prefix-relative contract held exactly as the spine
argued (both sides derive from `HOMEBREW_PREFIX`), and the fallback story is
documented for everyone outside it. The published tap picks the stub up with the
next release, alongside the MIT metadata and the CLI symlink.

## Correction (2026-06-12, same day)

The result above originally cited
`nu -c "nutorch tensor '[1,2]' | nutorch value"` → `[1.0,2.0]` as the
zero-config proof. **That test was a false positive**, discovered while
preparing the next website experiment:

- Under `nu -c` (and `nu -i -c`), Nushell 0.113 does NOT run vendor autoload at
  all — proven with an env-var probe file in the autoload dir (`NOT-LOADED` in
  both modes) and by `which "nutorch tensor"` returning empty under `-c`.
  Autoload rides config loading, which `-c` skips.
- The `[1.0,2.0]` output came from the EXTERNAL `nutorch` binary — the CLI
  symlink shipped by this same issue's deleted first draft. For the
  string-argument form, the external CLI and the module command are
  indistinguishable by output. The test had no discriminating power; the forms
  that discriminate are `which "nutorch tensor"` (module command existence) and
  the pipeline form `[1 2 3] | nutorch tensor` (module-only), and under `-c`
  both show the module absent.

**The corrected evidence for the goal:**

- Vendor autoload applies to config-loading startups — interactive sessions,
  exactly what the goal's sentence describes ("open a new Nushell…"). The user
  confirmed the behavior in his interactive REPL.
- The mechanism is even sturdier than the spine's XDG framing: `brew
  shellenv`
  exports no `XDG_DATA_DIRS` (verified) — the brew prefix autoload dir is BAKED
  INTO the brew-built `nu` at compile time, so every brew-installed Nushell
  scans it in every interactive session regardless of environment.
- The `-c`/script limitation is real and now documented here: scripts and
  one-liners load the module explicitly (`use … *`), which is the right practice
  for scripts in any case (`scripts/train-regression.nu` already does).

**Lesson recorded**: "right output" is not "right mechanism." A same-named
external binary (introduced by this very issue) could satisfy the test form
chosen; the verification should have used the discriminating checks from the
start. This is the exact class of error the waived adversarial result review has
repeatedly caught on other issues — the waiver stands (user decision), so
verification plans under it must carry the discrimination burden themselves.
