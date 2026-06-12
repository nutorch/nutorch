+++
[implementer]
agent = "claude-code"
model = "claude-fable-5"

[review]
waived = "user decision 2026-06-12: no adversarial review for issue 0013"
+++

# Experiment 6: NuTorch, the proper noun

## Description

Punch-list addition: the project's display name on the website becomes
**NuTorch** — a proper noun — everywhere the site uses the name AS A NAME.
Commands, handles, package/file names, domains, and code stay lowercase: a user
types `torch` and `nutorch`, installs `brew install nutorch`, visits
nutorch.com, and reads about NuTorch.

**The line between name and code** (the whole experiment is this distinction):

| Stays lowercase (code/identifier)              | Becomes NuTorch (display name)        |
| ---------------------------------------------- | ------------------------------------- |
| `torch` / `nutorch` commands in any code block | page `<title>`s, h1s, prose mentions  |
| `nutorch.nu`, `nutorchd`, `dist/nutorch.rb`    | the header/footer wordmark            |
| `brew … nutorch`, `github.com/nutorch/…`       | `og:site_name`, OG-image wordmark     |
| nutorch.com (the domain string)                | image alt text ("The NuTorch logo …") |
| CSS classes (`prose-nutorch`), ids, paths      | docs prose ("NuTorch puts tensors …") |

**Decisions, made here:**

1. **The wordmark keeps its two-tone split with capitals**: `Nu` (shell green) +
   `Torch` (flame orange) in the header — same treatment in the footer row and
   the 404 title. The OG card's SVG text in `process-images.ts` gets the same
   change and the image is regenerated.
2. **Inventory by category, applied at implementation** (from a design-time
   sweep): page titles (`index`, `404`, the `— nutorch docs` suffix in DocPage),
   `og:site_name`, the footer "nutorch · MIT", the hero image alt, and
   docs-prose mentions (e.g. getting-started's "nutorch puts GPU tensors in your
   shell", ops.md's "nutorch's operation surface") — every occurrence judged
   against the table above, not blind-replaced.
3. **The distinction gets an executable gate**: `check-content.ts` gains a brand
   check over BUILT HTML — after stripping `<code>`/`<pre>`/`<script>` content
   and attribute values (hrefs, ids, classes, srcs), a case-sensitive
   `\bnutorch\b` in rendered prose is a FAILURE (the exceptions live in markup
   the strip removes: code spans, fences, URLs, the domain). `nutorchd` in prose
   stays valid only inside code spans — which is where it already lives. The
   gate is what keeps future content honest about the brand without a human
   proofreading every page.
4. **Scope is the website** (the user's words: "titles used on the web page"):
   `website/src` + the OG pipeline. The repo README, formula `desc`, and
   `--version` output are NOT touched (recorded — they can follow in a later fix
   if wanted; `--version` prints a package name, which is arguably code anyway).

## Changes

1. **`website/src/components/Header.astro` + `Footer.astro`**: wordmark
   `Nu`/`Torch`; footer "NuTorch · MIT".
2. **`website/src/pages/index.astro`**: title, hero alt; **`404.astro`**: title.
3. **`website/src/layouts/Base.astro`**: `og:site_name`; **`DocPage.astro`**:
   the `— NuTorch docs` suffix.
4. **`website/src/content/docs/*.md`**: prose mentions only (code spans and
   fences untouched).
5. **`website/scripts/process-images.ts`**: OG SVG text → regenerated
   `og-nutorch.png` (filename unchanged — it is a path).
6. **`website/scripts/check-content.ts`**: the brand gate (decision 3).
7. **Nothing else** — no Rust, no repo README, no formula, no `v1/`.

## Verification

1. **The brand gate**: `check:content` green on the new build; then
   adversarially — plant a prose "nutorch" in a scratch copy and confirm the
   gate names it; confirm code spans, fences, and URLs do NOT trip it (the
   existing pages are full of them).
2. **Visual**: header/footer wordmark screenshots both modes ("NuTorch"
   two-tone); the regenerated OG card inspected by eye.
3. **Titles**: built `<title>`s and `og:site_name` read NuTorch; the docs suffix
   reads "— NuTorch docs" on every docs page.
4. **Nothing functional moved**: all commands in all fences byte-identical
   (`git diff` shows no fence lines); `check:links`, `check:ops-ref`,
   `check:theme` green; install block byte-equality still green.
5. **Hygiene**: dprint clean; zero `.rs` diffs; `v1/` untouched.

**Pass** = all five. **Fail** = the brand gate is inert, any fence line changed,
or a rendered-prose lowercase "nutorch" survives outside code/URLs.
