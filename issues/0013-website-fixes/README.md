+++
status = "open"
opened = "2026-06-12"
+++

# Issue 13: Website fixes — the punch list

## Goal

The nutorch.com site (built in issue 0012) is good; this issue makes it right. A
running punch list of user-requested fixes, applied through the normal
experiment workflow. The list is expected to GROW — the user has named three
fixes and reserved "other things I may want to fix."

## The punch list (from the user, 2026-06-12)

1. **Copyright is Astrohacker, not Ryan X. Charles** — with a link to
   [astrohacker.com](https://astrohacker.com). Today the site footer prints "©
   {year} Ryan X. Charles" (`website/src/components/Footer.astro`).
2. **An "add tensors" example at the very top of the home page** — the visitor
   should instantly see what matters: tensor operations at your terminal. Today
   the hero leads with the logo, headline, and prose; the first code on the page
   is the install block one section down.
3. **License changes from Apache to MIT.** This reaches beyond the website:
   - `LICENSE` at the repo root (currently Apache 2.0);
   - `license = "Apache-2.0"` in workspace/crate `Cargo.toml` manifests;
   - the Homebrew formula's `license "Apache-2.0"` line (`dist/nutorch.rb` here
     and `Formula/nutorch.rb` in the published tap — the tap update is an
     outward-facing step);
   - the website footer ("Apache-2.0") and the tap/repo README mentions.
4. _(Reserved for additions as the user names them.)_

## Background

Issue 0012 built the site: Astro 6 static + Bun + Tailwind v4 under `website/`,
brand tokens measured from the logo, both modes, 8 written docs pages + 9
generated reference pages, Pagefind search, and four executable gates
(`check:content`, `check:ops-ref`, `check:links`, the build). Issue 0012 is
closed and immutable; fixes happen here.

Relevant current facts:

- The footer line lives in `website/src/components/Footer.astro`; the
  hero/landing structure in `website/src/pages/index.astro`.
- The landing page's demo code goes through `CodeBlock.astro` (dual Vitesse
  themes — any new top-of-page example uses the same component).
- The 404 page and OG card carry no license/copyright text; the README install
  section and the tap README do mention the license context.
- v1 is frozen: `v1/LICENSE`/license texts in the archive are historical record
  and are NOT edited (the license change applies to the live project).

## Analysis

- Fixes 1 and 2 are website-only and low-risk; the existing gates plus both-mode
  screenshots verify them.
- Fix 3 is a PROJECT change with an outward-facing tail: the published tap repo
  (`nutorch/homebrew-nutorch`) declares `license "Apache-2.0"` in its formula,
  and the GitHub repo displays the LICENSE file. Updating the tap is a push to a
  public repo — sequenced and named explicitly in whatever experiment carries
  it, like issue 0011's publication steps. A license NOTICE question (whether
  any vendored component requires retaining attributions) gets checked in that
  experiment's design.
- The hero example (fix 2) should be REAL output, verified live — the
  `check-content.ts` honesty gate covers `index.astro` only via the install
  string today; the experiment should extend the verification habit to the new
  example (it is Astro source, not markdown, so the markdown checker does not
  see it).

## Scope

In: the punch-list items above, plus any further fixes the user names while this
issue is open; the license change across the live repo, website, and published
tap/release metadata.

Out (recorded): website deployment (still its own future issue); any v1/ edits
(frozen archive); re-releasing binaries for the license change (the next tagged
release naturally carries it; no retroactive re-bottle).
