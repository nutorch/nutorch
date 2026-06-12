+++
[implementer]
agent = "claude-code"
model = "claude-fable-5"

[review.design]
agent = "claude-code"
subagent = "adversarial-reviewer"
model = "claude-opus"

[review.result]
agent = "claude-code"
subagent = "adversarial-reviewer"
model = "claude-opus"
+++

# Experiment 4: Search, sitemap, and the finishing pass

## Description

The issue's last experiment: the three "cheap essentials" the spine pledged
(search, sitemap, OG polish), plus a 404 page, a link-integrity gate, and the
final whole-site QA in both modes. After this the issue closes — the site is
production-quality `dist/`, awaiting only the (separate, future) deployment
issue.

**Decisions, made here:**

1. **Search is Pagefind**, as the spine pledged: `pagefind` as a devDependency;
   the build becomes `astro build && pagefind --site dist` (index generated
   post-build, fully client-side at runtime — local-only discipline holds). The
   search UI mounts ONCE per page in DocPage — outside the twice-rendered
   DocNav, positioned above the article and visible at every breakpoint —
   avoiding the duplicate-id/double-init hazard of one instance per nav (review
   catch). It uses Pagefind's default UI themed with the brand tokens
   (`--pagefind-ui-*` CSS custom properties map onto our CSS variables, both
   modes); exact flag/property names re-verified against the pinned pagefind
   version at implementation. Search lives on docs pages only — the landing page
   pitches; the docs answer questions. Caveat recorded: the index exists only in
   `dist/` (after a build), so `astro dev` has no search — `bun run preview` is
   the place to try it.
2. **Indexing scope**: only the docs article body (`data-pagefind-body` on the
   DocPage article) — header/footer/sidebar boilerplate stays out of the index
   so results are content, not chrome.
3. **Sitemap is `@astrojs/sitemap`** (Astro-6-compatible release pinned at
   implementation): `site` is already `https://nutorch.com`; a `filter` drops
   the duplicate `/docs/` URL (the sitemap enumerates routes and does not read
   canonical links — review catch). `public/robots.txt` points at
   `/sitemap-index.xml`.
4. **404 page**: `src/pages/404.astro` — brand mark, "no such tensor", links
   home and to the docs. Astro emits `dist/404.html`, which the future host
   serves natively.
5. **Link integrity becomes a gate**: `scripts/check-links.ts` (bun) walks every
   built HTML file in `dist/`, extracts internal `href`s, and asserts each
   resolves to a built file (route dir, file, or anchor on a built page).
   External links are collected and listed but NOT fetched (local-only). Wired
   as `check:links`; catches the broken-link class that screenshots and HTML
   greps miss.
6. **OG polish**: per-page `og:title`/`og:description` already flow through
   Base; add `og:url` (canonical-aware), `og:site_name`, and switch docs pages'
   OG type to `article`. The OG image stays the brand card from Experiment 1.
7. **The final QA sweep**: rebuild everything from clean
   (`rm -rf dist && bun run build`), run ALL gates (`check:content`,
   `check:ops-ref`, `check:links`, dprint, frozen-lockfile), screenshot the
   landing page + one docs page + one reference page + the 404 + a SEARCH
   interaction (type a query, results visible) in BOTH modes, and review every
   shot by eye against the issue's beauty bar. Any visual defect found is fixed
   in this experiment — this is the pass that says "done".

## Changes

1. **`website/package.json`**: `pagefind` + `@astrojs/sitemap` deps; build
   script gains the pagefind step; `check:links` script.
2. **`website/astro.config.mjs`**: sitemap integration.
3. **`website/src/components/DocPage.astro`**: single search UI mount,
   `data-pagefind-body` on the article, `ogType="article"` passed to Base.
4. **`website/src/styles/global.css`**: Pagefind UI theme mapping (both modes).
5. **`website/src/layouts/Base.astro`**: og:url / og:site_name, and a new
   optional `ogType` prop (default `website`) mirroring `canonical`.
6. **`website/src/pages/404.astro`** (NEW).
7. **`website/public/robots.txt`** (NEW).
8. **`website/scripts/check-links.ts`** (NEW).
9. **No Rust changes; `v1/` untouched.**

## Verification

1. **Build gate**: clean rebuild exits 0; `dist/pagefind/` exists (index + UI
   assets); `dist/sitemap-index.xml` lists the site's routes; `dist/404.html`
   present; robots.txt present.
2. **Search works, proven by interaction**: serve `dist/`, headless-Chrome a
   docs page, type a query (e.g. "backward") into the search input, and
   screenshot visible results linking to the right pages — in BOTH modes (the
   themed UI must be legible in each). At least one result must point at a
   reference page (the generated content is indexed).
3. **Link integrity**: `check:links` green over the full `dist/`; then
   adversarially: a temp page with a bogus href fails it.
4. **OG/meta**: built pages carry og:url (matching their route), og:site_name,
   correct og:type; sitemap URLs use the `https://nutorch.com` origin.
5. **The QA sweep** (decision 7): all gates green from a clean build; all
   screenshots captured and reviewed; defects fixed or explicitly recorded.
6. **Hygiene**: dprint clean on touched files; frozen-lockfile green after the
   dependency additions (lock updated and committed); `v1/` and the Rust tree
   untouched.

**Pass** = all six. **Fail** = search returns nothing for indexed content, the
link gate is inert, or any built page ships an unresolvable internal link.

## Design Review

**Reviewer:** `adversarial-reviewer` subagent (fresh context, read-only).
**Verdict: APPROVED (first pass).** The reviewer confirmed the load-bearing
premises: `pagefind --site dist` and the `--pagefind-ui-*` theming are the
Pagefind-1.x contracts (to re-verify against the pinned version, since the dep
is not yet installed); `data-pagefind-body` on docs pages correctly excludes the
landing page from the index; `@astrojs/sitemap` with static output and the
configured `site` emits `sitemap-index.xml`; `404.astro` → `dist/404.html`; the
Base optional-prop pattern keeps the og changes backward-compatible; the spine's
outstanding Scope pledges (search, sitemap, OG) are all covered with deployment
correctly Out. Three Optionals folded: the search UI mounts ONCE outside the
twice-rendered DocNav (the duplicate-id/double-init hazard the design had walked
into); the `ogType` prop plumbing named explicitly in the Changes list; the
sitemap gains a `filter` dropping the duplicate `/docs/` URL (sitemaps enumerate
routes and ignore canonical links). Nit folded: dependency versions pinned and
the two CLI/CSS contracts re-verified at implementation.
