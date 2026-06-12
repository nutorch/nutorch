// Content honesty checks (issue 0012 exp 2):
// 1. The getting-started install block byte-matches src/lib/install.ts.
// 2. Every `torch <op>` used in docs fences is a real table op or a known
//    client/registry verb, per `torch ops --json` from the real binary.
import { execSync } from "node:child_process";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { INSTALL } from "../src/lib/install";

const DOCS = new URL("../src/content/docs/", import.meta.url).pathname;
let failed = false;

// 1. Install block.
const gettingStarted = readFileSync(`${DOCS}/getting-started.md`, "utf8");
const fence = gettingStarted.match(/```bash\n([\s\S]*?)\n```/);
if (!fence || fence[1] !== INSTALL) {
  console.error("FAIL: getting-started install block drifted from install.ts");
  failed = true;
}

// 2. Op-name membership. Non-op verbs are the client/registry surface,
// verified live in the experiment's verification.
const NON_OP_VERBS = new Set([
  "tensor",
  "value",
  "free",
  "tensors",
  "forward",
  "step",
  "daemon",
  "nn",
  "ops",
  "nu-module",
  "--version",
]);
const ops = new Set(
  (
    JSON.parse(
      execSync("torch ops --json", {
        env: { ...process.env, TMPDIR: execSync("mktemp -d").toString().trim() },
      }).toString(),
    ) as { name: string }[]
  ).map((o) => o.name),
);
execSync("torch daemon stop || true", { stdio: "ignore", shell: "/bin/zsh" });

for (const file of readdirSync(DOCS).filter((f) => f.endsWith(".md"))) {
  const text = readFileSync(`${DOCS}/${file}`, "utf8");
  for (const block of text.matchAll(/```(?:bash|nu)\n([\s\S]*?)\n```/g)) {
    for (const use of block[1].matchAll(/(?:torch|nutorch) ([a-z_-]+|--version)/g)) {
      const verb = use[1];
      if (!ops.has(verb) && !NON_OP_VERBS.has(verb)) {
        console.error(`FAIL: ${file}: unknown verb 'torch ${verb}'`);
        failed = true;
      }
    }
  }
}

// The landing page's demo code lives in Astro template LITERALS, not
// markdown fences — scan only the backtick strings (prose like the logo
// alt text would otherwise false-positive).
const INDEX = new URL("../src/pages/index.astro", import.meta.url).pathname;
const indexSource = readFileSync(INDEX, "utf8");
for (const literal of indexSource.matchAll(/`([\s\S]*?)`/g)) {
  for (const use of literal[1].matchAll(
    /(?:torch|nutorch) ([a-z_-]+|--version)/g,
  )) {
    const verb = use[1];
    if (!ops.has(verb) && !NON_OP_VERBS.has(verb)) {
      console.error(`FAIL: index.astro: unknown verb 'torch ${verb}'`);
      failed = true;
    }
  }
}

// 3. The brand gate (issue 0013 exp 6): in RENDERED prose, the name is
// NuTorch. Lowercase `nutorch` is code — it may appear only inside
// code/pre/script elements, attribute values, or URLs, all of which the
// strip below removes. Runs only when a build exists.
const DIST = new URL("../dist/", import.meta.url).pathname;
function distHtmlFiles(dir: string): string[] {
  const out: string[] = [];
  let entries: string[] = [];
  try {
    entries = readdirSync(dir);
  } catch {
    return out;
  }
  for (const name of entries) {
    const path = `${dir}/${name}`;
    if (statSync(path).isDirectory()) out.push(...distHtmlFiles(path));
    else if (name.endsWith(".html")) out.push(path);
  }
  return out;
}
for (const file of distHtmlFiles(DIST)) {
  let html = readFileSync(file, "utf8");
  html = html.replace(/<(code|pre|script|style)[\s\S]*?<\/\1>/g, " ");
  html = html.replace(/<[^>]*>/g, " "); // tags incl. all attribute values
  // URL/path-shaped tokens (domains, repo paths) are identifiers, not
  // prose: nutorch.com, github.com/nutorch/nutorch, ~/.nutorch, …
  html = html.replace(/\S*nutorch[./]\S*/g, " ");
  html = html.replace(/\S*[./~]nutorch\S*/g, " ");
  for (const m of html.matchAll(/.{0,30}\bnutorch\b.{0,30}/g)) {
    console.error(
      `FAIL: ${file.replace(DIST, "")}: prose lowercase brand: …${m[0].trim()}…`,
    );
    failed = true;
  }
}

if (failed) process.exit(1);
console.log("content checks passed");
