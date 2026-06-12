// Ops reference generator (issue 0012 exp 3): one markdown page per op
// category, generated from the real binaries — `torch ops --json` for
// name/category/summary, `torch <op> --help` for the usage line (line 1
// only; line 2 duplicates the summary). Output must be a dprint fixed
// point; if dprint ever disagrees, fix THIS script, not the files.
//
//   bun run scripts/gen-ops-reference.ts          # write pages
//   bun run scripts/gen-ops-reference.ts --check  # regenerate + byte-compare
import { execSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";

const OUT = new URL("../src/content/docs/reference/", import.meta.url)
  .pathname;
const CATEGORY_ORDER = [
  "creation",
  "pointwise",
  "comparison",
  "reduction",
  "linalg",
  "shape",
  "loss",
  "autograd",
  "utility",
];
const TITLES: Record<string, string> = {
  creation: "Creation ops",
  pointwise: "Pointwise ops",
  comparison: "Comparison ops",
  reduction: "Reduction ops",
  linalg: "Linear algebra ops",
  shape: "Shape ops",
  loss: "Loss ops",
  autograd: "Autograd ops",
  utility: "Utility ops",
};

interface Op {
  name: string;
  category: string;
  summary: string;
}

const env = { ...process.env, TMPDIR: mkdtempSync(`${tmpdir()}/nutorch-gen-`) };
const ops = JSON.parse(
  execSync("torch ops --json", { env }).toString(),
) as Op[];

const byCategory = new Map<string, Op[]>();
for (const op of ops) {
  if (!byCategory.has(op.category)) byCategory.set(op.category, []);
  byCategory.get(op.category)!.push(op);
}

const unknown = [...byCategory.keys()].filter(
  (c) => !CATEGORY_ORDER.includes(c),
);
if (unknown.length) {
  console.error(`unknown categories: ${unknown.join(", ")} — extend the list`);
  process.exit(1);
}

function usageLine(op: string): string {
  const help = execSync(`torch ${op} --help`, { env }).toString();
  const first = help.split("\n")[0];
  if (!first.startsWith("usage:")) {
    console.error(`unexpected --help shape for ${op}: ${first}`);
    process.exit(1);
  }
  return first;
}

function page(category: string, index: number, entries: Op[]): string {
  const lines: string[] = [
    "---",
    `title: ${TITLES[category]}`,
    `description: The ${entries.length} ${category} operation${
      entries.length === 1 ? "" : "s"
    }, generated from the op table.`,
    `order: ${20 + index}`,
    'section: "Reference"',
    "---",
    "",
    "Generated from the binaries by `scripts/gen-ops-reference.ts` — do not edit by",
    "hand. Every op also documents itself: `torch <op> --help`.",
  ];
  for (const op of entries) {
    // Display layer: the summary is rendered prose, where the project
    // name is the proper noun NuTorch (issue 0013 exp 6). The usage line
    // stays verbatim — it is CLI output.
    const summary = op.summary.replace(/\bnutorch\b/g, "NuTorch");
    lines.push("", `### ${op.name}`, "", summary, "", "```");
    lines.push(usageLine(op.name), "```");
  }
  lines.push("");
  return lines.join("\n");
}

const generated = new Map<string, string>();
CATEGORY_ORDER.forEach((category, i) => {
  const entries = byCategory.get(category);
  if (!entries) return;
  generated.set(`${category}.md`, page(category, i, entries));
});

if (process.argv.includes("--check")) {
  let stale = false;
  // Orphans: committed files no category produces anymore.
  const { readdirSync } = await import("node:fs");
  for (const file of readdirSync(OUT).filter((f) => f.endsWith(".md"))) {
    if (!generated.has(file)) {
      console.error(`ORPHAN: src/content/docs/reference/${file}`);
      stale = true;
    }
  }
  for (const [file, content] of generated) {
    let committed = "";
    try {
      committed = readFileSync(`${OUT}/${file}`, "utf8");
    } catch {}
    if (committed !== content) {
      console.error(`STALE: src/content/docs/reference/${file}`);
      stale = true;
    }
  }
  if (stale) process.exit(1);
  console.log(`ops reference current (${ops.length} ops, ${generated.size} pages)`);
} else {
  mkdirSync(OUT, { recursive: true });
  for (const [file, content] of generated) {
    writeFileSync(`${OUT}/${file}`, content);
  }
  console.log(`wrote ${generated.size} pages (${ops.length} ops) → src/content/docs/reference/`);
}
