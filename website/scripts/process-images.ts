// Logo pipeline (issue 0012 exp 1): raw-images/ → public/. Outputs are
// committed so builds never depend on rerunning this. Sharp resizes;
// png-to-ico packs the ICO container (sharp cannot encode ICO).
import pngToIco from "png-to-ico";
import sharp from "sharp";
import { mkdir, writeFile } from "node:fs/promises";

const OUT = new URL("../public/images/", import.meta.url).pathname;
const PUB = new URL("../public/", import.meta.url).pathname;
const MARK = new URL("../raw-images/nutorch-2d.png", import.meta.url).pathname;
const HERO = new URL("../raw-images/nutorch-3d.png", import.meta.url).pathname;

await mkdir(OUT, { recursive: true });

// Header/footer mark + favicon PNG sizes.
for (const size of [64, 128, 192]) {
  await sharp(MARK)
    .resize(size, size)
    .png()
    .toFile(`${OUT}/nutorch-2d-${size}.png`);
}

// Hero (1x/2x display sizes from the 850px render).
await sharp(HERO).resize(360, 360).png().toFile(`${OUT}/nutorch-hero.png`);
await sharp(HERO).resize(720, 720).png().toFile(`${OUT}/nutorch-hero@2x.png`);

// Favicon: 32px ICO.
const png32 = await sharp(MARK).resize(32, 32).png().toBuffer();
await writeFile(`${PUB}/favicon.ico`, await pngToIco([png32]));

// OG image: 1200x630, dark brand background, mark + wordmark + tagline.
const og = sharp({
  create: {
    width: 1200,
    height: 630,
    channels: 4,
    background: { r: 15, g: 19, b: 15, alpha: 1 },
  },
});
const markLarge = await sharp(MARK).resize(340, 340).png().toBuffer();
const ogText = Buffer.from(`<svg width="1200" height="630">
  <text x="450" y="300" font-family="Helvetica, Arial, sans-serif"
    font-weight="bold" font-size="110">
    <tspan fill="#6fd877">Nu</tspan><tspan fill="#ff8a4d">Torch</tspan>
  </text>
  <text x="452" y="370" font-family="Helvetica, Arial, sans-serif"
    font-size="40" fill="#97a591">GPU tensors for every shell</text>
</svg>`);
await og
  .composite([
    { input: markLarge, left: 70, top: 145 },
    { input: ogText, left: 0, top: 0 },
  ])
  .png()
  .toFile(`${OUT}/og-nutorch.png`);

console.log("images processed → public/");
