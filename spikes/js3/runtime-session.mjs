import { chromium } from "playwright";

const MSN_URL =
  "https://www.msn.com/es-mx/play/games/bubble-shooter-hd/cg-9nzvl6gzqhkj";
const BUNDLE_URL =
  "https://softgames.cdn.msnfun.com/9nzvl6gzqhkj/v187/assets/main-DKQrGX-t.js";
const PATCH_TARGET = ",app:e,root:v};MK(C),OK(C,r)";
const PATCH_REPLACEMENT =
  ",app:e,root:v};window.__bubblebot=C;MK(C),OK(C,r)";

function occurrenceCount(source, fragment) {
  return source.split(fragment).length - 1;
}

async function findPlayableFrame(page, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    for (const frame of page.frames()) {
      try {
        const playable = await frame.evaluate(() => {
          const runtime = window.__bubblebot;
          if (!runtime?.store?.getState || !runtime?.orchestrator?.simulate) return false;
          const state = runtime.store.getState();
          return Boolean(
            state?.grid?.sizeW === 17 &&
              Object.keys(state.grid.cells ?? {}).length > 0 &&
              state?.shooter?.ready &&
              state.shooter.currentType !== null,
          );
        });
        if (playable) return frame;
      } catch {
        // MSN frames may detach while the game shell initializes.
      }
    }
    await page.waitForTimeout(250);
  }
  throw new Error("Timed out waiting for an exposed playable Bubble Shooter round");
}

export async function openRuntimeSession() {
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const context = await browser.newContext({ viewport: { width: 1856, height: 1686 } });
  let intercepted = 0;

  await context.route(BUNDLE_URL, async (route) => {
    intercepted += 1;
    const response = await route.fetch();
    const source = await response.text();
    const matches = occurrenceCount(source, PATCH_TARGET);
    if (matches !== 1) {
      throw new Error(`Expected one runtime-construction marker, found ${matches}`);
    }
    await route.fulfill({
      response,
      body: source.replace(PATCH_TARGET, PATCH_REPLACEMENT),
    });
  });

  try {
    const page = await context.newPage();
    await page.goto(MSN_URL, { waitUntil: "domcontentloaded", timeout: 90_000 });
    const frame = await findPlayableFrame(page, 120_000);
    if (intercepted !== 1) {
      throw new Error(`Expected one main-bundle interception, observed ${intercepted}`);
    }
    return { browser, frame };
  } catch (error) {
    await browser.close();
    throw error;
  }
}
