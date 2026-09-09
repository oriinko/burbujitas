import { openRuntimeSession } from "./runtime-session.mjs";

const { browser, frame } = await openRuntimeSession();
try {
  const inspection = await frame.evaluate(() => {
    const runtime = window.__bubblebot;
    const queue = [{ value: runtime.root, path: "runtime.root", depth: 0 }];
    const seen = new Set();
    const sceneObjects = [];
    while (queue.length > 0) {
      const { value, path, depth } = queue.shift();
      if (!value || seen.has(value) || depth > 6) continue;
      seen.add(value);
      const own = Object.getOwnPropertyDescriptors(value);
      const prototype = Object.getPrototypeOf(value);
      const inherited = prototype ? Object.getOwnPropertyDescriptors(prototype) : {};
      const descriptors = { ...inherited, ...own };
      const names = Object.keys(descriptors);
      if (
        names.includes("consumeCurrentBubble") ||
        names.includes("addBubble") ||
        names.includes("updateFromBubble")
      ) {
        sceneObjects.push({
          path,
          methods: Object.fromEntries(
            names
              .filter((name) => typeof descriptors[name].value === "function")
              .map((name) => [
                name,
                {
                  arity: descriptors[name].value.length,
                  source: Function.prototype.toString.call(descriptors[name].value),
                },
              ]),
          ),
          onShoot:
            typeof value.onShoot === "function"
              ? Function.prototype.toString.call(value.onShoot)
              : null,
          hasActive: "hasActive" in value ? Boolean(value.hasActive) : null,
        });
      }
      if (Array.isArray(value.children)) {
        value.children.forEach((child, index) => {
          queue.push({ value: child, path: `${path}.children[${index}]`, depth: depth + 1 });
        });
      }
    }

    return {
      shooterReady: runtime.store.getState().shooter.ready,
      orchestratorShoot: {
        arity: runtime.orchestrator.shoot.length,
        source: Function.prototype.toString.call(runtime.orchestrator.shoot),
      },
      sceneObjects,
    };
  });
  process.stdout.write(`${JSON.stringify(inspection, null, 2)}\n`);
} finally {
  await browser.close();
}
