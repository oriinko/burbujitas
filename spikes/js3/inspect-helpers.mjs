import { openRuntimeSession } from "./runtime-session.mjs";

const { browser, frame } = await openRuntimeSession();
try {
  const inspection = await frame.evaluate(() => {
    const runtime = window.__bubblebot;
    const relevant = /(simulat|collision|land|snap|cluster|match|float|drop|neighbor|cell)/i;
    const inspectObject = (value, path) => {
      const own = Object.getOwnPropertyDescriptors(value);
      const prototype = Object.getPrototypeOf(value);
      const prototypeDescriptors = prototype
        ? Object.getOwnPropertyDescriptors(prototype)
        : {};
      const methods = [];
      for (const [name, descriptor] of Object.entries({ ...prototypeDescriptors, ...own })) {
        if (name !== "constructor" && relevant.test(name) && typeof descriptor.value === "function") {
          methods.push({ name, arity: descriptor.value.length });
        }
      }
      return methods.length > 0 ? { path, methods } : null;
    };

    const matches = [];
    for (const [name, value] of Object.entries(runtime)) {
      if (value && (typeof value === "object" || typeof value === "function")) {
        const match = inspectObject(value, `runtime.${name}`);
        if (match) matches.push(match);
      }
    }

    const queue = [{ value: runtime.root, path: "runtime.root", depth: 0 }];
    const seen = new Set();
    while (queue.length > 0) {
      const { value, path, depth } = queue.shift();
      if (!value || seen.has(value) || depth > 6) continue;
      seen.add(value);
      const match = inspectObject(value, path);
      if (match) matches.push(match);
      if (Array.isArray(value.children)) {
        value.children.forEach((child, index) => {
          queue.push({ value: child, path: `${path}.children[${index}]`, depth: depth + 1 });
        });
      }
    }

    return { relevantExposedMethods: matches };
  });
  process.stdout.write(`${JSON.stringify(inspection, null, 2)}\n`);
} finally {
  await browser.close();
}
