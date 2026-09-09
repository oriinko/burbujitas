import { openRuntimeSession } from "./runtime-session.mjs";

const { browser, frame } = await openRuntimeSession();
try {
  const inspection = await frame.evaluate(() => {
    const runtime = window.__bubblebot;
    const orchestrator = runtime.orchestrator;
    return {
      runtimeKeys: Object.keys(runtime).sort(),
      orchestratorMethods: Object.fromEntries(
        Object.keys(orchestrator)
          .sort()
          .map((name) => [
            name,
            {
              type: typeof orchestrator[name],
              arity: typeof orchestrator[name] === "function" ? orchestrator[name].length : null,
              source:
                typeof orchestrator[name] === "function"
                  ? Function.prototype.toString.call(orchestrator[name])
                  : null,
            },
          ]),
      ),
      stateGetterMatchesStore:
        orchestrator.getState() === runtime.store.getState(),
    };
  });
  process.stdout.write(`${JSON.stringify(inspection, null, 2)}\n`);
} finally {
  await browser.close();
}
