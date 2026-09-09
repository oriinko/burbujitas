import { openRuntimeSession } from "../js4/runtime-session.mjs";

const MAX_SHOTS = 10;
const MAX_STEPS = 200;
const AIM_ANGLES = Array.from({ length: 161 }, (_, index) => index - 80);
const READY_TIMEOUT_MS = 30_000;
const RESOLUTION_TIMEOUT_MS = 30_000;

const { browser, frame } = await openRuntimeSession();
try {
  const result = await frame.evaluate(
    async ({ aimAngles, maxShots, maxSteps, readyTimeoutMs, resolutionTimeoutMs }) => {
      const runtime = window.__bubblebot;
      const { orchestrator } = runtime;

      const simulatorSource = Function.prototype.toString.call(orchestrator.simulate);
      if (
        typeof orchestrator.simulate !== "function" ||
        orchestrator.simulate.length !== 1 ||
        !simulatorSource.includes("return L8(b,S,E,A)") ||
        simulatorSource.includes("dispatch")
      ) {
        throw new Error("The guarded v187 non-mutating simulator API is unavailable or changed");
      }

      const shootSource = Function.prototype.toString.call(orchestrator.shoot);
      if (
        typeof orchestrator.shoot !== "function" ||
        orchestrator.shoot.length !== 1 ||
        !shootSource.includes('dispatch({type:"BubbleShot",angle:b})') ||
        !shootSource.includes("return{x:I.x,y:I.y,vel:E")
      ) {
        throw new Error("The guarded v187 shooting API is unavailable or changed");
      }

      const findSceneObjects = () => {
        const queue = [runtime.root];
        const seen = new Set();
        const shooters = [];
        const flyingEngines = [];
        while (queue.length > 0) {
          const value = queue.shift();
          if (!value || seen.has(value)) continue;
          seen.add(value);
          if (
            typeof value.consumeCurrentBubble === "function" &&
            typeof value.isReadyToShoot === "function" &&
            typeof value.onShoot === "function"
          ) {
            shooters.push(value);
          }
          if (
            typeof value.addBubble === "function" &&
            Array.isArray(value.activeBubbles) &&
            typeof value.onCollision === "function"
          ) {
            flyingEngines.push(value);
          }
          if (Array.isArray(value.children)) queue.push(...value.children);
        }
        if (shooters.length !== 1 || flyingEngines.length !== 1) {
          throw new Error(
            `Expected one shooter and flying engine; found ${shooters.length}/${flyingEngines.length}`,
          );
        }
        return { shooter: shooters[0], flying: flyingEngines[0] };
      };

      const { shooter, flying } = findSceneObjects();
      if (
        !Function.prototype.toString
          .call(shooter.onShoot)
          .includes("addBubble(E.x,E.y,E.velX,E.velY,E.type)") ||
        !Function.prototype.toString.call(flying.onCollision).includes("s.landBubble(l,c,d)")
      ) {
        throw new Error("The guarded v187 shooter/flying execution chain is unavailable or changed");
      }

      const keyOf = (x, y) => `${x},${y}`;
      const neighborsOf = (x, y) => {
        const offsets =
          Math.abs(y % 2) === 0
            ? [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0]]
            : [[0, -1], [-1, 0], [0, 1], [1, -1], [1, 1], [1, 0]];
        return offsets.map(([dx, dy]) => ({ x: x + dx, y: y + dy }));
      };

      const canonicalize = (value) => {
        if (Array.isArray(value)) return value.map(canonicalize);
        if (value && typeof value === "object") {
          return Object.fromEntries(
            Object.keys(value)
              .sort()
              .filter((key) => typeof value[key] !== "function" && value[key] !== undefined)
              .map((key) => [key, canonicalize(value[key])]),
          );
        }
        return value;
      };

      const assertStateStructure = (state) => {
        if (
          !state ||
          !state.grid ||
          !state.shooter ||
          !state.config?.physics ||
          !state.layout?.shooter ||
          !state.round ||
          state.grid.sizeW !== 17 ||
          !Number.isInteger(state.grid.sizeH) ||
          typeof state.grid.cells !== "object" ||
          Array.isArray(state.grid.cells) ||
          !Array.isArray(state.grid.holds) ||
          !Number.isFinite(state.config.physics.bubbleSpeed) ||
          !Number.isFinite(state.config.physics.collisionDiameter) ||
          !Number.isFinite(state.layout.shooter.x) ||
          !Number.isFinite(state.layout.shooter.y) ||
          !Array.isArray(state.config.bubbleNames) ||
          !runtime.boosters ||
          typeof runtime.boosters.getArmed !== "function"
        ) {
          throw new Error("Runtime state is structurally inconsistent");
        }
        for (const [key, cell] of Object.entries(state.grid.cells)) {
          if (
            key !== keyOf(cell.cellX, cell.cellY) ||
            !Number.isInteger(cell.cellX) ||
            !Number.isInteger(cell.cellY) ||
            !Number.isInteger(cell.type)
          ) {
            throw new Error(`Runtime grid cell ${key} is structurally inconsistent`);
          }
        }
      };

      const relevantSnapshot = (state = runtime.store.getState()) => {
        assertStateStructure(state);
        return canonicalize({
          grid: {
            cells: state.grid.cells,
            holds: state.grid.holds,
            sizeW: state.grid.sizeW,
            sizeH: state.grid.sizeH,
            scrollRows: state.grid.scrollRows,
            refilledLinesTotal: state.grid.refilledLinesTotal,
            bubblesPopped: state.grid.bubblesPopped,
            lastLandedKey: state.grid.lastLandedKey,
            lastMatches: state.grid.lastMatches,
            lastFloating: state.grid.lastFloating,
          },
          shooter: state.shooter,
          round: {
            score: state.round.score ?? null,
            gameOver: state.round.gameOver ?? null,
            movesUsed: state.round.levelState?.movesUsed ?? null,
          },
          physics: {
            bubbleSpeed: state.config.physics.bubbleSpeed,
            collisionDiameter: state.config.physics.collisionDiameter,
          },
          bubbleNames: state.config.bubbleNames,
          shooterLayout: state.layout.shooter,
          armedBooster: runtime.boosters.getArmed(),
        });
      };

      const serializeRelevantState = () => JSON.stringify(relevantSnapshot());
      const boardSnapshot = (state = runtime.store.getState()) => {
        assertStateStructure(state);
        return {
          cells: canonicalize(state.grid.cells),
          lastLandedKey: state.grid.lastLandedKey,
          lastMatches: canonicalize(state.grid.lastMatches),
          lastFloating: canonicalize(state.grid.lastFloating),
          scrollRows: state.grid.scrollRows,
          refilledLinesTotal: state.grid.refilledLinesTotal,
          bubblesPopped: state.grid.bubblesPopped,
          score: state.round.score ?? null,
          movesUsed: state.round.levelState?.movesUsed ?? null,
          gameOver: Boolean(state.round.gameOver),
          shooter: canonicalize(state.shooter),
        };
      };

      const roundEnded = (state) =>
        Boolean(state.round?.gameOver) || Object.keys(state.grid.cells).length === 0;

      const readiness = (state) => {
        assertStateStructure(state);
        const storeReady =
          state.shooter.ready === true &&
          Number.isInteger(state.shooter.currentType) &&
          state.shooter.currentType >= 0 &&
          Number.isInteger(state.shooter.nextType) &&
          state.shooter.nextType >= 0;
        const visualReady = shooter.isReadyToShoot() === true;
        const typeAligned = shooter.getCurrentType() === state.shooter.currentType;
        const flyingIdle = flying.hasActive === false;
        const boosterIdle = runtime.boosters.getArmed() == null;
        return {
          ready: storeReady && visualReady && typeAligned && flyingIdle && boosterIdle,
          storeReady,
          visualReady,
          typeAligned,
          flyingIdle,
          boosterIdle,
        };
      };

      const waitForReadyBoundary = async () => {
        const deadline = performance.now() + readyTimeoutMs;
        while (performance.now() < deadline) {
          const state = runtime.store.getState();
          assertStateStructure(state);
          if (roundEnded(state)) return { ended: true, state };
          const signals = readiness(state);
          if (signals.ready) return { ended: false, state };
          if (signals.storeReady && signals.visualReady && !signals.typeAligned) {
            throw new Error("Shooter readiness is ambiguous: store and visual types disagree");
          }
          await new Promise((resolve) => setTimeout(resolve, 50));
        }
        throw new Error("Shooter readiness could not be positively established");
      };

      const cloneGrid = (cells) =>
        new Map(Object.entries(cells).map(([key, cell]) => [key, { ...cell }]));

      const sameColorComponent = (grid, start, bubbleType) => {
        const component = [start];
        const queue = [start];
        const visited = new Set([keyOf(start.x, start.y)]);
        for (let index = 0; index < queue.length; index += 1) {
          const cell = queue[index];
          for (const neighbor of neighborsOf(cell.x, cell.y)) {
            const key = keyOf(neighbor.x, neighbor.y);
            if (visited.has(key)) continue;
            visited.add(key);
            if (grid.get(key)?.type === bubbleType) {
              component.push(neighbor);
              queue.push(neighbor);
            }
          }
        }
        return component;
      };

      const disconnectedCells = (grid, holds) => {
        const connected = new Set();
        const queue = [];
        for (const [x, y] of holds) {
          const key = keyOf(x, y);
          if (grid.has(key) && !connected.has(key)) {
            connected.add(key);
            queue.push({ x, y });
          }
        }
        for (let index = 0; index < queue.length; index += 1) {
          const cell = queue[index];
          for (const neighbor of neighborsOf(cell.x, cell.y)) {
            const key = keyOf(neighbor.x, neighbor.y);
            if (grid.has(key) && !connected.has(key)) {
              connected.add(key);
              queue.push(neighbor);
            }
          }
        }
        return [...grid.keys()].filter((key) => !connected.has(key));
      };

      const maximumOccupiedRow = (grid) => {
        let maximum = -1;
        for (const cell of grid.values()) maximum = Math.max(maximum, cell.cellY);
        return maximum;
      };

      const compareTuples = (left, right) => {
        for (let index = 0; index < left.length; index += 1) {
          if (left[index] !== right[index]) return right[index] - left[index];
        }
        return 0;
      };

      const evaluateBestShot = (state, stableSnapshot) => {
        const uniqueLandings = new Map();
        const currentType = state.shooter.currentType;
        const speed = state.config.physics.bubbleSpeed;
        const collisionRadius = state.config.physics.collisionDiameter;
        const start = state.layout.shooter;
        const minRow = -state.grid.scrollRows;
        const maxRow = state.grid.sizeH - 1 - state.grid.scrollRows;

        for (const aimAngleDegrees of aimAngles) {
          const gameAngleDegrees = aimAngleDegrees - 90;
          const radians = (gameAngleDegrees * Math.PI) / 180;
          const simulated = orchestrator.simulate(
            {
              x: start.x,
              y: start.y,
              vel: {
                x: Math.cos(radians) * speed,
                y: Math.sin(radians) * speed,
              },
              cell: { x: 0, y: 0 },
              prevCell: { x: 0, y: 0 },
              bubbleType: currentType,
              collRadius: collisionRadius,
            },
            maxSteps,
          );
          if (serializeRelevantState() !== stableSnapshot) {
            throw new Error("Live state changed during pure trajectory evaluation");
          }
          const landing = simulated.collision;
          if (
            !landing ||
            !Number.isInteger(landing.x) ||
            !Number.isInteger(landing.y) ||
            landing.x < 0 ||
            landing.x >= state.grid.sizeW ||
            landing.y < minRow ||
            landing.y > maxRow ||
            state.grid.cells[keyOf(landing.x, landing.y)]
          ) {
            continue;
          }

          const grid = cloneGrid(state.grid.cells);
          const landingKey = keyOf(landing.x, landing.y);
          const sameColorAdjacent = neighborsOf(landing.x, landing.y).filter(
            (neighbor) => grid.get(keyOf(neighbor.x, neighbor.y))?.type === currentType,
          ).length;
          grid.set(landingKey, {
            type: currentType,
            cellX: landing.x,
            cellY: landing.y,
          });

          const component = sameColorComponent(grid, landing, currentType);
          const directPops = component.length >= 3 ? component : [];
          for (const cell of directPops) grid.delete(keyOf(cell.x, cell.y));
          const dropKeys = directPops.length > 0 ? disconnectedCells(grid, state.grid.holds) : [];
          for (const key of dropKeys) grid.delete(key);

          const resultingMaximumRow = maximumOccupiedRow(grid);
          const scoreTuple = [
            dropKeys.length,
            directPops.length,
            -resultingMaximumRow,
            directPops.length === 0 ? sameColorAdjacent : 0,
            -Math.abs(aimAngleDegrees),
          ];
          const candidate = {
            aimAngleDegrees,
            gameAngleDegrees,
            landing: { x: landing.x, y: landing.y },
            directPops,
            dropKeys,
            expectedCells: Object.fromEntries(
              [...grid.entries()].sort(([left], [right]) => left.localeCompare(right)),
            ),
            scoreTuple,
          };
          const existing = uniqueLandings.get(landingKey);
          if (!existing || compareTuples(candidate.scoreTuple, existing.scoreTuple) < 0) {
            uniqueLandings.set(landingKey, candidate);
          }
        }

        const ranked = [...uniqueLandings.values()].sort((left, right) =>
          compareTuples(left.scoreTuple, right.scoreTuple),
        );
        if (ranked.length === 0) throw new Error("No valid unique landing cell was simulated");
        return ranked[0];
      };

      const verifyImmediateResult = (selected, afterLanding) => {
        const predictedPopKeys = selected.directPops
          .map((cell) => keyOf(cell.x, cell.y))
          .sort();
        const predictedDropKeys = [...selected.dropKeys].sort();
        const actualPopKeys = afterLanding.lastMatches
          .map((cell) => keyOf(cell.x, cell.y))
          .sort();
        const actualDropKeys = afterLanding.lastFloating
          .map((cell) => keyOf(cell.x, cell.y))
          .sort();
        if (afterLanding.lastLandedKey !== keyOf(selected.landing.x, selected.landing.y)) {
          throw new Error("Actual landing cell differs from the selected simulation");
        }
        if (JSON.stringify(actualPopKeys) !== JSON.stringify(predictedPopKeys)) {
          throw new Error("Actual direct-pop cells differ from the pure evaluator prediction");
        }
        if (JSON.stringify(actualDropKeys) !== JSON.stringify(predictedDropKeys)) {
          throw new Error("Actual dropped cells differ from the pure evaluator prediction");
        }
        if (
          JSON.stringify(afterLanding.cells) !==
          JSON.stringify(canonicalize(selected.expectedCells))
        ) {
          throw new Error("Immediate landed grid contains an unrelated mutation");
        }
        for (const key of [...predictedPopKeys, ...predictedDropKeys]) {
          if (afterLanding.cells[key]) {
            throw new Error(`Predicted removed cell ${key} remains after landing`);
          }
        }
        return { actualPopKeys, actualDropKeys };
      };

      const verifyFinalGrid = (selected, afterLanding, afterReady) => {
        const refillRows = afterReady.refilledLinesTotal - afterLanding.refilledLinesTotal;
        const scrollDelta = afterReady.scrollRows - afterLanding.scrollRows;
        const expectedEntries = Object.entries(selected.expectedCells);
        if (refillRows === 0 && scrollDelta === 0) {
          if (JSON.stringify(afterReady.cells) !== JSON.stringify(afterLanding.cells)) {
            throw new Error("Final grid changed without a reported refill or scroll");
          }
          return;
        }
        if (refillRows < 0 || scrollDelta < 0) {
          throw new Error("Grid refill/scroll counters moved backward");
        }
        if (scrollDelta > 0) {
          for (const [key, cell] of expectedEntries) {
            if (afterReady.cells[key]?.type !== cell.type) {
              throw new Error(`Survivor ${key} disappeared during smooth grid scrolling`);
            }
          }
          return;
        }
        for (const [, cell] of expectedEntries) {
          const shiftedKey = keyOf(cell.cellX, cell.cellY + refillRows);
          if (afterReady.cells[shiftedKey]?.type !== cell.type) {
            throw new Error(
              `Survivor ${cell.cellX},${cell.cellY} disappeared during grid refill`,
            );
          }
        }
      };

      const signed = (value) => (value >= 0 ? `+${value}` : String(value));
      const lines = [];
      let shotsCompleted = 0;
      let totalScoreDelta = 0;
      let totalDirectPops = 0;
      let totalDrops = 0;
      let totalRefillEvents = 0;
      let totalScrollEvents = 0;
      let stopReason = "10-shot limit reached";

      for (let shotNumber = 1; shotNumber <= maxShots; shotNumber += 1) {
        try {
          const boundary = await waitForReadyBoundary();
          if (boundary.ended) {
            stopReason = "game over / round complete";
            break;
          }

          const evaluationState = boundary.state;
          const evaluationSnapshot = serializeRelevantState();
          const selected = evaluateBestShot(evaluationState, evaluationSnapshot);
          if (serializeRelevantState() !== evaluationSnapshot) {
            throw new Error("Live state changed between evaluation and execution");
          }

          const preShotState = runtime.store.getState();
          assertStateStructure(preShotState);
          if (roundEnded(preShotState)) {
            stopReason = "game over / round complete before execution";
            break;
          }
          if (!readiness(preShotState).ready) {
            throw new Error("Shooter readiness became ambiguous before execution");
          }
          const preShot = boardSnapshot(preShotState);
          const currentType = preShotState.shooter.currentType;
          const currentColor = preShotState.config.bubbleNames[currentType] ?? `TYPE_${currentType}`;

          let shootMethodInvocations = 0;
          let bubbleShotEvents = 0;
          let bubbleLandedEvents = 0;
          let refillEvents = 0;
          let refillRowsFromEvents = 0;
          let scrollEvents = 0;
          let afterLanding = null;

          const unsubscribeShot = runtime.store.on("BubbleShot", () => {
            bubbleShotEvents += 1;
          });
          const unsubscribeLanded = runtime.store.on("BubbleLanded", () => {
            bubbleLandedEvents += 1;
            afterLanding = boardSnapshot();
          });
          const unsubscribeRefilled = runtime.store.on("GridRefilled", (_state, event) => {
            refillEvents += 1;
            refillRowsFromEvents += event.rowsAdded ?? 0;
            if (event.scrolled) scrollEvents += 1;
          });

          try {
            if (serializeRelevantState() !== evaluationSnapshot) {
              throw new Error("Live state changed immediately before execution");
            }
            shootMethodInvocations += 1;
            const launched = orchestrator.shoot(selected.gameAngleDegrees);
            if (!launched) {
              throw new Error("Runtime rejected the selected shot; no retry attempted");
            }
            shooter.consumeCurrentBubble();
            shooter.onShoot({
              x: launched.x,
              y: launched.y,
              velX: launched.vel.x,
              velY: launched.vel.y,
              type: launched.bubbleType,
              angle: selected.gameAngleDegrees,
            });

            const deadline = performance.now() + resolutionTimeoutMs;
            while (true) {
              if (shootMethodInvocations > 1 || bubbleShotEvents > 1 || bubbleLandedEvents > 1) {
                throw new Error("More than one shot event occurred; no retry attempted");
              }
              const liveState = runtime.store.getState();
              assertStateStructure(liveState);
              const ended = roundEnded(liveState);
              const resolvedReady =
                bubbleLandedEvents === 1 &&
                flying.hasActive === false &&
                (ended || readiness(liveState).ready);
              if (resolvedReady) break;
              if (performance.now() >= deadline) {
                throw new Error("Shot resolution could not be positively established; no retry attempted");
              }
              await new Promise((resolve) => setTimeout(resolve, 50));
            }
          } finally {
            unsubscribeShot();
            unsubscribeLanded();
            unsubscribeRefilled();
          }

          if (
            shootMethodInvocations !== 1 ||
            bubbleShotEvents !== 1 ||
            bubbleLandedEvents !== 1 ||
            !afterLanding
          ) {
            throw new Error("Exactly-one-shot event verification failed");
          }

          const { actualPopKeys, actualDropKeys } = verifyImmediateResult(
            selected,
            afterLanding,
          );
          const afterReadyState = runtime.store.getState();
          const afterReady = boardSnapshot(afterReadyState);
          verifyFinalGrid(selected, afterLanding, afterReady);

          const ended = roundEnded(afterReadyState);
          if (!ended) {
            if (
              afterReadyState.shooter.currentType !== preShot.shooter.nextType ||
              !readiness(afterReadyState).ready
            ) {
              throw new Error("Shooter did not advance to the prior next bubble and become ready");
            }
          }

          const scoreDelta =
            typeof preShot.score === "number" && typeof afterReady.score === "number"
              ? afterReady.score - preShot.score
              : 0;
          const scrollDelta = afterReady.scrollRows - preShot.scrollRows;
          const refillRows = afterReady.refilledLinesTotal - preShot.refilledLinesTotal;
          const readyAfter = !ended && readiness(afterReadyState).ready;

          shotsCompleted += 1;
          totalScoreDelta += scoreDelta;
          totalDirectPops += actualPopKeys.length;
          totalDrops += actualDropKeys.length;
          totalRefillEvents += refillEvents;
          totalScrollEvents += scrollEvents;
          lines.push(
            `shot ${shotNumber} | ${currentColor} | aim ${signed(selected.aimAngleDegrees)}deg (${selected.gameAngleDegrees}deg API) | landing ${selected.landing.x},${selected.landing.y} | predicted ${selected.directPops.length}/${selected.dropKeys.length} pops/drops | score ${signed(scoreDelta)} | refill ${refillEvents} events/${signed(refillRows)} rows (observed ${signed(refillRowsFromEvents)}) / scroll ${scrollEvents} events/${signed(scrollDelta)} rows | ready ${readyAfter}`,
          );

          if (ended) {
            stopReason = "game over / round complete";
            break;
          }
        } catch (error) {
          stopReason = `safety stop: ${error instanceof Error ? error.message : String(error)}`;
          break;
        }
      }

      return {
        lines,
        summary: {
          shotsCompleted,
          totalScoreDelta,
          totalDirectPops,
          totalDrops,
          refillCount: totalRefillEvents,
          scrollEvents: totalScrollEvents,
          stopReason,
        },
      };
    },
    {
      aimAngles: AIM_ANGLES,
      maxShots: MAX_SHOTS,
      maxSteps: MAX_STEPS,
      readyTimeoutMs: READY_TIMEOUT_MS,
      resolutionTimeoutMs: RESOLUTION_TIMEOUT_MS,
    },
  );

  for (const line of result.lines) process.stdout.write(`${line}\n`);
  process.stdout.write(`shots completed: ${result.summary.shotsCompleted}\n`);
  process.stdout.write(`total score delta: ${result.summary.totalScoreDelta}\n`);
  process.stdout.write(`total direct pops: ${result.summary.totalDirectPops}\n`);
  process.stdout.write(`total drops: ${result.summary.totalDrops}\n`);
  process.stdout.write(`refill count: ${result.summary.refillCount}\n`);
  process.stdout.write(`scroll events: ${result.summary.scrollEvents}\n`);
  process.stdout.write(`stop reason: ${result.summary.stopReason}\n`);
} finally {
  await browser.close();
}
