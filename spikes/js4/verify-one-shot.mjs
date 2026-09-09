import { openRuntimeSession } from "./runtime-session.mjs";

const MAX_STEPS = 200;
const AIM_ANGLES = Array.from({ length: 161 }, (_, index) => index - 80);
const RESOLUTION_TIMEOUT_MS = 30_000;

const { browser, frame } = await openRuntimeSession();
try {
  const report = await frame.evaluate(
    async ({ aimAngles, maxSteps, resolutionTimeoutMs }) => {
      const runtime = window.__bubblebot;
      const orchestrator = runtime.orchestrator;
      const shootSource = Function.prototype.toString.call(orchestrator.shoot);
      if (
        orchestrator.shoot.length !== 1 ||
        !shootSource.includes('dispatch({type:"BubbleShot",angle:b})') ||
        !shootSource.includes("return{x:I.x,y:I.y,vel:E")
      ) {
        throw new Error("The live v187 orchestrator.shoot API is ambiguous or changed");
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
      const onShootSource = Function.prototype.toString.call(shooter.onShoot);
      const collisionSource = Function.prototype.toString.call(flying.onCollision);
      if (!onShootSource.includes("addBubble(E.x,E.y,E.velX,E.velY,E.type)")) {
        throw new Error("The shooter-to-flying callback no longer matches v187");
      }
      if (!collisionSource.includes("s.landBubble(l,c,d)")) {
        throw new Error("The flying collision-to-grid callback no longer matches v187");
      }

      const keyOf = (x, y) => `${x},${y}`;
      const neighborsOf = (x, y) => {
        const offsets =
          Math.abs(y % 2) === 0
            ? [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0]]
            : [[0, -1], [-1, 0], [0, 1], [1, -1], [1, 1], [1, 0]];
        return offsets.map(([dx, dy]) => ({ x: x + dx, y: y + dy }));
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
      const fullSnapshot = () => canonicalize(runtime.store.getState());
      const serializedSnapshot = () => JSON.stringify(fullSnapshot());
      const boardSnapshot = () => {
        const state = runtime.store.getState();
        return {
          cells: canonicalize(state.grid.cells),
          lastLandedKey: state.grid.lastLandedKey,
          lastMatches: canonicalize(state.grid.lastMatches),
          lastFloating: canonicalize(state.grid.lastFloating),
          scrollRows: state.grid.scrollRows,
          refilledLinesTotal: state.grid.refilledLinesTotal,
          bubblesPopped: state.grid.bubblesPopped,
          score: state.round?.score ?? null,
          movesUsed: state.round?.levelState?.movesUsed ?? null,
          shooter: canonicalize(state.shooter),
        };
      };

      const compareTuples = (left, right) => {
        for (let index = 0; index < left.length; index += 1) {
          if (left[index] !== right[index]) return right[index] - left[index];
        }
        return 0;
      };

      const evaluateBestShot = (state) => {
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
          const velocity = {
            x: Math.cos(radians) * speed,
            y: Math.sin(radians) * speed,
          };
          const simulated = orchestrator.simulate(
            {
              x: start.x,
              y: start.y,
              vel: velocity,
              cell: { x: 0, y: 0 },
              prevCell: { x: 0, y: 0 },
              bubbleType: currentType,
              collRadius: collisionRadius,
            },
            maxSteps,
          );
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
            expectedCells: Object.fromEntries([...grid.entries()].sort(([a], [b]) => a.localeCompare(b))),
            resultingMaximumRow,
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
        if (ranked.length === 0) throw new Error("No valid shot was available");
        return ranked[0];
      };

      const beforeEvaluation = serializedSnapshot();
      const initialState = runtime.store.getState();
      const selected = evaluateBestShot(initialState);
      if (serializedSnapshot() !== beforeEvaluation) {
        throw new Error("Live state changed during pre-shot evaluation");
      }

      const stateBeforeShot = runtime.store.getState();
      if (
        !stateBeforeShot.shooter.ready ||
        stateBeforeShot.shooter.currentType === null ||
        stateBeforeShot.shooter.nextType < 0 ||
        stateBeforeShot.round?.gameOver ||
        !shooter.isReadyToShoot() ||
        flying.hasActive ||
        shooter.getCurrentType() !== stateBeforeShot.shooter.currentType ||
        runtime.boosters.getArmed() != null
      ) {
        throw new Error("Pre-shot runtime readiness checks failed; no shot executed");
      }
      const beforeShot = fullSnapshot();
      const beforeShotSerialized = JSON.stringify(beforeShot);
      if (beforeShotSerialized !== beforeEvaluation) {
        throw new Error("Live state changed between evaluation and execution; no shot executed");
      }
      const beforeSummary = boardSnapshot();
      const visualChancesBefore = shooter.vLCur;

      let shootMethodInvocations = 0;
      let bubbleShotEvents = 0;
      let bubbleLandedEvents = 0;
      let gridRefilledEvents = 0;
      let afterLanding = null;
      const unsubscribeShot = runtime.store.on("BubbleShot", () => {
        bubbleShotEvents += 1;
      });
      const unsubscribeLanded = runtime.store.on("BubbleLanded", () => {
        bubbleLandedEvents += 1;
        afterLanding = boardSnapshot();
      });
      const unsubscribeRefilled = runtime.store.on("GridRefilled", () => {
        gridRefilledEvents += 1;
      });

      try {
        shootMethodInvocations += 1;
        const launched = orchestrator.shoot(selected.gameAngleDegrees);
        if (!launched) {
          throw new Error("orchestrator.shoot returned null; no retry attempted");
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
          if (bubbleShotEvents > 1 || bubbleLandedEvents > 1 || shootMethodInvocations > 1) {
            throw new Error("More than one shot was observed; stopping without retry");
          }
          const current = runtime.store.getState();
          if (
            bubbleLandedEvents === 1 &&
            !flying.hasActive &&
            current.shooter.ready &&
            shooter.isReadyToShoot()
          ) {
            break;
          }
          if (performance.now() >= deadline) {
            throw new Error("Timed out waiting for the one shot to resolve; no retry attempted");
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

      const sortedKeys = (cells) => Object.keys(cells).sort();
      const predictedPopKeys = selected.directPops.map((cell) => keyOf(cell.x, cell.y)).sort();
      const predictedDropKeys = [...selected.dropKeys].sort();
      const actualPopKeys = afterLanding.lastMatches
        .map((cell) => keyOf(cell.x, cell.y))
        .sort();
      const actualDropKeys = afterLanding.lastFloating
        .map((cell) => keyOf(cell.x, cell.y))
        .sort();
      if (afterLanding.lastLandedKey !== keyOf(selected.landing.x, selected.landing.y)) {
        throw new Error("Actual landing cell differs from the prediction");
      }
      const [actualLandingX, actualLandingY] = afterLanding.lastLandedKey
        .split(",")
        .map(Number);
      if (JSON.stringify(actualPopKeys) !== JSON.stringify(predictedPopKeys)) {
        throw new Error("Actual directly popped cells differ from the prediction");
      }
      if (JSON.stringify(actualDropKeys) !== JSON.stringify(predictedDropKeys)) {
        throw new Error("Actual dropped cells differ from the prediction");
      }
      if (
        JSON.stringify(afterLanding.cells) !== JSON.stringify(canonicalize(selected.expectedCells))
      ) {
        throw new Error("Immediate post-landing grid contains an unrelated mutation");
      }
      for (const key of [...predictedPopKeys, ...predictedDropKeys]) {
        if (afterLanding.cells[key]) {
          throw new Error(`Predicted removed cell ${key} remains after landing`);
        }
      }

      const afterSummary = boardSnapshot();
      const finalState = runtime.store.getState();
      const refillRows =
        afterSummary.refilledLinesTotal - afterLanding.refilledLinesTotal;
      const scrollDelta = afterSummary.scrollRows - afterLanding.scrollRows;
      const expectedEntries = Object.entries(selected.expectedCells);
      if (refillRows === 0 && scrollDelta === 0) {
        if (JSON.stringify(afterSummary.cells) !== JSON.stringify(afterLanding.cells)) {
          throw new Error("Final grid changed without a reported refill or scroll");
        }
      } else if (scrollDelta > 0) {
        for (const [key, cell] of expectedEntries) {
          if (afterSummary.cells[key]?.type !== cell.type) {
            throw new Error(`Pre-refill survivor ${key} disappeared during smooth scrolling`);
          }
        }
      } else {
        for (const [, cell] of expectedEntries) {
          const shiftedKey = keyOf(cell.cellX, cell.cellY + refillRows);
          if (afterSummary.cells[shiftedKey]?.type !== cell.type) {
            throw new Error(`Pre-refill survivor ${cell.cellX},${cell.cellY} disappeared during refill`);
          }
        }
      }

      const expectedCurrentType = beforeSummary.shooter.nextType;
      if (
        finalState.shooter.currentType !== expectedCurrentType ||
        !finalState.shooter.ready ||
        shooter.getCurrentType() !== finalState.shooter.currentType
      ) {
        throw new Error("Shooter current/next progression differs from the expected one-shot advance");
      }

      const delta = (before, after) =>
        before === null || after === null ? null : after - before;
      return {
        selected: {
          aimOffsetDegrees: selected.aimAngleDegrees,
          gameApiAngleDegrees: selected.gameAngleDegrees,
          predictedLandingCell: selected.landing,
          predictedDirectPopCount: selected.directPops.length,
          predictedDropCount: selected.dropKeys.length,
        },
        verification: {
          shootMethodInvocations,
          bubbleShotEvents,
          bubbleLandedEvents,
          actualLandingCell: { x: actualLandingX, y: actualLandingY },
          predictedPoppedCells: predictedPopKeys,
          actualPoppedCells: actualPopKeys,
          predictedDroppedCells: predictedDropKeys,
          actualDroppedCells: actualDropKeys,
          poppedAndDroppedCellsAbsentImmediately: true,
          noUnrelatedCellsDisappeared: true,
          shooter: {
            beforeCurrentType: beforeSummary.shooter.currentType,
            beforeNextType: beforeSummary.shooter.nextType,
            expectedCurrentType,
            afterCurrentType: afterSummary.shooter.currentType,
            afterNextType: afterSummary.shooter.nextType,
            readyAfter: afterSummary.shooter.ready,
          },
          grid: {
            gridRefilledEvents,
            scrollRowsBefore: beforeSummary.scrollRows,
            scrollRowsAfter: afterSummary.scrollRows,
            scrollRowsDelta: afterSummary.scrollRows - beforeSummary.scrollRows,
            refilledLinesBefore: beforeSummary.refilledLinesTotal,
            refilledLinesAfter: afterSummary.refilledLinesTotal,
            refilledLinesDelta: afterSummary.refilledLinesTotal - beforeSummary.refilledLinesTotal,
          },
          deltas: {
            score: delta(beforeSummary.score, afterSummary.score),
            movesUsed: delta(beforeSummary.movesUsed, afterSummary.movesUsed),
            chances: delta(beforeSummary.shooter.lives, afterSummary.shooter.lives),
            visualChances: shooter.vLCur - visualChancesBefore,
            bubblesPopped:
              afterSummary.bubblesPopped - beforeSummary.bubblesPopped,
          },
        },
      };
    },
    {
      aimAngles: AIM_ANGLES,
      maxSteps: MAX_STEPS,
      resolutionTimeoutMs: RESOLUTION_TIMEOUT_MS,
    },
  );

  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} finally {
  await browser.close();
}
