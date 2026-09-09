import { openRuntimeSession } from "./runtime-session.mjs";

const MAX_STEPS = 200;
const AIM_ANGLES = Array.from({ length: 161 }, (_, index) => index - 80);

const { browser, frame } = await openRuntimeSession();
try {
  const proof = await frame.evaluate(
    ({ aimAngles, maxSteps }) => {
      const runtime = window.__bubblebot;
      const orchestrator = runtime.orchestrator;
      if (typeof orchestrator.simulate !== "function" || orchestrator.simulate.length !== 1) {
        throw new Error("The inspected v187 simulate API is unavailable or changed");
      }
      const simulatorSource = Function.prototype.toString.call(orchestrator.simulate);
      if (!simulatorSource.includes("return L8(b,S,E,A)") || simulatorSource.includes("dispatch")) {
        throw new Error("The v187 simulator no longer matches the inspected non-mutating body");
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
        for (const hold of holds) {
          const [x, y] = hold;
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

      const canonicalSnapshot = () => {
        const state = runtime.store.getState();
        const cells = Object.fromEntries(
          Object.entries(state.grid.cells)
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([key, cell]) => [key, { ...cell }]),
        );
        const levelState = state.round?.levelState;
        return {
          grid: {
            cells,
            holds: state.grid.holds.map((hold) => [...hold]),
            sizeW: state.grid.sizeW,
            sizeH: state.grid.sizeH,
            scrollRows: state.grid.scrollRows,
            refilledLinesTotal: state.grid.refilledLinesTotal,
          },
          shooter: {
            currentType: state.shooter.currentType,
            nextType: state.shooter.nextType,
            ready: state.shooter.ready,
            lives: state.shooter.lives,
            maxLives: state.shooter.maxLives,
            chanceCycle: state.shooter.chanceCycle,
            needsRefill: state.shooter.needsRefill,
          },
          round: {
            score: state.round?.score ?? null,
            firstShot: state.round?.firstShot ?? null,
            gameOver: state.round?.gameOver ?? null,
            movesUsed: levelState?.movesUsed ?? null,
            maxMoves: levelState?.targets?.maxMoves ?? null,
          },
        };
      };

      const serializeSnapshot = () => JSON.stringify(canonicalSnapshot());
      const before = serializeSnapshot();
      const state = runtime.store.getState();
      if (!Array.isArray(state.grid.holds)) {
        throw new Error("The v187 ceiling-anchor list is unavailable");
      }

      const currentType = state.shooter.currentType;
      const currentColor = state.config.bubbleNames[currentType];
      const speed = state.config.physics.bubbleSpeed;
      const collisionRadius = state.config.physics.collisionDiameter;
      const shooter = state.layout.shooter;
      const minRow = -state.grid.scrollRows;
      const maxRow = state.grid.sizeH - 1 - state.grid.scrollRows;
      const uniqueLandings = new Map();

      const compareTuples = (left, right) => {
        for (let index = 0; index < left.length; index += 1) {
          if (left[index] !== right[index]) return right[index] - left[index];
        }
        return 0;
      };

      for (const aimAngleDegrees of aimAngles) {
        const gameAngleDegrees = aimAngleDegrees - 90;
        const radians = (gameAngleDegrees * Math.PI) / 180;
        const velocity = {
          x: Math.cos(radians) * speed,
          y: Math.sin(radians) * speed,
        };
        const bubble = {
          x: shooter.x,
          y: shooter.y,
          vel: velocity,
          cell: { x: 0, y: 0 },
          prevCell: { x: 0, y: 0 },
          bubbleType: currentType,
          collRadius: collisionRadius,
        };

        const simulation = orchestrator.simulate(bubble, maxSteps);
        if (serializeSnapshot() !== before) {
          throw new Error(`Live state mutated while simulating aim angle ${aimAngleDegrees}`);
        }
        const landing = simulation.collision;
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

        const drops = directPops.length > 0 ? disconnectedCells(grid, state.grid.holds) : [];
        for (const key of drops) grid.delete(key);
        const resultingMaximumRow = maximumOccupiedRow(grid);
        const adjacencyTieBreak = directPops.length === 0 ? sameColorAdjacent : 0;
        const scoreTuple = [
          drops.length,
          directPops.length,
          -resultingMaximumRow,
          adjacencyTieBreak,
          -Math.abs(aimAngleDegrees),
        ];
        const candidate = {
          angle: {
            aimDegrees: aimAngleDegrees,
            gameDegrees: gameAngleDegrees,
          },
          landingCell: { x: landing.x, y: landing.y },
          directPopCount: directPops.length,
          dropCount: drops.length,
          resultingMaximumOccupiedRow: resultingMaximumRow,
          finalScoreTuple: scoreTuple,
        };

        const existing = uniqueLandings.get(landingKey);
        if (!existing || compareTuples(candidate.finalScoreTuple, existing.finalScoreTuple) < 0) {
          uniqueLandings.set(landingKey, candidate);
        }
      }

      const ranked = [...uniqueLandings.values()].sort((left, right) =>
        compareTuples(left.finalScoreTuple, right.finalScoreTuple),
      );
      if (ranked.length === 0) throw new Error("No valid unique landing cells were simulated");
      if (serializeSnapshot() !== before) {
        throw new Error("Live state differs after pure shot evaluation");
      }

      return {
        currentBubble: { type: currentType, color: currentColor },
        anglesTested: aimAngles.length,
        uniqueLandingCells: ranked.length,
        top10: ranked.slice(0, 10),
        selectedBestShot: ranked[0],
        liveStateUnchanged: true,
      };
    },
    { aimAngles: AIM_ANGLES, maxSteps: MAX_STEPS },
  );

  process.stdout.write(`${JSON.stringify(proof, null, 2)}\n`);
} finally {
  await browser.close();
}
