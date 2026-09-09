import { openRuntimeSession } from "./runtime-session.mjs";

const REPRESENTATIVE_SHOTS = [
  { name: "center", angleDegrees: -90 },
  { name: "left-wall", angleDegrees: -150 },
  { name: "right-wall", angleDegrees: -30 },
];
const MAX_STEPS = 200;

const { browser, frame } = await openRuntimeSession();
try {
  const proof = await frame.evaluate(
    ({ shots, maxSteps }) => {
      const runtime = window.__bubblebot;
      const orchestrator = runtime.orchestrator;
      if (typeof orchestrator.simulate !== "function" || orchestrator.simulate.length !== 1) {
        throw new Error("The inspected v187 simulate API is not present with the expected arity");
      }

      const simulatorSource = Function.prototype.toString.call(orchestrator.simulate);
      if (!simulatorSource.includes("return L8(b,S,E,A)") || simulatorSource.includes("dispatch")) {
        throw new Error("The v187 simulator source no longer matches the inspected non-mutating body");
      }

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
      const speed = state.config.physics.bubbleSpeed;
      const shooter = state.layout.shooter;
      const bubbleType = state.shooter.currentType;
      const collisionRadius = state.config.physics.collisionDiameter;
      const round = (value) => Math.round(value * 1000) / 1000;

      const wallBouncePoints = (points) => {
        const bounces = [];
        let previousDirection = 0;
        for (let index = 1; index < points.length; index += 1) {
          const direction = Math.sign(points[index].x - points[index - 1].x);
          if (previousDirection !== 0 && direction !== 0 && direction !== previousDirection) {
            bounces.push({
              x: round(points[index - 1].x),
              y: round(points[index - 1].y),
            });
          }
          if (direction !== 0) previousDirection = direction;
        }
        return bounces;
      };

      const trajectories = [];
      for (const shot of shots) {
        const radians = (shot.angleDegrees * Math.PI) / 180;
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
          bubbleType,
          collRadius: collisionRadius,
        };

        const simulated = orchestrator.simulate(bubble, maxSteps);
        const afterThisSimulation = serializeSnapshot();
        if (afterThisSimulation !== before) {
          throw new Error(`Live state mutated during ${shot.name} simulation`);
        }

        const pathPoints = simulated.positions.map((point) => ({
          x: round(point.x),
          y: round(point.y),
        }));
        const bouncePoints = wallBouncePoints(simulated.positions);
        trajectories.push({
          name: shot.name,
          input: {
            angleDegrees: shot.angleDegrees,
            maxSteps,
            start: { x: bubble.x, y: bubble.y },
            velocity: { x: round(velocity.x), y: round(velocity.y) },
            bubbleType,
            collisionRadius,
          },
          pathPoints,
          collisionCell: simulated.collision,
          collisionPoint: simulated.collision
            ? { x: round(simulated.collisionX), y: round(simulated.collisionY) }
            : null,
          wallBounceCount: bouncePoints.length,
          wallBouncePoints: bouncePoints,
        });
      }

      const after = serializeSnapshot();
      if (after !== before) {
        throw new Error("Live state differs after the trajectory simulation set");
      }
      return { liveStateUnchanged: true, trajectories };
    },
    { shots: REPRESENTATIVE_SHOTS, maxSteps: MAX_STEPS },
  );

  process.stdout.write(`${JSON.stringify(proof, null, 2)}\n`);
} finally {
  await browser.close();
}
