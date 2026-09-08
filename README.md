# Bubblebot

Phase 0/1 implementation for local, passive Bubble Shooter analysis. It never clicks.

## Detect a screenshot

```powershell
python main.py --detect screenshot.png --debug-out debug/detected.png --state-out debug/state.json
```

If no screenshot is supplied, capture the primary monitor:

```powershell
python main.py --capture --debug-out debug/live.png --state-out debug/state.json
```

The command writes a parsed state JSON, prints an offset-grid console view, and writes an annotated image. Detection is intentionally conservative: low-confidence results are reported rather than invented.

## Tests

```powershell
pytest -q
```

Current scope: passive capture, board calibration, bubble detection, color clustering, launcher/current/next heuristics, diagnostics. Simulator, search, and input control are subsequent phases.
