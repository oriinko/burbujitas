from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from bubblebot.capture.screen import capture_primary_monitor
from bubblebot.diagnostics.overlay import draw_debug_overlay
from bubblebot.vision.detector import detect_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Passive Bubble Shooter detector; never clicks")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--detect", type=Path, help="input screenshot")
    source.add_argument("--capture", action="store_true", help="capture the primary monitor")
    parser.add_argument("--debug-out", type=Path, default=Path("debug/detected.png"))
    parser.add_argument("--state-out", type=Path, default=Path("debug/state.json"))
    args = parser.parse_args()

    image = cv2.imread(str(args.detect)) if args.detect else capture_primary_monitor()
    if image is None:
        raise SystemExit("Could not read screenshot")
    state = detect_image(image)
    args.debug_out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    cv2.imwrite(str(args.debug_out), draw_debug_overlay(image, state))
    print(state.console_grid())
    print(json.dumps({"current": state.current, "next": state.next_bubble, "calibration": state.calibration, "confidence": state.confidence}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
