"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import json
import numpy as np
"""

class LayoutContext:
    """Hold pillar geometry and the original layout image."""

    def __init__(self,
                 layout_img_path: str | Path,
                 layout_json_path: str | Path | None = None) -> None:
        self.layout_img_path = Path(layout_img_path)
        self.image = cv2.imread(str(self.layout_img_path))
        if self.image is None:
            raise RuntimeError("Could not load layout image")

        # ------------------------------------------------------------------ pillars
        if layout_json_path:
            self._load_pillars_from_json(layout_json_path)
        else:
            self._auto_detect_pillars()
        self._build_simple_index()

    # .................................................... pillar ingestion
    def _load_pillars_from_json(self, json_path: str | Path) -> None:
        data: Dict = json.loads(Path(json_path).read_text())
        self.pillars: List[Dict] = data.get("pillars", [])

    def _auto_detect_pillars(self) -> None:
        """Fallback: naïve circle detection on yellow blobs."""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # mask yellow
        mask = cv2.inRange(hsv, (18,  70,  80), (32, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        pillars = []
        for idx, cnt in enumerate(contours, 1):
            (x, y), rad = cv2.minEnclosingCircle(cnt)
            if rad < 5:          # discard noise
                continue
            pillars.append({
                "id": f"P{idx}",
                "center_px": [int(x), int(y)],
                "radius_px": int(rad)
            })
        self.pillars = sorted(pillars, key=lambda p: (p["center_px"][0],
                                                      p["center_px"][1]))

    def _build_simple_index(self):
        """Numpy arrays for quick maths."""
        self._centers_np = np.array([p["center_px"] for p in self.pillars])

    # .................................................... public helpers
    def nearest_pillar_id(self, x: int, y: int) -> str | None:
        """Return ID of the geometrically nearest pillar."""
        if len(self._centers_np) == 0:
            return None
        dists = np.linalg.norm(self._centers_np - (x, y), axis=1)
        return self.pillars[int(dists.argmin())]["id"]

    def draw_location_marker(self,
                             loc_px: Tuple[int, int],
                             out_img_path: str) -> None:
        """Save a copy of the layout image with a red dot where the activity is."""
        img = self.image.copy()
        cv2.circle(img, loc_px, 12, (0, 0, 255), -1)
        cv2.imwrite(out_img_path, img)