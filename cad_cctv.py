#!/usr/bin/env python3
"""
pillar_analysis_image.py

Identify structural pillars in a raster floor‑plan image using
GPT‑4o‑Vision for high‑level reasoning and OpenCV for quick local
circle/rectangle detection.

### Typical usage

```powershell
# API key comes from .env (python‑dotenv) or env var
python pillar_analysis_image.py "Annotated_Office.png" --prefix office

# Or pass the key explicitly
python pillar_analysis_image.py "Annotated_Office.png" --api-key sk-... --prefix office
```

Produces three artefacts side‑by‑side in the working directory:

| file                       | what it is                                   |
|----------------------------|----------------------------------------------|
| `office_annotated.png`     | original image with pillars highlighted       |
| `office_report.png`        | 3‑panel matplotlib report                    |
| `office_pillar_data.json`  | GPT JSON + locally detected pillar metadata   |
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

# -------------------------------------------------------------------- setup
# Optional: load OPENAI_API_KEY from .env if python‑dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass  # silently ignore – user may not need .env

plt.rcParams["figure.dpi"] = 150
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
LOG = logging.getLogger("pillar-img")


# =================================================================== class
class ImagePillarAnalyzer:
    """Analyse a floor‑plan PNG/JPG and mark structural pillars."""

    def __init__(self, image_path: str | Path, api_key: str):
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(self.image_path)

        self.image = cv2.imread(str(self.image_path))  # BGR
        if self.image is None:
            raise RuntimeError("OpenCV could not load the image – unsupported format?")

        self.client = OpenAI(api_key=api_key)

    # -------------------------------------------------------- GPT analysis
    def ask_gpt_to_find_pillars(self) -> Dict:
        """Call GPT‑4o Vision with the plan image and get pillar JSON."""
        # Encode current image as base64 PNG
        success, buf = cv2.imencode(".png", self.image)
        if not success:
            LOG.error("cv2.imencode failed – skipping GPT call.")
            return {}
        img_b64 = base64.b64encode(buf).decode()

        prompt = """You are a **structural engineer** examining an architectural floor‑plan image.  

### Symbol key
* **Yellow solid circles**  = structural pillars / columns.  
* **Ignore** all other graphics (red dots, blue wedges, furniture, text labels, walls, etc.).

---

## Objective
Return a **single JSON object** that captures only pillar information and does so **consistently** from run to run (so IDs remain stable if the same plan is analysed again).

```jsonc
{
  "pillar_summary": {
    "total_pillars": <int>,               // number of orange circles
    "arrangement": "grid | linear | irregular | along‑wall | corners", // best descriptor
    "average_spacing_ft": <float>,        // mean center‑to‑center distance (‑1 if scale unknown)
    "confidence": <float>                 // 0‑1 subjective confidence
  },
  "pillars": [
    {
      "id": "P1",                        // **stable** ID: assign left‑to‑right, top‑to‑bottom reading order
      "pixel_location": [x, y],           // image pixel coordinates of the circle centre
      "bounding_radius_px": <float>,      // radius of the circle in pixels
      "approx_world_location_ft": [x, y], // convert to feet if drawing scale present, else [‑1,‑1]
      "nearest_room_label": "Work Stations", // text label nearest to pillar; if none, "Unlabelled"
      "distance_to_nearest_pillar_ft": <float>, // ‑1 if scale unknown
      "notes": "on wall | freestanding | corner | within room interior"
    }
  ],
  "diagnostics": {
    "scale_found": true | false,          // did you detect a scale legend like "1/8'' = 1'"?
    "assumptions": "<list any approximations>",
    "confidence": <float>                // overall 0‑1 confidence that pillar list is correct
  }
}
```

## Consistency rules
1. **ID stability**  – Generate pillar IDs purely by image geometry: sort pillars left‑to‑right by `x` then top‑to‑bottom by `y` (reading order).  This ensures the same pillar gets the same ID every run.
2. If two pillars share the same `x` coordinate ±2 px, order by `y` ascending.
3. Use one decimal place for floats.  Use `‑1` for any numeric field you cannot compute.
4. Validate that the JSON is parsable and exactly follows the schema (no extra keys, no trailing commas).

### Output format
Return **only the JSON object** – no Markdown, no extra commentary.
"""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "high"}}
                    ]
                }]
            )
            raw = resp.choices[0].message.content
            data: Dict | str = json.loads(raw)
            # Some models double‑encode JSON as a string – handle that
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    LOG.warning("Nested JSON decode failed – treating as empty result.")
                    return {}
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # broad – network/API/JSON errors
            LOG.error("GPT‑4o Vision request failed: %s", exc)
            return {}

    # ---------------------------------------------------- local detection
    def _detect_circles(self) -> List[Tuple[int, int, int]]:
        gray = cv2.medianBlur(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 20,
                                   param1=100, param2=30, minRadius=5, maxRadius=300)
        return [tuple(map(int, c)) for c in circles[0]] if circles is not None else []

    def _detect_rectangles(self) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                if w * h > 100:  # ignore tiny blobs
                    rects.append((x, y, w, h))
        return rects

    # ------------------------------------------------------- visualisation
    def visualise(self, gpt_json: Dict, prefix: str):
        """Overlay pillar markers, save annotated PNG, report, and JSON."""
        if not gpt_json:
            LOG.warning("No GPT data – skipping visualisation.")
            return

        original = self.image
        overlay = original.copy()

        desc = gpt_json.get("pillar_analysis", {}).get("pillar_representation", "")
        total_gpt = gpt_json.get("pillar_analysis", {}).get("total_count", 0)

        local_hits: List[Dict] = []

        # ---- choose detection strategy based on GPT description --------
        if "circle" in desc.lower():
            circles = self._detect_circles()
            LOG.info("OpenCV Hough detected %d circles", len(circles))
            if circles:
                # bucket radii to choose the modal pillar size
                bucket = [round(r, -1) for *_, r in circles]
                mode_r = max(set(bucket), key=bucket.count)
                for idx, (x, y, r) in enumerate(circles, 1):
                    if round(r, -1) == mode_r:
                        cv2.circle(overlay, (x, y), r + 4, (0, 0, 255), 3)
                        cv2.circle(overlay, (x, y), r, (0, 255, 255), -1)
                        cv2.putText(overlay, f"P{idx}", (x - 10, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        local_hits.append({"id": f"P{idx}", "center": (x, y), "radius": r})
        elif any(k in desc.lower() for k in ("square", "rectangle")):
            rects = self._detect_rectangles()
            LOG.info("Contours detected %d rectangles", len(rects))
            if rects:
                bucket = [round((w + h) / 2, -1) for *_, w, h in rects]
                mode_s = max(set(bucket), key=bucket.count)
                for idx, (x, y, w, h) in enumerate(rects, 1):
                    if round((w + h) / 2, -1) == mode_s:
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.rectangle(overlay, (x + 2, y + 2), (x + w - 2, y + h - 2), (0, 255, 255), -1)
                        cv2.putText(overlay, f"P{idx}", (x + w // 3, y + h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        local_hits.append({"id": f"P{idx}", "bbox": (x, y, w, h)})

        # composite image
        annotated = cv2.addWeighted(original, 0.3, overlay, 0.7, 0)
        ann_path = f"{prefix}_annotated.png"
        cv2.imwrite(ann_path, annotated)
        LOG.info("Annotated image saved → %s", ann_path)

        # simple matplotlib report
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, :])

        ax0.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax0.axis("off"); ax0.set_title("Original")

        ax1.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax1.axis("off"); ax1.set_title(f"Pillars (GPT {total_gpt} / local {len(local_hits)})")

        ax2.text(0, 1, json.dumps(gpt_json, indent=2), va="top", family="monospace")
        ax2.axis("off")

        rep_path = f"{prefix}_report.png"
        plt.tight_layout(); plt.savefig(rep_path, bbox_inches="tight"); plt.close(fig)
        LOG.info("Report saved → %s", rep_path)

        # save JSON merge
        out_path = f"{prefix}_pillar_data.json"
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump({"gpt_analysis": gpt_json, "local_pillars": local_hits}, fp, indent=2)
        LOG.info("JSON saved → %s", out_path)

    # -------------------------------------------------------------- driver
    def run(self, prefix: str):
        gpt_data = self.ask_gpt_to_find_pillars()
        self.visualise(gpt_data, prefix)


# ================================================================= main

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect pillars in a floor‑plan image (PNG/JPG)")
    parser.add_argument("image", help="Path to plan PNG/JPG")
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI key (or set OPENAI_API_KEY / .env)")
    parser.add_argument("--prefix", default="pillar", help="Output file prefix")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("OpenAI API key missing – supply --api-key or set OPENAI_API_KEY/.env")

    analyzer = ImagePillarAnalyzer(args.image, args.api_key)
    analyzer.run(args.prefix)


if __name__ == "__main__":
    main()
