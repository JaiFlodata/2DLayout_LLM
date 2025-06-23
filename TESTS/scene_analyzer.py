"""
scene_analyzer.py
~~~~~~~~~~~~~~~~~

Send the *scene* photo (plus the layout for context) to Llama 4,
receive structured JSON, and draw bounding boxes.
Automatically downsizes/resizes and encodes as JPEG to avoid payload-too-large errors.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import base64
import cv2
import json
import logging
from groq import Groq

LOG = logging.getLogger("scene-analyzer")
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# Maximum dimension for image downscaling (pixels)
MAX_DIMENSION = 512
# JPEG quality (1-100)
JPEG_QUALITY = 50


def _resize_for_api(img: cv2.Mat) -> cv2.Mat:
    """Downscale image so max(width, height) <= MAX_DIMENSION."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_DIMENSION:
        return img
    scale = MAX_DIMENSION / max_dim
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _img_to_b64(img: cv2.Mat) -> str:
    """Resize + JPEG-encode + base64 to shrink payload."""
    small = _resize_for_api(img)
    # Encode as JPEG with compression
    ok, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode image")
    return base64.b64encode(buf).decode()


class SceneAnalyzer:
    """High-level narrative + per-object list + layout location."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def describe_scene(
        self,
        scene_img: cv2.Mat,
        layout_img: cv2.Mat,
        pillar_json: Dict
    ) -> Dict:
        """
        Call Llama to summarise activity, detect objects,
        and reference pillar IDs in the layout. Images are
        resized and JPEG-encoded client-side to avoid payload-too-large errors.
        """
        prompt = f"""
You are a **multimodal spatial analyst**.

### TASK
Given:
1. *Scene photo* – shows real activity.
2. *Annotated floor-plan* – shows labelled pillars P1…Pn.

Return **ONE JSON object** following this schema:

```jsonc
{{
  "overview": "<1-2 sentence summary of the main activity>",
  "detailed_description": "<≈4-8 sentences – clothing, motions, ambience>",
  "activity_location": {{
    "pillar_reference": "between P3 and P4"
  }},
  "objects": [
    {{
      "id": "O1",
      "label": "person",
      "bbox_px": [x, y, w, h],
      "attributes": {{
        "clothing": "black T-shirt, white track-pants, sports shoes",
        "pose_action": "doing push-ups"
      }}
    }}
  ],
  "diagnostics": {{
    "confidence": <float 0-1>
  }}
}}
```
Rules:
- Provide valid JSON only (no markdown fences).
- Use supplied pillar IDs for spatial context.
- Maintain ID stability via geometry.
"""
        raw: str = ""
        try:
            # Prepare encoded images
            scene_b64 = _img_to_b64(scene_img)
            layout_b64 = _img_to_b64(layout_img)

            resp = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{scene_b64}",
                            "detail": "low"}},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{layout_b64}",
                            "detail": "low"}},
                        {"type": "text", "text": json.dumps({"pillars": pillar_json.get("pillars", [])})}
                    ]
                }]
            )

            raw = resp.choices[0].message.content.strip()
            # Strip fences
            if raw.startswith("```"):
                raw = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("```"))

            data = json.loads(raw)
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                raise ValueError("Invalid JSON from Llama")
            return data
        except Exception as exc:
            LOG.error("Llama call failed: %s", exc)
            if raw:
                LOG.debug("Raw fragment:\n%s", raw[:400])
            return {}

    @staticmethod
    def annotate_scene(
        scene_img: cv2.Mat,
        objects: List[Dict],
        out_path: str
    ) -> None:
        img = scene_img.copy()
        for obj in objects:
            try:
                x, y, w, h = map(int, obj.get("bbox_px", [0, 0, 0, 0]))
            except Exception:
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                img,
                obj.get("id", ""),
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
        cv2.imwrite(out_path, img)  
