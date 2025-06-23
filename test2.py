# #!/usr/bin/env python3
# """
# pillar_camera_scene.py  ·  Llama-4 vision, zero-argument workflow
# -----------------------------------------------------------------
# • Build site_memory.json (pillars + cameras) from two annotated layout images.
# • Batch-analyse every image in frames/ to detect which pillar & camera it shows.
# • Also produce a detailed natural-language description of each frame’s scene.
# """

# from __future__ import annotations
# import base64
# import json
# import logging
# import os
# import sys
# from pathlib import Path
# from typing import Dict, List

# import cv2                                # pip install opencv-python
# from dotenv import load_dotenv            # pip install python-dotenv
# from groq import Groq                     # pip install groq

# # ─────────────────────────── configuration ────────────────────────────
# SCRIPT_DIR   = Path(__file__).parent.resolve()

# PILLAR_IMG   = SCRIPT_DIR / "office_annotated.png"      # yellow-pillar plan
# CAMERA_IMG   = SCRIPT_DIR / "Annotated_Office.png"      # red-dot / blue FOV plan

# FRAME_DIR    = SCRIPT_DIR / "frames"                    # CCTV frames folder
# MEMORY_JSON  = SCRIPT_DIR / "site_memory.json"          # generated once
# RESULTS_DIR  = SCRIPT_DIR / "results"                   # per-frame outputs
# RESULTS_DIR.mkdir(exist_ok=True)

# LMODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq model name

# logging.basicConfig(level=logging.INFO,
#                     format="%(levelname)s | %(message)s")
# LOG = logging.getLogger("pillar-scene-auto")


# # ───────────────────────── helper functions ───────────────────────────
# def b64_image(path: Path) -> str:
#     """Read image and return base64-encoded PNG string."""
#     img = cv2.imread(str(path))
#     if img is None:
#         raise RuntimeError(f"OpenCV could not read {path}")
#     ok, buf = cv2.imencode(".png", img)
#     if not ok:
#         raise RuntimeError(f"cv2.imencode failed for {path}")
#     return base64.b64encode(buf).decode()


# def llama_chat(client: Groq, blocks: List[Dict], max_tokens: int = 4096) -> str:
#     """Send a user message (blocks) to Llama-4 and return its raw text content."""
#     resp = client.chat.completions.create(
#         model=LMODEL,
#         max_tokens=max_tokens,
#         messages=[{"role": "user", "content": blocks}],
#     )
#     return resp.choices[0].message.content


# def extract_json(raw: str) -> Dict:
#     """Strip ``` fences if present and parse JSON → dict (or {})."""
#     txt = raw.strip()
#     if txt.startswith("```"):
#         txt = "\n".join(
#             line for line in txt.splitlines()
#             if not line.strip().startswith("```")
#         )
#     try:
#         obj = json.loads(txt)
#         # In case Llama returned a JSON-encoded string:
#         if isinstance(obj, str):
#             return json.loads(obj)
#         return obj
#     except json.JSONDecodeError:
#         LOG.error("Could not parse JSON from Llama:\n%s", raw[:400])
#         return {}


# # ────────────────────────── build site memory ─────────────────────────
# def build_memory(client: Groq) -> Dict:
#     """Ask Llama-4 for pillar & camera metadata, save to site_memory.json."""
#     prompt = (
#         "You will receive TWO floor-plan images.\n"
#         "• Image A: solid YELLOW circles → structural pillars.\n"
#         "• Image B: RED dots + BLUE wedges → CCTV cameras & their FOV.\n\n"
#         "Return ONE JSON object ONLY:\n"
#         "{\n"
#         '  "pillars":  [ {"id":"P1","pixel":[x,y],"radius_px":r}, … ],\n'
#         '  "cameras":  [ {"id":"C1","pixel":[x,y],"fov_polygon":[[x1,y1],…]}, … ],\n'
#         '  "summary":  { "total_pillars":<int>, "total_cameras":<int>, "confidence":<float> }\n'
#         "}"
#     )
#     blocks = [
#         {"type": "text", "text": prompt},
#         {"type": "image_url",
#          "image_url": {"url": f"data:image/png;base64,{b64_image(PILLAR_IMG)}",
#                        "detail": "high"}},
#         {"type": "image_url",
#          "image_url": {"url": f"data:image/png;base64,{b64_image(CAMERA_IMG)}",
#                        "detail": "high"}},
#     ]
#     raw = llama_chat(client, blocks)
#     memory = extract_json(raw)
#     if not memory:
#         raise RuntimeError("Llama returned no usable memory JSON.")
#     MEMORY_JSON.write_text(json.dumps(memory, indent=2))
#     LOG.info("Site memory saved → %s", MEMORY_JSON.name)
#     return memory


# # ───────────────────── analyse spatial (pillar & camera) ─────────────────────
# def analyse_spatial_frame(client: Groq, frame: Path, memory: Dict) -> Dict:
#     """Send one CCTV image + memory to Llama-4 → pillar & camera ID JSON."""
#     mem_subset = json.dumps({
#         "pillars": memory.get("pillars", []),
#         "cameras": [{"id": c["id"], "pixel": c["pixel"]} for c in memory.get("cameras", [])],
#     })
#     prompt = (
#         "Floor-plan memory (pillar & camera IDs with layout pixels):\n"
#         f"```json\n{mem_subset}\n```\n\n"
#         "You will now see ONE CCTV image. Decide:\n"
#         "• Is a pillar visible? If yes, which pillar ID?\n"
#         "• Which camera most likely captured the frame?\n\n"
#         "Respond ONLY with this JSON (no markdown):\n"
#         "{\n"
#         '  "pillar_id": "P4" | null,\n'
#         '  "camera_id": "C3" | null,\n'
#         '  "confidence": <float>,\n'
#         '  "notes": "<free text ≤40 chars>"\n'
#         "}"
#     )
#     blocks = [
#         {"type": "text", "text": prompt},
#         {"type": "image_url",
#          "image_url": {"url": f"data:image/png;base64,{b64_image(frame)}",
#                        "detail": "high"}},
#     ]
#     raw = llama_chat(client, blocks)
#     result = extract_json(raw)
#     if not result:
#         raise RuntimeError(f"Llama returned no spatial result for {frame.name}")
#     out_path = RESULTS_DIR / f"{frame.stem}_spatial.json"
#     out_path.write_text(json.dumps(result, indent=2))
#     LOG.info("✓ %s → %s", frame.name, out_path.name)
#     return result


# # ───────────────────── describe scene in detail ─────────────────────
# def describe_scene_frame(client: Groq, frame: Path) -> Dict:
#     """Send one image to Llama-4 for detailed scene understanding."""
#     prompt = (
#         "You will now see an image. Provide a comprehensive description of everything the image contains. "
#         "Your response should include:\n"
#         "1. A brief overview of the scene.\n"
#         "2. Detailed observations including:\n"
#         "   • People – how many, what they are doing, what they are wearing.\n"
#         "   • Objects – any visible furniture, tools, or equipment.\n"
#         "   • Activities – if any person is performing a specific action (e.g., pushups), describe it.\n"
#         "   • Spatial layout – relative positions if apparent.\n"
#         "Use natural, descriptive English. DO NOT respond in JSON or markdown format—just plain text.\n"
#     )
#     blocks = [
#         {"type": "text", "text": prompt},
#         {"type": "image_url",
#          "image_url": {"url": f"data:image/png;base64,{b64_image(frame)}",
#                        "detail": "high"}},
#     ]
#     raw = llama_chat(client, blocks)
#     out_path = RESULTS_DIR / f"{frame.stem}_description.txt"
#     out_path.write_text(raw)
#     LOG.info("✓ %s → %s", frame.name, out_path.name)
#     return {"description": raw}


# # ───────────────────────────── main flow ──────────────────────────────
# def main() -> None:
#     load_dotenv()  # pulls .env vars
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         LOG.error("Set GROQ_API_KEY in .env or environment.")
#         sys.exit(1)
#     client = Groq(api_key=api_key)

#     # 1. build or load memory
#     memory: Dict
#     if MEMORY_JSON.exists():
#         memory = json.loads(MEMORY_JSON.read_text())
#         LOG.info("Loaded existing memory (%s pillars, %s cameras).",
#                  len(memory.get("pillars", [])), len(memory.get("cameras", [])))
#     else:
#         LOG.info("Building site memory (first run)…")
#         memory = build_memory(client)

#     # 2. gather frames
#     frames = sorted([
#         f for f in FRAME_DIR.glob("*")
#         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
#     ])
#     if not frames:
#         LOG.warning("No frames found in %s — nothing to analyse.", FRAME_DIR)
#         return

#     LOG.info("Analysing %d frame(s)…", len(frames))
#     for frame in frames:
#         try:
#             # Spatial analysis (pillar & camera)
#             analyse_spatial_frame(client, frame, memory)
#         except Exception as exc:
#             LOG.error("Spatial analysis failed for %s: %s", frame.name, exc)
#         try:
#             # Detailed scene description
#             describe_scene_frame(client, frame)
#         except Exception as exc:
#             LOG.error("Scene description failed for %s: %s", frame.name, exc)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
scene_analysis_image.py
=======================

Analyse a *general* image with Llama 4 via Groq and return:

* A concise **overview** (1‑2 sentences) of the scene.
* An extended, highly **descriptive paragraph** (clothing, actions, ambience, etc.).
* A structured **object list** with
  * stable IDs,
  * pixel bounding boxes,
  * human‑readable attributes (colours, clothing items, poses, …),
  * and a coarse **spatial bucket** (“top‑left”, “centre‑right”, …).

It also produces three artefacts, written next to the source image:

| file                         | purpose                                    |
|------------------------------|--------------------------------------------|
| `<prefix>_annotated.png`     | image + labelled object bounding boxes     |
| `<prefix>_report.png`        | small 3‑panel matplotlib report            |
| `<prefix>_scene_data.json`   | raw JSON from Llama 4 + derived metadata   |
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
from groq import Groq

# ------------------------------------------------------------- configuration
try:  # optional .env
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

plt.rcParams["figure.dpi"] = 150
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
LOG = logging.getLogger("scene‑img")


# =================================================================== helpers
def bucket_position(img_w: int, img_h: int, bbox: Tuple[int, int, int, int]) -> str:
    """Return a 3 × 3 bucket label for the bbox centre (e.g. 'top‑left')."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    col = ["left", "centre", "right"][min(2, int(cx / img_w * 3))]
    row = ["top", "middle", "bottom"][min(2, int(cy / img_h * 3))]
    if row == "middle" and col == "centre":
        return "centre"
    return f"{row}-{col}".replace("middle-", "").replace("-centre", "")


# =================================================================== class
class ImageSceneAnalyzer:
    """Describe a generic scene and draw labelled bounding boxes."""

    LLAMA_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(self, image_path: str | Path, api_key: str):
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(self.image_path)

        self.image = cv2.imread(str(self.image_path))  # BGR
        if self.image is None:
            raise RuntimeError("OpenCV could not load the image – unsupported format?")

        self.client = Groq(api_key=api_key)
        self.h, self.w = self.image.shape[:2]

    # ------------------------------------------------------ Llama interaction
    def ask_llama_to_describe_scene(self) -> Dict:
        """Send the image to Llama 4 and get scene JSON back."""
        success, buf = cv2.imencode(".png", self.image)
        if not success:
            LOG.error("cv2.imencode failed – skipping Llama call.")
            return {}

        img_b64 = base64.b64encode(buf).decode()

        prompt = f"""
You are a **visual scene analyst**.  Examine the provided image carefully and
return a **single JSON object** with the following schema and rules:

```jsonc
{{
  "overview": "<1‑2 sentence high‑level summary>",
  "detailed_description": "<rich paragraph (≈4‑8 sentences) naming actions, clothing, mood, lighting, etc.>",
  "objects": [
    {{
      "id": "O1",                 // STABLE: order by x then y of bbox (left‑to‑right, top‑to‑bottom)
      "label": "person",          // concise noun or noun‑phrase
      "category": "human | animal | furniture | equipment | other",
      "bbox_px": [x, y, w, h],    // int pixel rectangle (top‑left origin)
      "relative_position": "top‑left | top | top‑right | left | centre | right | bottom‑left | bottom | bottom‑right",
      "attributes": {{
        "clothing": "black T‑shirt, white track‑pants, running shoes",
        "pose_action": "performing push‑ups",   // free‑text
        "colour": "predominantly dark clothing"
      }}
    }}
  ],
  "diagnostics": {{
    "confidence": <float 0‑1>,
    "notes": "<any assumptions or uncertainties>"
  }}
}}
