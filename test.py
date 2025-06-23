#!/usr/bin/env python3
"""
pillar_camera_auto.py  ·  Llama-4 vision, zero-argument workflow
---------------------------------------------------------------
• Build site_memory.json (pillars + cameras) from two annotated layout images.
• Batch-analyse every image in frames/ to detect which pillar & camera it shows.
"""

from __future__ import annotations
import base64, json, logging, os, sys
from pathlib import Path
from typing import Dict, List

import cv2                                # pip install opencv-python
from dotenv import load_dotenv            # pip install python-dotenv
from groq import Groq                     # pip install groq

# ─────────────────────────── configuration ────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()

PILLAR_IMG   = SCRIPT_DIR / "office_annotated.png"      # yellow-pillar plan
CAMERA_IMG   = SCRIPT_DIR / "Annotated_Office.png"      # red-dot / blue FOV plan

FRAME_DIR    = SCRIPT_DIR / "frames"                  # CCTV frames folder
MEMORY_JSON  = SCRIPT_DIR / "site_memory.json"        # generated once
RESULTS_DIR  = SCRIPT_DIR / "results"                 # per-frame JSONs
RESULTS_DIR.mkdir(exist_ok=True)

LMODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq model name

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(message)s")
LOG = logging.getLogger("pillar-auto")

# ───────────────────────── helper functions ───────────────────────────
def b64_image(path: Path) -> str:
    """Read image and return base64-encoded PNG string."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"OpenCV could not read {path}")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    return base64.b64encode(buf).decode()


def llama_chat(client: Groq, blocks: List[Dict], max_tokens: int = 4096) -> str:
    """Send a single user message to Llama-4 and return its raw text content."""
    resp = client.chat.completions.create(
        model=LMODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": blocks}],
    )
    return resp.choices[0].message.content


def extract_json(raw: str) -> Dict:
    """Strip ``` fences if present and parse JSON → dict (or {})."""
    txt = raw.strip()
    if txt.startswith("```"):
        txt = "\n".join(line for line in txt.splitlines()
                        if not line.strip().startswith("```"))
    try:
        obj = json.loads(txt)
        return json.loads(obj) if isinstance(obj, str) else obj
    except json.JSONDecodeError:
        LOG.error("Could not parse JSON from Llama:\n%s", raw[:400])
        return {}

# ────────────────────────── build site memory ─────────────────────────
def build_memory(client: Groq) -> Dict:
    """Ask Llama-4 for pillar & camera metadata, save to site_memory.json."""
    prompt = (
        "You will receive TWO floor-plan images.\n"
        "• Image A: solid YELLOW circles → structural pillars.\n"
        "• Image B: RED dots + BLUE wedges → CCTV cameras & their FOV.\n\n"
        "Return ONE JSON object ONLY:\n"
        "{\n"
        '  "pillars":  [ {"id":"P1","pixel":[x,y],"radius_px":r}, … ],\n'
        '  "cameras":  [ {"id":"C1","pixel":[x,y],"fov_polygon":[[x1,y1],…]}, … ],\n'
        '  "summary":  { "total_pillars":<int>, "total_cameras":<int>, "confidence":<float> }\n'
        "}"
    )

    blocks = [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64_image(PILLAR_IMG)}",
                       "detail": "high"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64_image(CAMERA_IMG)}",
                       "detail": "high"}},
    ]
    raw = llama_chat(client, blocks)
    memory = extract_json(raw)
    if not memory:
        raise RuntimeError("Llama returned no usable memory JSON.")

    MEMORY_JSON.write_text(json.dumps(memory, indent=2))
    LOG.info("Site memory saved → %s", MEMORY_JSON.name)
    return memory

# ────────────────────────── analyse one frame ─────────────────────────
def analyse_frame(client: Groq, frame: Path, memory: Dict) -> Dict:
    """Send one CCTV image + memory to Llama-4 → pillar & camera ID JSON."""
    mem_subset = json.dumps({
        "pillars": memory["pillars"],
        "cameras": [{"id": c["id"], "pixel": c["pixel"]} for c in memory["cameras"]],
    })

    prompt = (
        "Floor-plan memory (pillar & camera IDs with layout pixels):\n"
        f"```json\n{mem_subset}\n```\n\n"
        "You will now see ONE CCTV image.  Decide:\n"
        "• Is a pillar visible?  If yes, which pillar ID?\n"
        "• Which camera most likely captured the frame?\n\n"
        "Respond ONLY with this JSON (no markdown):\n"
        "{\n"
        '  "pillar_id": "P4" | null,\n'
        '  "camera_id": "C3" | null,\n'
        '  "confidence": <float>,\n'
        '  "notes": "<free text ≤40 chars>"\n'
        "}"
    )

    blocks = [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64_image(frame)}",
                       "detail": "high"}},
    ]
    raw = llama_chat(client, blocks)
    result = extract_json(raw)
    if not result:
        raise RuntimeError(f"Llama returned no result for {frame.name}")

    out_path = RESULTS_DIR / f"{frame.stem}_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    LOG.info("✓ %s → %s", frame.name, out_path.name)
    return result

# ───────────────────────────── main flow ──────────────────────────────
def main() -> None:
    load_dotenv()                                     # pulls .env vars
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        LOG.error("Set GROQ_API_KEY in .env or environment.")
        sys.exit(1)
    client = Groq(api_key=api_key)

    # 1. build or load memory
    if MEMORY_JSON.exists():
        memory = json.loads(MEMORY_JSON.read_text())
        LOG.info("Loaded existing memory (%s pillars, %s cameras).",
                 len(memory["pillars"]), len(memory["cameras"]))
    else:
        LOG.info("Building site memory (first run)…")
        memory = build_memory(client)

    # 2. gather frames
    frames = sorted([f for f in FRAME_DIR.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not frames:
        LOG.warning("No frames found in %s — nothing to analyse.", FRAME_DIR)
        return

    LOG.info("Analysing %d frame(s)…", len(frames))
    for frame in frames:
        try:
            analyse_frame(client, frame, memory)
        except Exception as exc:
            LOG.error("Frame %s failed: %s", frame.name, exc)

if __name__ == "__main__":
    main()