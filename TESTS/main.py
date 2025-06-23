#!/usr/bin/env python3
"""
main.py
~~~~~~~

Zero-argument driver with hardcoded paths and .env API key.

Run:
    python main.py

Before running, edit the CONFIGURATION block to match your file locations.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from layout_context import LayoutContext
from scene_analyzer import SceneAnalyzer

# ------------------------------------------------------------------------------
# CONFIGURATION: Edit these paths to match your environment
SCENE_PATH = r"C:\Users\jaigo\Desktop\KaamTrack\data\PXL_20250618_061533034.MP.jpg"
LAYOUT_PATH = r"C:\Users\jaigo\Desktop\KaamTrack\Annotated_Office2.png"
LAYOUT_JSON_PATH: str | None = r"C:\Users\jaigo\Desktop\KaamTrack\office_pillar_data.json"  # set to None if no JSON
PREFIX = "workout"  # output file prefix
# ------------------------------------------------------------------------------

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)
LOG = logging.getLogger("driver")


def build_report(
    scene_orig, scene_ann,
    layout_orig, layout_ann,
    json_data, out_path
) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    ax0, ax1, ax2, ax3 = [fig.add_subplot(gs[i, j])
                          for i in range(2) for j in range(2)]
    ax0.imshow(cv2.cvtColor(scene_orig, cv2.COLOR_BGR2RGB))
    ax0.set_title("Scene – original"); ax0.axis("off")

    ax1.imshow(cv2.cvtColor(scene_ann, cv2.COLOR_BGR2RGB))
    ax1.set_title("Scene – annotated"); ax1.axis("off")

    ax2.imshow(cv2.cvtColor(layout_orig, cv2.COLOR_BGR2RGB))
    ax2.set_title("Layout – original"); ax2.axis("off")

    ax3.imshow(cv2.cvtColor(layout_ann, cv2.COLOR_BGR2RGB))
    ax3.set_title("Layout – flagged location"); ax3.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Load Groq API key from .env
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Groq API key not found. Please set GROQ_API_KEY in your .env file."
        )

    # Load the scene image
    scene_img = cv2.imread(SCENE_PATH)
    if scene_img is None:
        raise RuntimeError(f"Could not load scene image: {SCENE_PATH}")
    LOG.info("Loaded scene image → %s", SCENE_PATH)

    # Initialize layout context (auto-detect pillars if JSON missing)
    layout_json = (
        LAYOUT_JSON_PATH
        if LAYOUT_JSON_PATH and Path(LAYOUT_JSON_PATH).exists()
        else None
    )
    lay = LayoutContext(LAYOUT_PATH, layout_json)
    LOG.info(
        "Loaded layout image → %s with %d pillar(s)",
        LAYOUT_PATH,
        len(lay.pillars)
    )

    # Describe the scene using Llama 4
    llama = SceneAnalyzer(api_key)
    scene_json = llama.describe_scene(
        scene_img,
        lay.image,
        {"pillars": lay.pillars}
    )
    if not scene_json:
        LOG.error("No scene JSON returned – aborting.")
        return

    # Annotate the scene
    out_scene_ann = f"{PREFIX}_annotated_scene.png"
    SceneAnalyzer.annotate_scene(
        scene_img,
        scene_json.get("objects", []),
        out_scene_ann
    )
    LOG.info("Annotated scene saved → %s", out_scene_ann)

    # Determine approximate location on the layout (centre of largest person bbox)
    loc_px = None
    persons = [
        obj for obj in scene_json.get("objects", [])
        if obj.get("label") == "person"
    ]
    if persons:
        largest = max(
            persons,
            key=lambda o: o["bbox_px"][2] * o["bbox_px"][3]
        )
        x, y, w, h = map(int, largest["bbox_px"])
        loc_px = (x + w // 2, y + h // 2)

    # Annotate the layout
    out_layout_ann = f"{PREFIX}_annotated_layout.png"
    if loc_px:
        lay.draw_location_marker(loc_px, out_layout_ann)
        scene_json.setdefault("activity_location", {})
        scene_json["activity_location"]["approx_pixel_on_layout"] = list(loc_px)
        scene_json["activity_location"]["nearest_pillar"] = (
            lay.nearest_pillar_id(*loc_px)
        )
        LOG.info("Annotated layout saved → %s", out_layout_ann)
    else:
        LOG.warning("No person detected; skipping layout annotation.")

    # Save structured JSON
    json_path = f"{PREFIX}_scene.json"
    Path(json_path).write_text(
        json.dumps(scene_json, indent=2, ensure_ascii=False)
    )
    LOG.info("Scene JSON saved → %s", json_path)

    # Build combined report
    layout_ann_img = (
        cv2.imread(out_layout_ann)
        if Path(out_layout_ann).exists()
        else lay.image
    )
    out_report = f"{PREFIX}_report.png"
    build_report(
        scene_img,
        cv2.imread(out_scene_ann),
        lay.image,
        layout_ann_img,
        scene_json,
        out_report
    )
    LOG.info("Combined report saved → %s", out_report)


if __name__ == "__main__":
    main()