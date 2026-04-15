"""
run_demo.py

Full pipeline demo — ExtractionAgent → ConsolidationAgent.

Discovers all images (.jpg / .jpeg / .png, case-insensitive) in samples/,
then runs the full extraction + consolidation pipeline for each one.

Prerequisites
-------------
1. Generate proto stubs (from project root):
       python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/mcp.proto

2. Start all five MCP servers (or use start_mcps.py):
       python start_mcps.py

3. Set your OpenAI API key (not required when using --mock):
       export OPENAI_API_KEY=sk-...

4. Place sample files:
       samples/<any>.jpg|jpeg|png  — one or more images (video frames work well)

Usage
-----
    python run_demo.py           # live pipeline (requires OPENAI_API_KEY)
    python run_demo.py --mock    # skip LLM; description = stringified bundle
"""

from __future__ import annotations

# Re-exec with the venv Python if we're not already running inside it.
# Compares sys.prefix (the active Python env) against the project venv dir.
# This means `python3 run_demo.py` works without activating the venv first.
import sys
from pathlib import Path as _Path
_venv_dir = _Path(__file__).parent / "venv"
if _venv_dir.exists() and _Path(sys.prefix).resolve() != _venv_dir.resolve():
    import os
    os.execv(str(_venv_dir / "bin" / "python3"), [str(_venv_dir / "bin" / "python3")] + sys.argv)

# Suppress Pydantic/LangChain UserWarning noise on Python 3.14
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

import asyncio
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from agents.consolidation_agent import ConsolidationAgent, MockConsolidationAgent
from agents.extraction_agent import ExtractionAgent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLES_DIR = Path(__file__).parent / "samples"

# Set to True to print raw MCP extraction data for every image without calling
# the consolidation agent (and without requiring OPENAI_API_KEY).
DRY_RUN = True

# Case-insensitive glob for common image extensions
IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _discover_images(directory: Path) -> list[Path]:
    """Return all image files found in directory, sorted by name."""
    found: set[Path] = set()
    for pattern in IMAGE_GLOBS:
        found.update(directory.glob(pattern))
    return sorted(found)


def _print_header(text: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _conf(value: float, low_threshold: float = 0.5) -> str:
    """Format a confidence value; append a warning marker if below threshold."""
    pct = f"{value:.0%}"
    return pct if value >= low_threshold else f"{pct} ⚠"


def _print_bundle_summary(bundle) -> None:
    """
    Print a clean, human-readable extraction summary.

    Strips internal noise (bounding boxes, LAB values, color_method,
    face_region coords, using_mock flags) and surfaces:
      - Scene-level info (scene type, people count, body build)
      - Per-person hair, race, age — cross-referenced by person_id
      - Count-mismatch warnings when MCPs disagree
      - Low-confidence warnings (< 50%)
      - MCP online/offline status on a single line
    Works correctly when any MCPs are offline.
    """
    flags = bundle.partial_flags

    # ── Scene line ──────────────────────────────────────────────────────────
    scene_parts: list[str] = []

    if bundle.people_count and flags.people_count:
        pc = bundle.people_count
        scene_parts.append(
            f"scene: {pc.get('scene_type', '?')}  |  "
            f"YOLO people: {pc.get('people_count', '?')}"
        )
    else:
        scene_parts.append("scene: OFFLINE")

    print("  " + "   |   ".join(scene_parts))

    # ── Body Build ───────────────────────────────────────────────────────────
    if bundle.body_build and flags.body_build:
        bb_people = bundle.body_build.get("people", [])
        print(f"\n  Body Build  ({len(bb_people)} person{'s' if len(bb_people) != 1 else ''} detected)")
        for p in bb_people:
            print(f"    #{p.get('person_id', '?')}  {p.get('body_build', '?')}   ({_conf(p.get('confidence', 0))})")
    else:
        print("\n  Body Build  OFFLINE")

    # ── Hair ────────────────────────────────────────────────────────────────
    if bundle.hair_color and flags.hair_color:
        people = bundle.hair_color.get("people", [])
        print(f"\n  Hair  ({len(people)} person{'s' if len(people) != 1 else ''} detected)")
        for p in people:
            color   = p.get("dominant_color", "?")
            texture = p.get("dominant_texture", "?")
            c_conf  = p.get("color_confidence", 0)
            t_conf  = p.get("texture_confidence", 0)
            occ     = "  [occluded]" if p.get("occluded") else ""
            note    = f"  [{p['note']}]" if p.get("note") else ""
            print(
                f"    #{p.get('person_id', '?')}  {color}, {texture}{occ}"
                f"   color: {_conf(c_conf)}  texture: {_conf(t_conf)}{note}"
            )
    else:
        print("\n  Hair    OFFLINE")

    # ── Race & Age (same face detector — show together) ─────────────────────
    race_ok = bool(bundle.race and flags.race)
    age_ok  = bool(bundle.age  and flags.age)

    race_people: list[dict] = bundle.race.get("people", []) if race_ok else []
    age_people:  list[dict] = bundle.age.get("people",  []) if age_ok  else []
    face_count = bundle.race.get("count", 0) if race_ok else (
                 bundle.age.get("count",  0) if age_ok  else None)

    # Cross-reference by person_id
    race_by_id = {p["person_id"]: p for p in race_people}
    age_by_id  = {p["person_id"]: p for p in age_people}
    person_ids = sorted(set(list(race_by_id) + list(age_by_id)))

    # Count-mismatch warning
    hair_count = len(bundle.hair_color.get("people", [])) if (bundle.hair_color and flags.hair_color) else None
    mismatch = (
        hair_count is not None
        and face_count is not None
        and hair_count != face_count
    )
    mismatch_note = f"  ⚠ hair detected {hair_count}" if mismatch else ""

    if race_ok or age_ok:
        face_label = "?" if face_count is None else face_count
        print(f"\n  Race & Age  ({face_label} face{'s' if face_count != 1 else ''} detected){mismatch_note}")
        if not person_ids:
            print("    (no faces detected)")
        else:
            for pid in person_ids:
                rp = race_by_id.get(pid, {})
                ap = age_by_id.get(pid, {})
                role      = rp.get("role") or ap.get("role", "?")
                gender    = rp.get("gender", "?")
                race      = rp.get("race", "?") if rp else "?"
                r_conf    = rp.get("race_confidence", 0)
                age_range = ap.get("age_range", "?") if ap else "?"
                a_conf    = ap.get("age_confidence", 0)
                is_main   = rp.get("is_main_actress") or ap.get("is_main_actress")
                main_tag  = "  [MAIN]" if is_main else ""

                race_str = f"{race} ({_conf(r_conf)})" if rp else "race: OFFLINE"
                age_str  = f"{age_range} ({_conf(a_conf)})" if ap else "age: OFFLINE"

                print(f"    #{pid}  {gender}, {race_str}, {age_str}{main_tag}  [{role}]")
    else:
        print("\n  Race & Age  OFFLINE")

    # Main actress summary (if present)
    if race_ok:
        main_race = bundle.race.get("main_actress_race")
        if main_race:
            main_age = bundle.age.get("main_actress_age", "?") if age_ok else "?"
            print(f"  → Main actress: {main_race}, {main_age}")

    # ── MCP status line ──────────────────────────────────────────────────────
    def _s(flag: bool, name: str) -> str:
        return f"✓ {name}" if flag else f"✗ {name}"

    status_line = "  ".join([
        _s(flags.hair_color,   "hair_color"),
        _s(flags.body_build,   "body_build"),
        _s(flags.people_count, "people_count"),
        _s(flags.race,         "race"),
        _s(flags.age,          "age"),
    ])
    if bundle.extraction_errors:
        errs = "  |  ".join(f"{k}: {v}" for k, v in bundle.extraction_errors.items())
        print(f"\n  Errors: {errs}")
    print(f"\n  MCPs: {status_line}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Video Insight Engine — batch demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip the OpenAI call; use MockConsolidationAgent instead.",
    )
    args = parser.parse_args()
    # -----------------------------------------------------------------------
    # Discover images before doing any work.
    # -----------------------------------------------------------------------
    images = _discover_images(SAMPLES_DIR)
    if not images:
        print(f"[ERROR] No images found in {SAMPLES_DIR}")
        print("  Add at least one .jpg / .jpeg / .png file and retry.")
        sys.exit(1)

    _print_header("Video Insight Engine — Batch Demo")
    mode_label = "DRY RUN (extraction only)" if DRY_RUN else ("MOCK (no LLM)" if args.mock else "LIVE (GPT-4o-mini)")
    print(f"\n  Mode   : {mode_label}")
    print(f"  Images : {len(images)} file(s) found in {SAMPLES_DIR}")
    for img in images:
        print(f"           • {img.name}")

    # Create agents once; both are stateless across calls.
    # ConsolidationAgent is only instantiated when DRY_RUN is False — this
    # avoids an OPENAI_API_KEY requirement during extraction-only runs.
    extraction_agent = ExtractionAgent()
    consolidation_agent: ConsolidationAgent | None = None
    if not DRY_RUN:
        consolidation_agent = (
            MockConsolidationAgent() if args.mock else ConsolidationAgent()
        )

    # -----------------------------------------------------------------------
    # Process each image through the full pipeline.
    # -----------------------------------------------------------------------
    for idx, image_path in enumerate(images, start=1):
        print(f"\n{'─' * 60}")
        print(f"  Processing [{idx}/{len(images)}]: {image_path.name}")
        print(f"{'─' * 60}")

        try:
            # Step 1 — Build the extraction input for this image.
            source_id = image_path.stem  # e.g. "brad" from "brad.jpeg"
            extraction_input = [
                {
                    "source_id":  source_id,
                    "image_path": str(image_path),
                }
            ]

            # Step 2 — Run ExtractionAgent (all 5 MCPs in parallel).
            print(f"\n  Extraction  →  source_id='{source_id}'")
            bundles = await extraction_agent.process(extraction_input)
            bundle  = bundles[0]
            _print_bundle_summary(bundle)

            if DRY_RUN:
                continue

            # Step 3 — Run ConsolidationAgent (GPT-4o-mini or mock).
            label = "mock (no LLM)" if args.mock else "GPT-4o-mini"
            print(f"\n  [2/2] Consolidation  →  {label} ...")
            output = await consolidation_agent.consolidate(bundle)

            # Step 4 — Print per-image summary (description + tags).
            print(f"\n  ✓ Result for '{image_path.name}':")
            print(f"    description : {output.description}")
            print(f"    tags        : {', '.join(output.tags)}")
            print(f"    confidence  : {output.confidence_score:.2f}")
            print(f"    partial     : {output.partial_result}")

            # Full JSON for reference
            print(f"\n  Full ConsolidatedOutput:")
            print(json.dumps(output.model_dump(), indent=4, ensure_ascii=False))

        except Exception as exc:  # noqa: BLE001
            # Any other unexpected error — log and continue.
            print(f"\n  [ERROR] Unexpected error for '{image_path.name}': {exc}")

    # -----------------------------------------------------------------------
    # Final batch summary.
    # -----------------------------------------------------------------------
    _print_header("Batch complete")
    print(f"  Processed {len(images)} image(s)\n")


if __name__ == "__main__":
    asyncio.run(main())
