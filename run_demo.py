"""
run_demo.py

Full pipeline demo — ExtractionAgent → ConsolidationAgent.

Discovers all images (.jpg / .jpeg / .png, case-insensitive) in samples/,
then runs the full extraction + consolidation pipeline for each one, using
samples/clip.wav as the shared audio source.

Prerequisites
-------------
1. Generate proto stubs (from project root):
       python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/mcp.proto

2. Start all five MCP servers in separate terminals:
       python -m mcps.hair_color.server
       python -m mcps.body_build.server
       python -m mcps.people_count.server
       python -m mcps.captions.server
       python -m mcps.audio.server

3. Set your OpenAI API key (not required when using --mock):
       export OPENAI_API_KEY=sk-...

4. Place sample files:
       samples/<any>.jpg|jpeg|png  — one or more images (video frames work well)
       samples/clip.wav            — a short audio clip (16 kHz mono WAV recommended)

Usage
-----
    python run_demo.py           # live pipeline (requires OPENAI_API_KEY)
    python run_demo.py --mock    # skip LLM; description = stringified bundle
"""

from __future__ import annotations

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
from agents.extraction_agent import ExtractionAgent, ExtractionError


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLES_DIR  = Path(__file__).parent / "samples"
SAMPLE_AUDIO = SAMPLES_DIR / "clip.wav"

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


def _print_extraction_flags(bundle) -> None:
    flags = bundle.partial_flags
    rows = [
        ("hair_color",   flags.hair_color),
        ("body_build",   flags.body_build),
        ("people_count", flags.people_count),
        ("captions",     flags.captions),
        ("audio",        flags.audio),
    ]
    for name, ok in rows:
        status = "OK" if ok else "FAILED"
        print(f"        {name:<14}: {status}")

    if bundle.extraction_errors:
        print("\n      Extraction errors:")
        for mcp, err in bundle.extraction_errors.items():
            print(f"        {mcp}: {err}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Video Insight Engine — batch demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip the OpenAI call; use MockConsolidationAgent instead.",
    )
    args = parser.parse_args()
    # -----------------------------------------------------------------------
    # Validate shared audio file and discover images before doing any work.
    # -----------------------------------------------------------------------
    if not SAMPLE_AUDIO.exists():
        print(f"[ERROR] Audio file not found: {SAMPLE_AUDIO}")
        print("  Place a WAV audio clip at samples/clip.wav and retry.")
        sys.exit(1)

    images = _discover_images(SAMPLES_DIR)
    if not images:
        print(f"[ERROR] No images found in {SAMPLES_DIR}")
        print("  Add at least one .jpg / .jpeg / .png file and retry.")
        sys.exit(1)

    _print_header("Video Insight Engine — Batch Demo")
    mode_label = "DRY RUN (extraction only)" if DRY_RUN else ("MOCK (no LLM)" if args.mock else "LIVE (GPT-4o-mini)")
    print(f"\n  Mode   : {mode_label}")
    print(f"  Audio  : {SAMPLE_AUDIO}")
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
                    "audio_path": str(SAMPLE_AUDIO),
                }
            ]

            # Step 2 — Run ExtractionAgent (all 5 MCPs in parallel).
            print(f"\n  [1/2] Extraction  →  source_id='{source_id}'")
            bundles = await extraction_agent.process(extraction_input)
            bundle  = bundles[0]
            _print_extraction_flags(bundle)
            hc = bundle.hair_color
            if hc:
                ppl = hc.get("people", [])
                dom_lab = ppl[0].get("dominant_lab", "?") if ppl else "?"
                print(f"  DEBUG - People detected: {len(ppl)}"
                      f" | color={hc.get('dominant_color', '?')}"
                      f" | texture={hc.get('dominant_texture', '?')}"
                      f" | Lab={dom_lab}")
            else:
                print("  DEBUG - People detected: 0 (hair_color MCP unavailable)")
            print(f"\n  Raw MCP data for '{image_path.name}':")
            print(bundle.model_dump_json(indent=4))

            if DRY_RUN:
                print(f"\n  [DRY RUN] Skipping consolidation for '{image_path.name}'.")
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

        except ExtractionError as exc:
            # Hard-dependency MCP failure — log and continue to the next image.
            print(f"\n  [SKIP] ExtractionError for '{image_path.name}': {exc}")
        except Exception as exc:  # noqa: BLE001
            # Any other unexpected error — log and continue.
            print(f"\n  [ERROR] Unexpected error for '{image_path.name}': {exc}")

    # -----------------------------------------------------------------------
    # Final batch summary.
    # -----------------------------------------------------------------------
    _print_header("Batch complete")
    print(f"  Processed {len(images)} image(s) with audio: {SAMPLE_AUDIO.name}\n")


if __name__ == "__main__":
    asyncio.run(main())
