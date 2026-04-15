"""
schemas/extraction_bundle.py

Pydantic v2 data models representing the output of the extraction pipeline.
ExtractionBundle carries all per-source feature results, partial-failure flags,
and any error messages accumulated during extraction.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PartialFlags(BaseModel):
    """
    Tracks which MCP extractions succeeded.
    False means that MCP failed (all retries exhausted or circuit open).
    """
    hair_color: bool = True
    body_build: bool = True
    people_count: bool = True
    captions: bool = True
    audio: bool = True
    race: bool = True
    age: bool = True


class ExtractionBundle(BaseModel):
    """
    The complete (possibly partial) extraction result for a single source.

    Fields
    ------
    source_id         : Identifier that traces back to the original video/image.
    hair_color        : JSON-decoded dict from the hair_color MCP, or None on failure.
                        Structure: {"people": [...], "count": n,
                                    "dominant_color": "<str>", "dominant_texture": "<str>"}
                        Each "people" entry: {person_id, dominant_color, color_palette,
                        color_confidence, dominant_lab ([L,a,b] of K-means centroid),
                        dominant_texture, texture_palette, texture_confidence,
                        hair_pixel_count, occluded}.
                        Color is determined via K-means clustering in CIELAB space,
                        then mapped to the nearest named color.
                        Texture is classified by SigLIP on a mask-filtered crop.
    body_build        : JSON-decoded dict from the body_build MCP, or None on failure.
    people_count      : JSON-decoded dict from the people_count MCP, or None on failure.
    captions          : JSON-decoded dict from the captions MCP, or None on failure.
    audio             : JSON-decoded dict from the audio MCP, or None on failure.
    partial_flags     : Boolean success flags per MCP.
    extraction_errors : Maps MCP name → human-readable error string for any failures.
    """
    source_id: str
    hair_color: dict | None = None
    body_build: dict | None = None
    people_count: dict | None = None
    captions: dict | None = None
    audio: dict | None = None
    race: dict | None = None
    age: dict | None = None
    partial_flags: PartialFlags = Field(default_factory=PartialFlags)
    extraction_errors: dict[str, str] = Field(default_factory=dict)
