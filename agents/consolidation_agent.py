"""
agents/consolidation_agent.py

ConsolidationAgent — synthesises an ExtractionBundle into a structured
ConsolidatedOutput using LangChain + OpenAI GPT-4o-mini.

Environment variables required
-------------------------------
    OPENAI_API_KEY  — your OpenAI API key

Usage
-----
    from agents.consolidation_agent import ConsolidationAgent
    agent = ConsolidationAgent()
    output = await agent.consolidate(bundle)
"""

from __future__ import annotations

import json
import logging
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from schemas.extraction_bundle import ExtractionBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt version — bump this string whenever the prompt changes.
# Never hardcode the version string inline; always reference PROMPT_VERSION.
# ---------------------------------------------------------------------------
PROMPT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ConsolidatedOutput(BaseModel):
    """
    Validated LLM output for a single video source.

    Fields
    ------
    description       : Human-readable narrative summarising the video content.
    tags              : Keyword/topic tags suitable for search indexing.
    confidence_score  : Overall confidence in the analysis [0.0, 1.0].
    dominant_features : Key visual/audio features driving the description.
    partial_result    : True when one or more MCP extractions failed.
    prompt_version    : Always set to PROMPT_VERSION at consolidation time.
    """
    description: str
    tags: list[str]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    dominant_features: list[str]
    partial_result: bool
    prompt_version: str

    @field_validator("confidence_score")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ConsolidationAgent:
    """
    Accepts an ExtractionBundle and returns a ConsolidatedOutput.

    Parameters
    ----------
    model_name : OpenAI model identifier (default: "gpt-4o-mini")
    temperature: Sampling temperature (default: 0.2 — low variance for structured output)
    timeout    : LLM call timeout in seconds (default: 30)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        timeout: int = 30,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running the consolidation agent."
            )

        self._llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            api_key=api_key,
        )

    async def consolidate(self, bundle: ExtractionBundle) -> ConsolidatedOutput:
        """
        Build a prompt from bundle fields, call the LLM, and validate the response.

        Returns a fully validated ConsolidatedOutput.
        Raises ValueError if the LLM response cannot be parsed or validated.
        """
        partial = any(
            not getattr(bundle.partial_flags, mcp)
            for mcp in ("hair_color", "body_build", "people_count", "race", "age")
        )

        system_prompt = _build_system_prompt()
        user_prompt   = _build_user_prompt(bundle)

        logger.info(
            "Calling LLM for source_id=%s partial=%s prompt_version=%s",
            bundle.source_id, partial, PROMPT_VERSION,
        )

        # FUTURE: if LLM call exceeds 8s timeout, fall back to lighter model
        #         (e.g. gpt-4o-mini with reduced prompt):
        #
        #   try:
        #       response = await asyncio.wait_for(
        #           self._llm.ainvoke([...]), timeout=8.0
        #       )
        #   except asyncio.TimeoutError:
        #       fallback_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=256)
        #       response = await fallback_llm.ainvoke([short_messages])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self._llm.ainvoke(messages)
        raw_text = response.content

        parsed = _parse_llm_response(raw_text, partial)
        return parsed


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return (
        "You are a video-content analyst. You receive structured feature data extracted "
        "from a video and must return a JSON object (no markdown fences) with these exact keys:\n"
        "  description       (string)  — one or two sentences describing the video content\n"
        "  tags              (array)   — 3–10 short keyword tags\n"
        "  confidence_score  (number)  — overall confidence between 0.0 and 1.0\n"
        "  dominant_features (array)   — 2–5 key features that drove your analysis\n"
        "\n"
        "Important notes on the input data:\n"
        "  - The hair_color field contains both color and texture data per person.\n"
        "    Each person entry has 'dominant_color' (named color determined by\n"
        "    K-means clustering in CIELAB color space), 'dominant_lab' ([L,a,b]\n"
        "    centroid), and 'dominant_texture' (e.g. 'Straight', 'Curly', 'Wavy',\n"
        "    'Bald' — classified by SigLIP on a mask-filtered crop).\n"
        "    If multiple people are present, describe each person's hair color and\n"
        "    texture individually (e.g. 'Person 1 has straight black hair; Person 2\n"
        "    has wavy blonde hair.'). Use the top-level 'dominant_color' and\n"
        "    'dominant_texture' fields as summaries when only one person matters.\n"
        "  - If a person's 'occluded' flag is true, their hair could not be\n"
        "    reliably classified (likely wearing a hat or scarf) — mention this.\n"
        "\n"
        "Return ONLY valid JSON. Do not include any other text or markdown."
    )


def _section_hair_color(data: dict | None, flag: bool) -> list[str]:
    """
    Render the hair_color section with a human-readable per-person summary
    (color + texture) followed by the full JSON payload.
    """
    if not flag or data is None:
        return ["[Hair Color & Texture]: NOT AVAILABLE (extraction failed)"]

    people: list[dict] = data.get("people", [])
    count: int = data.get("count", len(people))

    def _person_summary(p: dict) -> str:
        color = p.get("dominant_color", "unknown")
        texture = p.get("dominant_texture", "unknown")
        occluded = " (occluded)" if p.get("occluded") else ""
        return f"{texture.lower()} {color}{occluded}"

    if count == 0:
        summary = "no people detected"
    elif count == 1:
        summary = f"1 person — {_person_summary(people[0])}"
    else:
        parts = [
            f"Person {p.get('person_id', i+1)}: {_person_summary(p)}"
            for i, p in enumerate(people)
        ]
        summary = f"{count} people — " + "; ".join(parts)

    return [
        f"[Hair Color & Texture]: {summary}",
        f"  (full data: {json.dumps(data, ensure_ascii=False)})",
    ]


def _build_user_prompt(bundle: ExtractionBundle) -> str:
    """
    Serialise the ExtractionBundle into a structured text block for the LLM.
    Each section is clearly labelled so the model can parse it easily.
    """
    lines: list[str] = [
        f"SOURCE ID: {bundle.source_id}",
        "",
        "=== EXTRACTED FEATURES ===",
    ]

    def _section(title: str, data: dict | None, flag: bool) -> list[str]:
        if not flag or data is None:
            return [f"[{title}]: NOT AVAILABLE (extraction failed)"]
        return [f"[{title}]: {json.dumps(data, ensure_ascii=False)}"]

    flags = bundle.partial_flags
    lines += _section_hair_color(bundle.hair_color, flags.hair_color)
    lines += _section("Body Build",   bundle.body_build,   flags.body_build)
    lines += _section("People Count", bundle.people_count, flags.people_count)
    lines += _section("Race",         bundle.race,         flags.race)
    lines += _section("Age",          bundle.age,          flags.age)

    if bundle.extraction_errors:
        lines += ["", "=== EXTRACTION ERRORS ==="]
        for mcp, err in bundle.extraction_errors.items():
            lines.append(f"  {mcp}: {err}")

    lines += [
        "",
        "=== INSTRUCTIONS ===",
        "Analyse the features above and return the required JSON object.",
        "If some features are NOT AVAILABLE, note that in your description and "
        "reduce confidence_score accordingly.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_text: str, partial: bool) -> ConsolidatedOutput:
    """
    Parse the raw LLM text into a validated ConsolidatedOutput.

    Strips any accidental markdown fences before parsing.
    Raises ValueError on malformed JSON or schema violations.
    """
    # Strip markdown code fences if the model adds them despite instructions
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON response: {exc}\n\nRaw:\n{raw_text}") from exc

    # Inject fields that are always determined by this agent, not the LLM
    data["partial_result"]  = partial
    data["prompt_version"]  = PROMPT_VERSION

    return ConsolidatedOutput(**data)


# ---------------------------------------------------------------------------
# Mock agent — no LLM, no API key required
# ---------------------------------------------------------------------------

class MockConsolidationAgent(ConsolidationAgent):
    """
    Drop-in replacement for ConsolidationAgent that skips the OpenAI call.

    Useful for local development, CI, and demos where an API key is unavailable
    or billing is a concern.  The description field is set to the full JSON
    serialisation of the ExtractionBundle so every extracted field is still
    visible in the output.

    Usage
    -----
        agent = MockConsolidationAgent()
        output = await agent.consolidate(bundle)
    """

    def __init__(self) -> None:
        # Deliberately skip the parent __init__ — no LLM is needed and we
        # don't want to require OPENAI_API_KEY just to instantiate the mock.
        pass  # _llm is never set; consolidate() is fully overridden below

    async def consolidate(self, bundle: ExtractionBundle) -> ConsolidatedOutput:
        """
        Return a ConsolidatedOutput whose description is the stringified bundle.
        All other fields are filled with sensible defaults derived from the bundle.
        """
        partial = any(
            not getattr(bundle.partial_flags, mcp)
            for mcp in ("hair_color", "body_build", "people_count", "race", "age")
        )

        # Serialise the full bundle as the description so callers can see every
        # extracted field without needing to inspect the object separately.
        description = bundle.model_dump_json(indent=2)

        # Derive a minimal tag list from whichever MCP fields succeeded.
        tags: list[str] = ["mock"]
        if bundle.hair_color:
            color = bundle.hair_color.get("dominant_color")
            if color:
                tags.append(f"hair:{color}")
            texture = bundle.hair_color.get("dominant_texture")
            if texture:
                tags.append(f"texture:{texture}")
        if bundle.body_build:
            build = bundle.body_build.get("body_build")
            if build:
                tags.append(f"build:{build}")
        if bundle.people_count:
            count = bundle.people_count.get("people_count")
            if count is not None:
                tags.append(f"people:{count}")

        # Confidence: 1.0 when all MCPs succeeded, reduced by 0.1 per failure.
        failed = sum(
            1 for mcp in ("hair_color", "body_build", "people_count", "race", "age")
            if not getattr(bundle.partial_flags, mcp)
        )
        confidence = round(max(0.0, 1.0 - failed * 0.1), 2)

        dominant: list[str] = [
            mcp for mcp in ("hair_color", "body_build", "people_count", "race", "age")
            if getattr(bundle.partial_flags, mcp)
        ]

        return ConsolidatedOutput(
            description=description,
            tags=tags,
            confidence_score=confidence,
            dominant_features=dominant,
            partial_result=partial,
            prompt_version=PROMPT_VERSION,
        )
