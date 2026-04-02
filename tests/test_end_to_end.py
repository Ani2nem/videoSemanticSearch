"""
tests/test_end_to_end.py

End-to-end test: ExtractionAgent → ConsolidationAgent.

The LLM (OpenAI) and all MCP network calls are mocked so the test runs
without any external services or API keys.

Assertions:
  - Final ConsolidatedOutput matches the schema exactly (all required fields)
  - confidence_score is in [0.0, 1.0]
  - tags is a non-empty list
  - prompt_version equals the current PROMPT_VERSION constant

Run:
    pytest tests/test_end_to_end.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.consolidation_agent import ConsolidationAgent, ConsolidatedOutput, PROMPT_VERSION
from agents.extraction_agent import ExtractionAgent
from schemas.extraction_bundle import ExtractionBundle, PartialFlags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_extraction_bundle() -> ExtractionBundle:
    """Return a fully-populated bundle (all MCPs succeeded)."""
    return ExtractionBundle(
        source_id="e2e-test-001",
        hair_color={"dominant_color": "black", "palette": ["black"]},
        body_build={"body_build": "athletic", "landmarks_used": 20, "using_mock": False},
        people_count={"people_count": 1, "bounding_boxes": 1, "detections": [], "using_mock": False},
        captions={"text_lines": ["Hello World"], "line_count": 1, "raw_results": [], "using_mock": False},
        audio={"transcript": "Test audio", "language": "en", "language_prob": 0.99, "segments": [], "using_mock": False},
        partial_flags=PartialFlags(
            hair_color=True,
            body_build=True,
            people_count=True,
            captions=True,
            audio=True,
        ),
        extraction_errors={},
    )


def _mock_llm_json_response() -> str:
    """A valid JSON string the mocked LLM will return."""
    return json.dumps({
        "description": "A single athletic person with black hair speaking in English.",
        "tags": ["person", "athletic", "black-hair", "english", "caption"],
        "confidence_score": 0.92,
        "dominant_features": ["hair_color", "body_build", "audio_transcript"],
    })


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_schema_and_prompt_version(tmp_path):
    """
    Full pipeline test:
      1. ExtractionAgent returns a mock bundle (no gRPC calls made)
      2. ConsolidationAgent calls a mocked LLM
      3. Final ConsolidatedOutput is fully validated

    Assertions:
      - All required fields present and correctly typed
      - confidence_score in [0.0, 1.0]
      - tags is a list with at least one element
      - prompt_version == PROMPT_VERSION
      - partial_result == False (all flags True in this test)
    """
    bundle = _mock_extraction_bundle()

    # --- Mock ExtractionAgent.process to return the pre-built bundle ---
    mock_agent = ExtractionAgent.__new__(ExtractionAgent)
    mock_agent._semaphore = None  # not needed
    mock_agent.process = AsyncMock(return_value=[bundle])

    # --- Mock the LLM call inside ConsolidationAgent ---
    fake_llm_message = MagicMock()
    fake_llm_message.content = _mock_llm_json_response()

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-not-real"}):
        consolidation_agent = ConsolidationAgent()

    # Patch the underlying LangChain LLM instance
    consolidation_agent._llm = AsyncMock()
    consolidation_agent._llm.ainvoke = AsyncMock(return_value=fake_llm_message)

    # --- Run the pipeline ---
    bundles = await mock_agent.process([])
    output  = await consolidation_agent.consolidate(bundles[0])

    # --- Schema assertions ---
    assert isinstance(output, ConsolidatedOutput), (
        "Output must be a ConsolidatedOutput instance"
    )
    assert isinstance(output.description, str) and output.description, (
        "description must be a non-empty string"
    )
    assert isinstance(output.tags, list) and len(output.tags) > 0, (
        "tags must be a non-empty list"
    )
    assert 0.0 <= output.confidence_score <= 1.0, (
        f"confidence_score {output.confidence_score} must be in [0.0, 1.0]"
    )
    assert isinstance(output.dominant_features, list) and len(output.dominant_features) > 0, (
        "dominant_features must be a non-empty list"
    )
    assert isinstance(output.partial_result, bool), (
        "partial_result must be a bool"
    )
    assert output.partial_result is False, (
        "partial_result should be False when all MCP flags are True"
    )

    # --- prompt_version assertion ---
    assert output.prompt_version == PROMPT_VERSION, (
        f"prompt_version must equal PROMPT_VERSION ('{PROMPT_VERSION}') "
        f"but got '{output.prompt_version}'"
    )


@pytest.mark.asyncio
async def test_partial_bundle_sets_partial_result_true():
    """
    When the bundle has a partial flag (hair_color=False), the output's
    partial_result must be True.
    """
    bundle = _mock_extraction_bundle()
    bundle.partial_flags.hair_color = False
    bundle.extraction_errors["hair_color"] = "MCP unavailable"

    fake_llm_message = MagicMock()
    fake_llm_message.content = _mock_llm_json_response()

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-not-real"}):
        agent = ConsolidationAgent()

    agent._llm = AsyncMock()
    agent._llm.ainvoke = AsyncMock(return_value=fake_llm_message)

    output = await agent.consolidate(bundle)

    assert output.partial_result is True, (
        "partial_result must be True when any partial_flags entry is False"
    )
    assert output.prompt_version == PROMPT_VERSION
