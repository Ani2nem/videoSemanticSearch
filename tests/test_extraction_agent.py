"""
tests/test_extraction_agent.py

Unit tests for ExtractionAgent — circuit breaker, retry, soft/hard dependency
handling, and partial-result semantics.

All MCP network calls are mocked so no running servers are required.

Run:
    pytest tests/test_extraction_agent.py -v
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.extraction_agent import (
    CB_FAILURE_THRESHOLD,
    CB_RECOVERY_TIMEOUT,
    CircuitBreaker,
    CBState,
    ExtractionAgent,
    ExtractionError,
)
from schemas.extraction_bundle import ExtractionBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(overrides: dict | None = None) -> ExtractionAgent:
    """Create an agent with all MCP addresses pointing to non-existent hosts."""
    addresses = {
        "hair_color":   "localhost:59051",
        "body_build":   "localhost:59052",
        "people_count": "localhost:59053",
        "captions":     "localhost:59054",
        "audio":        "localhost:59055",
    }
    if overrides:
        addresses.update(overrides)
    return ExtractionAgent(mcp_addresses=addresses)


def _success_payload(name: str) -> dict:
    return {"feature": name, "value": "mock"}


def _make_input(tmp_path) -> dict:
    """Create a minimal test input with empty-but-existing sample files."""
    image_file = tmp_path / "frame.png"
    audio_file = tmp_path / "clip.wav"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
    audio_file.write_bytes(b"RIFF")               # minimal WAV header

    return {
        "source_id":  "test-src-001",
        "image_path": str(image_file),
        "audio_path": str(audio_file),
    }


# ---------------------------------------------------------------------------
# Test 1: Soft dependency failure (hair_color)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_soft_dependency_failure_does_not_raise(tmp_path):
    """
    When hair_color MCP returns an error, the pipeline should complete
    with partial_flags.hair_color == False and all other fields populated.
    ExtractionError must NOT be raised.
    """
    agent = _make_agent()

    async def mock_grpc_call(address, mcp_name, payload, payload_type, source_id):
        if mcp_name == "hair_color":
            raise RuntimeError("hair_color MCP unavailable")
        return _success_payload(mcp_name)

    with patch.object(ExtractionAgent, "_grpc_call", new=AsyncMock(side_effect=mock_grpc_call)):
        # Also bypass retry delays
        with patch("agents.extraction_agent.BACKOFF_DELAYS", (0.0, 0.0)):
            bundles = await agent.process([_make_input(tmp_path)])

    bundle = bundles[0]
    assert isinstance(bundle, ExtractionBundle)
    assert bundle.partial_flags.hair_color is False, "hair_color flag should be False"
    assert "hair_color" in bundle.extraction_errors
    # Soft deps that succeeded should be set
    assert bundle.body_build   is not None
    assert bundle.people_count is not None
    # Hard deps that succeeded should also be set
    assert bundle.captions is not None
    assert bundle.audio    is not None


# ---------------------------------------------------------------------------
# Test 2: Hard dependency failure (captions)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hard_dependency_failure_raises_extraction_error(tmp_path):
    """
    When captions MCP fails (hard dependency), ExtractionError must be raised.
    """
    agent = _make_agent()

    async def mock_grpc_call(address, mcp_name, payload, payload_type, source_id):
        if mcp_name == "captions":
            raise RuntimeError("captions service down")
        return _success_payload(mcp_name)

    with patch.object(ExtractionAgent, "_grpc_call", new=AsyncMock(side_effect=mock_grpc_call)):
        with patch("agents.extraction_agent.BACKOFF_DELAYS", (0.0, 0.0)):
            with pytest.raises(ExtractionError) as exc_info:
                await agent.process([_make_input(tmp_path)])

    err = exc_info.value
    assert err.mcp_name == "captions"
    assert err.source_id == "test-src-001"


# ---------------------------------------------------------------------------
# Test 3: Circuit breaker opens after 5 consecutive failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold(tmp_path):
    """
    After CB_FAILURE_THRESHOLD consecutive failures on body_build, the circuit
    should open and subsequent calls should fail immediately without contacting
    the MCP (i.e. _grpc_call is NOT invoked once the circuit is open).
    """
    agent = _make_agent()
    cb    = agent._circuit_breakers["body_build"]

    call_count = 0

    async def mock_grpc_call(address, mcp_name, payload, payload_type, source_id):
        nonlocal call_count
        if mcp_name == "body_build":
            call_count += 1
            raise RuntimeError("body_build is broken")
        return _success_payload(mcp_name)

    # Drive enough failures to open the circuit (5 consecutive failures, but
    # each _call_mcp attempt allows MAX_RETRIES=2 retries before giving up,
    # meaning each process() cycle contributes up to MAX_RETRIES+1 = 3 failures)
    with patch.object(ExtractionAgent, "_grpc_call", new=AsyncMock(side_effect=mock_grpc_call)):
        with patch("agents.extraction_agent.BACKOFF_DELAYS", (0.0, 0.0)):
            # Two passes: first gives 3 failures (attempts 1,2,3), second gives
            # 2 more — total 5 → circuit opens mid-second pass.
            # We run enough passes to guarantee the circuit opens.
            for _ in range(3):
                try:
                    await agent.process([_make_input(tmp_path)])
                except ExtractionError:
                    pass  # audio/captions might also fail; we only care about the CB

    # Circuit must be open now
    assert cb.state == CBState.OPEN, f"Expected OPEN but got {cb.state}"

    # Track calls after circuit is open
    calls_before = call_count

    with patch.object(ExtractionAgent, "_grpc_call", new=AsyncMock(side_effect=mock_grpc_call)):
        with patch("agents.extraction_agent.BACKOFF_DELAYS", (0.0, 0.0)):
            try:
                await agent.process([_make_input(tmp_path)])
            except ExtractionError:
                pass

    # _grpc_call for body_build must NOT have been called again (circuit is open)
    calls_for_body_build_after_open = call_count - calls_before
    assert calls_for_body_build_after_open == 0, (
        f"_grpc_call was invoked {calls_for_body_build_after_open} times after circuit opened"
    )


# ---------------------------------------------------------------------------
# Test 4: Half-open probe closes the circuit on success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_closes_on_success(tmp_path):
    """
    After the circuit is OPEN and CB_RECOVERY_TIMEOUT seconds have passed,
    one probe request should be sent. If it succeeds, the circuit closes.
    """
    agent = _make_agent()
    cb    = agent._circuit_breakers["body_build"]

    # Manually open the circuit
    for _ in range(CB_FAILURE_THRESHOLD):
        cb.record_failure()
    assert cb.state == CBState.OPEN

    # Simulate 30+ seconds elapsing by back-dating the opened_at timestamp
    cb._opened_at = time.monotonic() - (CB_RECOVERY_TIMEOUT + 1.0)

    # Now the circuit should transition to HALF_OPEN when is_open() is called
    assert cb.is_open() is False          # transitions to HALF_OPEN
    assert cb.state == CBState.HALF_OPEN

    # Probe succeeds
    async def mock_grpc_call(address, mcp_name, payload, payload_type, source_id):
        return _success_payload(mcp_name)

    with patch.object(ExtractionAgent, "_grpc_call", new=AsyncMock(side_effect=mock_grpc_call)):
        await agent.process([_make_input(tmp_path)])

    # Circuit must be CLOSED after a successful probe
    assert cb.state == CBState.CLOSED, (
        f"Expected CLOSED after successful probe but got {cb.state}"
    )
    assert cb._consecutive_failures == 0
