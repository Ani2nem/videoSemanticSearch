"""
agents/extraction_agent.py

ExtractionAgent — orchestrates concurrent calls to all five MCP servers.

Features
--------
- Async gRPC (grpc.aio) — no sync stubs anywhere in this module
- asyncio.Semaphore caps concurrent input processing at MAX_CONCURRENT_INPUTS
- Per-MCP retry logic: 2 retries, exponential backoff (100 ms → 300 ms)
- Per-MCP circuit breaker (Closed → Open → Half-Open) with:
    • Opens after 5 consecutive failures
    • Half-open probe fires after 30 s
    • Closes on successful probe; resets timer on failed probe
- Soft dependencies (hair_color, body_build, people_count): failure yields a
  partial ExtractionBundle (partial_flags[name]=False, error logged)
- Hard dependencies (captions, audio): failure raises ExtractionError,
  aborting the bundle for that source

Usage
-----
    from agents.extraction_agent import ExtractionAgent
    agent = ExtractionAgent()
    bundles = await agent.process([
        {"source_id": "clip_001", "image_path": "samples/frame.jpg", "audio_path": "samples/clip.wav"},
    ])
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import grpc
import grpc.aio

try:
    import mcp_pb2
    import mcp_pb2_grpc
except ImportError as exc:
    raise ImportError(
        "Generated proto stubs not found. Run:\n"
        "  python -m grpc_tools.protoc -I./proto "
        "--python_out=. --grpc_python_out=. ./proto/mcp.proto"
    ) from exc

from schemas.extraction_bundle import ExtractionBundle, PartialFlags

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MCP endpoints
MCP_ADDRESSES: dict[str, str] = {
    "hair_color":   "localhost:50051",
    "body_build":   "localhost:50052",
    "people_count": "localhost:50053",
    "captions":     "localhost:50054",
    "audio":        "localhost:50055",
    "race":         "localhost:50056",
}

# payload_type per MCP
MCP_PAYLOAD_TYPES: dict[str, str] = {
    "hair_color":   "image",
    "body_build":   "image",
    "people_count": "image",
    "captions":     "image",
    "audio":        "audio",
    "race":         "image",
}

# Which MCPs are hard dependencies (failure → raise ExtractionError)
HARD_DEPENDENCIES: frozenset[str] = frozenset({"captions", "audio"})

# Retry parameters
MAX_RETRIES = 2
BACKOFF_DELAYS = (0.1, 0.3)  # seconds: 100 ms after 1st failure, 300 ms after 2nd

# Circuit breaker parameters
CB_FAILURE_THRESHOLD = 5      # consecutive failures before opening
CB_RECOVERY_TIMEOUT  = 30.0   # seconds before attempting a half-open probe

# Max simultaneous inputs processed by the agent
MAX_CONCURRENT_INPUTS = 8

# Seconds to wait for an RPC to complete (covers both TCP connect + server response).
# Servers that are offline are detected within this window instead of hanging
# until the OS-level TCP timeout fires.
CONNECT_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CBState(Enum):
    CLOSED    = auto()
    OPEN      = auto()
    HALF_OPEN = auto()


@dataclass
class CircuitBreaker:
    """
    Per-MCP circuit breaker.

    State machine
    -------------
    CLOSED  → on 5th consecutive failure → OPEN
    OPEN    → after recovery_timeout seconds → HALF_OPEN
    HALF_OPEN → on success → CLOSED
              → on failure → OPEN (timer reset)
    """
    name: str
    failure_threshold: int = CB_FAILURE_THRESHOLD
    recovery_timeout: float = CB_RECOVERY_TIMEOUT

    _state: CBState = field(default=CBState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)

    @property
    def state(self) -> CBState:
        return self._state

    def is_open(self) -> bool:
        """
        Returns True if the circuit should block the call.
        Transitions OPEN → HALF_OPEN when the recovery timeout has elapsed.
        """
        if self._state == CBState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                logger.info("[CB:%s] Recovery timeout elapsed → HALF_OPEN", self.name)
                self._state = CBState.HALF_OPEN
                return False   # allow the half-open probe through
            return True
        return False

    def record_success(self) -> None:
        if self._state == CBState.HALF_OPEN:
            logger.info("[CB:%s] Probe succeeded → CLOSED", self.name)
        self._state = CBState.CLOSED
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._state == CBState.HALF_OPEN:
            logger.warning("[CB:%s] Probe failed → OPEN (timer reset)", self.name)
            self._state = CBState.OPEN
            self._opened_at = time.monotonic()
            return
        if self._consecutive_failures >= self.failure_threshold:
            if self._state != CBState.OPEN:
                logger.warning(
                    "[CB:%s] %d consecutive failures → OPEN",
                    self.name, self._consecutive_failures,
                )
            self._state = CBState.OPEN
            self._opened_at = time.monotonic()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ExtractionError(RuntimeError):
    """Raised when a hard-dependency MCP fails after all retries."""

    def __init__(self, source_id: str, mcp_name: str, reason: str) -> None:
        super().__init__(
            f"Hard dependency '{mcp_name}' failed for source_id='{source_id}': {reason}"
        )
        self.source_id = source_id
        self.mcp_name = mcp_name
        self.reason = reason


class ServerOfflineError(RuntimeError):
    """
    Raised when a gRPC server is unreachable (UNAVAILABLE) or did not respond
    within CONNECT_TIMEOUT (DEADLINE_EXCEEDED).

    Treated as a soft failure for every MCP — including hard dependencies —
    because an offline server is an infrastructure problem, not a data problem.
    The pipeline should continue with whatever servers are available.
    """

    def __init__(self, mcp_name: str, status_code: grpc.StatusCode, detail: str) -> None:
        super().__init__(f"[{mcp_name}] Server offline ({status_code.name}): {detail}")
        self.mcp_name = mcp_name
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Extraction Agent
# ---------------------------------------------------------------------------

class ExtractionAgent:
    """
    Orchestrates concurrent MCP calls for a batch of video inputs.

    Parameters
    ----------
    mcp_addresses : override default MCP address map (useful for testing)
    """

    def __init__(self, mcp_addresses: dict[str, str] | None = None) -> None:
        self._addresses = mcp_addresses or MCP_ADDRESSES
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(name=name) for name in self._addresses
        }
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_INPUTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, inputs: list[dict[str, Any]]) -> list[ExtractionBundle]:
        """
        Process a batch of inputs concurrently (capped at MAX_CONCURRENT_INPUTS).

        Each input dict must contain:
            source_id  : str
            image_path : str | Path  (can be empty string if no image)
            audio_path : str | Path  (can be empty string if no audio)

        Raises ExtractionError if a hard-dependency MCP fails on any input.
        """
        tasks = [self._process_one_guarded(inp) for inp in inputs]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_one_guarded(self, inp: dict[str, Any]) -> ExtractionBundle:
        async with self._semaphore:
            return await self._process_one(inp)

    async def _process_one(self, inp: dict[str, Any]) -> ExtractionBundle:
        source_id  = inp["source_id"]
        image_path = Path(inp.get("image_path", "") or "")
        audio_path = Path(inp.get("audio_path", "") or "")

        image_bytes = image_path.read_bytes() if image_path.is_file() else b""
        audio_bytes = audio_path.read_bytes() if audio_path.is_file() else b""

        bundle = ExtractionBundle(source_id=source_id)

        # Fan out to all MCPs concurrently
        results = await asyncio.gather(
            self._call_mcp("hair_color",   image_bytes, source_id),
            self._call_mcp("body_build",   image_bytes, source_id),
            self._call_mcp("people_count", image_bytes, source_id),
            self._call_mcp("captions",     image_bytes, source_id),
            self._call_mcp("audio",        audio_bytes, source_id),
            self._call_mcp("race",         image_bytes, source_id),
            return_exceptions=True,
        )

        mcp_names = ["hair_color", "body_build", "people_count", "captions", "audio", "race"]
        for name, outcome in zip(mcp_names, results):
            if isinstance(outcome, Exception):
                if isinstance(outcome, ServerOfflineError):
                    # Server is unreachable — always a soft failure regardless of
                    # dependency type. Record "Server Offline" and keep going.
                    bundle.extraction_errors[name] = "Server Offline"
                    setattr(bundle.partial_flags, name, False)
                    logger.warning(
                        "[%s] '%s' is offline → skipped (partial result)", source_id, name
                    )
                else:
                    error_msg = str(outcome)
                    bundle.extraction_errors[name] = error_msg
                    setattr(bundle.partial_flags, name, False)

                    if name in HARD_DEPENDENCIES:
                        raise ExtractionError(source_id, name, error_msg)

                    logger.warning(
                        "[%s] soft-dep '%s' failed → partial result. Error: %s",
                        source_id, name, error_msg,
                    )
            else:
                # outcome is a parsed dict from result_json
                setattr(bundle, name, outcome)

        return bundle

    async def _call_mcp(
        self,
        mcp_name: str,
        payload: bytes,
        source_id: str,
    ) -> dict:
        """
        Call a single MCP with retry + circuit breaker logic.

        Returns the parsed dict from FeatureResponse.result_json on success.
        Raises the last exception if all retries are exhausted or circuit is open.
        """
        cb = self._circuit_breakers[mcp_name]

        # Circuit is open — fail immediately without hitting the server
        if cb.is_open():
            raise RuntimeError(
                f"Circuit breaker for '{mcp_name}' is OPEN — skipping call"
            )

        address      = self._addresses[mcp_name]
        payload_type = MCP_PAYLOAD_TYPES[mcp_name]
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                result = await self._grpc_call(
                    address, mcp_name, payload, payload_type, source_id
                )
                cb.record_success()
                return result

            except ServerOfflineError:
                # Server is not up — no value in retrying, propagate immediately.
                cb.record_failure()
                raise

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                cb.record_failure()
                logger.warning(
                    "[%s] '%s' attempt %d/%d failed: %s",
                    source_id, mcp_name, attempt + 1, MAX_RETRIES + 1, exc,
                )
                if attempt < MAX_RETRIES:
                    delay = BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)]
                    await asyncio.sleep(delay)

                # Circuit may have just opened — stop retrying
                if cb.is_open():
                    break

        raise last_exc  # type: ignore[misc]

    @staticmethod
    async def _grpc_call(
        address: str,
        mcp_name: str,
        payload: bytes,
        payload_type: str,
        source_id: str,
    ) -> dict:
        """
        Open an async gRPC channel, call ExtractFeatures, return parsed dict.

        Uses grpc.aio exclusively — no synchronous stubs.
        The channel is opened and closed per-call to keep the design simple.
        In a production service, consider sharing channel objects across calls.

        A per-call timeout of CONNECT_TIMEOUT seconds is applied so that
        unreachable servers are detected quickly rather than hanging.
        UNAVAILABLE and DEADLINE_EXCEEDED are both re-raised as ServerOfflineError
        so callers can distinguish "server is down" from other failures.
        """
        try:
            async with grpc.aio.insecure_channel(address) as channel:
                stub = mcp_pb2_grpc.MCPServiceStub(channel)
                request = mcp_pb2.FeatureRequest(
                    payload=payload,
                    payload_type=payload_type,
                    source_id=source_id,
                )
                response: mcp_pb2.FeatureResponse = await stub.ExtractFeatures(
                    request, timeout=CONNECT_TIMEOUT
                )
        except grpc.aio.AioRpcError as exc:
            if exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                raise ServerOfflineError(mcp_name, exc.code(), exc.details() or "") from exc
            raise

        if not response.success:
            raise RuntimeError(
                f"MCP '{mcp_name}' returned success=False: {response.error_message}"
            )

        return json.loads(response.result_json)
