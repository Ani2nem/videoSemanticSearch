"""
mcps/audio/server.py

Async gRPC MCP server — Audio transcription.
Port: 50055

Responsibility: Accept raw audio bytes and return a transcript with detected
language using faster-whisper (base model, CPU-compatible).

Model: faster-whisper "base" (~145 MB). Downloaded automatically on first run
and cached in ~/.cache/huggingface/hub/.

Usage (standalone):
    python -m mcps.audio.server
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import tempfile
import os
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

try:
    from faster_whisper import WhisperModel  # type: ignore

    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

PORT = 50055

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [audio] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if not _WHISPER_AVAILABLE:
    logger.warning(
        "faster-whisper not installed — audio MCP running in MOCK mode. "
        "Install with: pip install faster-whisper"
    )

# Lazy-loaded Whisper singleton
_whisper_model: Any = None


def _get_whisper() -> Any:
    global _whisper_model  # noqa: PLW0603
    if _whisper_model is None and _WHISPER_AVAILABLE:
        # device="cpu", compute_type="int8" — works on any machine without a GPU.
        # TODO: Set device="cuda" and compute_type="float16" for GPU inference.
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("faster-whisper base model loaded")
    return _whisper_model


class AudioServicer(mcp_pb2_grpc.MCPServiceServicer):
    """
    Implements MCPService.ExtractFeatures for audio transcription.

    Input  : FeatureRequest.payload — raw audio bytes (WAV, MP3, FLAC, etc.)
    Output : FeatureResponse.result_json — JSON with keys:
               "transcript"       (str)        — full transcribed text
               "language"         (str)        — detected language code, e.g. "en"
               "language_prob"    (float)      — detection probability
               "segments"         (list[dict]) — [{start, end, text}, ...]
               "using_mock"       (bool)
    """

    async def ExtractFeatures(
        self,
        request: mcp_pb2.FeatureRequest,
        context: grpc.aio.ServicerContext,
    ) -> mcp_pb2.FeatureResponse:
        logger.info("Received request source_id=%s payload_type=%s",
                    request.source_id, request.payload_type)

        if request.payload_type != "audio":
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"audio MCP expects payload_type='audio', got '{request.payload_type}'",
            )

        if not request.payload:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty payload")

        try:
            result = await _transcribe_audio(request.payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Transcription failed for source_id=%s", request.source_id)
            return mcp_pb2.FeatureResponse(
                result_json="{}",
                confidence=0.0,
                success=False,
                error_message=str(exc),
            )

        return mcp_pb2.FeatureResponse(
            result_json=json.dumps(result["data"]),
            confidence=result["confidence"],
            success=True,
            error_message="",
        )


async def _transcribe_audio(audio_bytes: bytes) -> dict:
    """Offload CPU-bound Whisper inference to a thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe_audio_sync, audio_bytes)


def _transcribe_audio_sync(audio_bytes: bytes) -> dict:
    """
    Synchronous audio transcription using faster-whisper.

    Steps:
        1. Write audio bytes to a temporary file (faster-whisper needs a file path)
        2. Run model.transcribe() to get segments and language info
        3. Concatenate segment texts into a full transcript
        4. Clean up temp file

    Mock fallback (when faster-whisper unavailable):
        Returns stub data with using_mock=True.

    TODO: For long-form audio, pass beam_size=5 and vad_filter=True for better
          accuracy. For real-time streaming, consider WhisperModel.transcribe()
          with a generator over audio chunks.
    """
    if not _WHISPER_AVAILABLE:
        return {
            "data": {
                "transcript": "This is a mock transcription of the audio clip.",
                "language": "en",
                "language_prob": 0.99,
                "segments": [],
                "using_mock": True,
            },
            "confidence": 0.50,
        }

    model = _get_whisper()

    # faster-whisper requires a file path, not a bytes buffer
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments_iter, info = model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True,   # skip silent regions
        )
        segments = [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in segments_iter
        ]
    finally:
        os.unlink(tmp_path)

    transcript = " ".join(seg["text"] for seg in segments).strip()
    lang_prob = float(info.language_probability)

    return {
        "data": {
            "transcript": transcript,
            "language": info.language,
            "language_prob": lang_prob,
            "segments": segments,
            "using_mock": False,
        },
        "confidence": lang_prob,
    }


async def serve() -> None:
    server = grpc.aio.server()
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(AudioServicer(), server)
    listen_addr = f"[::]:{PORT}"
    server.add_insecure_port(listen_addr)
    logger.info("audio MCP server starting on %s", listen_addr)
    await server.start()
    logger.info("audio MCP server ready")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
