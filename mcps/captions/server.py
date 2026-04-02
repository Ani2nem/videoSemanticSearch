"""
mcps/captions/server.py

Async gRPC MCP server — On-screen text / caption extraction.
Port: 50054

Responsibility: Accept a video-frame image and extract all visible text using
PaddleOCR.

Model: PaddleOCR (lang='en' by default). Models are downloaded automatically on
first run and cached in ~/.paddleocr/.

Usage (standalone):
    python -m mcps.captions.server
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
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
    import numpy as np
    from paddleocr import PaddleOCR  # type: ignore
    from PIL import Image

    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False

PORT = 50054

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [captions] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if not _PADDLE_AVAILABLE:
    logger.warning(
        "paddleocr, Pillow, or numpy not installed — captions MCP running in MOCK mode. "
        "Install with: pip install paddleocr Pillow numpy"
    )

# Lazy-loaded PaddleOCR singleton (initialising is expensive)
_ocr_engine: Any = None


def _get_ocr() -> Any:
    global _ocr_engine  # noqa: PLW0603
    if _ocr_engine is None and _PADDLE_AVAILABLE:
        # use_angle_cls=True: handle rotated text; use_gpu=False: CPU-only by default
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
        logger.info("PaddleOCR engine initialised")
    return _ocr_engine


class CaptionsServicer(mcp_pb2_grpc.MCPServiceServicer):
    """
    Implements MCPService.ExtractFeatures for on-screen text extraction.

    Input  : FeatureRequest.payload — raw JPEG/PNG image bytes (video frame)
    Output : FeatureResponse.result_json — JSON with keys:
               "text_lines"  (list[str])  — extracted text lines, in reading order
               "line_count"  (int)        — number of distinct text lines found
               "raw_results" (list[dict]) — full OCR output per line
                                            [{text, confidence, bbox}, ...]
               "using_mock"  (bool)
    """

    async def ExtractFeatures(
        self,
        request: mcp_pb2.FeatureRequest,
        context: grpc.aio.ServicerContext,
    ) -> mcp_pb2.FeatureResponse:
        logger.info("Received request source_id=%s payload_type=%s",
                    request.source_id, request.payload_type)

        if request.payload_type != "image":
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"captions MCP expects payload_type='image', got '{request.payload_type}'",
            )

        if not request.payload:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty payload")

        try:
            result = await _extract_text(request.payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("OCR failed for source_id=%s", request.source_id)
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


async def _extract_text(image_bytes: bytes) -> dict:
    """Offload CPU-bound OCR to a thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _extract_text_sync, image_bytes)


def _extract_text_sync(image_bytes: bytes) -> dict:
    """
    Synchronous OCR using PaddleOCR.

    Steps:
        1. Decode image bytes → RGB numpy array
        2. Run PaddleOCR inference
        3. Parse results: each result entry is [[bbox_points], [text, score]]
        4. Return list of text lines + confidence

    Mock fallback (when PaddleOCR unavailable):
        Returns stub data with using_mock=True.

    TODO: For non-English video captions, set lang='ch', 'ja', 'ko', etc.
          For broadcast/subtitle text, consider EasyOCR or TrOCR as alternatives.
    """
    if not _PADDLE_AVAILABLE:
        return {
            "data": {
                "text_lines": ["Sample caption text"],
                "line_count": 1,
                "raw_results": [],
                "using_mock": True,
            },
            "confidence": 0.50,
        }

    ocr = _get_ocr()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    paddle_results = ocr.ocr(img_array, cls=True)

    text_lines: list[str] = []
    raw_results: list[dict] = []
    confidence_scores: list[float] = []

    # paddle_results is a list of pages; single-image → one page
    for page in (paddle_results or []):
        for line in (page or []):
            # line format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [text, score]]
            bbox, (text, score) = line
            text_lines.append(text)
            confidence_scores.append(float(score))
            raw_results.append({"text": text, "confidence": float(score), "bbox": bbox})

    avg_conf = (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0

    return {
        "data": {
            "text_lines": text_lines,
            "line_count": len(text_lines),
            "raw_results": raw_results,
            "using_mock": False,
        },
        "confidence": avg_conf if text_lines else 0.90,
    }


async def serve() -> None:
    server = grpc.aio.server()
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(CaptionsServicer(), server)
    listen_addr = f"[::]:{PORT}"
    server.add_insecure_port(listen_addr)
    logger.info("captions MCP server starting on %s", listen_addr)
    await server.start()
    logger.info("captions MCP server ready")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
