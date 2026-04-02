"""
mcps/people_count/server.py

Async gRPC MCP server — People Count feature extraction.
Port: 50053

Responsibility: Accept an image payload and return a count of people detected
using Ultralytics YOLOv8.

Model: yolov8n.pt (nano — fast, ~6 MB). The model is downloaded automatically
by Ultralytics on first run and cached in ~/.cache/ultralytics/.

Usage (standalone):
    python -m mcps.people_count.server
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
    from PIL import Image
    from ultralytics import YOLO  # type: ignore

    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

PORT = 50053

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [people_count] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if not _YOLO_AVAILABLE:
    logger.warning(
        "ultralytics or Pillow not installed — people_count MCP running in MOCK mode. "
        "Install with: pip install ultralytics Pillow"
    )

# Lazy-loaded YOLO model singleton
_yolo_model: Any = None


def _get_yolo():
    """Load yolov8n.pt once and reuse across requests."""
    global _yolo_model  # noqa: PLW0603
    if _yolo_model is None and _YOLO_AVAILABLE:
        _yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n model loaded")
    return _yolo_model

# YOLO class index for "person" in COCO
_PERSON_CLASS_ID = 0


class PeopleCountServicer(mcp_pb2_grpc.MCPServiceServicer):
    """
    Implements MCPService.ExtractFeatures for people counting.

    Input  : FeatureRequest.payload — raw JPEG/PNG image bytes
    Output : FeatureResponse.result_json — JSON with keys:
               "people_count"   (int)        — number of persons detected
               "bounding_boxes" (int)        — alias for people_count (one box per person)
               "detections"     (list[dict]) — [{x1,y1,x2,y2,confidence}, ...]
               "using_mock"     (bool)
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
                f"people_count MCP expects payload_type='image', got '{request.payload_type}'",
            )

        if not request.payload:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty payload")

        try:
            result = await _count_people(request.payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Detection failed for source_id=%s", request.source_id)
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


async def _count_people(image_bytes: bytes) -> dict:
    """Offload CPU-bound YOLO inference to a thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _count_people_sync, image_bytes)


def _count_people_sync(image_bytes: bytes) -> dict:
    """
    Synchronous people-counting using YOLOv8.

    Steps:
        1. Decode image bytes to a PIL Image
        2. Run YOLO inference (class filter: person)
        3. Collect bounding boxes and per-detection confidence scores
        4. Return summary dict

    Mock fallback (when ultralytics unavailable):
        Returns stub data with using_mock=True.

    TODO: For higher accuracy on dense crowds, swap yolov8n.pt for:
          - yolov8m.pt or yolov8l.pt (larger models)
          - or a crowd-counting specialist model (e.g. CSRNet, BayesCrowd)
    """
    if not _YOLO_AVAILABLE:
        return {
            "data": {
                "people_count": 2,
                "bounding_boxes": 2,
                "detections": [],
                "using_mock": True,
            },
            "confidence": 0.50,
        }

    model = _get_yolo()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image, classes=[_PERSON_CLASS_ID], verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

    count = len(detections)
    avg_conf = (sum(d["confidence"] for d in detections) / count) if count else 0.0

    return {
        "data": {
            "people_count": count,
            "bounding_boxes": count,
            "detections": detections,
            "using_mock": False,
        },
        "confidence": avg_conf if count else 0.90,  # 0.90 = high confidence of "zero people"
    }


async def serve() -> None:
    server = grpc.aio.server()
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(PeopleCountServicer(), server)
    listen_addr = f"[::]:{PORT}"
    server.add_insecure_port(listen_addr)
    logger.info("people_count MCP server starting on %s", listen_addr)
    await server.start()
    logger.info("people_count MCP server ready")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
