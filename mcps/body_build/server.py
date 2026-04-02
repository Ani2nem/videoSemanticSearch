"""
mcps/body_build/server.py

Async gRPC MCP server — Body Build feature extraction.
Port: 50052

Responsibility: Accept an image payload and infer a body-build category
(slim, athletic, muscular, etc.) from pose landmarks.

Primary implementation: MediaPipe Pose landmark extraction.
Fallback: clearly-labeled mock when MediaPipe is unavailable.

Usage (standalone):
    python -m mcps.body_build.server
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
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

# Optional heavy dependencies — gracefully degrade to mock if absent
try:
    from mediapipe.python.solutions import pose as mp_pose  # type: ignore
    import numpy as np
    from PIL import Image

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

PORT = 50052

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [body_build] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if not _MEDIAPIPE_AVAILABLE:
    logger.warning(
        "MediaPipe (or Pillow/numpy) not installed — body_build MCP running in MOCK mode. "
        "Install with: pip install mediapipe Pillow numpy"
    )


# ---------------------------------------------------------------------------
# MediaPipe Pose helper (only instantiated when library is present)
# ---------------------------------------------------------------------------
_pose_solution: Any = None


def _get_pose():
    """Lazy-initialise MediaPipe Pose (heavy; init once per process)."""
    global _pose_solution  # noqa: PLW0603
    if _pose_solution is None and _MEDIAPIPE_AVAILABLE:
        _pose_solution = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
        )
    return _pose_solution


class BodyBuildServicer(mcp_pb2_grpc.MCPServiceServicer):
    """
    Implements MCPService.ExtractFeatures for body-build inference.

    Input  : FeatureRequest.payload — raw JPEG/PNG image bytes
    Output : FeatureResponse.result_json — JSON with keys:
               "body_build"     (str)   — inferred category
               "landmarks_used" (int)   — number of visible landmarks
               "using_mock"     (bool)  — True when real model unavailable
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
                f"body_build MCP expects payload_type='image', got '{request.payload_type}'",
            )

        if not request.payload:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty payload")

        try:
            result = await _infer_body_build(request.payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference failed for source_id=%s", request.source_id)
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


async def _infer_body_build(image_bytes: bytes) -> dict:
    """
    Run body-build inference in a thread pool so the event loop stays free.
    Falls back to mock if MediaPipe is not installed.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _infer_body_build_sync, image_bytes)


def _infer_body_build_sync(image_bytes: bytes) -> dict:
    """
    Synchronous body-build inference — called from thread executor.

    MediaPipe path:
        1. Decode image bytes → RGB numpy array
        2. Run Pose landmark detection
        3. Compute shoulder-width / hip-width ratio to classify build type
        4. Return structured result

    Mock path (when MediaPipe unavailable):
        Returns a stub result clearly marked with using_mock=True.

    TODO: Enhance classification logic — current heuristics are a starting point.
          Consider replacing with a dedicated body-composition model trained on
          fitness/anthropometry datasets.
    """
    if not _MEDIAPIPE_AVAILABLE:
        # ---- MOCK FALLBACK ----
        return {
            "data": {
                "body_build": "athletic",
                "landmarks_used": 0,
                "using_mock": True,
            },
            "confidence": 0.50,
        }

    # ---- MEDIAPIPE PATH ----
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    pose = _get_pose()
    results = pose.process(img_array)

    if not results.pose_landmarks:
        # No person detected — return a neutral category
        return {
            "data": {
                "body_build": "unknown",
                "landmarks_used": 0,
                "using_mock": False,
            },
            "confidence": 0.20,
        }

    landmarks = results.pose_landmarks.landmark
    visible = [lm for lm in landmarks if lm.visibility > 0.5]

    # Shoulder width: landmarks 11 (left shoulder) and 12 (right shoulder)
    # Hip width: landmarks 23 (left hip) and 24 (right hip)
    try:
        left_shoulder  = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip       = landmarks[23]
        right_hip      = landmarks[24]

        shoulder_width = math.dist(
            (left_shoulder.x, left_shoulder.y),
            (right_shoulder.x, right_shoulder.y),
        )
        hip_width = math.dist(
            (left_hip.x, left_hip.y),
            (right_hip.x, right_hip.y),
        )

        ratio = shoulder_width / (hip_width + 1e-6)

        # Heuristic thresholds — TODO: calibrate on a labelled dataset
        if ratio > 1.4:
            build = "muscular"
        elif ratio > 1.15:
            build = "athletic"
        elif ratio > 0.9:
            build = "average"
        else:
            build = "slim"

        confidence = min(0.95, 0.5 + len(visible) / (len(landmarks) * 2))

    except IndexError:
        build = "unknown"
        confidence = 0.20

    return {
        "data": {
            "body_build": build,
            "landmarks_used": len(visible),
            "using_mock": False,
        },
        "confidence": confidence,
    }


async def serve() -> None:
    server = grpc.aio.server()
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(BodyBuildServicer(), server)
    listen_addr = f"[::]:{PORT}"
    server.add_insecure_port(listen_addr)
    logger.info("body_build MCP server starting on %s", listen_addr)
    await server.start()
    logger.info("body_build MCP server ready")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
