"""
mcps/body_build/server.py

Async gRPC MCP server — Body Build feature extraction.
Port: 50052

Responsibility: Accept an image payload and classify the body build of the
person(s) in the frame using CLIP zero-shot classification with template
ensembling.

Stack (zero new installs — everything already in the venv):
  CLIP model : openai/clip-vit-large-patch14 (already downloaded by race MCP)
  Classifies the full frame — body build is a whole-scene feature.

Categories: slim, petite, curvy, athletic, average, plus_size

Usage (standalone):
    python -m mcps.body_build.server
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
    import torch  # type: ignore
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    from PIL import Image
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

PORT = 50052

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [body_build] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if not _CLIP_AVAILABLE:
    logger.warning("torch/transformers/Pillow not installed — body_build MCP running in MOCK mode.")

# ---------------------------------------------------------------------------
# CLIP zero-shot classifier
# ---------------------------------------------------------------------------

_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

_BUILD_TEMPLATES: list[list[str]] = [
    # slim
    [
        "a photo of a slim person",
        "a very thin person's body",
        "a slender figure",
    ],
    # petite
    [
        "a photo of a petite person",
        "a short and slim person",
        "a small-framed person",
    ],
    # curvy
    [
        "a photo of a curvy person",
        "a person with curves",
        "a voluptuous figure",
    ],
    # athletic
    [
        "a photo of an athletic person",
        "a fit toned person's body",
        "a muscular athletic figure",
    ],
    # average
    [
        "a photo of an average build person",
        "a normal body type",
        "a medium build person",
    ],
    # plus_size
    [
        "a photo of a plus size person",
        "a large bodied person",
        "a full-figured person",
    ],
]
_BUILD_KEYS = ["slim", "petite", "curvy", "athletic", "average", "plus_size"]

_clip_model: Any = None
_clip_processor: Any = None
_build_text_embeds: Any = None


def _get_clip():
    global _clip_model, _clip_processor  # noqa: PLW0603
    if _clip_model is None and _CLIP_AVAILABLE:
        logger.info("Loading CLIP model %s (already cached from race MCP) …", _CLIP_MODEL_ID)
        _clip_model     = CLIPModel.from_pretrained(_CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        _clip_model.eval()
        logger.info("CLIP model ready")
    return _clip_model, _clip_processor


def _build_embeds(templates: list[list[str]]) -> "torch.Tensor":
    """
    Encode all prompt variants for each class, average + L2-normalise.
    Returns tensor of shape [n_classes, embed_dim].
    """
    model, processor = _get_clip()
    class_embeds = []
    for prompts in templates:
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            raw    = model.text_model(**inputs)
            embeds = model.text_projection(raw.pooler_output)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        embeds = embeds.mean(dim=0)
        embeds = embeds / embeds.norm()
        class_embeds.append(embeds)
    return torch.stack(class_embeds)


def _ensure_embeds() -> None:
    global _build_text_embeds  # noqa: PLW0603
    if _build_text_embeds is None:
        logger.info("Building body_build text embeddings …")
        _build_text_embeds = _build_embeds(_BUILD_TEMPLATES)
        logger.info("Body build embeddings ready")


def _clip_classify_image(image: "Image.Image") -> tuple[str, float]:
    """Classify a full image against body-build embeddings."""
    model, processor = _get_clip()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        raw       = model.vision_model(**inputs)
        img_embed = model.visual_projection(raw.pooler_output)
    img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
    sims  = (img_embed @ _build_text_embeds.T).squeeze(0)
    probs = (sims * model.logit_scale.exp()).softmax(dim=0)
    top_idx = int(probs.argmax())
    return _BUILD_KEYS[top_idx], round(float(probs[top_idx].detach()), 4)


def _warm_up() -> None:
    _get_clip()
    _ensure_embeds()


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------

class BodyBuildServicer(mcp_pb2_grpc.MCPServiceServicer):

    async def ExtractFeatures(
        self,
        request: mcp_pb2.FeatureRequest,
        context: grpc.aio.ServicerContext,
    ) -> mcp_pb2.FeatureResponse:
        logger.info("Received request source_id=%s", request.source_id)

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

async def _infer_body_build(image_bytes: bytes) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _infer_body_build_sync, image_bytes)


def _infer_body_build_sync(image_bytes: bytes) -> dict:
    if not _CLIP_AVAILABLE:
        return _mock_result()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    _ensure_embeds()

    try:
        build, confidence = _clip_classify_image(image)
    except Exception:
        logger.exception("CLIP classification failed")
        return _mock_result()

    logger.info("Body build: %s (conf=%.4f)", build, confidence)

    return {
        "data": {
            "body_build": build,
            "confidence": confidence,
            "using_mock": False,
        },
        "confidence": confidence,
    }


def _mock_result() -> dict:
    return {
        "data": {
            "body_build": "athletic",
            "confidence": 0.50,
            "using_mock": True,
        },
        "confidence": 0.50,
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

async def serve() -> None:
    if _CLIP_AVAILABLE:
        loop = asyncio.get_running_loop()
        logger.info("Loading CLIP model (already cached from race MCP) …")
        try:
            await loop.run_in_executor(None, _warm_up)
            logger.info("CLIP model ready")
        except Exception:
            logger.exception("CLIP model load failed.")

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
