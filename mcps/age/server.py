"""
mcps/age/server.py

Async gRPC MCP server — Age Range feature extraction.
Port: 50057

Responsibility: Detect age range for every face in a thumbnail.
Reports the main actress age (largest-face female) first in the output.

Stack (zero new installs — everything already in the venv):
  Face detection : OpenCV Haar cascade (haarcascade_frontalface_default.xml)
  Age classifier : openai/clip-vit-large-patch14 (already cached from race MCP)

Age ranges : 18-25, 26-35, 36-45, 46+

Output ordering: main actress is always people[0] and main_actress_age.

Usage (standalone):
    python -m mcps.age.server
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
    import cv2  # type: ignore
    import numpy as np
    from PIL import Image
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import torch  # type: ignore
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

_AVAILABLE = _CV2_AVAILABLE and _CLIP_AVAILABLE

PORT = 50057

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [age] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if not _CV2_AVAILABLE:
    logger.warning("opencv-python or Pillow not installed.")
if not _CLIP_AVAILABLE:
    logger.warning("torch/transformers not installed.")

# ---------------------------------------------------------------------------
# CLIP zero-shot classifier
# ---------------------------------------------------------------------------

_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# Template ensembling: multiple prompts per age range averaged in embedding space.
_AGE_TEMPLATES: list[list[str]] = [
    # 18-25
    [
        "a photo of a young adult woman in her early 20s",
        "a college-aged woman's face",
        "a woman who looks 18 to 25 years old",
    ],
    # 26-35
    [
        "a photo of a woman in her late 20s or early 30s",
        "a young professional woman's face",
        "a woman who looks 26 to 35 years old",
    ],
    # 36-45
    [
        "a photo of a middle-aged woman in her 40s",
        "a mature woman's face with some age lines",
        "a woman who looks 36 to 45 years old",
    ],
    # 46+
    [
        "a photo of an older woman over 45",
        "a senior woman's face",
        "a woman who looks 46 years old or older",
    ],
]
_AGE_KEYS = ["18-25", "26-35", "36-45", "46+"]

# Gender templates — used to identify main actress (same approach as race MCP)
_GENDER_TEMPLATES: list[list[str]] = [
    ["a photo of a woman", "a female person's face", "a woman's face"],
    ["a photo of a man",   "a male person's face",  "a man's face"],
]
_GENDER_KEYS = ["Woman", "Man"]

_clip_model: Any = None
_clip_processor: Any = None
_age_text_embeds: Any = None
_gender_text_embeds: Any = None


def _get_clip():
    global _clip_model, _clip_processor  # noqa: PLW0603
    if _clip_model is None and _CLIP_AVAILABLE:
        logger.info("Loading CLIP model %s (already cached from race MCP) …", _CLIP_MODEL_ID)
        _clip_model     = CLIPModel.from_pretrained(_CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        _clip_model.eval()
        logger.info("CLIP model ready")
    return _clip_model, _clip_processor


def _build_text_embeds(templates: list[list[str]]) -> "torch.Tensor":
    """
    For each class, encode all prompt variants and average + L2-normalise.
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
    global _age_text_embeds, _gender_text_embeds  # noqa: PLW0603
    if _age_text_embeds is None:
        logger.info("Building ensembled text embeddings …")
        _age_text_embeds    = _build_text_embeds(_AGE_TEMPLATES)
        _gender_text_embeds = _build_text_embeds(_GENDER_TEMPLATES)
        logger.info("Text embeddings ready")


def _clip_classify(face_crop: "Image.Image", text_embeds: "torch.Tensor", keys: list[str]) -> tuple[str, float]:
    """Classify a face crop against pre-built embeddings. Returns (top_key, confidence)."""
    model, processor = _get_clip()
    inputs = processor(images=face_crop, return_tensors="pt")
    with torch.no_grad():
        raw       = model.vision_model(**inputs)
        img_embed = model.visual_projection(raw.pooler_output)
    img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
    sims    = (img_embed @ text_embeds.T).squeeze(0)
    probs   = (sims * model.logit_scale.exp()).softmax(dim=0)
    top_idx = int(probs.argmax())
    return keys[top_idx], round(float(probs[top_idx].detach()), 4)


def _warm_up() -> None:
    _get_clip()
    _ensure_embeds()


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------

class AgeServicer(mcp_pb2_grpc.MCPServiceServicer):

    async def ExtractFeatures(
        self,
        request: mcp_pb2.FeatureRequest,
        context: grpc.aio.ServicerContext,
    ) -> mcp_pb2.FeatureResponse:
        logger.info("Received request source_id=%s", request.source_id)

        if request.payload_type != "image":
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"age MCP expects payload_type='image', got '{request.payload_type}'",
            )

        if not request.payload:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty payload")

        try:
            result = await _analyze(request.payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analysis failed for source_id=%s", request.source_id)
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

async def _analyze(image_bytes: bytes) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _analyze_sync, image_bytes)


def _analyze_sync(image_bytes: bytes) -> dict:
    """
    1. Decode image → PIL RGB + OpenCV greyscale
    2. Haar cascade: detect frontal faces → bounding boxes
    3. For each face crop: CLIP large zero-shot → gender + age range
    4. Find main actress (largest female face), place her first
    5. Return structured result
    """
    if not _AVAILABLE:
        return _mock_result()

    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np  = np.array(image_pil)
    gray      = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade      = cv2.CascadeClassifier(cascade_path)
    raw_faces    = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(40, 40),
    )

    # Drop detections smaller than 2% of the image area — these are almost
    # always background texture false positives, not real faces.
    img_w, img_h = image_pil.size
    min_area = 0.02 * img_w * img_h
    if len(raw_faces) > 0:
        raw_faces = [(x, y, w, h) for (x, y, w, h) in raw_faces if w * h >= min_area]

    if len(raw_faces) == 0:
        logger.info("No faces detected")
        return {
            "data": {
                "people": [],
                "count": 0,
                "main_actress_age": None,
                "using_mock": False,
            },
            "confidence": 0.0,
        }

    logger.info("Detected %d face(s)", len(raw_faces))

    people = []

    _ensure_embeds()

    for idx, (x, y, w, h) in enumerate(raw_faces, start=1):
        pad = int(max(w, h) * 0.30)
        x1 = max(0, int(x) - pad)
        y1 = max(0, int(y) - pad)
        x2 = min(img_w, int(x + w) + pad)
        y2 = min(img_h, int(y + h) + pad)

        face_crop = image_pil.crop((x1, y1, x2, y2))

        try:
            gender_label, gender_conf = _clip_classify(face_crop, _gender_text_embeds, _GENDER_KEYS)
            age_label,    age_conf    = _clip_classify(face_crop, _age_text_embeds,    _AGE_KEYS)
        except Exception:
            logger.exception("CLIP classification failed for face %d", idx)
            gender_label, gender_conf = "unknown", 0.0
            age_label,    age_conf    = "unknown", 0.0

        role = "actress" if gender_label == "Woman" else "actor"

        people.append({
            "person_id":        idx,
            "role":             role,
            "is_main_actress":  False,
            "age_range":        age_label,
            "age_confidence":   age_conf,
            "gender":           gender_label,
            "gender_confidence": gender_conf,
            "_area":            (x2 - x1) * (y2 - y1),
        })

    if not people:
        return {
            "data": {
                "people": [],
                "count": 0,
                "main_actress_age": None,
                "using_mock": False,
            },
            "confidence": 0.0,
        }

    # Main actress: largest face among females
    actresses = [p for p in people if p["role"] == "actress"]
    if actresses:
        main = max(actresses, key=lambda p: p["_area"])
        main["is_main_actress"] = True
        main_actress_age = main["age_range"]
    else:
        main_actress_age = None

    # Sort: main actress first, then other actresses, then actors
    def _sort_key(p: dict) -> int:
        if p.get("is_main_actress"):
            return 0
        if p["role"] == "actress":
            return 1
        return 2

    people.sort(key=_sort_key)

    for p in people:
        p.pop("_area", None)

    avg_conf = sum(p["age_confidence"] for p in people) / len(people)

    return {
        "data": {
            "people":           people,
            "count":            len(people),
            "main_actress_age": main_actress_age,
            "using_mock":       False,
        },
        "confidence": round(avg_conf, 4),
    }


def _mock_result() -> dict:
    return {
        "data": {
            "people": [
                {
                    "person_id":         1,
                    "role":              "actress",
                    "is_main_actress":   True,
                    "age_range":         "26-35",
                    "age_confidence":    0.70,
                    "gender":            "Woman",
                    "gender_confidence": 0.90,
                }
            ],
            "count":            1,
            "main_actress_age": "26-35",
            "using_mock":       True,
        },
        "confidence": 0.50,
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

async def serve() -> None:
    if _AVAILABLE:
        loop = asyncio.get_running_loop()
        logger.info("Loading CLIP model (already cached from race MCP) …")
        try:
            await loop.run_in_executor(None, _warm_up)
            logger.info("CLIP model ready")
        except Exception:
            logger.exception(
                "CLIP model load failed — check your internet connection and "
                "that torch/transformers are installed."
            )

    server = grpc.aio.server()
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(AgeServicer(), server)
    listen_addr = f"[::]:{PORT}"
    server.add_insecure_port(listen_addr)
    logger.info("age MCP server starting on %s", listen_addr)
    await server.start()
    logger.info("age MCP server ready")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
