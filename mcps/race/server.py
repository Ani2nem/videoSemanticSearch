"""
mcps/race/server.py

Async gRPC MCP server — Race/Ethnicity + Gender feature extraction.
Port: 50056

Responsibility: Detect race/ethnicity and gender for every face in a thumbnail.
Identifies each person as "actress" (female) or "actor" (male) and flags the
main actress — the largest-face female in the frame.

Stack (zero new installs — everything already in the venv):
  Face detection   : OpenCV Haar cascade (haarcascade_frontalface_default.xml)
                     Built into opencv-python, no download required.
  Race classifier  : dima806/fairface_race_image_detection  (HuggingFace ViT)
  Gender classifier: rizvandwiki/gender-classification  (HuggingFace ViT)
  Both HuggingFace models auto-download and cache on first run (~350 MB total).

Known limitation: Haar cascade is front-face biased — pure side profiles may
be missed. Revisit with a better detector when accuracy tuning begins.

Race labels : White, Black, Indian, East Asian, Southeast Asian,
              Middle Eastern, Latino_Hispanic
Gender      : Male → "actor" | Female → "actress"

Output ordering: main actress is always people[0] and actress_races[0].

Usage (standalone):
    python -m mcps.race.server
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

PORT = 50056

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [race] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if not _CV2_AVAILABLE:
    logger.warning("opencv-python or Pillow not installed.")
if not _CLIP_AVAILABLE:
    logger.warning("torch/transformers not installed.")

# ---------------------------------------------------------------------------
# CLIP zero-shot classifier
# ---------------------------------------------------------------------------

# Use the large model — significantly better accuracy than clip-vit-base-patch32
_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# Template ensembling: multiple prompts per class are averaged in embedding space
# before scoring. This is the technique from the original CLIP paper and
# significantly boosts accuracy over single-prompt zero-shot.
_RACE_TEMPLATES: list[list[str]] = [
    # White
    [
        "a photo of a white caucasian person",
        "a white european person's face",
        "a person with light skin and caucasian features",
    ],
    # Black
    [
        "a photo of a black person",
        "an african american person's face",
        "a person with dark skin and african features",
    ],
    # East Asian
    [
        "a photo of an east asian person",
        "a chinese, japanese, or korean person's face",
        "a person with east asian facial features",
    ],
    # South Asian
    [
        "a photo of a south asian person",
        "an indian or pakistani person's face",
        "a person with south asian facial features",
    ],
    # Middle Eastern
    [
        "a photo of a middle eastern person",
        "an arab person's face",
        "a person with middle eastern facial features",
    ],
    # Latino Hispanic
    [
        "a photo of a hispanic or latino person",
        "a latin american person's face",
        "a person with hispanic facial features",
    ],
    # Southeast Asian
    [
        "a photo of a southeast asian person",
        "a thai, vietnamese, or filipino person's face",
        "a person with southeast asian facial features",
    ],
]
_RACE_KEYS = ["White", "Black", "East Asian", "South Asian", "Middle Eastern", "Latino Hispanic", "Southeast Asian"]

_GENDER_TEMPLATES: list[list[str]] = [
    # Woman
    [
        "a photo of a woman",
        "a female person's face",
        "a woman's face",
    ],
    # Man
    [
        "a photo of a man",
        "a male person's face",
        "a man's face",
    ],
]
_GENDER_KEYS = ["Woman", "Man"]

_clip_model: Any = None
_clip_processor: Any = None
# Pre-computed text embeddings, shape [n_classes, embed_dim]; built once on load.
_race_text_embeds: Any = None
_gender_text_embeds: Any = None


def _get_clip():
    global _clip_model, _clip_processor  # noqa: PLW0603
    if _clip_model is None and _CLIP_AVAILABLE:
        logger.info("Loading CLIP model %s (first run may download ~1.7 GB) …", _CLIP_MODEL_ID)
        _clip_model     = CLIPModel.from_pretrained(_CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        _clip_model.eval()
        logger.info("CLIP model ready")
    return _clip_model, _clip_processor


def _build_text_embeds(templates: list[list[str]]) -> "torch.Tensor":
    """
    For each class, encode all its prompt variants and average + L2-normalise
    the resulting embeddings. This is the ensembling technique from the CLIP paper.
    Returns a tensor of shape [n_classes, embed_dim].
    """
    model, processor = _get_clip()
    class_embeds = []
    for prompts in templates:
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            raw = model.text_model(**inputs)
            # Use the internal text_model + projection to stay version-agnostic.
            # get_text_features() changed return type across transformers versions.
            embeds = model.text_projection(raw.pooler_output)  # [n_prompts, embed_dim]
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)    # L2 normalise
        embeds = embeds.mean(dim=0)                            # average
        embeds = embeds / embeds.norm()                        # re-normalise
        class_embeds.append(embeds)
    return torch.stack(class_embeds)   # [n_classes, embed_dim]


def _ensure_embeds() -> None:
    global _race_text_embeds, _gender_text_embeds  # noqa: PLW0603
    if _race_text_embeds is None:
        logger.info("Building ensembled text embeddings …")
        _race_text_embeds   = _build_text_embeds(_RACE_TEMPLATES)
        _gender_text_embeds = _build_text_embeds(_GENDER_TEMPLATES)
        logger.info("Text embeddings ready")


def _clip_classify(face_crop: Image.Image, text_embeds: "torch.Tensor", keys: list[str]) -> tuple[str, float]:
    """
    Classify a face crop against pre-built ensembled text embeddings.
    Returns (top_key, confidence).
    """
    model, processor = _get_clip()
    inputs = processor(images=face_crop, return_tensors="pt")
    with torch.no_grad():
        raw = model.vision_model(**inputs)
        img_embed = model.visual_projection(raw.pooler_output)
    img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
    # Cosine similarities → softmax probabilities
    sims    = (img_embed @ text_embeds.T).squeeze(0)   # [n_classes]
    probs   = (sims * model.logit_scale.exp()).softmax(dim=0)
    top_idx = int(probs.argmax())
    return keys[top_idx], round(float(probs[top_idx].detach()), 4)


def _warm_up() -> None:
    """Pre-load CLIP and build ensembled text embeddings before the first request."""
    _get_clip()
    _ensure_embeds()


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------

class RaceServicer(mcp_pb2_grpc.MCPServiceServicer):

    async def ExtractFeatures(
        self,
        request: mcp_pb2.FeatureRequest,
        context: grpc.aio.ServicerContext,
    ) -> mcp_pb2.FeatureResponse:
        logger.info("Received request source_id=%s", request.source_id)

        if request.payload_type != "image":
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"race MCP expects payload_type='image', got '{request.payload_type}'",
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
    3. For each face crop: CLIP large zero-shot → race + gender
    4. Assign role (actress/actor), find main actress (largest female face)
    5. Return structured result with main actress first
    """
    if not _AVAILABLE:
        return _mock_result()

    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np  = np.array(image_pil)
    gray      = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade      = cv2.CascadeClassifier(cascade_path)
    # minNeighbors=3 (was 4) catches more faces at the cost of a few more false
    # positives — acceptable for thumbnails where missed faces are worse.
    raw_faces    = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
    )

    if len(raw_faces) == 0:
        logger.info("No faces detected")
        return {
            "data": {
                "people": [],
                "count": 0,
                "main_actress_race": None,
                "actress_races": [],
                "actor_races": [],
                "using_mock": False,
            },
            "confidence": 0.0,
        }

    logger.info("Detected %d face(s)", len(raw_faces))

    img_w, img_h = image_pil.size
    people = []

    _ensure_embeds()

    for idx, (x, y, w, h) in enumerate(raw_faces, start=1):
        # Pad the bounding box by 30% so CLIP gets skin tone + surrounding
        # context, not just the tight face oval.
        pad = int(max(w, h) * 0.30)
        x1 = max(0, int(x) - pad)
        y1 = max(0, int(y) - pad)
        x2 = min(img_w, int(x + w) + pad)
        y2 = min(img_h, int(y + h) + pad)

        face_crop = image_pil.crop((x1, y1, x2, y2))

        try:
            race_label,   race_conf   = _clip_classify(face_crop, _race_text_embeds,   _RACE_KEYS)
            gender_label, gender_conf = _clip_classify(face_crop, _gender_text_embeds, _GENDER_KEYS)
        except Exception:
            logger.exception("CLIP classification failed for face %d", idx)
            race_label, race_conf     = "unknown", 0.0
            gender_label, gender_conf = "unknown", 0.0

        role = "actress" if gender_label == "Woman" else "actor"

        people.append({
            "person_id":         idx,
            "role":              role,
            "is_main_actress":   False,
            "race":              race_label,
            "race_confidence":   race_conf,
            "gender":            gender_label,
            "gender_confidence": gender_conf,
            "face_region":       {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            "_area":             (x2 - x1) * (y2 - y1),
        })

    if not people:
        return {
            "data": {
                "people": [],
                "count": 0,
                "main_actress_race": None,
                "actress_races": [],
                "actor_races": [],
                "using_mock": False,
            },
            "confidence": 0.0,
        }

    # Main actress: largest face among females
    actresses = [p for p in people if p["role"] == "actress"]
    if actresses:
        main = max(actresses, key=lambda p: p["_area"])
        main["is_main_actress"] = True
        main_actress_race = main["race"]
    else:
        main_actress_race = None

    # Sort people so main actress is first (index 0), then other actresses,
    # then actors — makes the main actress immediately accessible to callers.
    def _sort_key(p: dict) -> int:
        if p.get("is_main_actress"):
            return 0
        if p["role"] == "actress":
            return 1
        return 2

    people.sort(key=_sort_key)

    # actress_races: main actress first, then remaining actresses in order
    actress_races = [p["race"] for p in people if p["role"] == "actress"]
    actor_races   = [p["race"] for p in people if p["role"] == "actor"]

    for p in people:
        p.pop("_area", None)

    avg_conf = sum(p["race_confidence"] for p in people) / len(people)

    return {
        "data": {
            "people":           people,
            "count":            len(people),
            "main_actress_race": main_actress_race,
            "actress_races":    actress_races,
            "actor_races":      actor_races,
            "using_mock":       False,
        },
        "confidence": round(avg_conf, 4),
    }


def _mock_result() -> dict:
    return {
        "data": {
            "people": [
                {
                    "person_id":        1,
                    "role":             "actress",
                    "is_main_actress":  True,
                    "race":             "White",
                    "race_confidence":  0.85,
                    "gender":           "Woman",
                    "gender_confidence": 0.92,
                    "face_region":      {"x": 0, "y": 0, "w": 0, "h": 0},
                }
            ],
            "count":            1,
            "main_actress_race": "White",
            "actress_races":    ["White"],
            "actor_races":      [],
            "using_mock":       True,
        },
        "confidence": 0.50,
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

async def serve() -> None:
    # Warm up DeepFace on startup so model downloads happen before the first
    # real request, and any download failures are immediately visible in logs.
    if _AVAILABLE:
        loop = asyncio.get_running_loop()
        logger.info("Loading CLIP model (first run downloads ~1.7 GB) …")
        try:
            await loop.run_in_executor(None, _warm_up)
            logger.info("CLIP model ready")
        except Exception:
            logger.exception(
                "CLIP model load failed — check your internet connection and "
                "that torch/transformers are installed."
            )

    server = grpc.aio.server()
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(RaceServicer(), server)
    listen_addr = f"[::]:{PORT}"
    server.add_insecure_port(listen_addr)
    logger.info("race MCP server starting on %s", listen_addr)
    await server.start()
    logger.info("race MCP server ready")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
