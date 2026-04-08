"""
mcps/hair_color/server.py

Async gRPC MCP server — Hair Color & Texture analysis.
Port: 50051

Pipeline:
    1. MediaPipe Hair Segmenter → 2D boolean hair mask.
       Post-processing: dilation (9×9, 3 iters) + closing (11×11).
    2. MediaPipe Face Detector → per-person bounding boxes.
    3. Per person:
       a. Crop from hair mask bounds (10 px pad).  If local mask < 60 px,
          face-crop fallback: strictly above the face, with aggressive
          skin/background exclusion.
       b. COLOR — hybrid HSV primary + Lab K-means fallback (< 0.35 conf).
          Rule priority (later overwrites earlier):
            Brown (catch-all) < Dark Brown < Light Brown < Red/Auburn
            < Fantasy < Grey/White < Blonde < Black
       c. TEXTURE — EfficientNet-B0 on mask-filtered crop (grey background).
          Classes: Straight / Wavy / Curly / Bald.
          Blended labels (e.g. "Straight-Wavy") returned when top-2
          scores are within _BLEND_GAP of each other.
    4. No-face fallback when hair mask >= 20 px.

Texture model: EfficientNet-B0 (hair_texture_efficientnet_b0.pth)
    4 classes: Straight, Wavy, Curly, Bald.
    Fallback path always uses masked (grey-background) crop.

Usage:
    python -m mcps.hair_color.server
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import grpc
import grpc.aio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.models import efficientnet_b0

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import mcp_pb2
    import mcp_pb2_grpc
except ImportError as exc:
    raise ImportError(
        "Generated proto stubs not found. Run:\n"
        "  python -m grpc_tools.protoc -I./proto "
        "--python_out=. --grpc_python_out=. ./proto/mcp.proto"
    ) from exc

PORT = 50051

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [hair_analysis] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named hair colors in CIELAB — reference centroids for K-means fallback
# ---------------------------------------------------------------------------

NAMED_COLORS_LAB: list[tuple[str, float, float, float]] = [
    ("Black",             10.0,   0.5,    1.0),
    ("Dark Brown",        25.0,   6.0,   12.0),
    ("Brown",             38.0,   8.0,   18.0),
    ("Light Brown",       52.0,  10.0,   26.0),
    ("Dirty Blonde",      58.0,   4.0,   22.0),
    ("Blonde",            72.0,   4.0,   38.0),
    ("Strawberry Blonde", 62.0,  18.0,   28.0),
    ("Red",               38.0,  42.0,   32.0),
    ("Auburn",            28.0,  28.0,   22.0),
    ("Grey",              50.0,   0.0,    0.0),
    ("White",             92.0,   0.0,    1.0),
    ("Blue",              45.0,  -5.0,  -40.0),
    ("Pink",              55.0,  40.0,   -5.0),
    ("Purple",            30.0,  30.0,  -35.0),
]
_NAMED_LAB_ARRAY = np.array(
    [[L, a, b] for _, L, a, b in NAMED_COLORS_LAB], dtype=np.float32,
)
_NAMED_LABELS = [name for name, *_ in NAMED_COLORS_LAB]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

BALD_GATE_THRESHOLD = 80
OCCLUSION_CONFIDENCE_THRESHOLD = 0.30
MIN_HAIR_PIXELS = 20
FACE_CROP_FALLBACK_THRESH = 60
MASK_CROP_PAD = 10

MASK_DILATE_KERNEL = 9
MASK_DILATE_ITERS = 3
MASK_CLOSE_KERNEL = 11

FACE_SEARCH_PAD_X = 60
FACE_SEARCH_PAD_ABOVE = 120
FACE_SEARCH_PAD_BELOW_FRAC = 0.15

FACE_CROP_ABOVE_RATIO = 1.8
FACE_CROP_SIDE_RATIO = 0.4

HSV_MIN_CONF = 0.35

KMEANS_K = 3
KMEANS_MIN_PIXELS = 30
KMEANS_MAX_SAMPLE = 20_000


# ---------------------------------------------------------------------------
# Model asset paths
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).parent

_HAIR_SEG_PATH = _MODEL_DIR / "hair_segmenter.tflite"
_HAIR_SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"
)

_FACE_DET_PATH = _MODEL_DIR / "blaze_face_short_range.tflite"
_FACE_DET_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite"
)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

_TEXTURE_MODEL = None
_TEXTURE_DEVICE = None
_TEXTURE_TRANSFORM = None


def _load_texture_model(
    model_path: str | None = None,
) -> None:
    global _TEXTURE_MODEL, _TEXTURE_DEVICE, _TEXTURE_TRANSFORM

    if model_path is None:
        model_path = str(_MODEL_DIR / "hair_texture_efficientnet_b0.pth")

    _TEXTURE_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model = efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 5)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    _TEXTURE_DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(_TEXTURE_DEVICE)
    model.eval()
    _TEXTURE_MODEL = model
    logger.info("Texture model loaded on %s", _TEXTURE_DEVICE)


@lru_cache(maxsize=1)
def _load_hair_segmenter():
    if not _HAIR_SEG_PATH.exists():
        logger.info("Downloading MediaPipe hair segmenter (~4 MB) ...")
        urllib.request.urlretrieve(_HAIR_SEG_URL, _HAIR_SEG_PATH)
    base = mp_python.BaseOptions(model_asset_path=str(_HAIR_SEG_PATH))
    opts = mp_vision.ImageSegmenterOptions(
        base_options=base, output_category_mask=True,
    )
    logger.info("Hair segmenter loaded")
    return mp_vision.ImageSegmenter.create_from_options(opts)


@lru_cache(maxsize=1)
def _load_face_detector():
    if not _FACE_DET_PATH.exists():
        logger.info("Downloading MediaPipe BlazeFace detector ...")
        urllib.request.urlretrieve(_FACE_DET_URL, _FACE_DET_PATH)
    base = mp_python.BaseOptions(model_asset_path=str(_FACE_DET_PATH))
    opts = mp_vision.FaceDetectorOptions(
        base_options=base, min_detection_confidence=0.3,
    )
    logger.info("BlazeFace detector loaded")
    return mp_vision.FaceDetector.create_from_options(opts)


# ---------------------------------------------------------------------------
# HSV classifier
#
# OpenCV HSV: H 0-179, S 0-255, V 0-255.
#
# RULE ORDER (later overwrites earlier):
#   1. Brown          (catch-all default)
#   2. Dark Brown     (dark + clearly chromatic, S >= 40)
#   3. Light Brown    (narrow warm zone, V 120-165)
#   4. Red / Auburn
#   5. Fantasy        (Blue, Purple, Pink)
#   6. Grey / White   (S < 30 — runs AFTER Dark Brown to reclaim grey)
#   7. Blonde         (H 12-45, V > 125, S 30-180)
#   8. Black          (V < 60 — absolute highest priority)
# ---------------------------------------------------------------------------

def _hsv_classify(hair_pixels_rgb: np.ndarray) -> tuple[str, list[str], float]:
    n = len(hair_pixels_rgb)
    if n < KMEANS_MIN_PIXELS:
        return "Unknown", ["Unknown"], 0.0

    bgr = hair_pixels_rgb[:, ::-1].reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    h = hsv[:, 0].astype(np.int16)
    s = hsv[:, 1].astype(np.int16)
    v = hsv[:, 2].astype(np.int16)

    labels = np.empty(n, dtype=object)

    # 1. Brown — catch-all default
    labels[:] = "Brown"

    # 2. Dark Brown — dark, clearly chromatic (S >= 40 keeps it away from grey)
    labels[(v >= 60) & (v < 95) & (s >= 40) & (s < 150)] = "Dark Brown"

    # 3. Light Brown — narrow: warm hue, higher V, moderate S
    labels[
        (h >= 12) & (h < 25) & (s >= 35) & (s <= 110) & (v >= 120) & (v <= 165)
    ] = "Light Brown"

    # 4. Red / Auburn
    red_hue = (h < 10) | (h > 165)
    labels[red_hue & (s > 70) & (v >= 60) & (v < 115)] = "Auburn"
    labels[red_hue & (s > 70) & (v >= 115)] = "Red"

    # 5. Fantasy
    labels[(h >= 90) & (h < 135) & (s > 50) & (v > 40)] = "Blue"
    labels[(h >= 135) & (h < 160) & (s > 50) & (v > 40)] = "Purple"
    labels[(h >= 160) & (h <= 179) & (s > 50) & (v > 80)] = "Pink"
    labels[(h >= 0) & (h < 8) & (s > 60) & (v > 130)] = "Pink"

    # 6. Grey / White — desaturated.  AFTER Dark Brown so S < 30 reclaims grey.
    labels[(s < 30) & (v >= 60) & (v <= 200)] = "Grey"
    labels[(s < 30) & (v > 200)] = "White"

    # 7. Blonde / Dirty Blonde — wide warm hue, bright.
    #    v > 100 catches darker blondes that were falling into Brown.
    #    Dirty Blonde sits in the middle band (v 100-140, s 25-90).
    #    Pure Blonde overwrites Dirty Blonde at higher V (v > 140).
    labels[
        (h >= 12) & (h < 45) & (s >= 25) & (s <= 90) & (v >= 100) & (v <= 140)
    ] = "Dirty Blonde"
    labels[(h >= 12) & (h < 45) & (s >= 30) & (s <= 180) & (v > 140)] = "Blonde"
    # Catch lighter warm highlights (low-S, high-V) in blonde hair
    labels[(h >= 12) & (h < 45) & (s >= 15) & (s < 30) & (v > 150)] = "Dirty Blonde"

    # 8. Black — V < 60 is always Black, no exceptions.
    labels[v < 60] = "Black"

    counts = Counter(labels)

    # Treat Dirty Blonde as part of the Blonde family when computing
    # dominance — the Dirty Blonde rule steals votes from Blonde in
    # mid-brightness zones, artificially splitting confidence and causing
    # Lab fallback to kick in and return Brown instead.
    blonde_family_freq = counts.get("Blonde", 0) + counts.get("Dirty Blonde", 0)
    if blonde_family_freq > 0:
        # Temporarily collapse for dominance check only
        counts_collapsed = Counter(labels)
        for lbl in list(counts_collapsed.keys()):
            if lbl == "Dirty Blonde":
                counts_collapsed["Blonde"] += counts_collapsed.pop("Dirty Blonde")
        dominant, freq = counts_collapsed.most_common(1)[0]
    else:
        dominant, freq = counts.most_common(1)[0]

    palette = [c for c, _ in counts.most_common(3)]
    confidence = round(freq / n, 4)

    logger.info("HSV classify: %s (conf=%.3f, palette=%s)", dominant, confidence, palette)
    return dominant, palette, confidence


# ---------------------------------------------------------------------------
# Lab K-means (fallback)
# ---------------------------------------------------------------------------

def _lab_classify(
    hair_pixels_rgb: np.ndarray,
) -> tuple[str, list[str], float, list[float]]:
    n = len(hair_pixels_rgb)
    if n < KMEANS_MIN_PIXELS:
        return "Unknown", ["Unknown"], 0.0, [0.0, 0.0, 0.0]

    sample = hair_pixels_rgb
    if n > KMEANS_MAX_SAMPLE:
        idx = np.random.default_rng(42).choice(n, KMEANS_MAX_SAMPLE, replace=False)
        sample = hair_pixels_rgb[idx]

    bgr = sample[:, ::-1].reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1] -= 128.0
    lab[:, 2] -= 128.0

    k = min(KMEANS_K, len(lab))
    km = KMeans(n_clusters=k, n_init=4, max_iter=100, random_state=42)
    km.fit(lab)

    _, counts = np.unique(km.labels_, return_counts=True)
    biggest = int(np.argmax(counts))
    center = km.cluster_centers_[biggest]

    logger.info(
        "Lab K-means centroid: L=%.1f  a=%.1f  b=%.1f  (%d/%d px)",
        center[0], center[1], center[2], int(counts[biggest]), len(lab),
    )

    sorted_idxs = np.argsort(-counts)
    palette: list[str] = []
    for ci in sorted_idxs[:3]:
        name = _nearest_named_color(km.cluster_centers_[ci])
        if name not in palette:
            palette.append(name)

    dominant = _nearest_named_color(center)
    if dominant not in palette:
        palette.insert(0, dominant)
    palette = palette[:3]

    conf = round(float(counts[biggest]) / len(lab), 4)
    logger.info("Lab result: %s (conf=%.3f, palette=%s)", dominant, conf, palette)
    return dominant, palette, conf, [round(float(center[i]), 1) for i in range(3)]


def _nearest_named_color(lab: np.ndarray) -> str:
    dists = np.linalg.norm(_NAMED_LAB_ARRAY - lab, axis=1)
    return _NAMED_LABELS[int(np.argmin(dists))]


# ---------------------------------------------------------------------------
# Hybrid color
# ---------------------------------------------------------------------------

def _hybrid_color(
    hair_pixels_rgb: np.ndarray,
) -> tuple[str, list[str], float, list[float], str]:
    hsv_name, hsv_pal, hsv_conf = _hsv_classify(hair_pixels_rgb)
    lab_name, lab_pal, lab_conf, dom_lab = _lab_classify(hair_pixels_rgb)

    if hsv_conf >= HSV_MIN_CONF:
        # If HSV is confident but says Brown/Dark Brown while Lab points to
        # a lighter colour (Blonde, Dirty Blonde, Light Brown), trust Lab —
        # HSV browns and dark blondes heavily overlap in hue/sat space.
        _LIGHT_OVERRIDES = {"Blonde", "Dirty Blonde", "Strawberry Blonde", "Light Brown"}
        _HSV_DARK_NAMES  = {"Brown", "Dark Brown"}
        if hsv_name in _HSV_DARK_NAMES and lab_name in _LIGHT_OVERRIDES:
            logger.info(
                "-> Lab override: HSV=%s (%.3f) overridden by Lab=%s (%.3f)",
                hsv_name, hsv_conf, lab_name, lab_conf,
            )
            merged_pal = [lab_name] + [c for c in hsv_pal if c != lab_name][:2]
            return lab_name, merged_pal, min(0.98, round(lab_conf * 1.2, 4)), dom_lab, "lab"
        logger.info("-> Using HSV: %s (%.3f)", hsv_name, hsv_conf)
        return hsv_name, hsv_pal, min(0.98, round(hsv_conf * 1.3, 4)), dom_lab, "hsv"

    logger.info(
        "-> HSV conf %.3f < %.2f — Lab fallback: %s (%.3f)",
        hsv_conf, HSV_MIN_CONF, lab_name, lab_conf,
    )
    return lab_name, lab_pal, min(0.98, round(lab_conf * 1.2, 4)), dom_lab, "lab"


# ---------------------------------------------------------------------------
# Texture classification (EfficientNet-B0)
#
# Classes: Straight, Wavy, Curly, Bald, Dreadlocks
#
# Adjustments (applied in order):
#   1. Short-hair straight boost — pixels < 15 000 → +0.10 Straight
#      (only when pixel count is trustworthy and Bald is not leading)
#   2. Long-hair wavy+curly boost — pixels > 25 000 → +0.18 Wavy, +0.06 Curly
#      (only when pixel count is trustworthy and Bald is not leading)
#   3. Boost guard: pixel counts > _BOOST_PIXEL_CAP are synthetic/inflated
#      (face_crop_fallback) and must never trigger boosts.
#   4. Blend — top-2 within _BLEND_GAP and both in {Straight, Wavy, Curly}
#      → hyphenated label in canonical order (e.g. "Straight-Wavy")
# ---------------------------------------------------------------------------

_SHORT_HAIR_PIXEL_THRESH = 15_000
_STRAIGHT_SHORT_BOOST    = 0.0   # disabled — retrained model handles this

_LONG_HAIR_PIXEL_THRESH  = 25_000
_WAVY_LONG_BOOST         = 0.0   # disabled — retrained model handles this
_CURLY_LONG_BOOST        = 0.0   # disabled — retrained model handles this

_BLEND_GAP               = 0.12
_BOOST_PIXEL_CAP         = 500_000

_TEXTURE_CLASS_NAMES = ["Bald", "Curly", "Dreadlocks", "Straight", "Wavy"]
_BLEND_ORDER         = ["Straight", "Wavy", "Curly"]

# ---------------------------------------------------------------------------
# Temperature scaling + class prior correction
#
# The EfficientNet-B0 model is overtrained on Straight due to dataset
# imbalance. We correct this in two steps:
#
#   1. Temperature T > 1 applied to raw logits before softmax — flattens
#      the distribution so the model is less peaky/overconfident.
#
#   2. Class prior multipliers applied to post-softmax probs — subtract
#      the estimated training frequency bias per class. Values < 1.0
#      suppress over-represented classes; > 1.0 boost under-represented.
#      These are multiplicative (not additive) so they only matter when
#      the model already has some signal for a class.
#
# Tune _CLASS_PRIOR_SCALE by looking at val-set confusion matrix:
#   - If Wavy is consistently missed → increase Wavy scale
#   - If Straight is always winning  → decrease Straight scale
# ---------------------------------------------------------------------------

_TEMPERATURE       = 1.0   # no temperature scaling needed — model trained with balanced classes

_CLASS_PRIOR_SCALE = {
    "Bald":       1.0,
    "Curly":      1.0,
    "Dreadlocks": 1.0,
    "Straight":   1.0,
    "Wavy":       1.0,
}

# Bald hard threshold — only trigger when hair pixels are extremely low
# (below this, even the model can't distinguish Bald from sparse hair).
# The retrained model handles Bald detection via its own scores for
# normal cases — we only override here for near-zero pixel counts.
# Set deliberately low to avoid false positives on close-up shots.
_BALD_PIXEL_HARD_THRESH = 3_000


def _texture_classify(
    masked_crop: Image.Image,
    hair_pixel_count: int,
    face_area: int = 0,  # kept for API compatibility, no longer used internally
) -> tuple[str, list[str], float]:
    global _TEXTURE_MODEL, _TEXTURE_DEVICE, _TEXTURE_TRANSFORM
    if _TEXTURE_MODEL is None:
        _load_texture_model()

    # Hard Bald gate — only for truly near-zero hair pixel counts.
    # The ratio gate was removed because face bbox area varies wildly
    # with shot distance (close-up = huge bbox = low ratio even with
    # lots of hair). The retrained model handles normal Bald cases.
    if hair_pixel_count < _BALD_PIXEL_HARD_THRESH:
        logger.info(
            "Bald hard gate: hair_px=%d < %d → Bald",
            hair_pixel_count, _BALD_PIXEL_HARD_THRESH,
        )
        return "Bald", ["Bald"], 0.85

    img_tensor = (
        _TEXTURE_TRANSFORM(masked_crop.convert("RGB"))
        .unsqueeze(0)
        .to(_TEXTURE_DEVICE)
    )

    with torch.no_grad():
        logits = _TEXTURE_MODEL(img_tensor).squeeze()   # raw logits, shape (5,)

        # Step 1 — temperature scaling on logits before softmax
        scaled_logits = logits / _TEMPERATURE
        probs = (
            torch.nn.functional.softmax(scaled_logits, dim=0)
            .cpu()
            .numpy()
        )

    raw_scores = {name: float(probs[i]) for i, name in enumerate(_TEXTURE_CLASS_NAMES)}

    logger.info(
        "EfficientNet raw scores (T=%.1f, pixels=%d): %s",
        _TEMPERATURE, hair_pixel_count,
        "  ".join(f"{k}={v:.4f}" for k, v in raw_scores.items()),
    )

    # Step 2 — apply class prior correction (multiplicative)
    scores = {
        name: raw_scores[name] * _CLASS_PRIOR_SCALE.get(name, 1.0)
        for name in _TEXTURE_CLASS_NAMES
    }
    # Re-normalise so scores sum to ~1 (keeps confidence values interpretable)
    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}

    logger.info(
        "EfficientNet debiased scores: %s",
        "  ".join(f"{k}={v:.4f}" for k, v in scores.items()),
    )

    # Determine whether pixel-count boosts are safe to apply:
    #   - count must be a real (non-synthetic) value
    #   - Bald must not already be the leading prediction (it's structural, not texture)
    sorted_raw = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    _boost_eligible = (
        hair_pixel_count < _BOOST_PIXEL_CAP
        and sorted_raw[0][0] != "Bald"
    )

    if _boost_eligible and hair_pixel_count < _SHORT_HAIR_PIXEL_THRESH and "Straight" in scores:
        scores["Straight"] += _STRAIGHT_SHORT_BOOST
        logger.info(
            "Short-hair boost (pixels=%d < %d): Straight +%.2f",
            hair_pixel_count, _SHORT_HAIR_PIXEL_THRESH, _STRAIGHT_SHORT_BOOST,
        )

    if _boost_eligible and hair_pixel_count > _LONG_HAIR_PIXEL_THRESH:
        if "Wavy" in scores:
            scores["Wavy"] += _WAVY_LONG_BOOST
        if "Curly" in scores:
            scores["Curly"] += _CURLY_LONG_BOOST
        logger.info(
            "Long-hair boost (pixels=%d > %d): Wavy +%.2f, Curly +%.2f",
            hair_pixel_count, _LONG_HAIR_PIXEL_THRESH,
            _WAVY_LONG_BOOST, _CURLY_LONG_BOOST,
        )

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_items[0]

    logger.info(
        "EfficientNet adjusted scores: %s",
        "  ".join(f"{k}={v:.4f}" for k, v in sorted_items),
    )

    # Blend only when:
    #   1. top score is above a meaningful confidence floor (not noise)
    #   2. gap between top-2 is within _BLEND_GAP
    #   3. both labels are texture classes (not Bald/Dreadlocks)
    _MIN_BLEND_CONFIDENCE = 0.32   # below this the model is just guessing

    if len(sorted_items) > 1:
        second_label, second_score = sorted_items[1]
        gap = top_score - second_score
        if (
            top_score >= _MIN_BLEND_CONFIDENCE
            and top_label in _BLEND_ORDER
            and second_label in _BLEND_ORDER
            and gap < _BLEND_GAP
        ):
            a, b = sorted(
                [top_label, second_label],
                key=lambda x: _BLEND_ORDER.index(x),
            )
            blended = f"{a}-{b}"
            logger.info(
                "Texture blend: %s (%.4f) + %s (%.4f) → %s  gap=%.4f",
                top_label, top_score, second_label, second_score, blended, gap,
            )
            return blended, [blended], round(top_score, 4)

    logger.info(
        "Texture FINAL → %s (%.4f) | palette=%s",
        top_label, top_score, [k for k, _ in sorted_items[:3]],
    )
    return top_label, [top_label], round(top_score, 4)


# ---------------------------------------------------------------------------
# Mask / crop helpers
# ---------------------------------------------------------------------------

def _mask_bbox(
    mask_2d: np.ndarray, pad: int, img_h: int, img_w: int,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_2d)
    if len(ys) == 0:
        return None
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(img_w, int(xs.max()) + 1 + pad),
        min(img_h, int(ys.max()) + 1 + pad),
    )


_MASK_BG_COLOR = np.array([180, 180, 180], dtype=np.uint8)  # light grey

def _apply_mask_to_crop(
    img_np: np.ndarray,
    hair_mask: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> tuple[Image.Image, int]:
    crop_rgb = img_np[y1:y2, x1:x2].copy()
    crop_mask = hair_mask[y1:y2, x1:x2]
    crop_rgb[~crop_mask] = _MASK_BG_COLOR
    return Image.fromarray(crop_rgb), int(crop_mask.sum())


def _face_crop_above(
    face_x: int, face_y: int, face_w: int, face_h: int,
    img_h: int, img_w: int,
) -> tuple[int, int, int, int]:
    """Crop STRICTLY above the face — y2 = face_y (no skin overlap)."""
    above = int(face_h * FACE_CROP_ABOVE_RATIO)
    side = int(face_w * FACE_CROP_SIDE_RATIO)
    x1 = max(0, face_x - side)
    y1 = max(0, face_y - above)
    x2 = min(img_w, face_x + face_w + side)
    y2 = face_y
    if y2 <= y1 + 5:
        y2 = max(y1 + 10, min(img_h, face_y + int(face_h * 0.10)))
    return x1, y1, x2, y2


def _exclude_skin_and_bg(pixels_rgb: np.ndarray) -> np.ndarray:
    """Remove skin-toned and bright-background pixels from a flat (N,3) array."""
    if len(pixels_rgb) == 0:
        return pixels_rgb

    bgr = pixels_rgb[:, ::-1].reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    h = hsv[:, 0].astype(np.int16)
    s = hsv[:, 1].astype(np.int16)
    v = hsv[:, 2].astype(np.int16)

    # Aggressive skin exclusion: warm hues with moderate S and medium+ V
    is_skin = (h <= 25) & (s >= 20) & (s <= 180) & (v >= 60) & (v <= 245)

    # Bright desaturated background (white walls, overexposed areas)
    is_bright_bg = (s < 25) & (v > 170)

    keep = ~is_skin & ~is_bright_bg
    kept = pixels_rgb[keep]

    logger.info(
        "Skin/BG filter: %d total -> %d kept (-%d skin, -%d bg)",
        len(pixels_rgb), len(kept),
        int(is_skin.sum()), int(is_bright_bg.sum()),
    )
    return kept


def _simple_hair_mask(
    crop_rgb: np.ndarray,
    color_hint: str,
    bg_color: np.ndarray | None = None,
) -> np.ndarray:
    """Create a boolean mask for hair using HSV heuristics based on color_hint.

    bg_color: if provided, pixels within 15 RGB units of this color are
              excluded first — prevents matching the painted grey background
              when color_hint is 'Grey' or 'White'.
    """
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Build a background-exclusion mask so we never classify our own
    # painted background pixels as hair.
    if bg_color is not None:
        diff = np.abs(crop_rgb.astype(np.int16) - bg_color.astype(np.int16))
        is_bg_paint = diff.max(axis=2) < 20          # within 20 RGB units of bg
    else:
        is_bg_paint = np.zeros(crop_rgb.shape[:2], dtype=bool)

    if color_hint in ("Black", "Dark Brown", "Brown"):
        # Dark hair: low value, moderate saturation
        mask = (v < 90) & (s > 20)
    elif color_hint in ("Blonde", "Light Brown"):
        # Light hair: high value, moderate saturation
        mask = (v > 120) & (s > 30)
    elif color_hint in ("Grey", "White"):
        # Grey/white hair: low saturation BUT must also have non-trivial
        # texture variation relative to background — require s to be
        # *above* a minimal floor so we don't just select everything.
        # Also tighten v range: real grey hair sits 40–200, not 0–255.
        mask = (s < 30) & (v >= 40) & (v <= 200)
    elif color_hint in ("Blue", "Purple", "Pink"):
        h = hsv[:, :, 0]
        if color_hint == "Blue":
            mask = (h >= 90) & (h < 135) & (s > 50) & (v > 40)
        elif color_hint == "Purple":
            mask = (h >= 135) & (h < 160) & (s > 50) & (v > 40)
        else:  # Pink
            mask = ((h >= 160) | (h < 8)) & (s > 50) & (v > 80)
    else:
        # Default: keep darkest 60% of pixels
        threshold = np.percentile(v, 60)
        mask = v < threshold

    # Exclude painted background pixels
    mask = mask & ~is_bg_paint

    # Morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    return mask


# ---------------------------------------------------------------------------
# gRPC servicer
# ---------------------------------------------------------------------------

class HairAnalysisServicer(mcp_pb2_grpc.MCPServiceServicer):

    def __init__(self) -> None:
        _load_texture_model()
        _load_hair_segmenter()
        _load_face_detector()

    async def ExtractFeatures(
        self,
        request: mcp_pb2.FeatureRequest,
        context: grpc.aio.ServicerContext,
    ) -> mcp_pb2.FeatureResponse:
        logger.info(
            "Received request source_id=%s payload_type=%s",
            request.source_id, request.payload_type,
        )
        if request.payload_type != "image":
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"hair_color MCP expects payload_type='image', "
                f"got '{request.payload_type}'",
            )
        if not request.payload:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty payload")

        try:
            result = await _extract(request.payload, request.source_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Extraction failed for source_id=%s", request.source_id)
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
# Core extraction pipeline
# ---------------------------------------------------------------------------

def _run_extraction(image_bytes: bytes, source_id: str) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

    # ---- Step 1: Hair segmentation + aggressive morphology ----
    segmenter = _load_hair_segmenter()
    seg_result = segmenter.segment(mp_image)
    raw_mask = seg_result.category_mask.numpy_view()
    hair_mask_raw = np.squeeze(raw_mask) == 1
    raw_count = int(hair_mask_raw.sum())

    dilate_kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MASK_DILATE_KERNEL, MASK_DILATE_KERNEL),
    )
    dilated = cv2.dilate(
        hair_mask_raw.astype(np.uint8), dilate_kern, iterations=MASK_DILATE_ITERS,
    )
    close_kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MASK_CLOSE_KERNEL, MASK_CLOSE_KERNEL),
    )
    hair_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kern).astype(bool)

    total_hair = int(hair_mask.sum())
    mask_ok = total_hair >= FACE_CROP_FALLBACK_THRESH

    logger.info(
        "DEBUG - [%s] mask: raw=%d -> post=%d (+%d) | mask_success=%s",
        source_id, raw_count, total_hair, total_hair - raw_count, mask_ok,
    )

    # ---- Step 2: Face detection ----
    detector = _load_face_detector()
    det_result = detector.detect(mp_image)
    faces = det_result.detections
    logger.info("DEBUG - [%s] faces=%d", source_id, len(faces))

    # ---- Step 3: Per-person ----
    people: list[dict] = []

    for i, det in enumerate(faces):
        bb = det.bounding_box
        fx, fy = int(bb.origin_x), int(bb.origin_y)
        fw, fh = int(bb.width), int(bb.height)

        sx1 = max(0, fx - FACE_SEARCH_PAD_X)
        sy1 = max(0, fy - FACE_SEARCH_PAD_ABOVE)
        sx2 = min(img_w, fx + fw + FACE_SEARCH_PAD_X)
        sy2 = min(img_h, fy + int(fh * FACE_SEARCH_PAD_BELOW_FRAC))
        if sy2 <= sy1:
            sy2 = min(img_h, fy + fh)

        local_mask = hair_mask[sy1:sy2, sx1:sx2]
        local_count = int(local_mask.sum())
        bbox = _mask_bbox(local_mask, MASK_CROP_PAD, sy2 - sy1, sx2 - sx1)
        fallback = bbox is None or local_count < FACE_CROP_FALLBACK_THRESH

        if fallback:
            person = _face_crop_fallback(
                img_np, fx, fy, fw, fh, img_h, img_w, hair_mask,
                person_id=i + 1, local_count=local_count, source_id=source_id,
                total_hair=total_hair, face_area=fw * fh,
            )
        else:
            lx1, ly1, lx2, ly2 = bbox
            person = _analyze_region(
                img_np, hair_mask,
                sx1 + lx1, sy1 + ly1, sx1 + lx2, sy1 + ly2,
                person_id=i + 1,
                face_area=fw * fh,
            )

        logger.info(
            "DEBUG - [%s] person=%d | color=%s | texture=%s | "
            "hair_pixels=%d | method=%s | mask_success=%s | "
            "using_fallback=%s",
            source_id, i + 1,
            person["dominant_color"], person["dominant_texture"],
            person["hair_pixel_count"], person["color_method"],
            not fallback, fallback,
        )
        people.append(person)

    # ---- Step 4: No-face fallback ----
    if not people and total_hair >= MIN_HAIR_PIXELS:
        logger.warning("[%s] No faces — global mask fallback", source_id)
        gb = _mask_bbox(hair_mask, MASK_CROP_PAD, img_h, img_w)
        if gb:
            person = _analyze_region(
                img_np, hair_mask, gb[0], gb[1], gb[2], gb[3], person_id=1,
            )
            person["note"] = "no_face_detected"
            logger.info(
                "DEBUG - [%s] person=1 | color=%s | texture=%s | "
                "hair_pixels=%d | method=%s | using_fallback=no_face",
                source_id, person["dominant_color"],
                person["dominant_texture"], person["hair_pixel_count"],
                person["color_method"],
            )
            people.append(person)

    # ---- Step 5: Response ----
    if people:
        dom_color = Counter(p["dominant_color"] for p in people).most_common(1)[0][0]
        dom_tex = Counter(p["dominant_texture"] for p in people).most_common(1)[0][0]
        avg_c = round(sum(p["color_confidence"] for p in people) / len(people), 4)
    else:
        dom_color = dom_tex = "Unknown"
        avg_c = 0.0

    logger.info(
        "RESULT - [%s] %d person(s) | color=%s | texture=%s | conf=%.3f",
        source_id, len(people), dom_color, dom_tex, avg_c,
    )
    return {
        "data": {
            "people": people,
            "count": len(people),
            "dominant_color": dom_color,
            "dominant_texture": dom_tex,
        },
        "confidence": avg_c,
    }


# ---------------------------------------------------------------------------
# Face-crop fallback
#
# Always produces a masked (grey-background) crop so EfficientNet sees
# the same input distribution as during training.
#
# FIX: Dark-hair fast path — check brightness BEFORE skin/bg filtering.
# The skin exclusion filter (h<=25, s>=20, v>=60) correctly passes black
# pixels (v<60), but residual grey/neutral pixels then dominate the color
# read.  By checking the p25 brightness first we short-circuit to the dark
# pixel pool when the image is clearly dark-haired, skipping filtering
# entirely for those cases.
# ---------------------------------------------------------------------------

# Brightness threshold below which a pixel is considered "dark hair"
# Matches the HSV Black rule: v < 60 → ~mean RGB < 60
_DARK_PATH_BRIGHTNESS_CUT = 55.0
_DARK_PATH_PERCENTILE     = 25   # use bottom 25th percentile of brightness


def _face_crop_fallback(
    img_np: np.ndarray,
    fx: int, fy: int, fw: int, fh: int,
    img_h: int, img_w: int,
    hair_mask: np.ndarray,
    *,
    person_id: int,
    local_count: int,
    source_id: str,
    total_hair: int = 0,
    face_area: int = 0,
) -> dict:
    cx1, cy1, cx2, cy2 = _face_crop_above(fx, fy, fw, fh, img_h, img_w)

    logger.info(
        "[%s] Person %d: mask=%d px — face-crop (%d,%d)-(%d,%d)",
        source_id, person_id, local_count, cx1, cy1, cx2, cy2,
    )

    masked_crop, crop_pixel_count = _apply_mask_to_crop(
        img_np, hair_mask, cx1, cy1, cx2, cy2,
    )
    crop = np.array(masked_crop)
    ch, cw = crop.shape[:2]

    if ch < 8 or cw < 8:
        return _person_dict(
            person_id, "Unknown", "none", ["Unknown"], 0.10,
            [0.0, 0.0, 0.0], "Unknown", ["Unknown"], 0.10,
            hair_pixels=local_count, occluded=True,
        )

    # ----------------------------------------------------------------
    # ZERO-LOCAL-MASK PATH
    # When the segmenter found zero hair pixels in the face search zone
    # (local_count == 0), the face-crop region is dominated by
    # background (e.g. bright white walls). Using that crop for color
    # classification is meaningless — instead pull pixels directly from
    # the global hair mask, which may have found hair elsewhere in the
    # image (e.g. hair off to the sides or top of frame).
    # ----------------------------------------------------------------
    if local_count == 0 and total_hair >= MIN_HAIR_PIXELS:
        global_hair_rgb = img_np[hair_mask]
        logger.info(
            "[%s] Person %d: local_count=0 — using %d global mask px for color",
            source_id, person_id, len(global_hair_rgb),
        )
        color, c_pal, c_conf, dom_lab, method = _hybrid_color(global_hair_rgb)

        # Build texture crop from global mask bbox
        gb = _mask_bbox(hair_mask, MASK_CROP_PAD, img_h, img_w)
        if gb:
            global_masked_crop, global_px_count = _apply_mask_to_crop(
                img_np, hair_mask, gb[0], gb[1], gb[2], gb[3],
            )
            texture, t_pal, t_conf = _texture_classify(
                global_masked_crop, global_px_count, face_area=face_area,
            )
            hair_pixels_out = global_px_count
        else:
            texture, t_pal, t_conf = "Bald", ["Bald"], 0.5
            hair_pixels_out = 0

        person = _person_dict(
            person_id, color, method, c_pal, c_conf, dom_lab,
            texture, t_pal, t_conf,
            hair_pixels=hair_pixels_out, occluded=False,
        )
        person["note"] = "face_crop_fallback_global_mask"
        return person

    # ----------------------------------------------------------------
    # NORMAL FALLBACK PATH — local mask was sparse but non-zero,
    # or no global mask available. Use the face-crop region.
    # ----------------------------------------------------------------
    if crop_pixel_count < MIN_HAIR_PIXELS:
        crop_pixel_count = max(BALD_GATE_THRESHOLD + 1, ch * cw // 4)

    flat = crop.reshape(-1, 3)

    # --- Dark-hair fast path ---
    # If the bottom _DARK_PATH_PERCENTILE of brightness is very dark,
    # classify directly on those dark pixels before skin/bg filtering
    # strips them and leaves grey residuals dominating the color read.
    brightness = flat.astype(np.float32).mean(axis=1)
    dark_cut = float(np.percentile(brightness, _DARK_PATH_PERCENTILE))
    dark_pixels = flat[brightness <= dark_cut]

    if len(dark_pixels) >= KMEANS_MIN_PIXELS and dark_cut < _DARK_PATH_BRIGHTNESS_CUT:
        color, c_pal, c_conf, dom_lab, method = _hybrid_color(dark_pixels)
        logger.info(
            "[%s] Person %d: dark-path (p%d_cut=%.1f < %.1f) → %s",
            source_id, person_id,
            _DARK_PATH_PERCENTILE, dark_cut, _DARK_PATH_BRIGHTNESS_CUT,
            color,
        )
    else:
        # Normal path: skin/bg filter first, then classify.
        filtered = _exclude_skin_and_bg(flat)

        if len(filtered) < KMEANS_MIN_PIXELS:
            logger.info(
                "[%s] Person %d: %d px after filter — dark-pixel fallback",
                source_id, person_id, len(filtered),
            )
            dark_cut_30 = np.percentile(brightness, 30)
            filtered = flat[brightness <= dark_cut_30]
            if len(filtered) < KMEANS_MIN_PIXELS:
                filtered = flat

        color, c_pal, c_conf, dom_lab, method = _hybrid_color(filtered)

    color_hint = color

    # Guard: don't let Grey/White color_hint match the grey background
    # we painted ourselves — that would make _simple_hair_mask select
    # the entire crop. Fall back to the darkest-pixels strategy instead.
    if color_hint in ("Grey", "White"):
        simple_mask = _simple_hair_mask(crop, color_hint, bg_color=_MASK_BG_COLOR)
    else:
        simple_mask = _simple_hair_mask(crop, color_hint)
    simple_pixel_count = int(simple_mask.sum())

    if simple_pixel_count >= MIN_HAIR_PIXELS:
        hair_rgb_simple = crop[simple_mask]
        color, c_pal, c_conf, dom_lab, method = _hybrid_color(hair_rgb_simple)

        masked_crop_simple = crop.copy()
        masked_crop_simple[~simple_mask] = _MASK_BG_COLOR
        texture, t_pal, t_conf = _texture_classify(
            Image.fromarray(masked_crop_simple), simple_pixel_count,
            face_area=face_area,
        )
        logger.info("Fallback using simple mask: %s hair pixels", simple_pixel_count)
        hair_pixels_out = simple_pixel_count
    else:
        logger.info(
            "[%s] Person %d: fallback crop_pixel_count=%d (ch=%d cw=%d)",
            source_id, person_id, crop_pixel_count, ch, cw,
        )
        texture, t_pal, t_conf = _texture_classify(
            masked_crop, hair_pixel_count=crop_pixel_count,
            face_area=face_area,
        )
        hair_pixels_out = crop_pixel_count

    person = _person_dict(
        person_id, color, method, c_pal, c_conf, dom_lab,
        texture, t_pal, t_conf,
        hair_pixels=hair_pixels_out, occluded=False,
    )
    person["note"] = "face_crop_fallback"
    return person


# ---------------------------------------------------------------------------
# Normal mask-based analysis
# ---------------------------------------------------------------------------

def _analyze_region(
    img_np: np.ndarray,
    hair_mask: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    *,
    person_id: int,
    face_area: int = 0,
) -> dict:
    region_mask = hair_mask[y1:y2, x1:x2]
    hair_rgb = img_np[y1:y2, x1:x2][region_mask]

    color, c_pal, c_conf, dom_lab, method = _hybrid_color(hair_rgb)

    masked_crop, crop_count = _apply_mask_to_crop(img_np, hair_mask, x1, y1, x2, y2)
    texture, t_pal, t_conf = _texture_classify(masked_crop, crop_count, face_area=face_area)

    occluded = c_conf < OCCLUSION_CONFIDENCE_THRESHOLD

    return _person_dict(
        person_id, color, method, c_pal, c_conf, dom_lab,
        texture, t_pal, t_conf,
        hair_pixels=crop_count, occluded=occluded,
    )


def _person_dict(
    person_id: int,
    color: str, color_method: str,
    c_pal: list[str], c_conf: float, dom_lab: list[float],
    texture: str, t_pal: list[str], t_conf: float,
    *,
    hair_pixels: int,
    occluded: bool,
) -> dict:
    return {
        "person_id": person_id,
        "dominant_color": color,
        "color_method": color_method,
        "color_palette": c_pal,
        "color_confidence": c_conf,
        "dominant_lab": dom_lab,
        "dominant_texture": texture,
        "texture_palette": t_pal,
        "texture_confidence": t_conf,
        "hair_pixel_count": hair_pixels,
        "occluded": occluded,
    }


async def _extract(image_bytes: bytes, source_id: str = "") -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_extraction, image_bytes, source_id)


# ---------------------------------------------------------------------------
# Debug helpers for dump_training_crops.py
# ---------------------------------------------------------------------------

def debug_extract_masked_crop(
    image_bytes: bytes,
) -> tuple[Image.Image, int] | None:
    """
    Return (masked_grey_background_crop, hair_pixel_count) for a hair image,
    or None if no usable crop could be produced.

    Used by dump_training_crops.py to generate training data that matches
    exactly what the server sends to EfficientNet at inference time.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        img_h, img_w = img_np.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

        # Hair segmentation
        segmenter = _load_hair_segmenter()
        seg_result = segmenter.segment(mp_image)
        raw_mask = seg_result.category_mask.numpy_view()
        hair_mask_raw = np.squeeze(raw_mask) == 1

        dilate_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_DILATE_KERNEL, MASK_DILATE_KERNEL),
        )
        dilated = cv2.dilate(
            hair_mask_raw.astype(np.uint8), dilate_kern, iterations=MASK_DILATE_ITERS,
        )
        close_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_CLOSE_KERNEL, MASK_CLOSE_KERNEL),
        )
        hair_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kern).astype(bool)

        total_hair = int(hair_mask.sum())
        if total_hair < MIN_HAIR_PIXELS:
            return None

        gb = _mask_bbox(hair_mask, MASK_CROP_PAD, img_h, img_w)
        if gb is None:
            return None

        masked_crop, hair_px = _apply_mask_to_crop(img_np, hair_mask, *gb)
        if hair_px < MIN_HAIR_PIXELS:
            return None

        return masked_crop, hair_px

    except Exception as exc:
        logger.debug("debug_extract_masked_crop failed: %s", exc)
        return None


def debug_extract_bald_crop(
    image_bytes: bytes,
) -> tuple[Image.Image, int] | None:
    """
    For Bald training samples: return a face-region crop filled with grey
    background (simulating what the server produces for a bald person).

    Bald images have near-zero hair mask pixels so debug_extract_masked_crop
    returns None for them. This function instead crops the top of the face
    (where hair would be) and fills it entirely with the grey background color,
    which is exactly what EfficientNet sees at inference for a bald person.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        img_h, img_w = img_np.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

        # Detect face
        detector = _load_face_detector()
        det_result = detector.detect(mp_image)
        if not det_result.detections:
            # No face — just return a grey-filled centre crop
            cy, cx = img_h // 2, img_w // 2
            size = min(img_h, img_w, 224)
            crop = np.full((size, size, 3), _MASK_BG_COLOR, dtype=np.uint8)
            return Image.fromarray(crop), 0

        bb = det_result.detections[0].bounding_box
        fx = int(bb.origin_x)
        fy = int(bb.origin_y)
        fw = int(bb.width)
        fh = int(bb.height)

        # Crop strictly above the face — same region the server analyses
        cx1, cy1, cx2, cy2 = _face_crop_above(fx, fy, fw, fh, img_h, img_w)

        # Fill the whole crop with grey background — bald = no hair pixels
        ch = max(cy2 - cy1, 10)
        cw = max(cx2 - cx1, 10)
        bald_crop = np.full((ch, cw, 3), _MASK_BG_COLOR, dtype=np.uint8)

        return Image.fromarray(bald_crop), 0

    except Exception as exc:
        logger.debug("debug_extract_bald_crop failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Server entrypoint
# ---------------------------------------------------------------------------

async def serve() -> None:
    server = grpc.aio.server()
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(HairAnalysisServicer(), server)
    addr = f"[::]:{PORT}"
    server.add_insecure_port(addr)
    logger.info("Hair analysis MCP starting on %s", addr)
    await server.start()
    logger.info("Hair analysis MCP ready")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())