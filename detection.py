"""
Chip Rack Detector
==================
Pipeline:
  1. Detect every chip bag with an open-vocabulary detector (OWLv2)
  2. Filter boxes: size outliers, row-spanning boxes, duplicates/fragments,
     and (optionally) anything outside the rack foreground mask (rembg)
  3. Classify each bag by fusing two signals against the reference images:
       • CLIP image similarity (crop TTA vs. augmented references)
       • HSV colour-histogram similarity (separates same-brand flavours
         such as Doritos Nacho vs. Cool Ranch that CLIP confuses)
  4. Row-consistency smoothing: bags next to each other on a shelf are
     usually the same product, so neighbours nudge each other's scores
  5. Low-confidence bags are reported as "Unknown" instead of being forced
     into the wrong brand
  6. Return / print per-type counts + save an annotated image

Usage:
    python3 detection.py rack/1.png
    python3 detection.py rack/1.png --refs ./chip_refs --threshold 0.53
    python3 detection.py rack/1.png --save-crops   # dump each detected crop

Requirements:
    pip install torch transformers Pillow opencv-python-headless rembg
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import (CLIPModel, CLIPProcessor, Owlv2ForObjectDetection,
                          Owlv2Processor)

try:
    from rembg import new_session, remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_REFS_DIR  = Path(__file__).parent / "chip_refs"
SUPPORTED_EXTS    = {".png", ".jpg", ".jpeg", ".webp"}
CLIP_MODEL_ID     = "openai/clip-vit-base-patch32"
OWL_MODEL_ID      = "google/owlv2-base-patch16-ensemble"

# Text queries for the bag detector (scores are max-pooled across queries).
DETECT_QUERIES    = ["a bag of chips", "a snack bag", "a packet of potato chips"]

# OWLv2 objectness score a box needs to be considered a bag.
DETECT_THRESHOLD  = 0.20

# Box filters, relative to the median box area of confident detections.
MIN_AREA_FRAC     = 0.25    # smaller → fragment of a bag
MAX_AREA_FRAC     = 2.5     # larger  → covers several bags / a whole row
MAX_ASPECT_WH     = 1.5     # wider than tall → row-spanning box

# Duplicate suppression: IoU, and "small box mostly inside a bigger box".
NMS_IOU_THRESH    = 0.30
CONTAIN_THRESH    = 0.60

# A box must have at least this fraction of rack-foreground pixels (rembg).
MIN_FG_FRAC       = 0.25

# Classification: final score = (1-w)·CLIP + w·COLOR_SCALE·colour.
COLOR_WEIGHT      = 0.15
COLOR_SCALE       = 0.8     # brings Bhattacharyya coeffs onto CLIP's range
ROW_SMOOTHING     = 0.5     # weight of same-row neighbours' scores

# Fused score below which a bag is labelled "Unknown".
CLIP_THRESHOLD    = 0.53
UNKNOWN_LABEL     = "Unknown"

BATCH_SIZE        = 32

# HSV histogram layout: HUE_BINS×SAT_BINS chromatic bins + VAL_BINS grey bins.
HUE_BINS, SAT_BINS, VAL_BINS = 24, 4, 4

# BGR colour palette for annotation
PALETTE = [
    (0, 220, 80),   (0, 120, 255),  (255, 100, 0),
    (180, 0, 255),  (0, 210, 210),  (255, 220, 0),
    (255, 80, 180), (80, 255, 200), (0, 160, 160),
]

# Fixed BGR color mapping for premium visual consistency
BRAND_COLOR_MAP = {
    "Doritos Nacho": (0, 70, 230),                  # Warm Red/Orange
    "Doritos Cool Ranch": (240, 110, 0),            # Rich Blue
    "Lays Classic": (0, 220, 255),                 # Yellow/Golden
    "Lays Baked": (0, 190, 80),                    # Deep Green
    "Lays Baked Bbq": (80, 80, 180),               # Smoky Red
    "Lays Bbq": (50, 50, 130),                     # Barbecue Brown/Red
    "Lays Sourcream": (140, 210, 80),              # Greenish Lime
    "Miss Vickies Jalapeno": (0, 120, 255),        # Cyan Blue
    "Miss Vickies Pickle": (0, 160, 100),          # Soft Teal
    "Miss Vickies Sea Salt": (100, 100, 240),      # Lavender
    "Cheetos": (0, 140, 255),                      # Cheesy Orange
    "Ruffles Baked": (255, 100, 180),              # Baked Rose Pink
    "Ruffles Cheddar": (0, 110, 230),              # Vibrant Orange
    "Sun Chips Garden Salsa": (120, 200, 200),     # Muted Cyan
    "Sun Chips Harvest Cheddar": (180, 0, 180),    # Harvest Purple
    UNKNOWN_LABEL: (150, 150, 150),                # Grey
}


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    det_score: float
    label: str = UNKNOWN_LABEL
    score: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def pretty_name(filename: str) -> str:
    return Path(filename).stem.replace("-", " ").replace("_", " ").title()


def list_refs(refs_dir: Path) -> list[Path]:
    """Reference images: chip_refs/<label>.png and chip_refs/<label>/*.png."""
    return sorted(p for p in refs_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def ref_label(path: Path, refs_dir: Path) -> str:
    """
    A top-level file is labelled by its own name; an image inside a
    sub-folder is labelled by the folder, so extra shots of one product
    (e.g. a new packaging design) can live in chip_refs/<label>/.
    """
    rel = path.relative_to(refs_dir)
    return pretty_name(rel.parts[0] if len(rel.parts) > 1 else rel.name)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def box_area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def reference_augs(img: Image.Image) -> list[Image.Image]:
    """
    Views of a studio reference shot that resemble how bags look on a rack:
    bottom hidden by the wire basket, slightly tilted, dimmer lighting.
    """
    w, h = img.size
    white = (255, 255, 255)
    return [
        img,
        img.crop((0, 0, w, int(h * 0.55))),
        img.crop((0, 0, w, int(h * 0.70))),
        img.crop((0, 0, w, int(h * 0.85))),
        img.crop((int(w * 0.1), int(h * 0.05), int(w * 0.9), int(h * 0.75))),
        img.rotate(8, expand=True, fillcolor=white),
        img.rotate(-8, expand=True, fillcolor=white),
        ImageEnhance.Brightness(img).enhance(0.75),
        ImageEnhance.Color(img).enhance(0.8),
    ]


def crop_tta(crop: Image.Image) -> list[Image.Image]:
    """Full crop, top 70 % (above the basket wire), and a centred crop."""
    w, h = crop.size
    return [
        crop,
        crop.crop((0, 0, w, int(h * 0.70))),
        crop.crop((int(w * 0.1), int(h * 0.05), int(w * 0.9), int(h * 0.80))),
    ]


def color_hist(rgb: np.ndarray, is_ref: bool = False) -> np.ndarray:
    """
    Normalised hue/saturation histogram plus a few achromatic (grey-level)
    bins.  For rack crops only the central-top region is used, skipping the
    basket wire at the bottom and neighbouring bags at the sides.  For
    reference images the white studio background is masked out.
    """
    h, w = rgb.shape[:2]
    if not is_ref:
        rgb = rgb[int(h * 0.05):int(h * 0.75), int(w * 0.12):int(w * 0.88)]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    H, S, V = (hsv[..., i].astype(int) for i in range(3))

    valid  = ~((S < 25) & (V > 225)) if is_ref else np.ones_like(H, bool)
    chroma = (S >= 60) & (V >= 50) & valid
    achrom = ~chroma & valid

    hist = np.zeros(HUE_BINS * SAT_BINS + VAL_BINS, np.float32)
    hb = (H * HUE_BINS // 180).clip(0, HUE_BINS - 1)
    sb = ((S - 60) * SAT_BINS // 196).clip(0, SAT_BINS - 1)
    np.add.at(hist, (hb * SAT_BINS + sb)[chroma], 1)
    np.add.at(hist, HUE_BINS * SAT_BINS + (V * VAL_BINS // 256).clip(0, VAL_BINS - 1)[achrom], 1)

    # Blur along the hue axis so small lighting shifts don't flip bins.
    chrom_hist = hist[:HUE_BINS * SAT_BINS].reshape(HUE_BINS, SAT_BINS)
    hist[:HUE_BINS * SAT_BINS] = cv2.GaussianBlur(chrom_hist, (1, 3), 0).ravel()
    return hist / (hist.sum() + 1e-9)


# ── Model cache ───────────────────────────────────────────────────────────────

class _Models:
    """Loads models once per process and caches reference embeddings."""

    def __init__(self):
        self.lock = threading.Lock()
        self.device = pick_device()
        self.owl = self.owl_proc = None
        self.clip = self.clip_proc = None
        self.rembg_session = None
        self.refs_key = None
        self.ref_labels: list[str] = []
        self.ref_embs: torch.Tensor | None = None     # [N_views, D]
        self.ref_view_labels: np.ndarray | None = None
        self.ref_hists: np.ndarray | None = None      # [N_images, bins]
        self.ref_hist_labels: np.ndarray | None = None

    def load(self):
        if self.owl is None:
            print(f"🤖  Loading OWLv2 ({OWL_MODEL_ID}) …  [device: {self.device}]")
            self.owl_proc = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
            self.owl = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID).to(self.device).eval()
        if self.clip is None:
            print(f"🤖  Loading CLIP ({CLIP_MODEL_ID}) …")
            self.clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            self.clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device).eval()
        if HAS_REMBG and self.rembg_session is None:
            self.rembg_session = new_session("u2net")

    def encode(self, images: list[Image.Image]) -> torch.Tensor:
        """Normalised CLIP image embeddings, [N, D]."""
        out = []
        for i in range(0, len(images), BATCH_SIZE):
            px = self.clip_proc(images=images[i:i + BATCH_SIZE], return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                feats = self.clip.vision_model(pixel_values=px.to(self.device)).pooler_output
                feats = self.clip.visual_projection(feats)
            out.append(torch.nn.functional.normalize(feats, dim=-1))
        return torch.cat(out)

    def load_refs(self, refs_dir: Path):
        paths = list_refs(refs_dir)
        if not paths:
            raise FileNotFoundError(f"No reference images found in {refs_dir}")
        key = tuple((str(p), p.stat().st_mtime) for p in paths)
        if key == self.refs_key:
            return

        print(f"📚  Encoding {len(paths)} reference chip images …")
        labels, views, view_labels, hists, hist_labels = [], [], [], [], []
        for p in paths:
            img = Image.open(p).convert("RGB")
            label = ref_label(p, refs_dir)
            augs = reference_augs(img)
            if label not in labels:
                labels.append(label)
            views.extend(augs)
            view_labels.extend([label] * len(augs))
            hists.append(color_hist(np.array(img), is_ref=True))
            hist_labels.append(label)

        self.ref_labels = labels
        self.ref_embs = self.encode(views)
        self.ref_view_labels = np.array(view_labels)
        self.ref_hists = np.stack(hists)
        self.ref_hist_labels = np.array(hist_labels)
        self.refs_key = key


_MODELS = _Models()


# ── Pipeline stages ───────────────────────────────────────────────────────────

def foreground_mask(pil_img: Image.Image) -> np.ndarray | None:
    """Rack foreground alpha (H×W uint8) via rembg, or None if unavailable."""
    if _MODELS.rembg_session is None:
        return None
    print("🖼   Removing background …")
    rgba = np.array(rembg_remove(pil_img, session=_MODELS.rembg_session))
    alpha = rgba[:, :, 3]
    # A near-empty or near-full mask means rembg didn't find a clear rack;
    # don't use it to filter.
    frac = (alpha > 30).mean()
    return alpha if 0.10 < frac < 0.95 else None


def detect_bags(pil_img: Image.Image, threshold: float = DETECT_THRESHOLD) -> list[Detection]:
    """Run OWLv2 and return raw bag candidates in image coordinates."""
    m = _MODELS
    inputs = m.owl_proc(text=[DETECT_QUERIES], images=pil_img, return_tensors="pt").to(m.device)
    with torch.no_grad():
        outputs = m.owl(**inputs)
    # OWLv2 pads the image to a square, so boxes are relative to the long side.
    side = max(pil_img.size)
    res = m.owl_proc.image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=torch.tensor([[side, side]])
    )[0]
    W, H = pil_img.size
    dets = []
    for (x1, y1, x2, y2), s in zip(res["boxes"].cpu().tolist(), res["scores"].cpu().tolist()):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 - x1 > 4 and y2 - y1 > 4:
            dets.append(Detection((x1, y1, x2, y2), s))
    return dets


def filter_boxes(dets: list[Detection], alpha: np.ndarray | None) -> list[Detection]:
    """Drop size outliers, row-spanning boxes, off-rack boxes and duplicates."""
    if not dets:
        return dets
    med = float(np.median([box_area(d.box) for d in dets]))

    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.det_score, reverse=True):
        x1, y1, x2, y2 = d.box
        area = box_area(d.box)
        if not (MIN_AREA_FRAC * med <= area <= MAX_AREA_FRAC * med):
            continue
        if (x2 - x1) / (y2 - y1) > MAX_ASPECT_WH:
            continue
        if alpha is not None and (alpha[y1:y2, x1:x2] > 30).mean() < MIN_FG_FRAC:
            continue
        duplicate = False
        for k in kept:
            inter = intersection(d.box, k.box)
            k_area = box_area(k.box)
            if (inter / (area + k_area - inter) > NMS_IOU_THRESH
                    or inter / min(area, k_area) > CONTAIN_THRESH):
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept


def row_neighbour_weights(dets: list[Detection]) -> np.ndarray:
    """
    W[i, j] > 0 when bag j sits beside bag i on the same shelf, peaking when
    they are directly adjacent (centre distance ≈ one bag width).
    """
    n = len(dets)
    W = np.zeros((n, n))
    for i, a in enumerate(dets):
        ax1, ay1, ax2, ay2 = a.box
        for j, b in enumerate(dets):
            if i == j:
                continue
            bx1, by1, bx2, by2 = b.box
            y_overlap = max(0, min(ay2, by2) - max(ay1, by1)) / min(ay2 - ay1, by2 - by1)
            if y_overlap < 0.6:
                continue
            dx = abs((ax1 + ax2) - (bx1 + bx2)) / ((ax2 - ax1) + (bx2 - bx1))
            if dx < 2.5:
                W[i, j] = np.exp(-((dx - 1) ** 2) / 0.5)
    return W


def classify_bags(rgb: np.ndarray, dets: list[Detection], threshold: float):
    """Assign label + score to every detection in place."""
    if not dets:
        return
    m = _MODELS
    crops = [Image.fromarray(rgb[y1:y2, x1:x2]) for (x1, y1, x2, y2) in (d.box for d in dets)]

    # CLIP: average the TTA views, then take the best-matching view per label.
    views = [v for c in crops for v in crop_tta(c)]
    embs = m.encode(views).reshape(len(crops), 3, -1).mean(1)
    embs = torch.nn.functional.normalize(embs, dim=-1)
    view_sims = (embs @ m.ref_embs.T).cpu().numpy()

    # Colour: Bhattacharyya coefficient against each reference image.
    crop_hists = np.stack([color_hist(np.array(c)) for c in crops])
    image_color = np.sqrt(crop_hists[:, None, :] * m.ref_hists[None]).sum(-1)

    # Best-matching reference per label, CLIP and colour combined.
    clip_sims  = np.stack([view_sims[:, m.ref_view_labels == l].max(1) for l in m.ref_labels], 1)
    color_sims = np.stack([image_color[:, m.ref_hist_labels == l].max(1) for l in m.ref_labels], 1)
    fused = (1 - COLOR_WEIGHT) * clip_sims + COLOR_WEIGHT * COLOR_SCALE * color_sims

    # Same-row neighbours vote for the label, but the Unknown decision is made
    # on each bag's own score so a stray item can't borrow confidence.
    W = row_neighbour_weights(dets)
    norm = W.sum(1, keepdims=True)
    smoothed = fused + ROW_SMOOTHING * np.divide(W @ fused, norm, out=np.zeros_like(fused), where=norm > 0)

    for d, f, s in zip(dets, fused, smoothed):
        idx = int(s.argmax())
        d.score = float(f[idx])
        d.label = m.ref_labels[idx] if d.score >= threshold else UNKNOWN_LABEL


def annotate(rgb: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw translucent boxes + label badges.  Returns a BGR image."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Upscale small photos so the labels stay readable.
    scale = max(1.0, 1200 / max(img.shape[:2]))
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    label_colors: dict[str, tuple] = {}
    overlay = img.copy()
    boxes = []
    for d in dets:
        if d.label in BRAND_COLOR_MAP:
            color = BRAND_COLOR_MAP[d.label]
        else:
            color = label_colors.setdefault(d.label, PALETTE[len(label_colors) % len(PALETTE)])
        x1, y1, x2, y2 = (int(v * scale) for v in d.box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        boxes.append((x1, y1, x2, y2, color, d))
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    for x1, y1, x2, y2, color, d in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = d.label if d.label == UNKNOWN_LABEL else f"{d.label[:18]} {d.score:.2f}"
        fs = 0.38
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tw = min(tw, x2 - x1)
        cv2.rectangle(img, (x1, y1), (x1 + tw + 4, y1 + th + 6), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ── Core pipeline ─────────────────────────────────────────────────────────────

def detect_chips(
    rack_path: Path,
    refs_dir: Path,
    threshold: float = CLIP_THRESHOLD,
    save_crops: bool = False,
) -> dict[str, int]:
    """
    Full pipeline:
      rack image → OWLv2 bag detection → box filtering → CLIP + colour
      classification → row smoothing → counts dict

    Returns {chip_label: count}; low-confidence bags are counted as "Unknown".
    """
    with _MODELS.lock:
        _MODELS.load()
        _MODELS.load_refs(refs_dir)

        rack_pil = Image.open(rack_path).convert("RGB")
        rgb = np.array(rack_pil)
        print(f"\n📷  Loaded: {rack_path.name}  ({rack_pil.width}×{rack_pil.height})")

        alpha = foreground_mask(rack_pil)

        print("🔍  Detecting bags …")
        raw = detect_bags(rack_pil)
        dets = filter_boxes(raw, alpha)
        print(f"   Raw detections : {len(raw)}")
        print(f"   After filtering: {len(dets)}")

        classify_bags(rgb, dets, threshold)

    counts: dict[str, int] = defaultdict(int)
    for d in dets:
        counts[d.label] += 1

    print("\n" + "=" * 60)
    print("🛒  CHIP COUNT RESULTS")
    print("=" * 60)
    if not counts:
        print("  No chip bags detected.")
    else:
        for label, cnt in sorted(counts.items()):
            bar = "█" * min(cnt, 20)
            print(f"  • {label:<32} {cnt:>3}  {bar}")
        print("-" * 60)
        print(f"  {'TOTAL':<32} {sum(counts.values()):>3}")
    print("=" * 60)

    out_path = rack_path.with_name(rack_path.stem + "_detected.jpg")
    cv2.imwrite(str(out_path), annotate(rgb, dets))
    print(f"\n🖼   Annotated image  → {out_path}")

    if save_crops:
        crops_dir = rack_path.parent / f"{rack_path.stem}_crops"
        crops_dir.mkdir(exist_ok=True)
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d.box
            Image.fromarray(rgb[y1:y2, x1:x2]).save(crops_dir / f"{i:03d}_{d.label}_{d.score:.2f}.png")
        print(f"🗂   Crops            → {crops_dir}/")

    return dict(counts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Count chips on a rack: OWLv2 detection → CLIP + colour classification."
    )
    parser.add_argument("rack_image",
                        help="Path to the chip rack photo (jpg / png / webp).")
    parser.add_argument("--refs",
                        default=str(DEFAULT_REFS_DIR),
                        help="Directory of reference chip images (default: ./chip_refs).")
    parser.add_argument("--threshold",
                        type=float, default=CLIP_THRESHOLD,
                        help=f"Score below which a bag is 'Unknown' (default: {CLIP_THRESHOLD}).")
    parser.add_argument("--save-crops",
                        action="store_true",
                        help="Save every detected bag crop for inspection.")
    args = parser.parse_args()

    rack = Path(args.rack_image)
    refs = Path(args.refs)
    if not rack.exists(): print(f"❌  Not found: {rack}"); sys.exit(1)
    if not refs.exists(): print(f"❌  Not found: {refs}"); sys.exit(1)

    return detect_chips(rack, refs, threshold=args.threshold,
                        save_crops=args.save_crops)


if __name__ == "__main__":
    main()
