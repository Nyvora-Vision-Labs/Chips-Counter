"""
Chip Rack Detector
==================
Pipeline:
  1. Remove background from the rack image (rembg)
  2. Estimate chip-bag size from the foreground bounding box
  3. Slide a window over FOREGROUND-ONLY pixels at chip-bag scale
  4. CLIP-classify each candidate window
  5. Per-label NMS to collapse duplicates
  6. Return / print per-type counts + save an annotated image

Usage:
    python3 detection.py rack/1.png
    python3 detection.py rack/1.png --refs ./chip_refs --threshold 0.65
    python3 detection.py rack/1.png --save-crops   # dump each matched crop

Requirements:
    pip install torch torchvision transformers Pillow opencv-python-headless rembg
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_REFS_DIR  = Path(__file__).parent / "chip_refs"
SUPPORTED_EXTS    = {".png", ".jpg", ".jpeg", ".webp"}
CLIP_MODEL_ID     = "openai/clip-vit-base-patch32"

# Cosine-similarity threshold: window embedding vs best reference
CLIP_THRESHOLD    = 0.65

# Fraction of the foreground bounding-box dimensions used to estimate a
# single chip-bag size.  Tune if bags appear larger/smaller in your images.
BAG_WIDTH_FRAC    = 1 / 6    # ~6 bags side-by-side per row
BAG_HEIGHT_FRAC   = 1 / 5   # ~5 rows on the rack

# We slide windows at three scale factors around the estimated size
SCALE_FACTORS     = [0.75, 1.0, 1.3]

# Stride as a fraction of window size
# 0.5 means windows overlap by 50% — enough density without explosion of crops
STRIDE_FRAC       = 0.50

# A candidate window must contain at least this fraction of foreground pixels
# (eliminates windows that land mostly on transparent/background areas)
MIN_FG_FRAC       = 0.65

# Global IoU threshold: any two windows that overlap more than this are
# considered to cover the SAME bag — only the highest-scoring one survives
NMS_IOU_THRESH    = 0.20

# BGR colour palette for annotation
PALETTE = [
    (0, 220, 80),   (0, 120, 255),  (255, 100, 0),
    (180, 0, 255),  (0, 210, 210),  (255, 220, 0),
    (255, 80, 180), (80, 255, 200), (0, 160, 160),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def pretty_name(filename: str) -> str:
    return Path(filename).stem.replace("-", " ").replace("_", " ").title()


def encode_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    pil_img: Image.Image,
) -> torch.Tensor:
    """Return a normalised CLIP embedding tensor [1, 512]."""
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.vision_model(pixel_values=inputs["pixel_values"])
        emb = model.visual_projection(out.pooler_output)   # [1, 512]
    return emb / emb.norm(dim=-1, keepdim=True)


def classify(
    emb: torch.Tensor,
    ref_embs: torch.Tensor,
    ref_labels: list[str],
    threshold: float,
) -> tuple[str | None, float]:
    """Return (best_label, score) or (None, score) if below threshold."""
    sims  = (emb @ ref_embs.T)[0]
    idx   = int(sims.argmax().item())
    score = float(sims[idx].item())
    return (ref_labels[idx] if score >= threshold else None), score


def iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union of two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def nms_global(
    detections: list[tuple[float, str, tuple]],
    iou_thresh: float,
) -> list[tuple[float, str, tuple]]:
    """
    Global NMS across ALL labels.

    Sort all candidates by score (highest first).  A candidate is suppressed
    if it overlaps (IoU > iou_thresh) with ANY already-accepted candidate,
    regardless of label.  This ensures each physical bag position is counted
    exactly once, assigned to the label with the highest CLIP score.
    """
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    kept: list[tuple[float, str, tuple]] = []
    for det in detections:
        if all(iou(det[2], k[2]) < iou_thresh for k in kept):
            kept.append(det)
    return kept


def remove_background(pil_img: Image.Image) -> Image.Image:
    """Return an RGBA image with background removed via rembg."""
    if not HAS_REMBG:
        print("⚠️  rembg not installed — skipping background removal.")
        return pil_img.convert("RGBA")
    print("🖼   Removing background …")
    return rembg_remove(pil_img)   # RGBA


def foreground_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """
    Tight bounding box of the non-transparent region.
    Returns (x_min, y_min, x_max, y_max).
    """
    ys, xs = np.where(alpha > 30)
    if len(xs) == 0:
        h, w = alpha.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def sliding_windows(
    alpha: np.ndarray,
    win_w: int,
    win_h: int,
    stride_x: int,
    stride_y: int,
    fg_frac: float,
) -> Generator[tuple[int, int, int, int], None, None]:
    """
    Yield (x1, y1, x2, y2) windows that contain enough foreground pixels.
    Only iterates inside the foreground bounding box.
    """
    fx1, fy1, fx2, fy2 = foreground_bbox(alpha)
    win_area = win_w * win_h

    for y in range(fy1, max(fy1 + 1, fy2 - win_h + 1), stride_y):
        for x in range(fx1, max(fx1 + 1, fx2 - win_w + 1), stride_x):
            x2, y2 = x + win_w, y + win_h
            # Fraction of this window that is foreground
            patch_fg = alpha[y:y2, x:x2]
            if patch_fg.size == 0:
                continue
            if (patch_fg > 30).sum() / patch_fg.size >= fg_frac:
                yield x, y, x2, y2


# ── Core pipeline ─────────────────────────────────────────────────────────────

def detect_chips(
    rack_path: Path,
    refs_dir: Path,
    threshold: float = CLIP_THRESHOLD,
    save_crops: bool = False,
) -> dict[str, int]:
    """
    Full pipeline:
      rack image → bg removal → foreground sliding window → CLIP classify
      → per-label NMS → counts dict

    Returns {chip_label: count}.
    """

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🤖  Loading CLIP ({CLIP_MODEL_ID}) …  [device: {device}]")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    # ── Encode references ─────────────────────────────────────────────────────
    ref_paths = sorted(p for p in refs_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not ref_paths:
        print(f"❌  No reference images found in {refs_dir}"); sys.exit(1)

    print(f"\n📚  Encoding {len(ref_paths)} reference chip images …")
    ref_labels: list[str] = []
    ref_embs_list: list[torch.Tensor] = []
    for p in ref_paths:
        emb = encode_clip(clip_model, clip_processor, device,
                          Image.open(p).convert("RGB"))
        ref_labels.append(pretty_name(p.name))
        ref_embs_list.append(emb)
        print(f"   ✓ {pretty_name(p.name)}")
    ref_embs = torch.cat(ref_embs_list, dim=0)   # [N_refs, 512]

    # ── Load rack + remove background ────────────────────────────────────────
    rack_pil = Image.open(rack_path).convert("RGB")
    print(f"\n📷  Loaded: {rack_path.name}  ({rack_pil.width}×{rack_pil.height})")

    rack_rgba    = remove_background(rack_pil)
    rack_rgba_np = np.array(rack_rgba)            # H×W×4, uint8
    alpha        = rack_rgba_np[:, :, 3]          # H×W
    rack_bgr     = cv2.cvtColor(rack_rgba_np[:, :, :3], cv2.COLOR_RGB2BGR)
    rack_bgr[alpha == 0] = 0                      # zero-out transparent pixels

    # ── Estimate chip-bag size from the foreground bounding box ───────────────
    fx1, fy1, fx2, fy2 = foreground_bbox(alpha)
    fg_w = fx2 - fx1
    fg_h = fy2 - fy1
    base_win_w = max(40, int(fg_w * BAG_WIDTH_FRAC))
    base_win_h = max(50, int(fg_h * BAG_HEIGHT_FRAC))
    print(f"\n📐  Foreground region: {fg_w}×{fg_h} px")
    print(f"    Estimated bag size: {base_win_w}×{base_win_h} px")
    print(f"    Scale factors applied: {SCALE_FACTORS}")

    # ── Sliding window scan ───────────────────────────────────────────────────
    print("\n🔍  Scanning foreground with sliding windows …")
    raw_detections: list[tuple[float, str, tuple]] = []
    total_windows = 0

    crops_dir = rack_path.parent / f"{rack_path.stem}_crops"
    if save_crops:
        crops_dir.mkdir(exist_ok=True)

    for scale in SCALE_FACTORS:
        win_w = max(30, int(base_win_w * scale))
        win_h = max(40, int(base_win_h * scale))
        stride_x = max(8, int(win_w * STRIDE_FRAC))
        stride_y = max(8, int(win_h * STRIDE_FRAC))

        for x1, y1, x2, y2 in sliding_windows(alpha, win_w, win_h,
                                               stride_x, stride_y, MIN_FG_FRAC):
            total_windows += 1
            crop_bgr = rack_bgr[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

            emb          = encode_clip(clip_model, clip_processor, device, crop_pil)
            label, score = classify(emb, ref_embs, ref_labels, threshold)

            if label:
                raw_detections.append((score, label, (x1, y1, x2, y2)))
                if save_crops:
                    fname = f"raw_{total_windows:05d}_{label}_{score:.2f}.png"
                    crop_pil.save(crops_dir / fname)

    print(f"   Total windows evaluated : {total_windows}")
    print(f"   Windows above threshold : {len(raw_detections)}")

    # ── Global NMS ────────────────────────────────────────────────────────────
    # Suppress any window that overlaps (IoU > threshold) with a higher-scoring
    # window — across ALL labels, so each physical bag is counted once.
    detections = nms_global(raw_detections, NMS_IOU_THRESH)
    print(f"   After global NMS        : {len(detections)}")

    # ── Count results ─────────────────────────────────────────────────────────
    counts: dict[str, int] = defaultdict(int)
    for _, label, _ in detections:
        counts[label] += 1

    print("\n" + "=" * 60)
    print("🛒  CHIP COUNT RESULTS")
    print("=" * 60)
    if not counts:
        print(f"  Nothing matched above threshold {threshold:.2f}.")
        print(f"  Tip: try --threshold 0.58 or add more reference images.")
    else:
        total = 0
        for label, cnt in sorted(counts.items()):
            bar = "█" * min(cnt, 20)
            print(f"  • {label:<32} {cnt:>3}  {bar}")
            total += cnt
        print("-" * 60)
        print(f"  {'TOTAL':<32} {total:>3}")
    print("=" * 60)

    # ── Annotate and save output image ────────────────────────────────────────
    annotated    = rack_bgr.copy()
    label_colors: dict[str, tuple] = {}
    color_idx = 0

    for score, label, (x1, y1, x2, y2) in detections:
        if label not in label_colors:
            label_colors[label] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
        color = label_colors[label]

        # Translucent fill
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.30, annotated, 0.70, 0, annotated)

        # Solid border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label badge
        text = f"{label[:22]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        by = max(y1, th + 6)
        cv2.rectangle(annotated, (x1, by - th - 4), (x1 + tw + 4, by + 2), color, -1)
        cv2.putText(annotated, text, (x1 + 2, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = rack_path.with_name(rack_path.stem + "_detected.jpg")
    cv2.imwrite(str(out_path), annotated)
    print(f"\n🖼   Annotated image  → {out_path}")
    if save_crops:
        print(f"🗂   Raw crop dumps  → {crops_dir}/")

    return dict(counts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Count chips on a rack: bg removal → sliding window → CLIP."
    )
    parser.add_argument("rack_image",
                        help="Path to the chip rack photo (jpg / png / webp).")
    parser.add_argument("--refs",
                        default=str(DEFAULT_REFS_DIR),
                        help="Directory of reference chip images (default: ./chip_refs).")
    parser.add_argument("--threshold",
                        type=float, default=CLIP_THRESHOLD,
                        help=f"CLIP similarity threshold (default: {CLIP_THRESHOLD}).")
    parser.add_argument("--save-crops",
                        action="store_true",
                        help="Save every matched sliding-window crop for inspection.")
    args = parser.parse_args()

    rack = Path(args.rack_image)
    refs = Path(args.refs)
    if not rack.exists(): print(f"❌  Not found: {rack}"); sys.exit(1)
    if not refs.exists(): print(f"❌  Not found: {refs}"); sys.exit(1)

    return detect_chips(rack, refs, threshold=args.threshold,
                        save_crops=args.save_crops)


if __name__ == "__main__":
    main()