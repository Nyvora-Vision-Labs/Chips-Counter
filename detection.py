"""
Chip Rack Detector — 100% Local, No API Key
Usage:
    python3 detection.py chips-rack-1.png
    python3 detection.py chips-rack-1.png --refs ./chip_refs --threshold 0.75

Requirements:
    pip install torch torchvision transformers Pillow opencv-python-headless
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ── Config ──────────────────────────────────────────────────────────────────────
DEFAULT_REFS_DIR  = Path(__file__).parent / "chip_refs"
SUPPORTED_EXTS    = {".png", ".jpg", ".jpeg", ".webp"}
CLIP_MODEL_ID     = "openai/clip-vit-base-patch32"
CROP_SIZES        = [200, 280, 360]
STRIDE_RATIO      = 0.4
MATCH_THRESHOLD   = 0.60
NMS_IOU_THRESHOLD = 0.30


# ── Helpers ─────────────────────────────────────────────────────────────────────
def pretty_name(filename: str) -> str:
    return Path(filename).stem.replace("-", " ").replace("_", " ").title()


def encode_image(model, processor, device, pil_img: Image.Image) -> torch.Tensor:
    """Always returns a plain [1, D] float tensor."""
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]           # guaranteed key
    # Use the vision model directly — always returns a tensor
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
        # pooler_output is the [CLS] embedding → shape [1, 768]
        emb = vision_out.pooler_output              # plain Tensor, no ambiguity
    # Project to CLIP's shared embedding space
    emb = model.visual_projection(emb)              # [1, 512]
    return emb


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return a @ b.T                                  # [N_a, N_b]


def iou(b1, b2) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union else 0.0


def nms(detections, iou_thresh):
    detections = sorted(detections, key=lambda d: d[0], reverse=True)
    kept = []
    for det in detections:
        if all(iou(det[2], k[2]) < iou_thresh for k in kept):
            kept.append(det)
    return kept


def sliding_crops(image_np, sizes, stride_ratio):
    h, w = image_np.shape[:2]
    for size in sizes:
        stride = max(1, int(size * stride_ratio))
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                crop = image_np[y:y+size, x:x+size]
                pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                yield x, y, x+size, y+size, pil


# ── Core ────────────────────────────────────────────────────────────────────────
def detect_chips(rack_path: Path, refs_dir: Path, threshold: float):

    print(f"\n🤖  Loading CLIP ({CLIP_MODEL_ID}) ...")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")
    model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    # Encode references
    ref_paths = sorted(p for p in refs_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not ref_paths:
        print(f"❌  No images in {refs_dir}"); sys.exit(1)

    print("📚  Encoding reference images ...")
    ref_labels, ref_embs = [], []
    for p in ref_paths:
        emb = encode_image(model, processor, device, Image.open(p).convert("RGB"))
        ref_labels.append(pretty_name(p.name))
        ref_embs.append(emb)
        print(f"   ✓ {pretty_name(p.name)}")
    ref_embs = torch.cat(ref_embs, dim=0)           # [N_refs, 512]

    # Load rack
    rack_np = cv2.imread(str(rack_path))
    if rack_np is None:
        print(f"❌  Cannot read {rack_path}"); sys.exit(1)
    print(f"\n🔍  Scanning {rack_path.name}  ({rack_np.shape[1]}x{rack_np.shape[0]}) ...")

    # Slide & match
    detections = []
    n = 0
    for x1, y1, x2, y2, crop_pil in sliding_crops(rack_np, CROP_SIZES, STRIDE_RATIO):
        emb   = encode_image(model, processor, device, crop_pil)
        sims  = cosine_sim(emb, ref_embs)[0]        # [N_refs]
        idx   = sims.argmax().item()
        score = sims[idx].item()
        if score >= threshold:
            detections.append((score, ref_labels[idx], (x1, y1, x2, y2)))
        n += 1
        if n % 200 == 0:
            print(f"   ... {n} crops done")

    print(f"   Total crops: {n}")

    detections = nms(detections, NMS_IOU_THRESHOLD)

    # Results
    print("\n" + "="*60)
    print("🛒  CHIPS DETECTED ON RACK")
    print("="*60)
    if not detections:
        print(f"  Nothing above threshold {threshold}. Try --threshold 0.70")
    else:
        counts = {}
        for score, label, box in detections:
            counts.setdefault(label, []).append(score)
        for label, scores in sorted(counts.items()):
            facings = len(scores)
            avg     = sum(scores) / facings
            print(f"  • {label:<30}  {facings:>2} facing(s)  [sim {avg:.2f}]  {'█'*min(facings,10)}")
    print("="*60)

    # Annotate and save
    annotated  = rack_np.copy()
    colors     = [(0,200,0),(0,100,255),(255,100,0),(150,0,255),(0,200,200)]
    label_list = list({d[1] for d in detections})
    for score, label, (x1,y1,x2,y2) in detections:
        c = colors[label_list.index(label) % len(colors)]
        cv2.rectangle(annotated, (x1,y1), (x2,y2), c, 2)
        cv2.putText(annotated, label[:20], (x1, max(y1-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    out = rack_path.with_name(rack_path.stem + "_detected.jpg")
    cv2.imwrite(str(out), annotated)
    print(f"\n🖼   Saved → {out}")


# ── Entry ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("rack_image")
    p.add_argument("--refs",      default=str(DEFAULT_REFS_DIR))
    p.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    args = p.parse_args()

    rack = Path(args.rack_image)
    refs = Path(args.refs)
    if not rack.exists(): print(f"❌  {rack} not found"); sys.exit(1)
    if not refs.exists(): print(f"❌  {refs} not found"); sys.exit(1)

    detect_chips(rack, refs, args.threshold)

if __name__ == "__main__":
    main()