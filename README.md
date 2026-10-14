---
title: Chips Counter
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 5050
---
# 🛒 AI Chip Rack Counter

A high-precision, computer vision-based chip inventory system. This tool automates the process of counting chip bags in retail racks by detecting every bag with an open-vocabulary detector (OWLv2) and identifying each one with CLIP plus colour matching against a library of reference images.

![UI Preview](https://img.shields.io/badge/Interface-Modern_Web-orange?style=flat-square)
![Core](https://img.shields.io/badge/Engine-Python_3.10+-blue?style=flat-square)
![ML](https://img.shields.io/badge/ML-OWLv2_+_CLIP-green?style=flat-square)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Gauurab/chips-counter)

## 🌐 Live Demo

Try out the application directly in your browser without any setup:
[**Gauurab/chips-counter on Hugging Face Spaces**](https://huggingface.co/spaces/Gauurab/chips-counter)

## 🌟 Features

- **Bag Detection**: `google/owlv2-base-patch16-ensemble` finds each individual bag, with no fixed grid or bag-size guess.
- **Smart Box Filtering**: Drops row-spanning boxes, fragments and duplicates; `rembg` removes anything off the rack (drink cups, store clutter).
- **CLIP + Colour Classification**: `openai/clip-vit-base-patch32` similarity fused with HSV colour histograms, so same-brand flavours (Doritos Nacho vs. Cool Ranch, Sun Chips Garden Salsa vs. Harvest Cheddar) are told apart.
- **Shelf-Row Consistency**: Neighbouring bags on a shelf vote for each other's label, fixing isolated mistakes.
- **Unknown Detection**: Bags that don't match any reference are counted as `Unknown` instead of being forced into the wrong brand.
- **Multi-Rack Wizard**: Support for inventorying multiple racks with "Front" and "Back" view aggregation.
- **Stack Depth Logic**: Calculate total stock by multiplying visible bags by their stack depth.

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 2. Prepare Reference Images
Place clean, front-facing product shots (ideally on a white background) in the `chip_refs/` directory. The filename (without extension) is used as the label (e.g., `lays-classic.png` → `Lays Classic`).

To give one product several reference shots (e.g. old and new packaging), put them in a sub-folder named after the product: `chip_refs/lays-classic/front.png`, `chip_refs/lays-classic/new-design.png`.

> Use studio/product photos, not crops from shelf photos. Low-resolution shelf crops share the rack photo's blur and lighting, so CLIP matches *every* bag to them.

### 3. Run the Server
Start the Flask backend:

```bash
python3 server.py
```

### 4. Open the Interface
Navigate to `http://localhost:5050` in your browser.

## 🛠 How it Works

1. **Detection**: OWLv2 is prompted with "a bag of chips" and returns a box per bag.
2. **Filtering**: Boxes far from the median bag size, wider than tall, mostly inside another box, or off the rack foreground (`rembg`) are dropped.
3. **Identification**: Each bag crop (plus two test-time crops) is compared to augmented `chip_refs` views with CLIP; this is fused with a colour-histogram score. Same-row neighbours smooth the result, and bags scoring below `CLIP_THRESHOLD` become `Unknown`.
4. **Aggregation**: If you provided a "Front" and "Back" view, the counts are summed and multiplied by the **Stack Depth** provided in the UI.

> **Tip:** if a product shows up as `Unknown` (or as a look-alike brand), add a reference image for it to `chip_refs/`. Accuracy depends heavily on having a reference for every product on the rack.

## ⚙️ Continuous Deployment

This project uses **GitHub Actions** (`.github/workflows/sync_to_huggingface.yml`) for seamless continuous deployment.
Whenever new code is pushed to the `main` branch, the workflow automatically synchronizes the latest changes to the live Hugging Face Space using the `huggingface_hub` Python package. This pipeline easily handles updating large model files directly via the Hugging Face API, completely bypassing Git LFS size limits.

## 📁 Project Structure

- `detection.py`: Core CV pipeline (OWLv2 detection + filtering + CLIP/colour classification).
- `server.py`: Flask backend serving the API and web UI.
- `index.html`: Modern, multi-step wizard interface.
- `chip_refs/`: Library of reference images for classification.
- `uploads/`: Temporary storage for processed images.

## ⚖️ License
MIT
