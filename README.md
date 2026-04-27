---
title: Chips Counter
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 5050
---
# 🛒 AI Chip Rack Counter

A high-precision, computer vision-based chip inventory system. This tool automates the process of counting chip bags in retail racks by isolating the rack from the background, segmenting individual bags, and classifying them using OpenAI's CLIP.

![UI Preview](https://img.shields.io/badge/Interface-Modern_Web-orange?style=flat-square)
![Core](https://img.shields.io/badge/Engine-Python_3.10+-blue?style=flat-square)
![ML](https://img.shields.io/badge/ML-CLIP_+_rembg-green?style=flat-square)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Gauurab/chips-counter)

## 🌐 Live Demo

Try out the application directly in your browser without any setup:
[**Gauurab/chips-counter on Hugging Face Spaces**](https://huggingface.co/spaces/Gauurab/chips-counter)

## 🌟 Features

- **Background Removal**: Uses `rembg` to strip away store clutter and focus solely on the rack.
- **Foreground-Aware Segmentation**: Custom sliding-window algorithm that detects individual bags based on rack dimensions.
- **CLIP Classification**: Leverages `openai/clip-vit-base-patch32` for zero-shot classification against a library of reference images.
- **Global NMS (Non-Maximum Suppression)**: Ensures each physical bag is counted only once, even if multiple labels match.
- **Multi-Rack Wizard**: Support for inventorying multiple racks with "Front" and "Back" view aggregation.
- **Stack Depth Logic**: Calculate total stock by multiplying visible bags by their stack depth.

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python 3.10+ installed.

```bash
pip install flask torch torchvision transformers rembg opencv-python numpy pillow
```

### 2. Prepare Reference Images
Place high-quality crops of individual chip bags in the `chip_refs/` directory. The filename (without extension) will be used as the label (e.g., `Lays Classic.jpg`).

### 3. Run the Server
Start the Flask backend:

```bash
python3 server.py
```

### 4. Open the Interface
Navigate to `http://localhost:5050` in your browser.

## 🛠 How it Works

1. **Isolation**: `rembg` generates an alpha mask of the rack.
2. **Scanning**: The system slides windows over the foreground areas. It skips regions that are mostly background.
3. **Identification**: Every valid window is passed to CLIP. CLIP compares the window against the `chip_refs` embeddings.
4. **Aggregation**: The results are cleaned using Global NMS. If you provided a "Front" and "Back" view, the counts are summed and multiplied by the **Stack Depth** provided in the UI.

## ⚙️ Continuous Deployment

This project uses **GitHub Actions** (`.github/workflows/sync_to_huggingface.yml`) for seamless continuous deployment.
Whenever new code is pushed to the `main` branch, the workflow automatically synchronizes the latest changes to the live Hugging Face Space using the `huggingface_hub` Python package. This pipeline easily handles updating large model files directly via the Hugging Face API, completely bypassing Git LFS size limits.

## 📁 Project Structure

- `detection.py`: Core CV pipeline (rembg + sliding window + CLIP + NMS).
- `server.py`: Flask backend serving the API and web UI.
- `index.html`: Modern, multi-step wizard interface.
- `chip_refs/`: Library of reference images for classification.
- `uploads/`: Temporary storage for processed images.

## ⚖️ License
MIT
