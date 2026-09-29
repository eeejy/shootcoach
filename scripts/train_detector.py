"""Train the bullet-hole detector (Ultralytics YOLO) on the synthetic (or real) dataset.

    python scripts/train_detector.py --data data/synth/data.yaml --epochs 60
Outputs runs/detect/<name>/weights/best.pt; copy it to models/hole_detector.pt.
"""
import argparse
import os

from ultralytics import YOLO

from shootcoach.device import best_device, model_path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synth/data.yaml")
ap.add_argument("--model", default=model_path("yolo11n.pt"), help="이어 학습하려면 models/hole_detector.pt")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=832)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None, help="기본: 자동 (NVIDIA → CUDA, 맥 → MPS, 그 외 CPU)")
ap.add_argument("--name", default="holes")
args = ap.parse_args()
args.device = best_device() if args.device is None else args.device
print("device:", args.device)

YOLO(args.model).train(
    data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
    name=args.name, patience=15, close_mosaic=10, workers=min(4, os.cpu_count() or 1), degrees=3.0, fliplr=0.5, flipud=0.5,
    project="runs/detect", exist_ok=True, plots=True,
)
