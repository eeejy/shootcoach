"""Train the bullet-hole detector (Ultralytics YOLO) on the synthetic (or real) dataset.

    python scripts/train_detector.py --data data/synth/data.yaml --epochs 60
Outputs runs/detect/<name>/weights/best.pt; copy it to models/hole_detector.pt.
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synth/data.yaml")
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=832)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="mps")
ap.add_argument("--name", default="holes")
args = ap.parse_args()

YOLO(args.model).train(
    data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
    name=args.name, patience=15, close_mosaic=10, workers=4, degrees=3.0, fliplr=0.5, flipud=0.5,
    project="runs/detect", exist_ok=True, plots=True,
)
