#!/usr/bin/env python3
"""
sam3_auto_annotate.py
Generate training annotations using SAM3 (Segment Anything Model 3) Automatic Mask Generator.
This script does NOT depend on GroundingDINO or Autodistill. It uses SAM's automatic mask generation
and simple area filtering to create either YOLO detection labels (from mask bboxes) or per-image
semantic masks suitable for segmentation training.

Usage examples:
  python sam3_auto_annotate.py --input_dir ./images --output_root ./annotations --sam_checkpoint /path/to/sam3.pt --task detect
  python sam3_auto_annotate.py --input_dir ./images --output_root ./annotations --sam_checkpoint /path/to/sam3.pt --task segment --class_id 1

Notes:
- This script attempts to import the common SAM/SegmentAnything Python API (Segment Anything repo).
  If the import fails, install segment-anything or adjust imports to your local SAM3 repo API.
- Filtering is done by mask area fraction relative to image area. You can also filter by
  SAM-provided 'stability_score' or 'predicted_iou' if available.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

# Try to import SAM/segment-anything API. Adjust these imports if your SAM3 repo exposes a different surface.
try:
    # many SAM releases expose SamModel and SamAutomaticMaskGenerator in segment_anything
    from segment_anything import SamModel, SamAutomaticMaskGenerator
except Exception:
    SamModel = None
    SamAutomaticMaskGenerator = None


def init_sam(checkpoint_path, device="cuda"):
    if SamModel is None or SamAutomaticMaskGenerator is None:
        raise RuntimeError("Could not import Segment Anything API. Install the repo or adapt the imports to your SAM3 implementation.")
    sam = SamModel(checkpoint_path)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return sam, mask_generator


def filter_masks_by_area(masks_info, H, W, min_area_frac, max_area_frac, score_key=None, score_thresh=None):
    keep = []
    for m in masks_info:
        # m expected to be a dict with keys like 'segmentation' (bool mask HxW), 'area', 'bbox', optionally 'predicted_iou' or 'stability_score'
        area = float(m.get("area", np.array(m["segmentation"]).sum()))
        area_frac = area / float(H * W)
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        if score_key and score_thresh is not None:
            score = m.get(score_key, None)
            if score is None or score < score_thresh:
                continue
        keep.append(m)
    return keep


def masks_to_semantic_mask(masks_info, H, W, class_id=1):
    sem = np.zeros((H, W), dtype=np.uint8)
    # last mask wins
    for m in masks_info:
        seg = np.array(m["segmentation"]).astype(bool)
        if seg.shape != (H, W) and seg.size == H * W:
            seg = seg.reshape((H, W))
        sem[seg] = int(class_id)
    return sem


def masks_to_yolo_labels(masks_info, H, W, class_id=0):
    # produces list of strings 'class cx cy w h' normalized
    labels = []
    for m in masks_info:
        # bbox expected [x1, y1, x2, y2]
        bbox = m.get("bbox", None)
        if bbox is None:
            # compute bbox from segmentation
            seg = np.array(m["segmentation"]).astype(bool)
            ys, xs = np.where(seg)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
        else:
            x1, y1, x2, y2 = bbox
        # convert to YOLO normalized cx cy w h
        bw = float(W)
        bh = float(H)
        cx = ((x1 + x2) / 2.0) / bw
        cy = ((y1 + y2) / 2.0) / bh
        w = (x2 - x1) / bw
        h = (y2 - y1) / bh
        labels.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return labels


def process_image(path, mask_generator, task, out_masks_dir, out_labels_dir, class_id, min_area_frac, max_area_frac, score_key=None, score_thresh=None):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    H, W = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # generate masks
    masks_info = mask_generator.generate(img_rgb)

    kept = filter_masks_by_area(masks_info, H, W, min_area_frac, max_area_frac, score_key=score_key, score_thresh=score_thresh)

    out_meta = {"image": str(path.resolve()), "kept": len(kept)}

    stem = Path(path).stem
    if task == "segment":
        sem = masks_to_semantic_mask(kept, H, W, class_id=class_id)
        out_path = Path(out_masks_dir) / (stem + ".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sem).save(out_path)
        out_meta["mask"] = str(out_path.resolve())
    else:
        # detect -> YOLO labels
        labels = masks_to_yolo_labels(kept, H, W, class_id=class_id)
        out_path = Path(out_labels_dir) / (stem + ".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write("\n".join(labels))
        # copy image into same images dir
        out_img_path = Path(out_labels_dir).parent / "images" / Path(path).name
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_img_path.exists():
            shutil.copyfile(path, out_img_path)
        out_meta["label"] = str(out_path.resolve())
        out_meta["image_out"] = str(out_img_path.resolve())

    return out_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", choices=["detect","segment"], default="segment")
    p.add_argument("--class_id", type=int, default=1, help="Numeric class id for semantic masks or YOLO labels")
    p.add_argument("--class_name", default="object")
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_area_frac", type=float, default=0.0005)
    p.add_argument("--max_area_frac", type=float, default=0.5)
    p.add_argument("--score_key", type=str, default=None, help="Optional mask score key (e.g. 'stability_score' or 'predicted_iou')")
    p.add_argument("--score_thresh", type=float, default=None)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    if len(images) == 0:
        print("No images found in input_dir")
        return

    # init sam
    sam, mask_generator = init_sam(args.sam_checkpoint, device=args.device)

    # output dirs
    if args.task == "segment":
        masks_out = output_root / "masks"
        masks_out.mkdir(parents=True, exist_ok=True)
    else:
        labels_out = output_root / "yolo" / "annotations"
        labels_out.mkdir(parents=True, exist_ok=True)
        (labels_out.parent / "images").mkdir(parents=True, exist_ok=True)

    # split
    import random
    random.seed(args.seed)
    random.shuffle(images)
    split = int(len(images) * args.train_frac)
    train = images[:split]
    val = images[split:]

    manifest = []
    for subset, imgs, subdir in [("train", train, None), ("val", val, None)]:
        for im in tqdm(imgs, desc=f"Processing {subset}"):
            if args.task == "segment":
                meta = process_image(im, mask_generator, "segment", masks_out if args.task=="segment" else None, None, args.class_id, args.min_area_frac, args.max_area_frac, score_key=args.score_key, score_thresh=args.score_thresh)
            else:
                meta = process_image(im, mask_generator, "detect", None, labels_out, args.class_id, args.min_area_frac, args.max_area_frac, score_key=args.score_key, score_thresh=args.score_thresh)
            meta["split"] = subset
            manifest.append(meta)

    # write manifest and id2label
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    id2label = {"0": "background", str(int(args.class_id)): args.class_name}
    with open(meta_dir / "id2label.json", "w") as fh:
        json.dump(id2label, fh, indent=2)

    print("Done. Output:", output_root)


if __name__ == "__main__":
    main()
