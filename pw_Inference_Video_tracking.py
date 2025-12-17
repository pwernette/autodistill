'''
Example usage:

python sam3_auto_annotate.py --input_dir /mnt/d/sfm_greeen_2025harvest/20250926_oxley/for_annotation --task detect --workers 24 --device cuda --min_area_frac 1e-6 --max_area_frac 0.4 --score_thresh 0.5 --dedupe_iou 0.3
python sam3_auto_annotate.py --input_dir /mnt/d/sfm_greeen_2025harvest/20250926_oxley/for_annotation --task segment --workers 24 --device cuda --min_area_frac 1e-6 --max_area_frac 0.4 --score_thresh 0.5 --dedupe_iou 0.3

python sam3_auto_annotate.py --input_dir /mnt/d/sfm_greeen_2025harvest/GoPRo/20251004/low_3 --task detect --workers 8 --device cuda --min_area_frac 1e-6 --max_area_frac 0.6 --score_thresh 0.2 --dedupe_iou 0.5
python sam3_auto_annotate.py --input_dir /mnt/d/sfm_greeen_2025harvest/GoPRo/20251004/low_3 --task segment --workers 8 --device cuda --min_area_frac 1e-6 --max_area_frac 0.6 --score_thresh 0.2 --dedupe_iou 0.5

'''
import cv2, sys, argparse
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np


def draw_segmentation_masks(frame, masks, ids=None):
    """
    Draw segmentation masks on the frame.
    Optionally draw instance IDs if provided.
    """
    for idx, mask in enumerate(masks):
        overlay = np.zeros_like(frame, dtype=np.uint8)
        overlay[:, :, 1] = (mask * 255).astype(np.uint8)  # green mask
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)

        if ids is not None:
            # Draw ID at centroid
            y, x = np.where(mask)
            if len(y) > 0 and len(x) > 0:
                cy, cx = int(np.mean(y)), int(np.mean(x))
                cv2.putText(frame, f"ID:{ids[idx]}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def assign_detection_ids(boxes, active_tracks, next_instance_id, iou_threshold=0.5, dist_threshold=100):
    """
    Assign instance IDs to detection boxes based on IoU matching with a centroid-distance fallback.
    active_tracks: dict mapping track_id -> box
    Returns: ids (list), new_active_tracks (dict id->box), next_instance_id
    """
    new_tracks = {}
    ids = []

    # Precompute centroids for active tracks
    active_centroids = {tid: ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for tid, b in active_tracks.items()}

    for box in boxes:
        best_id = None
        best_iou = 0
        for old_id, old_box in active_tracks.items():
            # compute IoU
            x1 = max(box[0], old_box[0])
            y1 = max(box[1], old_box[1])
            x2 = min(box[2], old_box[2])
            y2 = min(box[3], old_box[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (box[2]-box[0]) * (box[3]-box[1])
            a2 = (old_box[2]-old_box[0]) * (old_box[3]-old_box[1])
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_id = old_id

        if best_id is None:
            # fallback: centroid-distance matching
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            best_dist = None
            for tid, (acx, acy) in active_centroids.items():
                dist = np.hypot(acx - cx, acy - cy)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_dist is not None and best_dist > dist_threshold:
                best_id = None

        if best_id is None:
            best_id = next_instance_id
            next_instance_id += 1

        new_tracks[best_id] = box
        ids.append(best_id)

    return ids, new_tracks, next_instance_id


def assign_segmentation_ids(masks, active_tracks, next_instance_id, iou_threshold=0.5, dist_threshold=100):
    """
    Assign instance IDs to segmentation masks based on IoU matching with centroid-distance fallback.
    active_tracks: dict mapping track_id -> mask
    Returns: ids (list), new_active_tracks (dict id->mask), next_instance_id
    """
    new_tracks = {}
    ids = []

    # precompute centroids for active masks
    active_centroids = {}
    for tid, old_mask in active_tracks.items():
        ys, xs = np.where(old_mask)
        if len(xs) > 0:
            active_centroids[tid] = (np.mean(xs), np.mean(ys))

    for mask in masks:
        best_id = None
        best_iou = 0
        for old_id, old_mask in active_tracks.items():
            intersection = np.logical_and(mask, old_mask).sum()
            union = np.logical_or(mask, old_mask).sum()
            iou = intersection / union if union > 0 else 0
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_id = old_id

        if best_id is None:
            # fallback: centroid distance
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx = np.mean(xs)
                cy = np.mean(ys)
                best_dist = None
                for tid, (acx, acy) in active_centroids.items():
                    dist = np.hypot(acx - cx, acy - cy)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_id = tid
                if best_dist is not None and best_dist > dist_threshold:
                    best_id = None
            else:
                best_id = None

        if best_id is None:
            best_id = next_instance_id
            next_instance_id += 1

        new_tracks[best_id] = mask
        ids.append(best_id)

    return ids, new_tracks, next_instance_id


def bbox_iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0


def nms_keep_indices(boxes, scores, iou_thresh):
    """Greedy NMS. boxes: (N,4) numpy, scores: (N,) numpy."""
    if len(boxes) == 0:
        return []
    idxs = np.argsort(-scores)
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        if len(idxs) == 1:
            break
        others = idxs[1:]
        rem = []
        for j in others:
            if bbox_iou(boxes[i], boxes[j]) <= iou_thresh:
                rem.append(j)
        idxs = np.array(rem)
    return keep


def mask_iou(maskA, maskB):
    """Compute IoU between two boolean masks."""
    a = maskA.astype(bool)
    b = maskB.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def mask_nms_keep_indices(masks, scores, iou_thresh):
    """Greedy NMS for masks using mask IoU."""
    if len(masks) == 0:
        return []
    # ensure masks is list/array of 2D boolean arrays
    idxs = np.argsort(-scores)
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        if len(idxs) == 1:
            break
        others = idxs[1:]
        rem = []
        for j in others:
            if mask_iou(masks[i], masks[j]) <= iou_thresh:
                rem.append(j)
        idxs = np.array(rem)
    return keep


def process_video(input_path, output_path, model_path, device="cpu", conf_threshold=0.25, use_tracking=False, side_by_side=False, nms_iou=0.0, debug=False):
    # Load YOLO11 model
    model = YOLO(model_path)
    model.to(device)

    model_type = model.task  # "detect" or "segment"
    print(f"🔍 Detected model type: {model_type}")

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video (side-by-side or annotated-only)
    out_width = width * (2 if side_by_side else 1)
    out_height = height

    # Ensure dimensions are even
    if out_width % 2 != 0:
        out_width -= 1
    if out_height % 2 != 0:
        out_height -= 1

    # Cap output width to a conservative maximum for codecs like mpeg4/mp4v
    MAX_OUT_WIDTH = 4096
    need_resize = False
    if out_width > MAX_OUT_WIDTH:
        scale = MAX_OUT_WIDTH / float(out_width)
        new_out_w = int(out_width * scale)
        new_out_h = int(out_height * scale)
        new_out_w -= new_out_w % 2
        new_out_h -= new_out_h % 2
        need_resize = True
    else:
        new_out_w = out_width
        new_out_h = out_height

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (new_out_w, new_out_h)
    )

    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

    # Tracking state
    next_instance_id = 0
    active_tracks = {}

    # Counting state
    total_count = 0
    counted_ids = set()  # for tracked instances
    # For crossing-based counting we keep last-x for tracked IDs and simple short-lived untracked tracks
    tracked_last_x = {}  # id -> last known centroid x
    untracked_tracks = []  # [{'id', 'cx','cy','counted','frames'}]
    next_untracked_id = 0
    counted_untracked_ids = set()
    UNTRACKED_MIN_DIST = 30  # px, distance threshold to associate untracked detections across frames
    UNTRACKED_MAX_MISSES = 30  # frames before dropping an untracked ephemeral track

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection/tracking
        if use_tracking:
            results = model.track(frame, persist=True, conf=conf_threshold, device=device, verbose=False, show=False)[0]
        else:
            results = model(frame, conf=conf_threshold, device=device, verbose=False, show=False)[0]

        # Extract per-detection arrays to optionally run NMS
        all_boxes = [box.xyxy[0].cpu().numpy() for box in results.boxes]
        all_scores = [float(box.conf[0]) for box in results.boxes]
        all_cls = [int(box.cls[0]) for box in results.boxes]

        # If segmentation model, read masks now so we can run mask-NMS when requested
        if model_type == "segment":
            if hasattr(results, "masks") and results.masks is not None:
                masks_all = results.masks.data.cpu().numpy()
                # if masks have extra channel dim, squeeze
                if masks_all.ndim == 4 and masks_all.shape[1] == 1:
                    masks_all = masks_all[:, 0, :, :]
            else:
                masks_all = []
        else:
            masks_all = []

        # Choose NMS method: for segment use mask IoU NMS when requested, otherwise bbox NMS
        if model_type == "segment" and nms_iou and nms_iou > 0 and len(masks_all) > 0:
            keep_idx = mask_nms_keep_indices(masks_all, np.array(all_scores), float(nms_iou))
        else:
            if nms_iou and nms_iou > 0 and len(all_boxes) > 0:
                keep_idx = nms_keep_indices(np.array(all_boxes), np.array(all_scores), float(nms_iou))
            else:
                keep_idx = list(range(len(all_boxes)))

        # Keep lists filtered by keep_idx for use below
        filtered_boxes = [all_boxes[i] for i in keep_idx]
        filtered_scores = [all_scores[i] for i in keep_idx]
        filtered_cls = [all_cls[i] for i in keep_idx]

        annotated = frame.copy()

        # Define counting band (vertical narrow band around center, default 10% of width)
        band_width = max(1, int(width * 0.10))
        band_center = int(width * 0.5)
        band_left = max(0, band_center - band_width // 2)
        band_right = min(width - 1, band_left + band_width)
        # draw translucent filled purple band plus outline for prominence
        overlay = annotated.copy()
        cv2.rectangle(overlay, (band_left, 0), (band_right, height), (128, 0, 128), -1)
        annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)
        cv2.rectangle(annotated, (band_left, 0), (band_right, height), (128, 0, 128), 4)

        # Collect centroids and ids for unified crossing logic
        centroids = []  # list of (cx, cy)
        ids = []

        if model_type == "segment":
            # masks: list of HxW bool arrays
            if hasattr(results, "masks") and results.masks is not None:
                masks_all = results.masks.data.cpu().numpy()
            else:
                masks_all = []

            # apply NMS filtering to masks + boxes
            if len(masks_all) > 0 and len(filtered_boxes) > 0:
                masks = [masks_all[i] for i in keep_idx]
            else:
                masks = []

            # Use simple IoU-based assignment every frame to maintain stable IDs even without external tracker
            if len(masks) > 0:
                ids, active_tracks, next_instance_id = assign_segmentation_ids(
                    masks, active_tracks, next_instance_id
                )
            else:
                ids = [None] * len(masks)

            for mi, mask in enumerate(masks):
                y, x = np.where(mask)
                if len(y) == 0 or len(x) == 0:
                    centroids.append((None, None))
                    continue
                cy, cx = int(np.mean(y)), int(np.mean(x))
                centroids.append((cx, cy))

            annotated = draw_segmentation_masks(annotated, masks, ids)

            # Draw bounding boxes for segment objects (use filtered boxes)
            for out_idx, keep_i in enumerate(keep_idx):
                # ensure mapping exists
                if out_idx >= len(filtered_boxes):
                    break
                x1, y1, x2, y2 = map(int, filtered_boxes[out_idx])
                conf = float(filtered_scores[out_idx])
                cls = int(filtered_cls[out_idx])
                tid = ids[out_idx] if out_idx < len(ids) else None
                label = f"{model.names[cls]} {conf:.2f}"
                label = f"ID:{tid} {label}" if tid is not None else label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, label, (x1, max(y1-10,20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 3)

        elif model_type == "detect":
            # Use filtered boxes for detection branch
            boxes = filtered_boxes
            # Use simple IoU-based assignment every frame to maintain stable IDs even without external tracker
            if len(boxes) > 0:
                ids, active_tracks, next_instance_id = assign_detection_ids(
                    boxes, active_tracks, next_instance_id
                )
            else:
                ids = [None] * len(boxes)

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centroids.append((cx, cy))

                conf = float(filtered_scores[idx])
                cls = int(filtered_cls[idx])
                tid = ids[idx] if idx < len(ids) else None
                label = f"{model.names[cls]} {conf:.2f}"
                label = f"ID:{tid} {label}" if tid is not None else label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), 3)
                cv2.putText(annotated, label, (x1, max(y1-10,20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 3)

        # Debug: print raw counts
        if debug:
            print(f"[DEBUG] frame={int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames} boxes_before_nms={len(all_boxes)} nms_kept={len(keep_idx)} masks_all={len(masks_all) if 'masks_all' in locals() else 0}")

        # Unified crossing detection: require an object to move from one side of the vertical line to the other
        for idx, (cx, cy) in enumerate(centroids):
            if cx is None:
                continue
            obj_id = ids[idx] if idx < len(ids) else None
            if obj_id is not None:
                # tracked: use tracked_last_x
                prev_x = tracked_last_x.get(obj_id)
                if prev_x is not None:
                    # count when object ENTERS the band (was outside, now inside)
                    crossed = ((prev_x < band_left and band_left <= cx <= band_right) or (prev_x > band_right and band_left <= cx <= band_right))
                    if crossed and obj_id not in counted_ids:
                        counted_ids.add(obj_id)
                        total_count += 1
                        if debug:
                            print(f"[DEBUG] COUNT tracked id={obj_id} prev_x={prev_x} cx={cx} band=({band_left},{band_right}) total={total_count}")
                tracked_last_x[obj_id] = cx
            else:
                # untracked: match to short-lived untracked_tracks by distance
                matched = False
                for t in untracked_tracks:
                    # match to any existing untracked track by proximity
                    dist = np.hypot(t['cx'] - cx, t['cy'] - cy)
                    if dist < UNTRACKED_MIN_DIST:
                        prev_x = t['cx']
                        # count when untracked object ENTERS the band
                        crossed = ((prev_x < band_left and band_left <= cx <= band_right) or (prev_x > band_right and band_left <= cx <= band_right))
                        if crossed and (not t.get('counted', False)) and (t['id'] not in counted_untracked_ids):
                            t['counted'] = True
                            counted_untracked_ids.add(t['id'])
                            total_count += 1
                            if debug:
                                print(f"[DEBUG] COUNT untracked id={t['id']} prev_x={prev_x} cx={cx} band=({band_left},{band_right}) total={total_count}")
                        # update track
                        t['cx'] = cx
                        t['cy'] = cy
                        t['frames'] = 0
                        matched = True
                        break
                if not matched:
                    # create a new untracked short-lived track
                    # Before creating, ensure we don't already have a very-close track (safety)
                    close_existing = False
                    for t in untracked_tracks:
                        if np.hypot(t['cx'] - cx, t['cy'] - cy) < (UNTRACKED_MIN_DIST / 2):
                            # merge into that track
                            t['cx'] = cx
                            t['cy'] = cy
                            t['frames'] = 0
                            close_existing = True
                            break
                    if not close_existing:
                        untracked_tracks.append({'id': next_untracked_id, 'cx': cx, 'cy': cy, 'counted': False, 'frames': 0})
                        next_untracked_id += 1

        # Debug: show tracked_last_x and untracked_tracks
        if debug:
            print(f"[DEBUG] tracked_last_x keys={list(tracked_last_x.keys())} untracked_count={len(untracked_tracks)} total_count={total_count}")

        # age and prune untracked ephemeral tracks
        for t in untracked_tracks:
            t['frames'] = t.get('frames', 0) + 1
        untracked_tracks = [t for t in untracked_tracks if t['frames'] <= UNTRACKED_MAX_MISSES]

        # Display total counter in upper-right (larger and more visible)
        counter_text = f"Total: {total_count}"
        font_scale = 4.0
        font_thickness = 8
        (tw, th), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        tx = new_out_w - tw - 20 if 'new_out_w' in locals() else width - tw - 20
        ty = 60
        bg_pad = 12
        cv2.rectangle(annotated, (tx - bg_pad, ty - th - bg_pad), (tx + tw + bg_pad, ty + bg_pad), (0, 0, 0), -1)
        cv2.putText(annotated, counter_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # Side-by-side output; resize if output was scaled
        if side_by_side:
            if need_resize:
                target_single_w = new_out_w // 2
                target_single_h = new_out_h
                frame_resized = cv2.resize(frame, (target_single_w, target_single_h), interpolation=cv2.INTER_AREA)
                annotated_resized = cv2.resize(annotated, (target_single_w, target_single_h), interpolation=cv2.INTER_AREA)
                combined = cv2.hconcat([frame_resized, annotated_resized])
            else:
                combined = cv2.hconcat([frame, annotated])
        else:
            # only export annotated video (default)
            if need_resize:
                annotated_resized = cv2.resize(annotated, (new_out_w, new_out_h), interpolation=cv2.INTER_AREA)
                combined = annotated_resized
            else:
                combined = annotated

        out.write(combined)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"✔ Output saved to: {output_path}")


def get_parser():
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    examples = '''
Example usage:

python sam3_auto_annotate.py --input_dir /mnt/d/sfm_greeen_2025harvest/20250926_oxley/for_annotation --task detect --workers 24 --device cuda --min_area_frac 1e-6 --max_area_frac 0.4 --score_thresh 0.5 --dedupe_iou 0.3
python sam3_auto_annotate.py --input_dir /mnt/d/sfm_greeen_2025harvest/20250926_oxley/for_annotation --task segment --workers 24 --device cuda --min_area_frac 1e-6 --max_area_frac 0.4 --score_thresh 0.5 --dedupe_iou 0.3

python sam3_auto_annotate.py --input_dir /mnt/d/sfm_greeen_2025harvest/GoPRo/20251004/low_3 --task detect --workers 8 --device cuda --min_area_frac 1e-6 --max_area_frac 0.6 --score_thresh 0.2 --dedupe_iou 0.5
python sam3_auto_annotate.py --input_dir /mnt/d/sfm_greeen_2025harvest/GoPRo/20251004/low_3 --task segment --workers 8 --device cuda --min_area_frac 1e-6 --max_area_frac 0.6 --score_thresh 0.2 --dedupe_iou 0.5
'''

    parser = argparse.ArgumentParser(
        description="YOLO11 Video Inference with optional instance tracking",
        formatter_class=CustomFormatter,
        epilog=examples
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--model", required=True, help="YOLO11 .pt model path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="Device: cpu or cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--use-tracking", action="store_true", help="Enable instance-ID tracking across frames")
    parser.add_argument("--side-by-side", dest="side_by_side", action="store_true", help="Export side-by-side original+annotated (default: annotated only)")
    parser.add_argument("--nms-iou", type=float, default=0.0, help="IoU threshold for per-frame NMS (0.0 to disable)")
    parser.add_argument("--debug", action="store_true", help="Print debug information per-frame")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    # if run with no arguments, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        use_tracking=args.use_tracking,
        side_by_side=args.side_by_side,
        nms_iou=args.nms_iou,
        debug=args.debug
    )
