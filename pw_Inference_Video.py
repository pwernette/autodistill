import cv2
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np


def draw_segmentation_masks(frame, results):
    """
    Draw segmentation masks if present.
    """
    if not hasattr(results, "masks") or results.masks is None:
        return frame

    masks = results.masks.data.cpu().numpy()  # [N, H, W]

    for mask in masks:
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)  # green channel
        frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    return frame


def process_video(
    input_video_path,
    output_video_path,
    model_path,
    device="cpu",
    conf_threshold=0.25
):
    # Load model
    model = YOLO(model_path)
    model.to(device)

    # AUTO-DETECT MODEL TYPE
    model_type = model.task  # "detect", "segment", "pose", ...

    print(f"🔍 Detected model type: {model_type}")

    seg_enabled = (model_type == "segment")

    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = width * 2
    out_height = height

    # Ensure dimensions are even (many codecs require even dims)
    if out_width % 2 != 0:
        out_width -= 1
    if out_height % 2 != 0:
        out_height -= 1

    # Some codecs (mpeg4/mp4v) error if dimensions are too large for the container or codec.
    # If concatenated width is very large, scale both frames down to a safe maximum.
    MAX_OUT_WIDTH = 4096  # conservative upper bound; adjust if you know your codec supports more
    need_resize = False
    if out_width > MAX_OUT_WIDTH:
        scale = MAX_OUT_WIDTH / float(out_width)
        new_out_w = int(out_width * scale)
        new_out_h = int(out_height * scale)
        # ensure even
        new_out_w -= new_out_w % 2
        new_out_h -= new_out_h % 2
        need_resize = True
    else:
        new_out_w = out_width
        new_out_h = out_height

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (new_out_w, new_out_h)
    )

    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        # pass verbose=False/show=False to suppress ultralytics per-inference logs
        results = model(frame, conf=conf_threshold, device=device, verbose=False, show=False)[0]

        annotated = frame.copy()

        # Segmentation (automatically triggered)
        if seg_enabled:
            annotated = draw_segmentation_masks(annotated, results)

        # Bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw a thicker red bounding box and label for visibility
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 3
            )

        # Optionally resize frames if output dimensions were scaled
        if need_resize:
            target_single_w = new_out_w // 2
            target_single_h = new_out_h
            frame_resized = cv2.resize(frame, (target_single_w, target_single_h), interpolation=cv2.INTER_AREA)
            annotated_resized = cv2.resize(annotated, (target_single_w, target_single_h), interpolation=cv2.INTER_AREA)
            combined = cv2.hconcat([frame_resized, annotated_resized])
        else:
            combined = cv2.hconcat([frame, annotated])

        out.write(combined)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

    print(f"✔ Output saved to: {output_video_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11 video inference with auto model type detection."
    )

    parser.add_argument("--input",  required=True, help="Input video file.")
    parser.add_argument("--model",  required=True, help="YOLO11 model (.pt).")
    parser.add_argument("--output", required=True, help="Output video path.")

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (cpu or cuda). Default: cpu"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf
    )
