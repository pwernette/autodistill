import os
import argparse
from PIL import Image
from tqdm import tqdm

import numpy as np

import supervision as sv
from supervision.detection.utils import *
from ultralytics import YOLO
# from autodistill_yolov8 import YOLOv8

from pw_Auto_Distill import str_to_bool


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def filter_detections(image, annotations, area_thresh_max, conf_thresh):
    """

    :param image:
    :param annotations:
    :param area_thresh_min:
    :param area_thresh_max:
    :param conf_thresh:
    :return annotations:
    """
    # print('\nFiltering detection for {} with area:{}'.format(image, area_thresh))

    # height, width = Image.open(filt_image).size
    # print(image.shape)
    height,width,_ = image.shape


    # Filter by area (minimum and maximum)
    # annotations = annotations[(annotations.box_area / (height * width)) > area_thresh_min]
    annotations = annotations[(annotations.box_area / (height * width)) < area_thresh_max]

    # Filter by confidence
    # print(annotations.confidence)
    annotations = annotations[annotations.confidence >= conf_thresh]

    return annotations

def sanitize_class_ids(detections, fallback_id=0):
    if detections.mask is None:
        detections.class_id = np.array([], dtype=int)
        return

    num_masks = len(detections.mask)
    class_ids = getattr(detections, "class_id", None)

    if class_ids is None or len(class_ids) != num_masks:
        detections.class_id = np.full((num_masks,), fallback_id, dtype=int)
    else:
        # Replace None or invalid entries
        sanitized = []
        for cid in class_ids:
            try:
                val = int(cid)
                sanitized.append(val if val >= 0 else fallback_id)
            except (TypeError, ValueError):
                sanitized.append(fallback_id)
        detections.class_id = np.array(sanitized, dtype=int)


# def process_dir(model_dir, input_dir, output_dir, task, 
#                 owrite=False, model_weights_type='best', img_size=None, 
#                 conf=0.3, iou=0.5, area_thresh_min=0.0, area_thresh_max=0.4, conf_thresh=0.1,
#                 use_sahi=False):
#     """
    
#     :param model_dir:
#     :param input_dir:
#     :param output_dir:
#     :param task:
#     :param owrite:
#     :param model_weights_type:
#     :param img_size:
#     :param conf:
#     :param iou:
#     :param use_sahi:
#     :return:
#     """

#     # Create the target directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     print(os.path.join(model_dir, 'weights', model_weights_type+'.pt'))
#     assert os.path.exists(os.path.join(model_dir, 'weights', model_weights_type+'.pt'))

#     # Load the model
#     try:
#         model = YOLO(os.path.join(model_dir, 'weights', model_weights_type+'.pt'))
#     except Exception as e:
#         print(f"ERROR: Could not load model!\n{e}")
#         return

#     if task == 'detect':
#         # Load the tracker
#         tracker = sv.ByteTrack()
#         # Create the annotator for detection
#         box_annotator = sv.BoxAnnotator()
#         # Adds label to annotation (tracking)
#         labeler = sv.LabelAnnotator()
#     elif task == 'segment':
#         # Create the annotators for segmentation
#         mask_annotator = sv.MaskAnnotator()
#         box_annotator = sv.BoxAnnotator()
#     else:
#         raise Exception("ERROR: Specify --task [detect, segment]")
    
#     # Area threshold
#     area_thresh = 1.1

#     imgs = sv.list_files_with_extensions(directory=input_dir, extensions=["png", "PNG", "jpg", "JPG", "jpeg", "JPEG"])
#     print(f"\nFound {len(imgs)} images in {input_dir}")
    
#     # get image dimensions
#     # if img_size is None:
#     #     img_size = Image.open(imgs[0]).size
#     # print('Using images with size: {}'.format(img_size))

#     # with sv.ImageSink(target_dir_path=output_dir, overwrite=owrite) as sink:
#     # Loop through all the images
#     for img in tqdm(imgs, total=len(imgs)):
#         with Image.open(img) as imag:
#             imag = np.array(imag)
#             # print(type(imag))
#             # Run the frame through the model and make predictions
#             result = model(imag,
#                             conf=0.1,
#                             iou=iou,
#                             # imgsz=img_size,
#                             half=True,
#                             augment=False,
#                             max_det=1000,
#                             verbose=False,
#                             show=False)[0]

#             # Version issues
#             result.obb = None

#             # Convert the results
#             detections = sv.Detections.from_ultralytics(result)

#             # Filter the detections
#             # print(f"Original annotations: {len(detections)}")
#             # detections = filter_detections(imag, detections, area_thresh_min, area_thresh_max, conf_thresh)
#             detections = filter_detections(imag, detections, area_thresh_max, conf_thresh)
#             # print(f"Remaining annotations after filtering: {len(detections)}")

#             if task == 'detect':
#                 # Track the detections
#                 detections = tracker.update_with_detections(detections)
#                 labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

#                 # Create an annotated version of the frame (boxes)
#                 annotated_img = box_annotator.annotate(scene=imag.copy(), detections=detections)
#                 annotated_img = labeler.annotate(scene=annotated_img, detections=detections, labels=labels)
#             else:
#                 # Create an annotated version of the frame (masks and boxes)
#                 annotated_img = mask_annotator.annotate(scene=imag.copy(), detections=detections)
#                 annotated_img = box_annotator.annotate(scene=annotated_img, detections=detections)

#         # convert the annotated image to a PIL image
#         if annotated_img.ndim == 2:
#             annotated_pil = Image.fromarray(annotated_img, mode='L')
#         elif annotated_img.ndim == 3 and annotated_img.shape[2] == 3:
#             annotated_pil = Image.fromarray(annotated_img, mode='RGB')
#         elif annotated_img.ndim == 3 and annotated_img.shape[2] == 4:
#             annotated_pil = Image.fromarray(annotated_img, mode='RGBA')
#         else:
#             raise ValueError(f"Unsupported image shape: {annotated_img.shape}")

#         # save the annotated image
#         annotated_pil.save(os.path.join(output_dir, os.path.basename(img)))


def process_dir(model_dir, input_dir, output_dir, task,
                owrite=False, model_weights_type='best', img_size=None,
                conf=0.25, iou=0.5, area_thresh_min=0.0, area_thresh_max=0.2, conf_thresh=0.1,
                use_sahi=False, slice_height=512, slice_width=512, overlap_ratio=0.2):
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, 'weights', model_weights_type + '.pt')
    assert os.path.exists(weights_path), f"Weights not found at {weights_path}"

    # Load the YOLO model
    model = YOLO(weights_path)

    if use_sahi:
        print("✅ Using SAHI slicing for inference")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",  # adjust if needed: "yolov5", "yolov11", etc.
            model_path=weights_path,
            confidence_threshold=conf_thresh,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        sahi_model = None

    if task == 'detect':
        tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator()
        labeler = sv.LabelAnnotator()
    elif task == 'segment':
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
    else:
        raise Exception("ERROR: Specify --task [detect, segment]")

    imgs = sv.list_files_with_extensions(input_dir, ["png","jpg","jpeg","PNG","JPG","JPEG"])
    print(f"\nFound {len(imgs)} images in {input_dir}")

    for img in tqdm(imgs):
        imag = np.array(Image.open(img))

        if use_sahi:
            # --- Sliced inference ---
            sahi_result = get_sliced_prediction(
                image=imag,
                detection_model=sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio
            )
            # Convert SAHI predictions to supervision Detections
            object_predictions = sahi_result.object_prediction_list
            if len(object_predictions) > 0:
                xyxy = np.array([p.bbox.to_xyxy() for p in object_predictions], dtype=float)
                confidence = np.array([p.score for p in object_predictions], dtype=float)
                class_id = np.array([p.category.id for p in object_predictions], dtype=int)
                masks = [p.mask.array for p in object_predictions if p.mask is not None]
                masks = np.stack(masks, axis=0) if masks else None
                detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id, mask=masks)
            else:
                detections = sv.Detections.empty()

        else:
            # --- Standard YOLO inference ---
            result = model(imag, conf=conf_thresh, iou=iou, half=True, verbose=False, show=False)[0]
            result.obb = None
            detections = sv.Detections.from_ultralytics(result)

        detections = filter_detections(imag, detections, area_thresh_max, conf_thresh)

        # Annotation
        if task == 'detect':
            detections = tracker.update_with_detections(detections)
            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated_img = box_annotator.annotate(scene=imag.copy(), detections=detections)
            annotated_img = labeler.annotate(scene=annotated_img, detections=detections, labels=labels)
        else:
            annotated_img = mask_annotator.annotate(scene=imag.copy(), detections=detections)
            annotated_img = box_annotator.annotate(scene=annotated_img, detections=detections)

        Image.fromarray(annotated_img).save(os.path.join(output_dir, os.path.basename(img)))

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video Processing with YOLO and ByteTrack")

    parser.add_argument('-m','-input_model','-inmodel','-model','-trained_model','-mod',
                        dest='input_model',
                        type=str,
                        # required=True,
                        default="/mnt/h/GrapeFinder_v2/2025-11-13_15-45-32_yolo11s-seg_GrapeMapper_segment_0.35_0.5_0.01_0.3_JPG",
                        help="Path to the source weights file"
                        )
    parser.add_argument('-wt','-weights','-weightstype',
                        dest='weights_type',
                        type=str,
                        choices=['best','last'],
                        default='best',
                        help="Source weights option [choices: best, last] [default: best]"
                        )
    parser.add_argument('-i','-input_dir','-indir','-idir','-inputdir','-in',
                        dest='input_directory',
                        type=str,
                        # required=True,
                        default="/mnt/d/sfm_greeen_2025harvest/20250926_oxley/photos_geotagged_3",
                        help="Path to the directory of images (input)"
                        )
    parser.add_argument('-o','-output_dir','-outdir','-odir','-outputdir','-out',
                        dest='output_directory',
                        type=str,
                        # required=True,
                        default="/mnt/h/GrapeFinder_v2/2025-11-13_15-45-32_yolo11s-seg_GrapeMapper_segment_0.35_0.5_0.01_0.3_JPG/results",
                        help="Path to the target video directory (output)"
                        )
    parser.add_argument('-t','-task',
                        dest='task',
                        type=str,
                        choices=['detect','segment'],
                        default='segment',
                        help="Task to perform [choices: detect, segment]"
                        )
    parser.add_argument('-c','-conf','-confidence',
                        dest='confidence',
                        type=float,
                        default=0.25,
                        help="Confidence threshold for the model"
                        )
    parser.add_argument('-iou',
                        dest='iou',
                        type=float,
                        # default=0.7,
                        default=0.5,
                        help="IOU threshold for the model"
                        )
    parser.add_argument('-sahi',
                        dest='sahi',
                        action='store_true',
                        help="Use SAHI"
                        )
    parser.add_argument('-owrite','-overwrite','-overwrite_output',
                        dest='overwrite_output',
                        action='store_true',
                        help="Overwrite outputs [default = False]"
                        )
    parser.add_argument('--slice_height', type=int, default=512, help="Slice height for SAHI")
    parser.add_argument('--slice_width', type=int, default=512, help="Slice width for SAHI")
    parser.add_argument('--overlap', type=float, default=0.2, help="Overlap ratio for SAHI slices")

    args = parser.parse_args()

    if not args.output_directory:
        args.output_directory = os.path.join(args.input_directory, os.path.basename(args.input_model), "results")
    
    # print args to console
    print("\nArguments:")
    for arg in vars(args):
        print(arg, ": ", getattr(args, arg))
    
    if args.sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

    process_dir(model_dir=args.input_model, 
                input_dir=args.input_directory, 
                output_dir=args.output_directory, 
                task=args.task, 
                owrite=args.overwrite_output, 
                model_weights_type=args.weights_type, 
                img_size=None, 
                conf=args.confidence, 
                iou=args.iou)

if __name__ == "__main__":
    main()





# import os
# import argparse
# from PIL import Image
# from tqdm import tqdm

# import numpy as np

# import supervision as sv
# from ultralytics import YOLO

# from pw_Auto_Distill import str_to_bool

# # Optional SAHI imports guarded by flag later
# # robust_sahi_imports.py
# SAHI_AVAILABLE = False
# Yolov8DetectionModel = None
# get_sliced_prediction = None
# from sahi.models.yolov8
# try:
#     # preferred: explicit submodule import
#     from sahi.models.yolov8 import Yolov8DetectionModel as _Yolov8DetectionModel
#     from sahi.predict import get_sliced_prediction as _get_sliced_prediction
#     SAHI_AVAILABLE = True
#     Yolov8DetectionModel = _Yolov8DetectionModel
#     get_sliced_prediction = _get_sliced_prediction
# except Exception:
#     try:
#         # older/alternate layout (rare)
#         from sahi.model import Yolov8DetectionModel as _Yolov8DetectionModel
#         from sahi.predict import get_sliced_prediction as _get_sliced_prediction
#         SAHI_AVAILABLE = True
#         Yolov8DetectionModel = _Yolov8DetectionModel
#         get_sliced_prediction = _get_sliced_prediction
#     except Exception:
#         SAHI_AVAILABLE = False

# if not SAHI_AVAILABLE:
#     print("SAHI imports failed. Ensure 'sahi' is installed in the same interpreter and try `from sahi.models.yolov8 import Yolov8DetectionModel`.")


# # ----------------------------------------------------------------------------------------------------------------------
# # Functions
# # ----------------------------------------------------------------------------------------------------------------------
# def filter_detections(image, annotations, area_thresh_max, conf_thresh):
#     """
#     Filter detections by relative area and confidence.

#     :param image: np.ndarray HxWxC
#     :param annotations: sv.Detections
#     :param area_thresh_max: float, maximum relative area
#     :param conf_thresh: float, minimum confidence
#     :return annotations: sv.Detections
#     """
#     height, width, _ = image.shape

#     # Filter by area (max only here; min can be added as needed)
#     annotations = annotations[(annotations.box_area / (height * width)) < area_thresh_max]

#     # Filter by confidence
#     annotations = annotations[annotations.confidence > conf_thresh]

#     return annotations


# def sanitize_class_ids(detections, fallback_id=0):
#     """
#     Ensure class_id matches the length of masks for segmentation outputs.
#     """
#     if detections.mask is None:
#         detections.class_id = np.array([], dtype=int)
#         return

#     num_masks = len(detections.mask)
#     class_ids = getattr(detections, "class_id", None)

#     if class_ids is None or len(class_ids) != num_masks:
#         detections.class_id = np.full((num_masks,), fallback_id, dtype=int)
#     else:
#         sanitized = []
#         for cid in class_ids:
#             try:
#                 val = int(cid)
#                 sanitized.append(val if val >= 0 else fallback_id)
#             except (TypeError, ValueError):
#                 sanitized.append(fallback_id)
#         detections.class_id = np.array(sanitized, dtype=int)


# def sahi_to_sv(mod_results):
#     """
#     Convert SAHI SlicedPrediction (result of get_sliced_prediction)
#     into supervision.Detections, including masks if available.
#     """
#     object_predictions = mod_results.object_prediction_list

#     if len(object_predictions) == 0:
#         return sv.Detections(
#             xyxy=np.empty((0, 4), dtype=float),
#             confidence=np.empty((0,), dtype=float),
#             class_id=np.empty((0,), dtype=int)
#         )

#     # Bounding boxes in xyxy
#     xyxy = np.array([pred.bbox.to_xyxy() for pred in object_predictions], dtype=float)
#     # Confidence as float
#     confidence = np.array([pred.score for pred in object_predictions], dtype=float)
#     # Class ids as int
#     class_id = np.array([pred.category.id for pred in object_predictions], dtype=int)

#     # Masks (if segmentation model)
#     masks = None
#     # Check if any prediction includes a mask
#     has_mask = any(getattr(pred, "mask", None) is not None for pred in object_predictions)
#     if has_mask:
#         # Each mask is SAHI Mask object; the numpy array is at .array with shape (H, W)
#         mask_arrays = [pred.mask.array if pred.mask is not None else None for pred in object_predictions]
#         # Filter out Nones to avoid stacking failure
#         mask_arrays = [m for m in mask_arrays if m is not None]
#         if len(mask_arrays) > 0:
#             # Stack into (N, H, W)
#             masks = np.stack(mask_arrays, axis=0)

#     if masks is not None:
#         detections_results = sv.Detections(
#             xyxy=xyxy,
#             confidence=confidence,
#             class_id=class_id,
#             mask=masks
#         )
#     else:
#         detections_results = sv.Detections(
#             xyxy=xyxy,
#             confidence=confidence,
#             class_id=class_id
#         )
#     return detections_results


# def process_dir(model_dir,
#                 input_dir,
#                 output_dir,
#                 task,
#                 owrite=False,
#                 model_weights_type='best',
#                 img_size=None,
#                 conf=0.3,
#                 iou=0.5,
#                 area_thresh_min=0.0,
#                 area_thresh_max=0.4,
#                 conf_thresh=0.1,
#                 use_sahi=False,
#                 slice_height=512,
#                 slice_width=512,
#                 overlap_ratio=0.2):
#     """
#     Process images with either standard Ultralytics inference or SAHI slicing.

#     :param use_sahi: bool, enable SAHI sliced inference
#     :param slice_height: int, SAHI slice height
#     :param slice_width: int, SAHI slice width
#     :param overlap_ratio: float, SAHI overlap ratio for height and width
#     """

#     # Create the target directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     weights_path = os.path.join(model_dir, 'weights', model_weights_type + '.pt')
#     print(weights_path)
#     assert os.path.exists(weights_path), f"Weights not found at {weights_path}"

#     # Load the base Ultralytics model
#     try:
#         base_model = YOLO(weights_path)
#     except Exception as e:
#         print(f"ERROR: Could not load model!\n{e}")
#         return

#     # SAHI wrapper (only if requested)
#     sahi_model = None
#     if use_sahi:
#         if not SAHI_AVAILABLE:
#             print("ERROR: SAHI is not installed. Please `pip install sahi` or run without --sahi.")
#             return
#         sahi_model = Yolov8DetectionModel(
#             model_path=weights_path,
#             confidence_threshold=conf,
#             image_size=img_size or 640
#         )

#     if task == 'detect':
#         tracker = sv.ByteTrack()
#         box_annotator = sv.BoxAnnotator()
#         labeler = sv.LabelAnnotator()
#     elif task == 'segment':
#         mask_annotator = sv.MaskAnnotator()
#         box_annotator = sv.BoxAnnotator()
#     else:
#         raise Exception("ERROR: Specify --task [detect, segment]")

#     imgs = sv.list_files_with_extensions(
#         directory=input_dir,
#         extensions=["png", "PNG", "jpg", "JPG", "jpeg", "JPEG"]
#     )
#     print(f"\nFound {len(imgs)} images in {input_dir}")

#     for img in tqdm(imgs, total=len(imgs)):
#         with Image.open(img) as imag_pil:
#             imag = np.array(imag_pil)

#             if use_sahi:
#                 # SAHI sliced inference
#                 sahi_result = get_sliced_prediction(
#                     image=imag,
#                     detection_model=sahi_model,
#                     slice_height=slice_height,
#                     slice_width=slice_width,
#                     overlap_height_ratio=overlap_ratio,
#                     overlap_width_ratio=overlap_ratio
#                 )
#                 detections = sahi_to_sv(sahi_result)

#             else:
#                 # Standard Ultralytics inference
#                 result = base_model(
#                     imag,
#                     conf=conf,
#                     iou=iou,
#                     imgsz=img_size,
#                     half=True,
#                     augment=False,
#                     max_det=1000,
#                     verbose=False,
#                     show=False
#                 )[0]

#                 # Version issues
#                 result.obb = None

#                 # Convert the results
#                 detections = sv.Detections.from_ultralytics(result)

#             # Filter the detections (you can restore min-area if needed)
#             detections = filter_detections(imag, detections, area_thresh_max, conf_thresh)

#             # Annotate
#             if task == 'detect':
#                 # Track the detections
#                 detections = tracker.update_with_detections(detections)
#                 labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

#                 annotated_img = box_annotator.annotate(scene=imag.copy(), detections=detections)
#                 annotated_img = labeler.annotate(scene=annotated_img, detections=detections, labels=labels)
#             else:
#                 # Segmentation + boxes
#                 annotated_img = mask_annotator.annotate(scene=imag.copy(), detections=detections)
#                 annotated_img = box_annotator.annotate(scene=annotated_img, detections=detections)

#         # Convert annotated image to PIL for saving
#         if annotated_img.ndim == 2:
#             annotated_pil = Image.fromarray(annotated_img, mode='L')
#         elif annotated_img.ndim == 3 and annotated_img.shape[2] == 3:
#             annotated_pil = Image.fromarray(annotated_img, mode='RGB')
#         elif annotated_img.ndim == 3 and annotated_img.shape[2] == 4:
#             annotated_pil = Image.fromarray(annotated_img, mode='RGBA')
#         else:
#             raise ValueError(f"Unsupported image shape: {annotated_img.shape}")

#         out_path = os.path.join(output_dir, os.path.basename(img))
#         if (not owrite) and os.path.exists(out_path):
#             # Respect overwrite flag
#             continue
#         annotated_pil.save(out_path)


# # ----------------------------------------------------------------------------------------------------------------------
# # Main
# # ----------------------------------------------------------------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Image Processing with YOLO, ByteTrack, and optional SAHI slicing")

#     parser.add_argument('-m','-input_model','-inmodel','-model','-trained_model','-mod',
#                         dest='input_model',
#                         type=str,
#                         default="/mnt/h/GrapeFinder_v2/2025-11-10_16-51-10_yolo11s-seg_GrapeMapper_segment_0.5_0.5_0.01_0.3_JPG",
#                         help="Path to the source weights directory (expects weights/<best|last>.pt inside)"
#                         )
#     parser.add_argument('-wt','-weights','-weightstype',
#                         dest='weights_type',
#                         type=str,
#                         choices=['best','last'],
#                         default='best',
#                         help="Source weights option [choices: best, last] [default: best]"
#                         )
#     parser.add_argument('-i','-input_dir','-indir','-idir','-inputdir','-in',
#                         dest='input_directory',
#                         type=str,
#                         default="/mnt/d/sfm_greeen_2025harvest/20250926_oxley/photos_geotagged_3",
#                         help="Path to the directory of images (input)"
#                         )
#     parser.add_argument('-o','-output_dir','-outdir','-odir','-outputdir','-out',
#                         dest='output_directory',
#                         type=str,
#                         default="/mnt/h/GrapeFinder_v2/2025-11-10_16-51-10_yolo11s-seg_GrapeMapper_segment_0.5_0.5_0.01_0.3_JPG/results",
#                         help="Path to the target image directory (output)"
#                         )
#     parser.add_argument('-t','-task',
#                         dest='task',
#                         type=str,
#                         choices=['detect','segment'],
#                         default='segment',
#                         help="Task to perform [choices: detect, segment]"
#                         )
#     parser.add_argument('-c','-conf','-confidence',
#                         dest='confidence',
#                         type=float,
#                         default=0.35,
#                         help="Confidence threshold for the model"
#                         )
#     parser.add_argument('-iou',
#                         dest='iou',
#                         type=float,
#                         default=0.3,
#                         help="IOU threshold for the model"
#                         )
#     parser.add_argument('-sahi',
#                         dest='sahi',
#                         action='store_true',
#                         help="Use SAHI sliced inference"
#                         )
#     parser.add_argument('--slice_height',
#                         dest='slice_height',
#                         type=int,
#                         default=512,
#                         help="SAHI slice height (pixels)"
#                         )
#     parser.add_argument('--slice_width',
#                         dest='slice_width',
#                         type=int,
#                         default=512,
#                         help="SAHI slice width (pixels)"
#                         )
#     parser.add_argument('--overlap',
#                         dest='overlap',
#                         type=float,
#                         default=0.2,
#                         help="SAHI overlap ratio for height and width"
#                         )
#     parser.add_argument('-owrite','-overwrite','-overwrite_output',
#                         dest='overwrite_output',
#                         action='store_true',
#                         help="Overwrite outputs [default = False]"
#                         )

#     args = parser.parse_args()

#     if not args.output_directory:
#         args.output_directory = os.path.join(args.input_directory, os.path.basename(args.input_model), "results")

#     print("\nArguments:")
#     for arg in vars(args):
#         print(arg, ": ", getattr(args, arg))

#     process_dir(model_dir=args.input_model,
#                 input_dir=args.input_directory,
#                 output_dir=args.output_directory,
#                 task=args.task,
#                 owrite=args.overwrite_output,
#                 model_weights_type=args.weights_type,
#                 img_size=None,
#                 conf=args.confidence,
#                 iou=args.iou,
#                 use_sahi=args.sahi,
#                 slice_height=args.slice_height,
#                 slice_width=args.slice_width,
#                 overlap_ratio=args.overlap)


# if __name__ == "__main__":
#     main()