import os
import glob
import shutil
import argparse
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

import supervision as sv
from supervision.detection.utils import *
from autodistill import helpers
from autodistill_grounded_sam import GroundedSAM
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import torch
from torchvision.ops import box_iou



def str_to_bool(s):
    """
    Converts str ['t','true','f','false'] to boolean, not case sensitive.
    Checks first if already a boolean.
    Raises exception if unexpected entry.
        args:
            s: str
        returns:
            out_boolean: output boolean [True or False]
    """
    #check to see if already boolean
    if isinstance(s, bool):
        out_boolean = s
    else:
        # remove quotes, commas, and case from s
        sf = s.lower().replace('"', '').replace("'", '').replace(',', '')
        # True
        if sf in ['t', 'true']:
            out_boolean = True
        # False
        elif sf in ['f', 'false']:
            out_boolean = False
        # Unexpected arg
        else:
            # print exception so it will be visible in console, then raise exception
            print('ArgumentError: Argument invalid. Expected boolean '
                  + 'got ' + '"' + str(s) + '"' + ' instead')
            raise Exception('ArgumentError: Argument invalid. Expected boolean '
                            + 'got ' + '"' + str(s) + '"' + ' instead')
    return out_boolean


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


# def non_max_suppression(predictions: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
#   rows,_ = predictions.shape

#   sort_index = np.flip(predictions[:, 4].argsort())
#   predictions = predictions[sort_index]

#   boxes = predictions[:, :4]
#   if isinstance(boxes, np.ndarray):
#     boxes = torch.from_numpy(boxes).float()
#   elif not isinstance(boxes, torch.Tensor):
#     boxes = torch.tensor(boxes, dtype=torch.float32)

#   categories = predictions[:, 4]
#   ious = box_iou(boxes, boxes)
#   ious = ious - np.eye(rows)

#   keep = np.ones(rows, dtype=bool)
#   if isinstance(keep, np.ndarray):
#     keep = torch.from_numpy(keep).bool()
#   condition = condition.bool()

#   for index, (iou, category) in enumerate(zip(ious, categories)):
#       if not keep[index]:
#           continue

#       condition = (iou > iou_threshold) & (categories == category)
#       keep = keep & ~condition

#   return keep[sort_index.argsort()]


def non_max_suppression(
    predictions,
    iou_threshold: float = 0.5,
    image_shape=None,
    min_area: float = 0.0,
    max_area: float = 1.0,
) -> list:

    """
    predictions: [N,5] -> [x1, y1, x2, y2, score]
    iou_threshold: NMS IoU
    image_shape: (H, W) required if min_area/max_area filtering is used
    min_area/max_area: box area fraction relative to full image

    Returns list of indices (relative to the original predictions array) to keep.
    """

    # Ensure numpy array
    predictions = np.array(predictions)
    if predictions.size == 0:
        return []

    N = predictions.shape[0]
    orig_indices = np.arange(N)

    # ---------- Area Filtering (optional) ----------
    if image_shape is not None and (min_area > 0.0 or max_area < 1.0):
        # image_shape may be (H, W) or (H, W, C)
        if len(image_shape) == 3:
            H, W, _ = image_shape
        elif len(image_shape) == 2:
            H, W = image_shape
        else:
            raise ValueError("image_shape must be (H,W) or (H,W,C)")
        img_area = float(H) * float(W)

        boxes_xyxy = predictions[:, :4]
        box_w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        box_h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        box_area = box_w * box_h
        area_frac = box_area / img_area

        area_mask = (area_frac >= float(min_area)) & (area_frac <= float(max_area))
    else:
        area_mask = np.ones(N, dtype=bool)

    # Filter predictions and track original indices
    filtered_preds = predictions[area_mask]
    filtered_orig_idx = orig_indices[area_mask]

    if filtered_preds.shape[0] == 0:
        return []

    # ---------- Continue With NMS on filtered predictions ----------
    # Sort by score descending
    sort_index = np.flip(filtered_preds[:, 4].argsort())
    preds_sorted = filtered_preds[sort_index]
    orig_sorted_idx = filtered_orig_idx[sort_index]

    boxes = preds_sorted[:, :4]
    scores = preds_sorted[:, 4]

    if isinstance(boxes, np.ndarray):
        boxes_t = torch.from_numpy(boxes).float()
    elif isinstance(boxes, torch.Tensor):
        boxes_t = boxes.float()
    else:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)

    if isinstance(scores, np.ndarray):
        scores_t = torch.from_numpy(scores).float()
    elif isinstance(scores, torch.Tensor):
        scores_t = scores.float()
    else:
        scores_t = torch.tensor(scores, dtype=torch.float32)

    keep_local = []
    idxs = scores_t.argsort(descending=True)
    while idxs.numel() > 0:
        current = idxs[0]
        keep_local.append(int(current.item()))
        if idxs.numel() == 1:
            break
        rest = idxs[1:]
        ious = box_iou(boxes_t[current].unsqueeze(0), boxes_t[rest]).squeeze(0)
        idxs = rest[ious <= iou_threshold]

    # Map kept local indices (relative to preds_sorted) back to original predictions indices
    # preds_sorted was built from filtered_preds[sort_index], so map: orig_sorted_idx[keep_local]
    keep_global = [int(orig_sorted_idx[i]) for i in keep_local]

    return keep_global


# def non_max_suppression(predictions, iou_threshold=0.5) -> np.ndarray:
#     """
#     boxes: Tensor[N, 4] in [x1, y1, x2, y2] format
#     scores: Tensor[N]
#     iou_thresh: float
#     Returns: indices of boxes to keep
#     """
#     rows,_ = predictions.shape

#     sort_index = np.flip(predictions[:, 4].argsort())
#     predictions = predictions[sort_index]

#     boxes = predictions[:, :4]
#     scores = predictions[:, 4]

#     if isinstance(boxes, np.ndarray):
#         boxes = torch.from_numpy(boxes).float()
#     if isinstance(scores, np.ndarray):
#         scores = torch.from_numpy(scores).float()

#     keep = []
#     idxs = scores.argsort(descending=True)

#     while idxs.numel() > 0:
#         current = idxs[0]
#         keep.append(current.item())

#         if idxs.numel() == 1:
#             break

#         rest = idxs[1:]
#         ious = box_iou(boxes[current].unsqueeze(0), boxes[rest]).squeeze(0)
#         idxs = rest[ious <= iou_threshold]

#     # print("🔍 keep:", keep)
#     return keep

def extract_frames_from_video(video_path, image_dir, start_ratio=.15, end_ratio=.85, frame_stride=15):
    """

    :param video_path:
    :param image_dir:
    :param start_ratio:
    :param end_ratio:
    :param frame_stride:
    :return:
    """
    # Create a name pattern
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    image_name_pattern = video_name + "-{:05d}."+file_ext

    # Get the video feed
    cap = cv2.VideoCapture(video_path)

    # Figure out the start and end points
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_at = int(start_ratio * total_frames)
    end_at = int(end_ratio * total_frames)

    # Create output directory
    os.makedirs(image_dir, exist_ok=True)

    # Video is opened, frame extracted
    success, image = cap.read()
    frame_number = 0

    if not success:
        raise Exception("Could not open video!")

    while success:
        # Every Nth frame, that is between N and N
        if frame_number % frame_stride == 0 and start_at <= frame_number <= end_at:
            # Write the image to disk if it doesn't already exist
            frame_filename = os.path.join(image_dir, image_name_pattern.format(frame_number))
            if not os.path.exists(frame_filename):
                cv2.imwrite(frame_filename, image)

        # Continue with the video feed
        success, image = cap.read()
        frame_number += 1

    # Release me!
    cap.release()


def extract_frames(video_paths, image_dir, start_ratio=.15, end_ratio=.85, frame_stride=15):
    """

    :param video_paths:
    :param image_dir:
    :param start_ratio:
    :param end_ratio:
    :param frame_stride:
    :return:
    """
    with ThreadPoolExecutor() as executor:
        # Use executor to process each video in parallel
        executor.map(
            lambda video_path: extract_frames_from_video(video_path, image_dir, start_ratio, end_ratio, frame_stride),
            video_paths
        )


def batch_and_copy_images(source_folder, output_folder, batch_size=64, file_ext='JPG'):
    """

    :param source_folder:
    :param batch_size:
    :return:
    """
    print('\nBatching data with size = {}'.format(batch_size))

    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print("Source folder '{}' not found.".format(source_folder))
        return

    # Create a list of image files in the source folder
    image_files = glob.glob(os.path.join(source_folder, '*.'+file_ext))

    # Check if there are any image files
    if not image_files:
        print("No image files found in '{}'.".format(source_folder))
        return
    else:
        print("Found {} image files in '{}'.".format(len(image_files),source_folder))

    # Create output folders
    output_folder_base = os.path.join(output_folder,"images")
    output_folder_index = 1

    # Iterate over image files and copy them into batches
    for index, image_file in enumerate(image_files):
        # Check if a new batch needs to be created
        if index % batch_size == 0:
            current_output_folder = f"{output_folder_base}_{output_folder_index}"
            # print(current_output_folder)
            os.makedirs(current_output_folder, exist_ok=True)
            output_folder_index += 1

        # Create the path for the copy in the current output folder
        copy_path = os.path.join(current_output_folder, os.path.basename(image_file))
        # print(copy_path)

        # Copy the image file to the current output folder
        shutil.copyfile(image_file, copy_path)

    print("Copies of images successfully created in batches of {} into {} folders.".format(batch_size,output_folder_index-1))
    print('Batched data saved in {}.\n'.format(output_folder_base))


def filter_detections(image, annotations, area_thresh_min, area_thresh_max, conf_thresh):
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
    annotations = annotations[(annotations.box_area / (height * width)) > area_thresh_min]
    annotations = annotations[(annotations.box_area / (height * width)) < area_thresh_max]

    # Filter by confidence
    # print(annotations.confidence)
    annotations = annotations[annotations.confidence > conf_thresh]

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

def filter_detections_by_indices(detections, keep, task='segment'):
    def safe_slice(attr):
        return [attr[i] for i in keep] if attr is not None else []

    xyxy = np.array(safe_slice(detections.xyxy)).reshape(-1, 4)

    # if task == 'segment':
    mask = np.array([detections.mask[i] for i in keep]) if detections.mask is not None else None
        
    confidence = np.array(safe_slice(detections.confidence)) if detections.confidence is not None else None
    class_id = np.array(safe_slice(detections.class_id)) if detections.class_id is not None else np.zeros(len(keep), dtype=int)
    tracker_id = np.array(safe_slice(detections.tracker_id)) if detections.tracker_id is not None else None

    data = {k: [v[i] for i in keep] for k, v in detections.data.items()} if detections.data else {}

    # if task == 'detect':
    return sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
        data=data,
        metadata=detections.metadata
    )
    # else:
    #     return sv.Detections(
    #         xyxy=xyxy,
    #         confidence=confidence,
    #         class_id=class_id,
    #         tracker_id=tracker_id,
    #         data=data,
    #         metadata=detections.metadata
    #     )

# def filter_invalid_class_ids(detections):
#     valid_indices = [
#         i for i, cid in enumerate(detections.class_id)
#         if isinstance(cid, (int, np.integer)) and cid is not None and cid >= 0
#     ]

#     # Slice all fields by valid indices
#     return sv.Detections(
#         xyxy=np.array([detections.xyxy[i] for i in valid_indices]).reshape(-1, 4),
#         mask=np.array([detections.mask[i] for i in valid_indices]) if detections.mask is not None else None,
#         confidence=np.array([detections.confidence[i] for i in valid_indices]) if detections.confidence is not None else None,
#         class_id=np.array([detections.class_id[i] for i in valid_indices], dtype=int),
#         tracker_id=np.array([detections.tracker_id[i] for i in valid_indices]) if detections.tracker_id is not None else None,
#         data={k: [v[i] for i in valid_indices] for k, v in detections.data.items()} if detections.data else {},
#         metadata=detections.metadata
#     )

def filter_invalid_class_ids(detections):
    valid_indices = [
        i for i, cid in enumerate(detections.class_id)
        if isinstance(cid, (int, np.integer)) and cid is not None and cid >= 0
    ]
    
    # --- FIX START ---
    num_valid = len(valid_indices)
    
    # 1. Handle XYXY: Ensure it's a (N, 4) array.
    xyxy_filtered = np.array([detections.xyxy[i] for i in valid_indices]).reshape(-1, 4)
    
    # 2. Handle Mask: Correctly shape the empty mask array if no detections remain.
    mask_filtered = None
    if detections.mask is not None:
        if num_valid > 0:
            mask_filtered = np.array([detections.mask[i] for i in valid_indices])
        else:
            # If mask is present but valid_indices is empty, create an empty 3D array (0, H, W).
            # We can't know H, W here, but using np.empty((0, 0, 0)) often works for 
            # supervision/numpy arrays where the first dimension is 0. 
            # However, for safety against the ValueError, we'll try to match the expected empty shape.
            # supervision's internal logic expects (0, H, W). A generic (0, 0, 0) often passes.
            mask_filtered = np.empty((0, 0, 0), dtype=bool) 
            # Alternatively: mask_filtered = np.array([detections.mask[i] for i in valid_indices], dtype=bool).reshape((0,0,0))
    # --- FIX END ---

    # Slice all other fields by valid indices
    return sv.Detections(
        xyxy=xyxy_filtered,
        mask=mask_filtered,
        confidence=np.array([detections.confidence[i] for i in valid_indices]) if detections.confidence is not None else None,
        class_id=np.array([detections.class_id[i] for i in valid_indices], dtype=int),
        tracker_id=np.array([detections.tracker_id[i] for i in valid_indices]) if detections.tracker_id is not None else None,
        data={k: [v[i] for i in valid_indices] for k, v in detections.data.items()} if detections.data else {},
        metadata=detections.metadata
    )

def rebuild_detections(d):
    N = len(d.mask)
    return sv.Detections(
        xyxy=np.array(d.xyxy[:N]).reshape(-1, 4),
        mask=np.array(d.mask[:N]),
        class_id=np.array(d.class_id[:N], dtype=int),
        confidence=np.array(d.confidence[:N]) if d.confidence is not None else None,
        tracker_id=np.array(d.tracker_id[:N]) if d.tracker_id is not None else None,
        data={k: v[:N] for k, v in d.data.items()} if d.data else {},
        metadata=d.metadata
    )

def render_dataset(dataset, output_dir, class_map=None, include_boxes=True, include_masks=False, col_map=None):
    """

    :param dataset:
    :param output_dir:
    :param include_boxes:
    :param include_masks:
    :return:
    """

    # Create the annotation object
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS)
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
    # mask_annotator = sv.MaskAnnotator()
    # box_annotator = sv.BoxAnnotator()

    with sv.ImageSink(target_dir_path=output_dir, overwrite=False) as sink:
        for path, imag, annot in dataset:
            # print("🔍 Detections keys:", annot.__dict__.keys())
            # print("🔍 class_id:", getattr(annot, "class_id", None))
            # print("🔍 mask count:", len(getattr(annot, "mask", [])))
            
            # Read the original EXIF data
            try:
                original_image_pil = Image.open(path)
                exif_data = original_image_pil.info.get('exif') # Get raw EXIF byte data
            except Exception as e:
                print(f"Warning: Could not read or extract EXIF from {path}. Error: {e}")
                exif_data = None
            
            # Remove invalid class_ids
            annot = filter_invalid_class_ids(annot)

            if include_boxes:
                # Get the boxes for the annotations
                imag = box_annotator.annotate(scene=imag, detections=annot)

            if include_masks:
                if "class_id" in annot.data:
                    del annot.data["class_id"]
                if "color" in annot.data:
                    del annot.data["color"]

                sanitize_class_ids(annot)

                annot.class_id = np.array([
                    0 if cid is None or not isinstance(cid, (int, np.integer)) or cid < 0 else int(cid)
                    for cid in annot.class_id
                ], dtype=int)
                
                # print("✅ Final class_id:", annot.class_id)
                # print("✅ Type:", type(annot.class_id), "Shape:", annot.class_id.shape)
                # print("🧪 About to annotate:")
                # print("class_id:", annot.class_id)
                # print("type:", type(annot.class_id))
                # print("contains None:", any(cid is None for cid in annot.class_id))
                # print("contains invalid:", any(not isinstance(cid, (int, np.integer)) or cid < 0 for cid in annot.class_id))
                
                annot = rebuild_detections(annot)
                
                # print("🔍 Final sanity check:")
                # print("xyxy:", len(annot.xyxy))
                # print("mask:", len(annot.mask))
                # print("class_id:", len(annot.class_id))
                # print("confidence:", len(annot.confidence) if annot.confidence is not None else "None")
                
                # Get the masks for the annotations
                imag = mask_annotator.annotate(scene=imag, detections=annot)

            # output_file = os.path.basename(image_name)
            output_file = os.path.basename(path)

            # Convert the annotated NumPy array (imag) to a PIL image
            # annotated_pil = Image.fromarray(cv2.cvtColor(imag, cv2.COLOR_BGR2RGB))
            
            # Save the image, passing the original EXIF data
            # try:
            #     if exif_data is not None:
            #         annotated_pil.save(output_file, exif=exif_data)
            #     else:
            #         annotated_pil.save(output_file)
            # except Exception as e:
            #     print(f"Error saving image with EXIF: {output_file}. Falling back to standard save. Error: {e}")
            #     # annotated_pil.save(output_file)
            sink.save_image(image=imag, image_name=output_file)


def remove_bad_data(data_dir, fileext='JPG'):
    """

    :param data_dir:
    :return:
    """
    # Set the paths
    train_dir = os.path.join(data_dir,"train")
    valid_dir = os.path.join(data_dir,"valid")
    render_dir = os.path.join(data_dir,"rendered")

    # Make sure they exist
    assert os.path.exists(train_dir)
    assert os.path.exists(valid_dir)
    assert os.path.exists(render_dir)

    print('train_dir = {}'.format(train_dir))
    print('valid_dir = {}'.format(valid_dir))
    print('render_dir = {}'.format(render_dir))

    combined_dict = {}

    # Get all the training images and labels paths
    train_images = glob.glob(os.path.join(train_dir,"images","*."+fileext))
    train_labels = glob.glob(os.path.join(train_dir,"labels","*.txt"))
    # print('found {} train images in {}'.format(len(train_images),os.path.join(train_dir,"images")))

    for img, lab in zip(train_images, train_labels):
        basename = os.path.basename(img).split(".")[0]
        # print('training image: {}'.format(basename))
        combined_dict[basename] = {
            "image": img,
            "label": lab
        }

    # Get all the validation images and labels paths
    valid_images = glob.glob(os.path.join(valid_dir,"images","*."+fileext))
    valid_labels = glob.glob(os.path.join(valid_dir,"labels","*.txt"))

    for img, lab in zip(valid_images, valid_labels):
        basename = os.path.basename(img).split(".")[0]
        print('validation image: {}'.format(basename))
        combined_dict[basename] = {
            "image": img,
            "label": lab
        }

    # Get the rendered images
    print(os.path.join(render_dir,"*."+fileext))
    render_images = glob.glob(os.path.join(render_dir,"*."+fileext))
    # print('found {} rendered images in {}'.format(len(render_images),render_dir))

    # print('combined_dict {}'.format(combined_dict))

    # print('pre-filter: combined_dict length = {}'.format(len(combined_dict)))

    # Loop through the rendered images and removes those that
    # exist from the combined dictionary.
    for render_image in render_images:
        basename = os.path.basename(render_image).split(".")[0]
        # print('rendered image: {}'.format(basename))
        if basename in combined_dict:
            combined_dict.pop(basename)
    
    # print('post-filter: combined_dict length = {}'.format(len(combined_dict)))

    # Finally, loop though the remaining image / labels,
    # representing the bad data, and delete them.
    for value in combined_dict.values():
        print(f"NOTE: Deleting {value['image']}")
        os.remove(value['image'])
        print(f"NOTE: Deleting {value['label']}")
        os.remove(value['label'])


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing with YOLO and ByteTrack")

    parser.add_argument('-dir',  
                        dest='dir', type=str, default='/mnt/d/sfm_greeen_2025harvest/20250926_oxley/for_annotation', help='Directory to process')
    parser.add_argument('-ont',  
                        dest='ont', type=str, choices=['grape','grapes','green grapes','riesling','rocks','mussels','fish','shells','trees','viticulture'], default='grape', 
                        help='Ontology to use')
    parser.add_argument('-detect', action='store_true', help='Detect objects')
    parser.add_argument('-segment', action='store_true', default=True, help='Segment objects')
    parser.add_argument('-amin','-areamin','-area_thresh_min',  
                        dest='area_threshold_minimum', type=float, default=0.01, help='Minimum area threshold for detections')
    parser.add_argument('-amax','-areamax','-area_thresh_max',
                        dest='area_threshold_maximum', type=float, default=0.3, help='Maximum area threshold for detections')
    parser.add_argument('-conf','-conf_thresh',  
                        dest='conf_thresh', type=float, default=0.35, help='Confidence threshold for detections')
    parser.add_argument('-nms','-nms_thresh',  
                        dest='nms_thresh', type=float, default=0.5, help='NMS threshold for detections')
    parser.add_argument('-bsize',  
                        dest='bsize', type=int, default=32, help='Batch size')
    parser.add_argument('-use_sahi', action='store_true', help='Use SAHI')
    parser.add_argument('-file_ext',  
                        dest='file_ext', type=str, default='JPG', help='File extension')
    parser.add_argument('-response', '-response_required', 
                        dest='response_required', action='store_true', help='Response required')
    parser.add_argument('-verbose', '-verbose_mode', 
                        dest='verbose_mode', action='store_true', help='Run in verbose mode')

    args = parser.parse_args()

    assert args.detect or args.segment, ValueError('Specify either -detect or -segment')
    if args.detect:
        mode = 'detect'
    else:
        mode = 'segment'

    # ------------------------------------------------------
    # Modify each of these as needed!

    # Define the workflow
    EXTRACT_FRAMES = False
    CREATE_LABELS = True

    # Debug
    SAVE_LABELS = True

    # CV Tasks
    # DETECTION = False
    # SEGMENTATION = True

    # There can only be one
    # assert DETECTION != SEGMENTATION

    # Set up the labeling ontology
    ont_dict = {
        'coastal': CaptionOntology({
            "rock": "rock",
            "tiny rock": "rock",
            "small rock": "rock",
            "big rock": "rock",
            "fuzzy rock": "rock",
            "smooth rock": "rock",
            "person": "person",
            "people": "person",
            "umbrella": "umbrella",
            "umbrellas": "umbrella",
            "grass": "grass",
            "grasses": "grass",
            "wave": "wave",
            "waves": "wave",
            "big wave": "wave",
            "small wave": "wave",
            "white water": "breaking wave",
            "breaking wave": "breaking wave",
            "waves breaking": "breaking wave",
            "wave breaking": "breaking wave",
            "breaking waves": "breaking wave",
            "white water wave": "breaking wave",
            "chair": "chair",
            "chairs": "chair",
        }),
        'grape': CaptionOntology({
            "grape": "grape",
            "grapes": "grape",
            "fuzzy grape": "grape",
            "fuzzy grape cluster": "grape",
            "grape cluster": "grape",
            "purple grape": "grape",
            "purple grape cluster": "grape",
            "purple grapes": "grape",
            "red grape": "grape",
            "red grape cluster": "grape",
            "red grapes": "grape",
        }),
        'grapes': CaptionOntology({
            "grapes": "grape",
        }),
        'green grapes': CaptionOntology({
            "green grape": "grape",
            "green grapes": "grape",
            "green grape cluster": "grape",
            "green grape bunch": "grape",
            "green grape bunches": "grape",
            "green grape bunch cluster": "grape",
            "fuzzy green grape": "grape",
            "fuzzy green grapes": "grape",
        }),
        'riesling': CaptionOntology({
            "riesling": "riesling",
            "riesling cluster": "riesling",
            "fuzzy riesling": "riesling",
            "fuzzy riesling cluster": "riesling",
            "green grapes": "riesling",
            "green grape": "riesling",
            "green grape cluster": "riesling",
            "green grape bunch": "riesling",
            "green grape bunches": "riesling",
            "green grape bunch cluster": "riesling",
            "fuzzy green grape": "riesling",
            "fuzzy green grapes": "riesling",
            "fuzzy green grape cluster": "riesling",
            "grape": "riesling",
            "grapes": "riesling",
            "grape cluster": "riesling",
            "fuzzy grape": "riesling",
            "fuzzy grape cluster": "riesling",
            "fuzzy grapes": "riesling",
            "yellow grape": "riesling",
            "yellow grapes": "riesling",
            "yellow grape cluster": "riesling",
            "yellow grape bunch": "riesling",
            "yellow grape bunches": "riesling",
            "yellow grape bunch cluster": "riesling",
            "fuzzy yellow grape": "riesling",
            "fuzzy yellow grapes": "riesling",
            "fuzzy yellow grape cluster": "riesling",
        }),
        'rocks': CaptionOntology({
            "rock": "rock",
            "tiny rock": "rock",
            "small rock": "rock",
            "big rock": "rock",
            "fuzzy rock": "rock",
            "smooth rock": "rock",
        }),
        'mussels': CaptionOntology({
            "mussel": "mussel",
            "mussel": "clam",
            "mussel": "oyster"
        }),
        'fish': CaptionOntology({
            "shell": "shell",
            "shell hash": "shell hash",
        }),
        'shells': CaptionOntology({
            "shell": "shell",
            "shell hash": "shell hash",
        }),
        'trees': CaptionOntology({
            "tree": "tree",
            "big tree": "tree",
            "small tree": "tree",
            "tiny tree": "tree",
            "fuzzy tree": "tree",
            "smooth tree": "tree",
        }),
        'viticulture': CaptionOntology({
            "grape": "grape",
            "vine": "stem",
            "grapevine": "stem",
            "wine grape": "grape",
            "wine vine": "stem",
            "grape vine": "stem",
            "vine grape": "grape",
            "vine wine grape": "grape",
            "grapevine grape": "grape",
            "grapevine vine": "stem",
            "wine grapevine grape": "grape",
            "wine grapevine vine": "stem",
            "grape grapevine": "grape",
            "grape wine grape": "grape",
            "grape wine vine": "stem",
            "grape wine grapevine": "stem",
            "vine grapevine grape": "grape",
            "vine grapevine vine": "stem",
            "stem": "stem",
            "stems": "stem",
            "trunk": "stem",
            "trunks": "stem",
            "branch": "stem",
            "branches": "stem",
            "branchlet": "stem",
            "branchlets": "stem",
            "sky": "sky",
            "blue sky": "sky",
            "cloud": "sky",
            "foliage": "foliage",
            "foliages": "foliage",
            "leaves": "foliage",
            "leaf": "foliage",
            "post": "stem",
            "posts": "stem",
        }),
    }

    # cmap = {
    #     "grape": sv.Color(255, 0, 0),     # Red
    #     "foliage": sv.Color(0, 255, 0),   # Green
    #     "sky": sv.Color(0, 0, 255),       # Blue
    #     "stem": sv.Color(255, 255, 0),    # Cyan
    # }

    cmap = {
        0: sv.Color(255, 0, 0),     # Red
        1: sv.Color(0, 255, 0),   # Green
        2: sv.Color(0, 0, 255),       # Blue
        3: sv.Color(255, 255, 0),    # Cyan
    }

    # coastal_labels = set(ont_dict['coastal'].values())
    coastal_cmap = {
        0: sv.Color(128, 64, 64),           # Brownish
        1: sv.Color(255, 192, 203),       # Pink
        2: sv.Color(255, 165, 0),       # Orange
        3: sv.Color(0, 128, 0),            # Dark Green
        4: sv.Color(0, 191, 255),           # Deep Sky Blue
        5: sv.Color(135, 206, 250),# Light Sky Blue
        6: sv.Color(160, 82, 45),          # Saddle Brown
    }

    model_name_dict = {
        'coastal': "CoastalMapper",
        'grape': "GrapeMapper",
        'grapes': "GrapeMapper",
        'green grapes': "GrapeMapper_green",
        'riesling': "GrapeMapper_riesling",
        'rocks': "RockMapper",
        'mussels': "MusselMapper",
        'fish': "FishMapper",
        'shells': "ShellMapper",
        'trees': "TreeMapper",
        'viticulture': "VinesMapper",
    }
    # ontology = CaptionOntology({
    #     "rock": "rock",
    #     "tiny rock": "rock",
    #     "small rock": "rock",
    #     "big rock": "rock",
    #     "fuzzy rock": "rock",
    #     "smooth rock": "rock",
    # })
    # ontology = CaptionOntology({
    #     "mussel": "mussel",
    #     "mussel": "clam",
    #     "mussel": "oyster"
    # })
    # ontology = CaptionOntology({
    #     "shell": "shell",
    #     "shell hash": "shell hash",
    # })
    # ontology = CaptionOntology({
    #     "fish": "fish",
    #     "big fish": "fish",
    #     "small fish": "fish",
    #     "tiny fish": "fish",
    #     "fuzzy fish": "fish",
    # })

    # ontology = CaptionOntology({
    #     "tree": "tree",
    #     "car": "car",
    #     "telephone pole": "pole",
    #     "pole":"pole",
    #     "deer":"deer",
    #     "person":"person",
    # })

    # ontology = CaptionOntology({
    #     "grapes": "grapes",
    #     "grape": "grapes",
    # })
    # ontology = ont_dict[args.ont]

    # # Polygon's size as a ratio of the image
    # # Large polygons shouldn't be included...
    # # area_thresh = 0.4
    # # area_thresh = 0.01
    # area_thresh = 0.01

    # # conf_thresh = 0.15
    # # conf_thresh = 0.3
    # conf_thresh = 0.35
    # # conf_thresh = 0.6

    # # Non-maximum suppression threshold
    # # nms_thresh = 0.1
    # # nms_thresh = 0.9
    # nms_thresh = 0.5

    # print('615 ontology_values: ', dict(ont_dict[args.ont].promptMap).values())

    # # Build a unified class_id map
    # class_name_to_id = {}
    # next_id = 0
    # for class_name in set(dict(ont_dict[args.ont].promptMap).values()):
    #     class_name_to_id[class_name] = next_id
    #     next_id += 1
    # print('623 class_name_to_id: ', class_name_to_id)

    # # Build a unified caption_to_id map
    # caption_to_id = {
    #     caption: class_name_to_id[class_name]
    #     for caption, class_name in dict(ont_dict[args.ont].promptMap).items()
    # }
    # print('630 caption_to_id: ', caption_to_id)

    ordered_captions = list(dict(ont_dict[args.ont].promptMap).keys())

    # class name → unified ID
    class_name_to_id = {}
    next_id = 0
    for class_name in set(dict(ont_dict[args.ont].promptMap).values()):
        class_name_to_id[class_name] = next_id
        next_id += 1

    # caption → unified ID
    caption_to_id = {
        caption: class_name_to_id[class_name]
        for caption, class_name in dict(ont_dict[args.ont].promptMap).items()
    }


    if args.verbose_mode:
        print('\nRoot dir = {}'.format(args.dir))

    print('\nInput directory = {}'.format(args.dir))

    # Frames are batched (RAM) and temporarily placed here
    # batched_dir = os.path.join(rdir,"images_resize_05_png_b"+str(bsize))
    # batched_dir = os.path.join(rdir,"images_resize_03_b"+str(bsize))
    # batched_dir = f'{rdir}/images_resize_05_b{str(bsize)}'
    batched_dir = f'{args.dir}/images_b{str(args.bsize)}'

    # If it exists from last time (exited early) delete
    if os.path.exists(batched_dir):
        shutil.rmtree(batched_dir)

    # Create the directory
    os.makedirs(batched_dir, exist_ok=True)

    # Auto labeled data; this is also temporary until being filtered
    # auto_labeled_dir = os.path.join(rdir,"Auto_Labeled")
    if args.use_sahi:
        auto_labeled_dir = f'{args.dir}/{model_name_dict[args.ont]}_Auto_Labeled_sahi'
    else:
        auto_labeled_dir = f'{args.dir}/{model_name_dict[args.ont]}_Auto_Labeled'
    if os.path.isdir(auto_labeled_dir):
        print('\n{} found. Deleting existing directory.'.format(auto_labeled_dir))
        shutil.rmtree(auto_labeled_dir, ignore_errors=True)
    os.makedirs(auto_labeled_dir, exist_ok=True)

    # The root folder containing *all* post-processed dataset for training
    # training_data_dir = os.path.join(rdir,"Training_Data")
    training_data_dir = f'{args.dir}/Training_Data'
    os.makedirs(training_data_dir, exist_ok=True)

    # ------------------------------------------------------
    # UPDATE THIS AND ONLY THIS

    # Currently we're creating single-class datasets, and
    # merging them together right before training the model

    if args.use_sahi:
        dataset_name = model_name_dict[args.ont]+'_'+mode+'_'+str(args.conf_thresh)+'_'+str(args.nms_thresh)+'_'+str(args.area_threshold_minimum)+'_'+str(args.area_threshold_maximum)+"_sahi_"+args.file_ext
    else:
        dataset_name = model_name_dict[args.ont]+'_'+mode+'_'+str(args.conf_thresh)+'_'+str(args.nms_thresh)+'_'+str(args.area_threshold_minimum)+'_'+str(args.area_threshold_maximum)+"_"+args.file_ext

    # The directory for the current dataset being created
    # current_data_dir = os.path.join(training_data_dir, dataset_name)
    current_data_dir = f'{training_data_dir}/{dataset_name}'
    os.makedirs(current_data_dir, exist_ok=True)

    # Directory contains visuals of images and labels (for QA/QC)
    # rendered_data_dir = os.path.join(current_data_dir,"rendered")
    rendered_data_dir = f'{current_data_dir}/rendered'
    

    # ---------------------------------------
    # Workflow
    # ---------------------------------------
    # image_paths = sv.list_files_with_extensions(directory=input_dir, extensions=["png", "jpg", "jpeg"])
    # print('\nFound {} images'.format(len(image_paths)))





    if CREATE_LABELS:
        # Make copies of the extracted frames, make in batches of N
        # This has to be done because the auto labeler is RAM heavy
        batch_and_copy_images(args.dir, batched_dir, batch_size=args.bsize, file_ext=args.file_ext)
        temporary_image_folders = glob.glob(f"{batched_dir}/images_*")
        print("Batch Folders Found: ", len(temporary_image_folders))

        if args.detect:
            # Initialize the foundational base model, set the thresholds
            base_model = GroundingDINO(ontology=ont_dict[args.ont],
                                       box_threshold=args.conf_thresh,
                                       text_threshold=args.conf_thresh)
            # For rendering
            include_boxes = True
            include_masks = False

        else:
            # Initialize the foundational base model, set the thresholds
            base_model = GroundedSAM(ontology=ont_dict[args.ont],
                                     box_threshold=args.conf_thresh,
                                     text_threshold=args.conf_thresh)
            
            # For rendering
            include_boxes = False
            include_masks = True

        # Loop through the temp folders of images
        for temporary_image_folder in temporary_image_folders:
            print(f'\nGenerating labels for: {temporary_image_folder}')
            # Create labels for the images in temp folder
            dataset = base_model.label(input_folder=temporary_image_folder,
                                       extension="."+args.file_ext,
                                       output_folder=auto_labeled_dir,
                                       sahi=args.use_sahi)
            
            # --- LABEL FILTERING AND NMS (Existing Logic) ---
            
            # 1. Remap class_id using captions for filtering
            ordered_captions = list(dict(ont_dict[args.ont].promptMap).keys())
            for path, image, annotations in dataset:
                annotations.class_id = [
                    caption_to_id.get(ordered_captions[cid], 0)
                    for cid in annotations.class_id
                ]

                # 3. Filter based on NMS
                predictions = np.column_stack((annotations.xyxy, annotations.confidence))
                indices = non_max_suppression(predictions, args.nms_thresh, image_shape=image.shape, min_area=args.area_threshold_minimum, max_area=args.area_threshold_maximum)

                # If no indices kept, clear annotations in-place
                if len(indices) == 0:
                    # Ensure annotations object is emptied so dataset consumers see the change
                    annotations.xyxy = np.empty((0, 4))
                    annotations.mask = None
                    annotations.confidence = np.array([])
                    annotations.class_id = np.array([], dtype=int)
                    annotations.tracker_id = None
                    annotations.data = {}
                else:
                    # Get filtered detections
                    if args.detect:
                        filtered = filter_detections_by_indices(annotations, indices, task='detect')
                    else:
                        filtered = filter_detections_by_indices(annotations, indices, task='segment')

                    # Mutate the existing annotations object so dataset reflects changes
                    annotations.xyxy = np.array(filtered.xyxy).reshape(-1, 4) if filtered.xyxy is not None else np.empty((0, 4))
                    annotations.mask = np.array(filtered.mask) if getattr(filtered, 'mask', None) is not None else None
                    annotations.confidence = np.array(filtered.confidence) if getattr(filtered, 'confidence', None) is not None else np.array([])
                    annotations.class_id = np.array(filtered.class_id, dtype=int) if getattr(filtered, 'class_id', None) is not None else np.array([], dtype=int)
                    annotations.tracker_id = np.array(filtered.tracker_id) if getattr(filtered, 'tracker_id', None) is not None else None
                    annotations.data = filtered.data if getattr(filtered, 'data', None) else {}
            
            # 4. Final Class ID Sanity Check/Remapping
            if len(set(dataset.classes)) == 1:
                dataset.classes = [f'{dataset_name}']

            for _, _, detections in dataset:
                if hasattr(detections, "data") and "caption" in detections.data:
                    captions = detections.data["caption"]
                    detections.class_id = [caption_to_id.get(caption, 0) for caption in captions]

            # --- END LABEL FILTERING ---

            # 5. Save the filtered dataset to the YOLO structure (pre-split)
            if SAVE_LABELS:
                dataset.as_yolo(os.path.join(current_data_dir, "images"),
                                os.path.join(current_data_dir, "annotations"),
                                min_image_area_percentage=0.01,
                                data_yaml_path=os.path.join(current_data_dir, "data.yaml"))

            # 6. Display the filtered dataset (renders images)
            render_dataset(dataset,
                           rendered_data_dir,
                           class_map=caption_to_id,
                           include_boxes=include_boxes,
                           include_masks=include_masks,
                           col_map=cmap)
            
            # --- PAUSE, REVIEW, AND CLEAN (New/Reorganized Logic) ---
            
            # 7. Split the filtered dataset into training / valid.
            # This creates the train/valid folders that `remove_bad_data` expects.
            if SAVE_LABELS:
                print("\nSplitting data for quality check...")
                if args.detect:
                    helpers.split_data(current_data_dir, record_confidence=True, file_extension=args.file_ext)
                else:
                    helpers.split_data(current_data_dir, record_confidence=False, file_extension=args.file_ext)
            
            # Delete the temporary copies from the batch folder
            print(f"Deleting temporary image folder: {temporary_image_folder}")
            shutil.rmtree(temporary_image_folder)

        # 8. PAUSE for manual review
        if args.response_required:
            input_prompt = (
                f"\n**************************************************************\n"
                f"ACTION REQUIRED: Please review the rendered images in:\n"
                f"    Rendered Folder: {rendered_data_dir}\n"
                f"-> DELETE any unwanted image/annotation files from this FOLDER.\n"
                f"The script will automatically remove corresponding files from\n"
                f"the 'train' and 'valid' folders based on what you delete here.\n"
                f"Press ENTER to continue after cleanup...\n"
                f"**************************************************************\n"
            )
            input(input_prompt)

        # 9. Clean up the data based on what was deleted from the rendered folder
        if SAVE_LABELS:
            print("\nCleaning up training/validation data based on deleted rendered files...")
            remove_bad_data(current_data_dir, args.file_ext)

        # Remove the main temporary batch directory when done with all batches
        if os.path.exists(batched_dir):
            shutil.rmtree(batched_dir)
            
    print("Done.")






    # if CREATE_LABELS:
    #     # Make copies of the extracted frames, make in batches of N
    #     # This has to be done because the auto labeler is RAM heavy
    #     batch_and_copy_images(args.dir, batched_dir, batch_size=args.bsize, file_ext=args.file_ext)
    #     temporary_image_folders = glob.glob(f"{batched_dir}/images_*")
    #     # temporary_image_folders = [batched_dir]
    #     print("Batch Folders Found: ", len(temporary_image_folders))

    #     if args.detect:
    #         # Initialize the foundational base model, set the thresholds
    #         base_model = GroundingDINO(ontology=ont_dict[args.ont],
    #                                    box_threshold=args.conf_thresh,
    #                                    text_threshold=args.conf_thresh)
    #         # For rendering
    #         include_boxes = True
    #         include_masks = False

    #     else:
    #         # Initialize the foundational base model, set the thresholds
    #         base_model = GroundedSAM(ontology=ont_dict[args.ont],
    #                                  box_threshold=args.conf_thresh,
    #                                  text_threshold=args.conf_thresh)
            
    #         # For rendering
    #         include_boxes = False
    #         include_masks = True

    #     # Loop through the temp folders of images
    #     for temporary_image_folder in temporary_image_folders:
    #         print(f'\nGenerating labels for: {temporary_image_folder}')
    #         # Create labels for the images in temp folder
    #         dataset = base_model.label(input_folder=temporary_image_folder,
    #                                    extension="."+args.file_ext,
    #                                    output_folder=auto_labeled_dir,
    #                                    sahi=args.use_sahi)
            
    #         # print('729 dataset_classes: ', dataset.classes)
    #         # print(type(dataset))
            
    #         # Remap the class_id to unified IDs
    #         # for _, _, detections in dataset:
    #         #     if "caption" in detections.data:
    #         #         print(detections.data.keys())
    #         #         captions = detections.data["caption"]
    #         #         detections.class_id = [caption_to_id.get(caption, 0) for caption in captions]
    #         #     else:
    #         #         print("WARNING: No captions found in detections")

    #         # print(len(list(dataset.images.keys())))
    #         # Delete the temporary copies
    #         # shutil.rmtree(temporary_image_folder)

    #         # Filter the dataset
    #         # print(len(dataset))

    #         # for _, _, detections in dataset:
    #         #     detections.class_id = [
    #         #         caption_to_id.get(ordered_captions[cid], 0)
    #         #         for cid in detections.class_id
    #         #     ]
    #         # print("✅ Collapsed class_id:", detections.class_id)

    #         # for image_name in tqdm(image_names):
    #         for path, image, annotations in dataset:
    #             annotations.class_id = [
    #                 caption_to_id.get(ordered_captions[cid], 0)
    #                 for cid in annotations.class_id
    #             ]
    #             # numpy arrays for this image
    #             # class_id = annotations.class_id
    #             # print('752 class_id: ', class_id)

    #             # Filter based on area and confidence (removes large and unconfident)
    #             if args.detect:
    #                 annotations = filter_detections(image, annotations, area_thresh=args.area_thresh, conf_thresh=args.conf_thresh)

    #             # Filter based on NMS (removes all the duplicates, faster than with_nms)
    #             predictions = np.column_stack((annotations.xyxy, annotations.confidence))
    #             indices = non_max_suppression(predictions, args.nms_thresh)
                
    #             mask = [i in indices for i in range(len(annotations))]

    #             if len(indices) > 0:
    #                 # annotations = annotations[indices]
    #                 annotations = filter_detections_by_indices(annotations, indices)

    #                 # annotations = annotations
    #                 # annotations.class_id = np.zeros_like(class_id)
    #                 # print('765 annotation_class: ', annotations.class_id)
    #             else:
    #                 annotations = None
                    
    #         # Change the dataset classes
    #         if len(set(dataset.classes)) == 1:
    #             dataset.classes = [f'{dataset_name}']
    #         # print('773 dataset.classes: ', dataset.classes)

    #         for _, _, detections in dataset:
    #             if hasattr(detections, "data") and "caption" in detections.data:
    #                 captions = detections.data["caption"]
    #                 detections.class_id = [caption_to_id.get(caption, 0) for caption in captions]
    #         # print('782 dataset.classes: ', dataset.classes)

    #         if SAVE_LABELS:
    #             # Save the filtered dataset (this is used for training)
    #             dataset.as_yolo(current_data_dir + "/images",
    #                             current_data_dir + "/annotations",
    #                             min_image_area_percentage=0.01,
    #                             data_yaml_path=current_data_dir + "/data.yaml")

    #         # Display the filtered dataset
    #         render_dataset(dataset,
    #                        rendered_data_dir,
    #                     #    class_map=class_name_to_id,
    #                        class_map=caption_to_id,
    #                        include_boxes=include_boxes,
    #                        include_masks=include_masks,
    #                        col_map=cmap)

    #     if SAVE_LABELS:
    #         # Split the filtered dataset into training / valid
    #         if args.detect:
    #             helpers.split_data(current_data_dir, record_confidence=True, file_extension=args.file_ext)
    #         else:
    #             helpers.split_data(current_data_dir, record_confidence=False, file_extension=args.file_ext)

    #         # -----------------------------------------
    #         # Manually delete any images as needed!
    #         # -----------------------------------------
    #         if args.response_required:
    #             response = input("Delete any bad labeled frames from {} now...".format(current_data_dir))
    #         # Remove images and labels from train/valid if they were deleted from rendered
    #         remove_bad_data(current_data_dir, args.file_ext)

    # print("Done.")
