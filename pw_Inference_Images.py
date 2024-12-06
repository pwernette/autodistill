import os
import argparse
from PIL import Image
from tqdm import tqdm

import numpy as np

import supervision as sv
from supervision.detection.utils import *
from ultralytics import YOLO

from pw_Auto_Distill import filter_detections


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def process_dir(model_dir, input_dir, output_dir, task, owrite=False, model_weights_type='best', img_size=None, conf=0.3, iou=0.7):
    """

    :param source_weights:
    :param source_video:
    :param output_dir:
    :param task:
    :param start_at:
    :param end_at:
    :param conf:
    :param iou:
    :return:
    """

    # Create the target path
    # target_path = f"{output_dir}/{os.path.basname(model_dir)}"
    target_path = os.path.join(output_dir, os.path.basename(model_dir))
    # Create the target directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(os.path.join(model_dir, 'weights', model_weights_type+'.pt'))
    assert os.path.exists(os.path.join(model_dir, 'weights', model_weights_type+'.pt'))
    # Load the model
    model = YOLO(os.path.join(model_dir, 'weights', model_weights_type+'.pt'))

    if task == 'detect':
        # Load the tracker
        tracker = sv.ByteTrack()
        # Create the annotator for detection
        box_annotator = sv.BoundingBoxAnnotator()
        # Adds label to annotation (tracking)
        labeler = sv.LabelAnnotator()
    elif task == 'segment':
        # Create the annotators for segmentation
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoundingBoxAnnotator()
    else:
        raise Exception("ERROR: Specify --task [detect, segment]")
    
    # Area threshold
    area_thresh = 1.1

    imgs = sv.list_files_with_extensions(directory=input_dir, extensions=["png", "jpg", "jpeg"])
    
    # get image dimensions
    if img_size is None:
        img_size = Image.open(imgs[0]).size
    print('Using images with size: {}'.format(img_size))

    # with sv.ImageSink(target_dir_path=output_dir, overwrite=owrite) as sink:
    # Loop through all the images
    for img in tqdm(imgs, total=len(imgs)):
        with Image.open(img) as imag:
            # Run the frame through the model and make predictions
            result = model(imag,
                            conf=conf,
                            iou=iou,
                            imgsz=img_size,
                            half=True,
                            augment=False,
                            max_det=1000,
                            verbose=False,
                            show=False)[0]

            # Version issues
            result.obb = None

            # Convert the results
            detections = sv.Detections.from_ultralytics(result)

            # Filter the detections
            detections = filter_detections(img, detections, area_thresh)

            if task == 'detect':
                # Track the detections
                detections = tracker.update_with_detections(detections)
                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

                # Create an annotated version of the frame (boxes)
                annotated_img = box_annotator.annotate(scene=imag.copy(), detections=detections)
                annotated_img = labeler.annotate(scene=annotated_img, detections=detections, labels=labels)
            else:
                # Create an annotated version of the frame (masks and boxes)
                # annotated_img = mask_annotator.annotate(scene=imag.copy(), detections=detections)
                annotated_img = box_annotator.annotate(scene=imag.copy(), detections=detections)

        # Write the frame to the video
        # sink.save_image(image=np.array(annotated_img), image_name=os.path.basename(img))
        annotated_img.save(os.path.join(output_dir,os.path.basename(img)))


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video Processing with YOLO and ByteTrack")

    parser.add_argument('-m','-input_model','-inmodel','-model','-trained_model','-mod',
                        dest='input_model',
                        type=str,
                        required=True,
                        help="Path to the source weights file"
                        )
    parser.add_argument('-wt','-weights','-weightstype',
                        dest='weights_type',
                        type=str,
                        choices=['best','last'],
                        default='best',
                        help="Source weights option [choices: best, last]"
                        )
    parser.add_argument('-i','-input_dir','-indir','-idir','-inputdir','-in',
                        dest='input_directory',
                        type=str,
                        required=True,
                        help="Path to the directory of images (input)"
                        )
    parser.add_argument('-o','-output_dir','-outdir','-odir','-outputdir','-out',
                        dest='output_directory',
                        type=str,
                        required=True,
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
                        default=0.15,
                        help="Confidence threshold for the model"
                        )
    parser.add_argument('-iou',
                        dest='iou',
                        type=float,
                        default=0.3,
                        help="IOU threshold for the model"
                        )

    args = parser.parse_args()

    process_dir(model_dir=args.input_model, 
                input_dir=args.input_directory, 
                output_dir=args.output_directory, 
                task=args.task, 
                owrite=False, 
                model_weights_type=args.weights_type, 
                img_size=None, 
                conf=args.confidence, 
                iou=args.iou)

if __name__ == "__main__":
    main()