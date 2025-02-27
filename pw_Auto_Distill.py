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

import supervision as sv
from supervision.detection.utils import *
from autodistill import helpers
from autodistill_grounded_sam import GroundedSAM
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology



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


def non_max_suppression(predictions: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
  rows,_ = predictions.shape

  sort_index = np.flip(predictions[:, 4].argsort())
  predictions = predictions[sort_index]

  boxes = predictions[:, :4]
  categories = predictions[:, 4]
  ious = box_iou_batch(boxes, boxes)
  ious = ious - np.eye(rows)

  keep = np.ones(rows, dtype=bool)

  for index, (iou, category) in enumerate(zip(ious, categories)):
      if not keep[index]:
          continue

      condition = (iou > iou_threshold) & (categories == category)
      keep = keep & ~condition

  return keep[sort_index.argsort()]

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


def batch_and_copy_images(source_folder, output_folder, batch_size=64):
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


def filter_detections(filt_image, annotations, area_thresh):
    """

    :param image:
    :param annotations:
    :param area_thresh:
    :param conf_thresh:
    :return annotations:
    """
    # print('\nFiltering detection for {} with area:{}'.format(image, area_thresh))

    # height, width = Image.open(image).size
    # print(image.shape)
    height,width,_ = filt_image.shape

    # Filter by area
    annotations = annotations[(annotations.box_area / (height * width)) < area_thresh]

    # Filter by confidence
    # print(annotations.confidence)
    annotations = annotations[annotations.confidence > conf_thresh]

    return annotations


def render_dataset(dataset, output_dir, include_boxes=True, include_masks=False):
    """

    :param dataset:
    :param output_dir:
    :param include_boxes:
    :param include_masks:
    :return:
    """
    # Images
    # image_names = list(dataset.images.keys())

    # Create the annotation object
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()

    with sv.ImageSink(target_dir_path=output_dir, overwrite=False) as sink:
        for path, imag, annot in dataset:
        # for i_idx, image_name in enumerate(image_names):

            # Get the images and annotation
            # image = dataset.images[image_name]
            # annotations = dataset.annotations[image_name]

            if include_boxes:
                # Get the boxes for the annotations
                imag = box_annotator.annotate(scene=imag, detections=annot)

            if include_masks:
                # Get the masks for the annotations
                imag = mask_annotator.annotate(scene=imag, detections=annot)

            # output_file = os.path.basename(image_name)
            output_file = os.path.basename(path)
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
    print('found {} train images in {}'.format(len(train_images),os.path.join(train_dir,"images")))

    for img, lab in zip(train_images, train_labels):
        basename = os.path.basename(img).split(".")[0]
        print('training image: {}'.format(basename))
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
    render_images = glob.glob(os.path.join(render_dir,"*."+fileext))
    print('found {} rendered images in {}'.format(len(render_images),render_dir))

    print('combined_dict {}'.format(combined_dict))

    print('pre-filter: combined_dict length = {}'.format(len(combined_dict)))

    # Loop through the rendered images and removes those that
    # exist from the combined dictionary.
    for render_image in render_images:
        basename = os.path.basename(render_image).split(".")[0]
        print('rendered image: {}'.format(basename))
        if basename in combined_dict:
            combined_dict.pop(basename)
    
    print('post-filter: combined_dict length = {}'.format(len(combined_dict)))

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
                        dest='dir', type=str, help='Directory to process')
    parser.add_argument('-ont',  
                        dest='ont', type=str, choices=['grapes','rocks','mussels','fish','shells','trees'], default='grapes', 
                        help='Ontology to use')
    parser.add_argument('-detect', action='store_true', help='Detect objects')
    parser.add_argument('-segment', action='store_true', help='Segment objects')
    parser.add_argument('-area_thresh',  
                        dest='area_thresh', type=float, default=0.01, help='Area threshold for detections')
    parser.add_argument('-conf_thresh',  
                        dest='conf_thresh', type=float, default=0.35, help='Confidence threshold for detections')
    parser.add_argument('-nms_thresh',  
                        dest='nms_thresh', type=float, default=0.5, help='NMS threshold for detections')
    parser.add_argument('-bsize',  
                        dest='bsize', type=int, default=32, help='Batch size')
    parser.add_argument('-use_sahi', action='store_true', help='Use SAHI')
    parser.add_argument('-file_ext',  
                        dest='file_ext', type=str, default='JPG', help='File extension')

    args = parser.parse_args()

    assert args.detect or args.segment, ValueError('Specify either -detect or -segment')

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
        'grapes': CaptionOntology({
            "grapes": "grapes",
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
    }

    model_name_dict = {
        'grapes': "GrapeMapper",
        'rocks': "RockMapper",
        'mussels': "MusselMapper",
        'fish': "FishMapper",
        'shells': "ShellMapper",
        'trees': "TreeMapper",
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
    ontology = ont_dict[args.ont]

    # Polygon's size as a ratio of the image
    # Large polygons shouldn't be included...
    # area_thresh = 0.4
    # area_thresh = 0.01
    area_thresh = 0.01

    # conf_thresh = 0.15
    # conf_thresh = 0.3
    conf_thresh = 0.35
    # conf_thresh = 0.6

    # Non-maximum suppression threshold
    # nms_thresh = 0.1
    # nms_thresh = 0.9
    nms_thresh = 0.5

    # Extract every N frames
    frame_stride = 15
    
    # batch size
    bsize = 32

    # use SAHI
    use_sahi = False

    file_ext = 'JPG'


    # Get the root data directory (Data); OCD
    # rdir = os.path.dirname("B:/RockFinder/images")
    # root = root.replace("\\", "/")
    # rdir = "B:/RockFinder/images"
    # rdir = "G:/sfm_agu/images_a/raw_dup"
    
    # rdir = "D:/sfm_deer/rgb_e85s70"
    
    # rdir = "/mnt/d/greeen/autodist_orig"
    rdir = "/mnt/h/GrapeFinder/CabFranc_original"
    print('\nRoot dir = {}'.format(args.dir))

    # model_name_base = 'RockFinder'
    # model_name_base = 'FishFinder'
    # model_name_base = 'DeerMapper'
    model_name_base = 'GrapeMapper'

    # Converted videos from TATOR get placed here
    # converted_video_dir = f"{root}/Converted_Videos"
    # os.makedirs(converted_video_dir, exist_ok=True)

    # Extracted frames from Converted videos go here
    # input_dir = os.path.join(rdir,"images_resize_05_png")
    # input_dir = os.path.join(rdir,"images_resize_03")
    # input_dir = f'{rdir}/images_resize_05'
    input_dir = rdir
    os.makedirs(input_dir, exist_ok=True)
    print('\nInput directory = {}'.format(args.dir))

    # Frames are batched (RAM) and temporarily placed here
    # batched_dir = os.path.join(rdir,"images_resize_05_png_b"+str(bsize))
    # batched_dir = os.path.join(rdir,"images_resize_03_b"+str(bsize))
    # batched_dir = f'{rdir}/images_resize_05_b{str(bsize)}'
    batched_dir = f'{args.dir}/images_b{str(bsize)}'

    # If it exists from last time (exited early) delete
    if os.path.exists(batched_dir):
        shutil.rmtree(batched_dir)

    # Create the directory
    os.makedirs(batched_dir, exist_ok=True)

    # Auto labeled data; this is also temporary until being filtered
    # auto_labeled_dir = os.path.join(rdir,"Auto_Labeled")
    if use_sahi:
        auto_labeled_dir = f'{args.dir}/{model_name_base}_Auto_Labeled_05_sahi'
    else:
        auto_labeled_dir = f'{args.dir}/{model_name_base}_Auto_Labeled_05'
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
    if use_sahi:
        dataset_name = model_name_base+"_05_sahi_"+args.file_ext
    else:
        dataset_name = model_name_base+"_05_"+args.file_ext

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
        batch_and_copy_images(args.dir, batched_dir, batch_size=args.bsize)
        temporary_image_folders = glob.glob(f"{batched_dir}/images_*")
        # temporary_image_folders = [batched_dir]
        print("Batch Folders Found: ", len(temporary_image_folders))

        if DETECTION:
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
                                       sahi=use_sahi)
            # print(len(list(dataset.images.keys())))
            # Delete the temporary copies
            # shutil.rmtree(temporary_image_folder)

            # Filter the dataset
            # image_names = list(dataset.images.keys())
            # print(len(image_names))
            print(len(dataset))

            # for image_name in tqdm(image_names):
            for path, image, annotations in dataset:
                # numpy arrays for this image
                class_id = annotations.class_id

                # Filter based on area and confidence (removes large and unconfident)
                if args.detect:
                    annotations = filter_detections(image, annotations, args.area_thresh)

                # Filter based on NMS (removes all the duplicates, faster than with_nms)
                predictions = np.column_stack((annotations.xyxy, annotations.confidence))
                indices = non_max_suppression(predictions, args.nms_thresh)
                if len(indices) > 0:
                    annotations = annotations[indices]
                    # annotations = annotations
                    annotations.class_id = np.zeros_like(class_id)
                else:
                    annotations = None
                    
            # Change the dataset classes
            dataset.classes = [f'{dataset_name}']

            if SAVE_LABELS:
                # Save the filtered dataset (this is used for training)
                dataset.as_yolo(current_data_dir + "/images",
                                current_data_dir + "/annotations",
                                min_image_area_percentage=0.01,
                                data_yaml_path=current_data_dir + "/data.yaml")

            # Display the filtered dataset
            render_dataset(dataset,
                           rendered_data_dir,
                           include_boxes=include_boxes,
                           include_masks=include_masks)

        if SAVE_LABELS:
            # Split the filtered dataset into training / valid
            if args.detect:
                helpers.split_data(current_data_dir, record_confidence=True, file_extension=args.file_ext)
            else:
                helpers.split_data(current_data_dir, record_confidence=False, file_extension=args.file_ext)

            # -----------------------------------------
            # Manually delete any images as needed!
            # -----------------------------------------
            response = input("Delete any bad labeled frames from {} now...".format(current_data_dir))
            # Remove images and labels from train/valid if they were deleted from rendered
            remove_bad_data(current_data_dir, args.file_ext)

    print("Done.")
