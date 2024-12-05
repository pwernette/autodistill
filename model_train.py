from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from ultralytics import YOLO
import os, shutil, tqdm
import numpy as np
# import supervision as sv
# from supervision.detection.utils import non_max_suppression

import shutil
# import supervision as sv

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations


# basedir = "/mnt/b/RockFinder/"
basedir = "B:/RockFinder/"
imgdir = os.path.join(basedir,"images","images_resize_05")
# imgdir = os.path.join(basedir,"images","images_resize_075")
# imgdir = os.path.join(basedir,"images","images_full")
dsetdir = os.path.join(basedir,"datasets")

# ont = {"rock": "rock",
#        "shell": "shell", 
#        "mussel": "mussel",
#        "fish": "fish",
#        "plant": "plant"}
# ont = {"rock": "rock"}
ont = {"rock": "rock",
       "tiny rock": "rock",
       "small rock": "rock",
       "big rock": "rock",
       "fuzzy rock": "rock",
       "smooth rock": "rock"}

# get length of output class labels
ont_list = list(set(lab for lab in {ont.values()} for lab in lab))

# Polygon's size as a ratio of the image
# Large polygons shouldn't be included...
area_thresh = 0.4
box_threshold = 0.1
text_threshold = 0.1

# Non-maximum suppression threshold
nms_thresh = 0.1
num_epochs = 200
use_sahi = True

foundational_model = 'dino'
model_type = 'detect'
model_level = 'x'  # options are: n, s, m, l, x (in order of increasing complexity)



def filter_detections(image, annotations, area_thresh, conf_thresh=0.0):
    """
    :param image:
    :param annotations:
    :param area_thresh:
    :param conf_thresh:
    :return annotations:
    """

    height, width, channels = image.shape
    image_area = height * width

    # Filter by area
    annotations = annotations[(annotations.box_area / image_area) < area_thresh]

    # Filter by confidence
    annotations = annotations[annotations.confidence > conf_thresh]

    return annotations


if 'dino' in foundational_model:
  from autodistill_grounding_dino import GroundingDINO
  base_model = GroundingDINO(ontology=CaptionOntology(ont),
                             box_threshold=text_threshold, 
                             text_threshold=text_threshold)
  outname = "autodistill_dino_"+str(len(ont_list))+"class"
elif 'sam' in foundational_model:
  from autodistill_grounded_sam import GroundedSAM
  base_model = GroundedSAM(ontology=CaptionOntology(ont), 
                           box_threshold=text_threshold, 
                           text_threshold=text_threshold)
  outname = "autodistill_sam_"+str(len(ont_list))+"class"
elif 'sam' in foundational_model and '2' in foundational_model:
  from autodistill_grounded_sam_2 import GroundedSAM2
  base_model = GroundedSAM2(ontology=CaptionOntology(ont),
                            box_threshold=text_threshold, 
                            text_threshold=text_threshold)
  outname = "autodistill_sam2_"+str(len(ont))+"class"
elif 'detic' in foundational_model:
  from autodistill_detic import DETIC
  base_model = DETIC(ontology=CaptionOntology(ont),
                     box_threshold=text_threshold, 
                     text_threshold=text_threshold)
  outname = "autodistill_detic_"+str(len(ont_list))+"class"
# For rendering
include_boxes = False
include_masks = True

if use_sahi:
  outname = outname+"_sahi"
datadir = os.path.join(dsetdir, outname)
prjdir = os.path.join(basedir, "trained_models_detect")

print('\nOutput data directory:\n  {}'.format(dsetdir))
print('Output trained model directory:\n  {}'.format(prjdir))
print('Outputs name:\n  {}'.format(outname))
print('\nUsing SAHI?: {}'.format(use_sahi))
print('Ontology:')
for i,v in ont.items():
  print('  {} --> {}'.format(i,v))

print(ont_list)

if os.path.isdir(datadir):
  print('\nRemoving existing directory: {}'.format(datadir))
  shutil.rmtree(datadir, ignore_errors=True)

# label all images in the input folder
if use_sahi:
  print('\nGenerating labelled dataset using Grounded SAM with SAHI...\n')
else:
  print('\nGenerating labelled dataset using Grounded SAM...\n')

dataset = base_model.label(
  input_folder=imgdir,
  output_folder=datadir,
  # extension=".png",
  record_confidence=True,
  sahi=use_sahi
)

print(len(dataset))

# Filter the dataset
image_names = list(dataset.images.keys())

# for image_name in tqdm.tqdm(image_names):
#     # numpy arrays for this image
#     image = dataset.images[image_name]
#     annotations = dataset.annotations[image_name]
#     class_id = dataset.annotations[image_name].class_id

#     # Filter based on area and confidence (removes large and unconfident)
#     annotations = filter_detections(image, annotations, area_thresh)

#     # Filter based on NMS (removes all the duplicates, faster than with_nms)
#     predictions = np.column_stack((annotations.xyxy, annotations.confidence))
#     indices = non_max_suppression(predictions, nms_thresh)
#     annotations = annotations[indices]

#     # Update the annotations and class IDs in dataset
#     dataset.annotations[image_name] = annotations
#     dataset.annotations[image_name].class_id = np.zeros_like(class_id)

# Change the dataset classes
# dataset.classes = [f'{outname}']

print(len(dataset))

if os.path.isdir(os.path.join(prjdir, outname)):
  print('\nRemoving existing directory: {}'.format(os.path.join(prjdir, outname)))
  shutil.rmtree(os.path.join(prjdir, outname), ignore_errors=True)

if 'det' in model_type.lower():
  target_model = YOLO("yolov8x.pt")
elif 'seg' in model_type.lower():
  target_model = YOLO("yolov8x-seg.pt")
else:
  # default
  target_model = YOLO("yolov8x.pt")

target_model.train(data=os.path.join(datadir,"data.yaml"), 
                             device=0,
                             epochs=num_epochs, 
                             patience=int(num_epochs * .3),
                             batch=16,
                             imgsz=1632,
                             project=prjdir, 
                             name=outname,
                             plots=True,
                             single_cls=True)
# except Exception as e:
#   print('\nNo GroundedSAM model trained.')
#   print(e)




'''
Foundational Model: DETIC
'''
# from autodistill_detic import DETIC
# # try:
# base_model = DETIC(ontology=CaptionOntology(ont))
# # For rendering
# # include_boxes = False
# # include_masks = True

# outname = "autodistill_detic_"+str(len(ont))+"class"
# if use_sahi:
#   outname = outname+"_sahi"
# datadir = os.path.join(dsetdir, outname)
# prjdir = os.path.join(basedir, "trained_models_detect")

# print('\nOutput data directory:\n  {}'.format(dsetdir))
# print('Output trained model directory:\n  {}'.format(prjdir))
# print('Outputs name:\n  {}'.format(outname))
# print('\nUsing SAHI?: {}'.format(use_sahi))
# print('Ontology:')
# for i,v in ont.items():
#   print('  {} --> {}'.format(i,v))

# if os.path.isdir(datadir):
#   shutil.rmtree(datadir, ignore_errors=True)

# # label all images in a folder called `context_images`
# if use_sahi:
#   print('\nGenerating labelled dataset using DETIC with SAHI...\n')
# else:
#   print('\nGenerating labelled dataset using DETIC...\n')

# dataset = base_model.label(
#   input_folder=imgdir,
#   output_folder=datadir,
#   record_confidence=True,
#   sahi=use_sahi
# )

# # # Filter the dataset
# # image_names = list(dataset.images.keys())

# # for image_name in tqdm(image_names):
# #     # numpy arrays for this image
# #     image = dataset.images[image_name]
# #     annotations = dataset.annotations[image_name]
# #     class_id = dataset.annotations[image_name].class_id

# #     # Filter based on area and confidence (removes large and unconfident)
# #     annotations = filter_detections(image, annotations, area_thresh)

# #     # Filter based on NMS (removes all the duplicates, faster than with_nms)
# #     predictions = np.column_stack((annotations.xyxy, annotations.confidence))
# #     indices = sv.detection.utils.box_non_max_suppression(predictions, nms_thresh)
# #     annotations = annotations[indices]

# #     # Update the annotations and class IDs in dataset
# #     dataset.annotations[image_name] = annotations
# #     dataset.annotations[image_name].class_id = np.zeros_like(class_id)

# # # Change the dataset classes
# # dataset.classes = [f'{outname}']

# if os.path.isdir(os.path.join(prjdir, outname)):
#   shutil.rmtree(os.path.join(prjdir, outname), ignore_errors=True)

# target_model = YOLO("yolov8x-seg.pt")
# results = target_model.train(data=os.path.join(datadir,"data.yaml"), 
#                              device=0,
#                              epochs=num_epochs, 
#                              patience=int(num_epochs * .3),
#                              batch=16,
#                              imgsz=2000,
#                              project=prjdir, 
#                              name=outname,
#                              plots=True,
#                              single_cls=True)
# # except Exception as e:
# #   print('\nNo GroundedSAM model trained.')
# #   print(e)






'''
Foundational Model: Grounded SAM-2
'''
# from autodistill_grounded_sam_2 import GroundedSAM2
# try:
# base_model = GroundedSAM2(ontology=CaptionOntology(ont))

# outname = "autodistill_sam2_"+str(len(ont))+"class"
# if use_sahi:
#   outname = outname+"_sahi"
# datadir = os.path.join(dsetdir,outname)
# prjdir = os.path.join(basedir,"trained_models_detect")

# if os.path.isdir(datadir):
#   shutil.rmtree(datadir, ignore_errors=True)

# # label all images in a folder called `context_images`
# print('\nGenerating labelled dataset using Grounded SAM-2 with SAHI...\n')
# base_model.label(
#   input_folder=imgdir,
#   output_folder=datadir,
#   sahi=use_sahi
# )

# if os.path.isdir(os.path.join(prjdir, outname)):
#   shutil.rmtree(os.path.join(prjdir, outname), ignore_errors=True)

# target_model = YOLO("yolov8x-seg.pt")
# target_model.train(data=os.path.join(datadir,"data.yaml"), 
#                   epochs=200, 
#                   project=prjdir, name=outname)
# except Exception as e:
#   print('\nNo GroundedSAM-2 model trained.')
#   print(e)



'''
Foundational Model: Grounding DINO
'''
# from autodistill_grounding_dino import GroundingDINO
# try:
#   base_model = GroundingDINO(ontology=CaptionOntology(ont))

#   outname = "autodistill_dino_"+str(len(ont))+"class"
#   if use_sahi:
#     outname = outname+"_sahi"
#   datadir = os.path.join(dsetdir,outname)
#   prjdir = os.path.join(basedir,"trained_models_detect")

#   # label all images in a folder called `context_images`
#   print('\nGenerating labelled dataset using Grounding DINO with SAHI...')
#   base_model.label(
#     input_folder=imgdir,
#     output_folder=datadir,
#     sahi=use_sahi
#   )

#   print('   Training YOLO model with ')
#   target_model = YOLO("yolov8x-seg.pt")
#   target_model.train(data=os.path.join(datadir,"data.yaml"), 
#                     epochs=200, 
#                     project=prjdir, name=outname)
# except Exception as e:
#   print('\nNo GroundingDINO model trained.')
#   print(e)



'''
Foundational Model: Efficient SAM
'''
# from autodistill_efficientsam import EfficientSAM
# try:
# base_model = EfficientSAM(ontology=CaptionOntology(ont))

# outname = "autodistill_efficientsam_"+str(len(ont))+"class"
# if use_sahi:
#   outname = outname+"_sahi"
# datadir = os.path.join(dsetdir, outname)
# prjdir = os.path.join(basedir, "trained_models_detect")

# if os.path.isdir(datadir):
#   os.removedirs(datadir)

# # label all images in a folder called `context_images`
# print('\nGenerating labelled dataset using Grounded SAM with SAHI...\n')
# base_model.label(
#   input_folder=imgdir,
#   output_folder=datadir,
#   sahi=use_sahi
# )

# if os.path.isdir(os.path.join(prjdir, outname)):
#   os.removedirs(os.path.join(prjdir, outname))

# target_model = YOLO("yolov8x-seg.pt")
# target_model.train(data=os.path.join(datadir,"data.yaml"), 
#                   epochs=200, 
#                   project=prjdir, name=outname)
# except Exception as e:
#   print('\nNo GroundedSAM model trained.')
#   print(e)



# run inference on the new model
# pred = target_model.predict("./dataset/valid/your-image.jpg", confidence=0.5)
# print(pred)

# # optional: upload your model to Roboflow for deployment
# from roboflow import Roboflow

# rf = Roboflow(api_key="API_KEY")
# project = rf.workspace().project("PROJECT_ID")
# project.version(DATASET_VERSION).deploy(model_type="yolov8", model_path=f"./runs/detect/train/")