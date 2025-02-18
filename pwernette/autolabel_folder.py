import os, cv2

import autodistill
from autodistill_yolov8 import YOLOv8
from autodistill.detection import CaptionOntology
from autodistill.utils import plot


# define an ontology to map class names to our DETIC prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
ont_person = CaptionOntology({
    "person":"person",
})
ont_waves = CaptionOntology({
    "wave":"wave",
    "waves":"wave",
    "big wave":"wave",
    "small wave":"wave",
    "white water":"wave",
    "breaking wave":"wave"
})
ont_grapes = CaptionOntology({
    "grape": "grape",
    "grapes": "grapes",
    "grape cluster": "grapes",
})


''' Update these parameters '''
use_ontology = ont_grapes
base_model_type = 'sam'
# folder_to_label = '/mnt/c/Users/werne/OneDrive/Documents/coastal_cameras/test_images - Copy'
# folder_to_place = '/mnt/c/Users/werne/OneDrive/Documents/coastal_cameras/test_images - outputs'
folder_to_label = '/mnt/d/greeen/autodist_orig'
folder_to_place = '/mnt/d/greeen/autodist_out'
image_extension = '.JPG'
''' end update parameters '''


if base_model_type == 'sam':
    from autodistill_grounded_sam import GroundedSAM
    base_model = GroundedSAM(
        ontology = use_ontology
    )
elif base_model_type == 'detic':
    from autodistill_detic import DETIC
    base_model = DETIC(
        ontology = use_ontology
    )
elif base_model_type == 'dino':
    from autodistill_grounding_dino import GroundingDINO
    base_model = GroundingDINO(
        ontology = use_ontology
    )
elif base_model_type == 'sam2':
    from autodistill_grounded_sam_2 import GroundedSAM2
    base_model = GroundedSAM2(
        ontology = use_ontology
    )

# plot(
#     image=cv2.imread("logistics.jpeg"),
#     classes=base_model.ontology.classes(),
#     detections=results
# )

# base_model.label(input_folder=folder_to_label, 
#                  extension=image_extension,
#                  output_folder=folder_to_place)

# base_model.label('/c/Users/werne/OneDrive/Documents/coastal_cameras/test_images - Copy/202407151150.jpg')

# import supervision as sv
# from tqdm.notebook import tqdm

# image = cv2.imread('/mnt/c/Users/werne/OneDrive/Documents/coastal_cameras/test_images - Copy/202407151150.jpg')

# classes = ['person']

# detections = base_model.predict('/mnt/c/Users/werne/OneDrive/Documents/coastal_cameras/test_images - Copy/202407151150.jpg')

# box_annotator = sv.BoxAnnotator()

# labels = [f"{classes[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]

# annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# sv.plot_image(annotated_frame)

target_model = YOLOv8("yolov8n.pt")
target_model.train(folder_to_place, epochs=100)

# from IPython.display import Image

# Image(filename='{HOME}/runs/detect/train/confusion_matrix.png', width=600)
