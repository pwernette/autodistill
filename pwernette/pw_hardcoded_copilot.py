import os
import cv2
import json
import shutil
import numpy as np
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from pycocotools import mask as mask_utils
from ultralytics import YOLO

# Set up the labeling ontology
ont_dict = {
    'grape': CaptionOntology({
        "grape": "grapes",
        "fuzzy grape": "grapes",
        "fuzzy grape cluster": "grapes",
        "grape cluster": "grapes",
    }),
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

model_name_dict = {
    'grape': "GrapeMapper",
    'grapes': "GrapeMapper",
    'rocks': "RockMapper",
    'mussels': "MusselMapper",
    'fish': "FishMapper",
    'shells': "ShellMapper",
    'trees': "TreeMapper",
    'viticulture': "VinesMapper",
}

# --- User Selection ---
selected_key = "grape"
ontology = ont_dict[selected_key]
model_name = model_name_dict[selected_key]

# --- Get Class List and Map ---
class_list = ontology.classes()
class_map = {name: i+1 for i, name in enumerate(class_list)}

# --- Label Images ---
image_folder = "/mnt/d/sfm_greeen_2025harvest/20250926_oxley/test_images"
label_folder = "/mnt/d/sfm_greeen_2025harvest/20250926_oxley/test_images/labels"
vis_folder = "/mnt/d/sfm_greeen_2025harvest/20250926_oxley/test_images/visualizations"

model = GroundedSAM(ontology=ontology)



# from PIL import Image
# image = Image.open("/mnt/d/sfm_greeen_2025harvest/20250926_oxley/test_images/DSC_2848.JPG")
# predictions = model.predict(image)
# print("Predictions:", predictions)



model.label(input_folder=image_folder, output_folder=label_folder)
print("Labeling complete. Files generated:", os.listdir(label_folder))

# --- Format for YOLOv11 Segmentation ---
yolo_dataset = "/mnt/d/sfm_greeen_2025harvest/20250926_oxley/test_images/"
yolo_dataset = os.path.join(yolo_dataset, model_name)
os.makedirs(yolo_dataset, exist_ok=True)
img_out = os.path.join(yolo_dataset, "images", "train")
lbl_out = os.path.join(yolo_dataset, "labels", "train")
os.makedirs(img_out, exist_ok=True)
os.makedirs(lbl_out, exist_ok=True)

for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_folder, image_name)
    label_path = os.path.join(label_folder, image_name + ".json")
    if not os.path.exists(label_path):
        continue

    # Copy image to YOLO folder
    shutil.copy(image_path, os.path.join(img_out, image_name))

    # Convert annotation
    with open(label_path, "r") as f:
        annotations = json.load(f)

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    yolo_lines = []

    for ann in annotations:
        class_id = ann["category_id"]
        rle = ann["segmentation"]
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        mask = mask_utils.decode(rle)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3:
                continue
            contour = contour.squeeze()
            norm_pts = [(x / width, y / height) for x, y in contour]
            flat = [str(round(coord, 6)) for pt in norm_pts for coord in pt]
            line = f"{class_id} " + " ".join(flat)
            yolo_lines.append(line)

    # Save YOLO annotation
    label_txt = os.path.join(lbl_out, image_name.rsplit(".", 1)[0] + ".txt")
    with open(label_txt, "w") as f:
        f.write("\n".join(yolo_lines))

# --- Create Dataset YAML ---
yaml_path = os.path.join(yolo_dataset, "grapefinder.yaml")
with open(yaml_path, "w") as f:
    f.write("path: {}\n".format(yolo_dataset))
    f.write("train: images/train\n")
    f.write("val: images/train\n")  # Use train as val for now
    f.write("names:\n")
    for i, name in enumerate(class_list):
        f.write(f"  {i+1}: {name}\n")
