import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from models.keras_ssd300 import ssd_300
from models.keras_layer_DecodeDetections import decode_detections

from helpers import SSD300Config
from helpers import annotate_image as image_annotator

import numpy as np
import glob
import os



config = SSD300Config(pos_iou_threshold=0.5, neg_iou_limit=0.3)


training_model = keras.models.load_model('./checkpoints/final_ssd', compile=False)

model  = decode_detections(
    training_model,
    config,
    confidence_thresh=0.5,
    iou_threshold=0.45,
    top_k=200,
    nms_max_output_size=400)


orig_images = [] # Store the images here.
input_images = []
pil_input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_dir = './example_images'
write_out_dir = './results'

os.makedirs(write_out_dir, exist_ok=True)

# orig_images.append(imread(img_path))

for image_path in glob.glob(os.path.join(img_dir,'*.jpg')):
    pil_image = image.load_img(image_path, target_size=(config.height, config.width))
    pil_input_images.append(pil_image)
    img = np.array(pil_image)
    input_images.append(img)

input_images = preprocess_input(np.array(input_images))

bboxes, scores, labels = model.predict(input_images)


confidence_threshold = 0.75


for index in range(input_images.shape[0]):

    bbox = bboxes[index]
    confidence = scores[index]
    label = labels[index]

    print(bbox.shape)
    # confidence

    annotated_image = image_annotator(
        image=pil_input_images[index], 
        bboxes=bbox, 
        scores=confidence, 
        labels=label, 
        threshold=0.75)


    annotated_image.save(os.path.join(write_out_dir,f"{index}_predicted.jpg"))


