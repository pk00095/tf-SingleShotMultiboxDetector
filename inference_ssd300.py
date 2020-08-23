import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

from models.keras_layer_AnchorBoxes import AnchorBoxes
# from models.keras_layer_L2Normalization import L2Normalization
from models.keras_layer_DecodeDetections import DecodeDetections
from models.keras_ssd300 import ssd_300

from keras_ssd_loss import SSDLoss
from helpers import SSD300Config
from helpers import annotate_image as image_annotator

import numpy as np
import pdb
import glob
import os


config = SSD300Config(pos_iou_threshold=0.5, neg_iou_limit=0.3)

num_classes = 13

def load_ssd300(config, checkpoint_file, num_classes):

    model = ssd_300(image_size=config.input_shape,
                    n_classes=num_classes,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=config.scales,
                    aspect_ratios_per_layer=config.aspect_ratios,
                    two_boxes_for_ar1=config.two_boxes_for_ar1,
                    steps=config.strides,
                    offsets=config.offsets,
                    clip_boxes=config.clip_boxes,
                    variances=config.variances,
                    normalize_coords=config.normalize_coords,
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    # 2: Load the trained weights into the model.

    model.load_weights(checkpoint_file, by_name=True)

    return model

    # ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    # model = keras.models.load_model(
    #     './checkpoints/final_ssd.h5', 
    #     custom_objects={'AnchorBoxes': AnchorBoxes,
    #                    # 'L2Normalization': L2Normalization,
    #                    'DecodeDetections': DecodeDetections,
    #                    'compute_loss': ssd_loss.compute_loss}
    #                    )


model = load_ssd300(config, './checkpoints/final_ssd.h5', num_classes)

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

input_images = np.array(input_images)/255 

y_pred = model.predict(input_images)


confidence_threshold = 0.75


for index in range(y_pred.shape[0]):

    bbox = y_pred[index,:,2:]
    confidence = y_pred[index,:,1]
    label = y_pred[index,:,0].astype(int)

    print(bbox.shape)

    annotated_image = image_annotator(
        image=pil_input_images[index], 
        bboxes=bbox, 
        scores=confidence, 
        labels=label, 
        threshold=0.75)


    annotated_image.save(os.path.join(write_out_dir,f"{index}_predicted.jpg"))
# pdb.set_trace()


# onehot_classes = y_pred[:,:13]

# score = np.amax(onehot_classes, axis=-1)
# label = np.argmax(onehot_classes, axis=-1)


# thres_label = label[score>confidence_threshold]
# thres_confidence = score[score>confidence_threshold]
# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('   class   conf ')
# print(thres_label, thres_confidence)

# print(np.unique(thres_label))
# print(np.max(thres_confidence), np.min(thres_confidence))


