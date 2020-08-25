import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

from models.keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from helpers import draw_boxes_on_image_v2, SSD300Config
from tfrecord_parser import Tfrpaser

import os, pdb
import numpy as np

config = SSD300Config(pos_iou_threshold=0.5, neg_iou_limit=0.3)

num_classes = 21

batch_size = 8

initial_epoch   = 0
final_epoch     = 1
steps_per_epoch = 1

checkpoint_path = './checkpoints/final_ssd'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)


config = SSD300Config(pos_iou_threshold=0.5, neg_iou_limit=0.3)

model, preprocess_input, predictor_sizes = ssd_300(
    weights='imagenet',
    image_size=config.input_shape,
    n_classes=num_classes,
    mode='training',
    l2_regularization=0.0005,
    scales=config.scales,
    aspect_ratios_per_layer=config.aspect_ratios,
    two_boxes_for_ar1=config.two_boxes_for_ar1,
    steps=config.strides,
    offsets=config.offsets,
    clip_boxes=config.clip_boxes,
    variances=config.variances,
    normalize_coords=config.normalize_coords,
    return_predictor_sizes=True)


parser = Tfrpaser(
    config=config, 
    predictor_sizes=predictor_sizes, 
    num_classes=num_classes, 
    batch_size=batch_size, 
    preprocess_input=preprocess_input)

dataset = parser.parse_tfrecords(filename=os.path.join(os.getcwd(),'DATA','train*.tfrecord'))


ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
# model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)
model.compile(optimizer=Adam(lr=0.001, clipnorm=0.001), loss=ssd_loss.compute_loss)

history = model.fit(
    dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=final_epoch,
    # callbacks=callbacks,
    # validation_data=val_generator,
    # validation_steps=ceil(val_dataset_size/batch_size),
    initial_epoch=initial_epoch)

# model.save(checkpoint_path,save_format='tf')
tf.keras.models.save_model(
    model,
    checkpoint_path,
    overwrite=True,
    include_optimizer=True,
    save_format='tf',
    signatures=None,
    options=None,
)
