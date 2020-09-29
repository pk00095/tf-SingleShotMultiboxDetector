import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

from tf_ssd.models.keras_ssd300 import ssd_300
from tf_ssd.keras_ssd_loss import SSDLoss
from tf_ssd.helpers import draw_boxes_on_image_v2, SSD300Config
from tf_ssd.tfrecord_parser import Tfrpaser

from segmind_track import KerasCallback
from segmind_track import set_experiment

import os, pdb
import numpy as np


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", help="the directory containing images", type=str, required=True)
    parser.add_argument("--ex_id", help="the directory containing images", type=str, required=True)
    parser.add_argument("--epochs", help="number of epochs to run training", type=int, required=True)
    parser.add_argument("--steps_per_epoch", help="the number of steps for a complete epoch", type=int, required=True)
    # parser.add_argument("--snapshot_epoch", help="take snapshot every nth epoch", type=int, default=5)
    parser.add_argument("--batch_size", help="number of training instances per batch", type=int, default=2)

    args = parser.parse_args()

    set_experiment(args.ex_id)

    # num_classes = 13
    # batch_size = 8

    # initial_epoch   = 0
    # final_epoch     = 5
    # steps_per_epoch = 1000

    # checkpoint_path = './checkpoints/final_ssd.h5'
    checkpoint_path = './checkpoints/final_ssd'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     os.path.join(checkpoint_path, prefix), 
    #     # monitor='val_loss', 
    #     verbose=1, 
    #     save_best_only=True,
    #     save_weights_only=False, 
    #     mode='auto', 
    #     period=snapshot_epoch)

    # callbacks.append(model_checkpoint_callback)

    config = SSD300Config(pos_iou_threshold=0.5, neg_iou_limit=0.3)

    model, preprocess_input, predictor_sizes = ssd_300(
        weights='imagenet',
        image_size=config.input_shape,
        n_classes=args.num_classes,
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
        num_classes=args.num_classes, 
        batch_size=args.batch_size, 
        preprocess_input=preprocess_input)

    dataset = parser.parse_tfrecords(filename=os.path.join(os.getcwd(),'DATA','train*.tfrecord'))


    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    # sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    # model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)
    model.compile(optimizer=Adam(lr=0.001, clipnorm=0.001), loss=ssd_loss.compute_loss)



    history = model.fit(
        dataset,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        callbacks=[KerasCallback],
        # validation_data=val_generator,
        # validation_steps=ceil(val_dataset_size/batch_size),
        initial_epoch=0)

    model.save(checkpoint_path)