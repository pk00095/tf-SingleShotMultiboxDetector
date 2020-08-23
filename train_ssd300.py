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

model = ssd_300(image_size=config.input_shape,
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
                normalize_coords=config.normalize_coords)

predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]


parser = Tfrpaser(config=config, predictor_sizes=predictor_sizes, num_classes=num_classes, batch_size=batch_size)

dataset = parser.parse_tfrecords(filename=os.path.join(os.getcwd(),'DATA','train*.tfrecord'))

# for data, annotation in dataset.take(1):
#     image_batch = data.numpy()
#     abxs_batch = annotation.numpy()
#     # print(image_batch.shape)
#     # print(abxs_batch[0])

#     index_zeros = np.where(np.all(abxs_batch, axis=-1))

#     # print(abxs_batch[index_zeros])
#     print(abxs_batch[index_zeros].shape)

#     classification = np.argmax(abxs_batch[:,:,:13], axis=-1)

#     uniques = np.unique(classification)
#     print(index_zeros)
#     print(uniques)

# exit()

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
# model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)
model.compile(optimizer=Adam(lr=0.001, clipnorm=0.001), loss=ssd_loss.compute_loss)



initial_epoch   = 0
final_epoch     = 5
steps_per_epoch = 1000

history = model.fit(dataset,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              # callbacks=callbacks,
                              # validation_data=val_generator,
                              # validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)
output = './checkpoints/final_ssd.h5'
direc = '/'
direc = direc.join(output.split('/')[:-1])
if not os.path.isdir(direc):
  print(direc)
  os.mkdir(direc)

model.save(output)