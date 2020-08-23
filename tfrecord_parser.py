import tensorflow as tf
from tensorflow import keras
import cv2, os, glob
import numpy as np

from helpers import SSD300Config
from ssd_input_encoder import SSDInputEncoder

config = SSD300Config()



def pad_resize(image, height, width, resize_width, resize_height):
    """Summary

    Args:
        image (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        scale (TYPE): Description

    Returns:
        numpy nd.array: Description
    """
    padded_image = np.zeros(shape=(height.astype(int), width.astype(int),3), dtype=image.dtype)
    h,w,_ =  image.shape
    padded_image[:h,:w,:] = image
    resized_image = cv2.resize(padded_image, (resize_width, resize_height)).astype(keras.backend.floatx())
    return resized_image


@tf.function
def decode_pad_resize(image_string, pad_height, pad_width, resize_width, resize_height):
    """Summary

    Args:
      image_string (TYPE): Description
      pad_height (TYPE): Description
      pad_width (TYPE): Description
      esize_width, resize_height (TYPE): Description

    Returns:
      tf.tensor: Description
    """
    image = tf.image.decode_jpeg(image_string)
    image = tf.numpy_function(pad_resize, [image, pad_height, pad_width, resize_width, resize_height], Tout=keras.backend.floatx())
    #image.set_shape([None, None, 3])
    return image


def print_tensor(x):
    print(x.shape)
    return x




class Tfrpaser(object):
    """docstring for Tfrpaser"""
    def __init__(self, num_classes, config, predictor_sizes, batch_size):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_box_per_image = config.max_boxes_per_image
        self.boxes_list = []
        self.variances = config.variances

        self.img_height = config.height
        self.img_width = config.width

        self.ssd_encoder_layer = SSDInputEncoder(
                                    img_height=config.height,
                                    img_width=config.width,
                                    n_classes=num_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=config.scales,
                                    aspect_ratios_per_layer=config.aspect_ratios,
                                    two_boxes_for_ar1=config.two_boxes_for_ar1,
                                    strides=config.strides,
                                    offsets=config.offsets,
                                    clip_boxes=config.clip_boxes,
                                    variances=config.variances,
                                    matching_type='multi',
                                    pos_iou_threshold=config.pos_iou_threshold,
                                    neg_iou_limit=config.neg_iou_limit,
                                    normalize_coords=config.normalize_coords)

    # def make_gt(self, bboxes, height_scale, width_scale):
    #     """Summary
    #     Create a bbox batch of the form [xmin,ymin, xmax,ymax,labels]
    #     """

    #     # delete bboxes containing [-1,-1,-1,-1, -1] added in **[1]
    #     bboxes = bboxes[~np.all(bboxes==-1, axis=1)]

    #     bboxes[:,[1,2]] *= width_scale
    #     bboxes[:,[3,4]] *= height_scale

    #     num_boxes, boxes_per_row = bboxes.shape

    #     assert boxes_per_row == 5

    #     arr = np.zeros((self.max_box_per_image,5), keras.backend.floatx())
    #     arr[:,0] -= 1

    #     max_index = min(num_boxes, self.max_box_per_image)

    #     arr[:max_index] = bboxes[:max_index]

    #     return arr #.astype(keras.backend.floatx())

    # @tf.function
    # def tf_make_gt(self, xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch, height_scale, width_scale):
        
    #     annotation_batch = list()

    #     for index in range(self.batch_size):

    #         xmins, ymins, xmaxs, ymaxs, labels = xmin_batch[index], ymin_batch[index], xmax_batch[index], ymax_batch[index], label_batch[index]

    #         # labels = tf.cast(labels, keras.backend.floatx())
    #         # print(labels.shape)

    #         # bboxes = tf.convert_to_tensor([labels, xmins,ymins,xmaxs,ymaxs], dtype=keras.backend.floatx())
    #         # bboxes = tf.transpose(bboxes)

    #         # bboxes = tf.numpy_function(self.make_gt, [bboxes, height_scale, width_scale], Tout=keras.backend.floatx())

    #         # annotation_batch.append(bboxes)

    #         _ = tf.numpy_function(print_tensor, [xmins], Tout=keras.backend.floatx())

    #     return xmin_batch        

    def _parse_fn(self, serialized):
        """Summary
            
            Args:
                serialized (TYPE): Description
            
            Returns:
                TYPE: Description
        """
        features = {
              'image/height': tf.io.FixedLenFeature([], tf.int64),
              'image/width': tf.io.FixedLenFeature([], tf.int64),
              'image/encoded': tf.io.FixedLenFeature([],tf.string),
              'image/object/bbox/xmin': tf.io.VarLenFeature(keras.backend.floatx()),
              'image/object/bbox/xmax': tf.io.VarLenFeature(keras.backend.floatx()),
              'image/object/bbox/ymin': tf.io.VarLenFeature(keras.backend.floatx()),
              'image/object/bbox/ymax': tf.io.VarLenFeature(keras.backend.floatx()),
              'image/f_id': tf.io.FixedLenFeature([], tf.int64),
              'image/object/class/label':tf.io.VarLenFeature(tf.int64)}


        parsed_example = tf.io.parse_example(serialized=serialized, features=features)

        max_height = tf.cast(tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(tf.keras.backend.max(parsed_example['image/width']), tf.int32)

        height_scale = self.img_width/max_height
        width_scale = self.img_height/max_width

        height_scale = keras.backend.cast_to_floatx(height_scale)
        width_scale = keras.backend.cast_to_floatx(width_scale)


        image_batch = tf.map_fn(lambda x: decode_pad_resize(x, max_height, max_width, config.width, config.width), parsed_example['image/encoded'], dtype=keras.backend.floatx())

        # **[1] pad with -1 to batch properly
        xmin_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/bbox/xmin']*width_scale, default_value=-1), axis=-1)
        xmax_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/bbox/xmax']*width_scale, default_value=-1), axis=-1)
        ymin_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/bbox/ymin']*height_scale, default_value=-1), axis=-1)
        ymax_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/bbox/ymax']*height_scale, default_value=-1), axis=-1)

        label_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=-1), axis=-1)
        label_batch = keras.backend.cast_to_floatx(label_batch)

        # print(label_batch.shape, xmin_batch.shape, xmax_batch.shape, ymin_batch.shape, ymax_batch.shape)

        annotation_batch = tf.concat([label_batch, xmin_batch, ymin_batch, xmax_batch, ymax_batch], axis=-1)

        y_true = tf.numpy_function(self.ssd_encoder_layer.generate_ytrue, [annotation_batch],  Tout= keras.backend.floatx())

        # _ = self.tf_make_gt(
        #     xmin_batch, 
        #     xmax_batch, 
        #     ymin_batch, 
        #     ymax_batch, 
        #     label_batch, 
        #     height_scale, 
        #     width_scale)

        return image_batch/255, y_true


    def parse_tfrecords(self, filename):

        dataset = tf.data.Dataset.list_files(filename).shuffle(buffer_size=8).repeat(-1)
        dataset = dataset.interleave(
                    tf.data.TFRecordDataset,
                    num_parallel_calls = tf.data.experimental.AUTOTUNE,
                    deterministic=False)

        dataset = dataset.batch(
                    self.batch_size,
                    drop_remainder=True)
        
        dataset = dataset.map(
                    self._parse_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == '__main__':

    from helpers import draw_boxes_on_image_v2, SSD300Config

    config = SSD300Config()

    parser = Tfrpaser(config=config, predictor_sizes=[(512,512)]*6,num_classes=13,batch_size=2)
    
    dataset = parser.parse_tfrecords(filename=os.path.join(os.getcwd(),'DATA','train*.tfrecord'))

    for data, annotation in dataset.take(1):
        image_batch = data.numpy()
        abxs_batch = annotation.numpy()
        # print(image_batch.shape)
        print(abxs_batch.shape)
        # # print(image_batch)
        # print(abxs_batch)

        for index in range(parser.batch_size):
            im = draw_boxes_on_image_v2(image_batch[index]*255, abxs_batch[index])
            cv2.imwrite(f"{index}.jpg", im)
        