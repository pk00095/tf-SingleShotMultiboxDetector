import tensorflow as tf
import zipfile, tempfile, os, glob, tqdm
import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFont

font = ImageFont.load_default()

interpolation_options = {
    'nearest':cv2.INTER_NEAREST,
    'linear':cv2.INTER_LINEAR,
    'cubic':cv2.INTER_CUBIC,
    'area':cv2.INTER_AREA,
    'lanczos4':cv2.INTER_LANCZOS4
}


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]
# from utils import get_random_data

def draw_boxes_on_image_v2(image, boxes):
    image = image.astype('uint8')
    # num_boxes = boxes.shape[0]
    for l,x1,x2,y1,y2 in boxes:
        if x1==y1==x2==y2==-1:
            break

        class_and_score = f"label :{l}"
        cv2.rectangle(img=image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=2)
        cv2.putText(img=image, text=class_and_score, org=(int(x1), int(y1) - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
    return image

# def annotate_image(image_path, bboxes, scores, labels, threshold=0.5, label_dict=None):
def annotate_image(image, bboxes, scores, labels, threshold=0.5, label_dict=None):
  """Summary
  
  Args:
      image_path (str): path to image to annotate
      bboxes (TYPE): Description
      scores (TYPE): Description
      labels (TYPE): Description
      threshold (float, optional): Description
      label_dict (None, optional): Description
  
  Returns:
      TYPE: Description
  """
  # image = Image.open(image_path)
  Imagedraw = ImageDraw.Draw(image)

  for box, label, score in zip(bboxes, labels, scores):
    if score < threshold:
      continue

    (left,top,right,bottom) = box

    label_to_display = label
    if isinstance(label_dict, dict):
      label_to_display = label_dict[label]

    caption = "{}|{:.3f}".format(label_to_display, score)
    #draw_caption(draw, b, caption)

    colortofill = STANDARD_COLORS[label]
    Imagedraw.rectangle([left,top,right,bottom], fill=None, outline=colortofill)

    display_str_heights = font.getsize(caption)[1]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * display_str_heights

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    text_width, text_height = font.getsize(caption)
    margin = np.ceil(0.05 * text_height)
    Imagedraw.rectangle([(left, text_bottom-text_height-2*margin), (left+text_width,text_bottom)], fill=colortofill)

    Imagedraw.text((left+margin, text_bottom-text_height-margin),caption,fill='black',font=font)

  return image

# scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
# scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
# The anchor box aspect ratios used in the original SSD300; the order matters


class SSD300Config(object):
    """docstring for SSDConfig"""
    def __init__(self, 
        aspect_ratios=[[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]], 
        strides = [8, 16, 32, 64, 100, 300],
        scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
        offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        max_boxes=300, 
        variances=[0.1, 0.1, 0.2, 0.2],
        # score=0.3,
        pos_iou_threshold=0.5,
        neg_iou_limit=0.5):

        self.height = 300
        self.width = 300

        self.input_shape = (self.height, self.width, 3)

        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.max_boxes_per_image = max_boxes

        self.aspect_ratios = aspect_ratios
        self.strides = strides

        self.variances = variances

        assert len(aspect_ratios) == len(offsets)
        self.offsets = offsets
        self.scales = scales

        self.two_boxes_for_ar1 = True
        self.clip_boxes = False
        self.normalize_coords = True

        self.coords='centroids'



def download_aerial_dataset(dataset_path=tempfile.gettempdir()):
    zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/aerial-vehicles-dataset.zip'
    path_to_zip_file = tf.keras.utils.get_file(
        'aerial-vehicles-dataset.zip',
        zip_url,
        cache_dir=dataset_path, 
        cache_subdir='',
        extract=False)
    directory_to_extract_to = os.path.join(dataset_path,'aerial-vehicles-dataset')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    images_dir = os.path.join(dataset_path, 'aerial-vehicles-dataset','images')
    annotation_dir = os.path.join(dataset_path, 'aerial-vehicles-dataset','annotations','pascalvoc_xml')

    return images_dir, annotation_dir


def download_chess_dataset(dataset_path=tempfile.gettempdir()):
    zip_url = 'https://public.roboflow.ai/ds/uBYkFHtqpy?key=HZljsh2sXY'
    path_to_zip_file = tf.keras.utils.get_file(
        'chess_pieces.zip',
        zip_url,
        cache_dir=dataset_path, 
        cache_subdir='',
        extract=False)
    directory_to_extract_to = os.path.join(dataset_path,'chess_pieces')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    images_dir = os.path.join(dataset_path, 'chess_pieces','train')
    annotation_dir = os.path.join(dataset_path, 'chess_pieces','train')

    for image in tqdm.tqdm(glob.glob(os.path.join(images_dir, '*.jpg'))):
        new_name = image.replace('_jpg.rf.', '')
        os.rename(image, new_name)

        annotation = image.replace('.jpg', '.xml')
        new_name = annotation.replace('_jpg.rf.', '')
        os.rename(annotation, new_name)

    return images_dir, annotation_dir

