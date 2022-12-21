import common
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from dethelper import get_area_from_bbox
from imagehelper import *
from google.colab.patches import cv2_imshow
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops


def visualize_results_on_image(image, ann_list, skip_scores, skip_labels, imshow):
  """ Visualize the results of detections on image given a list of annotations

      Parameters
      ----------
      image: the image as an array
      ann_list: the annotation list
      skip_scores: skip score when drawing a single detection
      skip_labels: skip label when drawing a single detection
      imshow: show the image using cv2_imshow or return it instead

      Returns
      -------
      the image with detections
  """
  det_bboxes = np.array([get_det_box_from_bbox(ann['bbox']) for ann in ann_list])
  det_scores = np.array([ann['score'] for ann in ann_list])
  det_classes = np.array([ann['category_id'] for ann in ann_list])

  image_boxes_labels = viz_utils.visualize_boxes_and_labels_on_image_array(
    image,
    det_bboxes,
    det_classes,
    det_scores,
    common.CATEGORY_INDEX,
    use_normalized_coordinates=False,
    max_boxes_to_draw=100,
    min_score_thresh=common.THRESHOLD_LOW,
    agnostic_mode=False,
    line_thickness=2,
    skip_scores=skip_scores,
    skip_labels=skip_labels
  )

  if imshow:
    cv2_imshow(image_boxes_labels)
    return None
  else:
    return image_boxes_labels


def get_annotations_by_image_id(image_id, det_dict):
  ann_list = det_dict['annotations']
  return [ann for ann in ann_list if ann['image_id'] == image_id]

def get_det_box_from_bbox(bbox):
  # detection box format: (ymin, xmin, ymax, xmax)
  # bbox COCO format: (x-top left, y-top left, width, height)
  xmin = bbox[0]
  ymin = bbox[1]
  xmax = xmin + bbox[2]
  ymax = ymin + bbox[3]

  return [ymin, xmin, ymax, xmax]

def get_det_count_by_size(ann_list):
  small_count = 0
  medium_count = 0
  large_count = 0
  for ann in ann_list:
    area = ann['area']
    if area <= 32**2:
      small_count += 1
    elif area <= 96**2:
      medium_count += 1
    else:
      large_count += 1

  return small_count, medium_count, large_count


def get_det_small_count(ann_list):
  small_count = 0
  for ann in ann_list:
    if get_area_from_bbox(ann['bbox']) < 32**2:
      small_count += 1

def get_average_detection_score(ann_list):
  score = 0
  if len(ann_list) == 0:
    return score
    
  for ann in ann_list:
    score += ann['score']
  
  return score / len(ann_list)


def get_sr_crop_image(file_name, model, scale):
  full_path = common.SREVAL_DIR + 'srplot/' + file_name + '/' + model + '_x' + str(scale) + '.jpg'

  return get_image_from_file(full_path, True, False, 'uint8')


def plot_sr_comparison(file_name, x, y, crop_size, scale):
  model_list = ['BICUBIC', 'EDSR', 'ESPCN', 'FSRCNN', 'LapSRN', 'DBPI', 'ESRGAN', 'RealESRGAN']

  fig = plt.figure(figsize=(12, 12))
  columns = 4
  rows = 2
  for i in range(1, columns*rows +1):
      model = model_list[i-1]
      img = get_sr_crop_image(file_name, x, y, crop_size, model, 2)
      if img is None:
        img = 255 * np.ones((200,200,3), np.uint8)

      fig.add_subplot(rows, columns, i)
      plt.axis('off')
      plt.imshow(img)

  plt.show()


def figure_sr_comparsion(file_name, x, y, crop_size, scale):
  model_list = ['BICUBIC', 'EDSR', 'ESPCN', 'FSRCNN', 'LapSRN', 'FeMaSR', 'BSRGAN', 'RealESRGAN']

  fig, axes  = plt.subplots(2, 4, figsize=(20, 10))
  fig.subplots_adjust()

  for ax, model in zip(axes.flatten(), model_list):
      ax.axis('off')
      ax.set_title('{}'.format(model))
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      img = get_sr_crop_image(file_name, model, scale)
      ax.imshow(img)


  plt.tight_layout(pad=1, h_pad=1, w_pad=1)
  plt.show()