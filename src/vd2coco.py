import common
import dethelper
from filehelper import *
from imagehelper import *


def get_dict_from_gt(file_name, image_files, category_list):
  """ Get the VisDrone Ground-Truth annotations dict

      Parameters
      ----------
      file_name: the file name
      image_files: the list of image files to be processed
      category_list: the whitelist of categories

      Returns
      -------
      the dict from ground-truth annotations
  """

  # check if the gt dict has been already generated
  file_path = os.path.join(common.DETEVAL_DIR, file_name) + '.json'
  if os.path.isfile(file_path):
    return read_annotations_from_json(common.DETEVAL_DIR, file_name)
  
  # generate a new dict from the image_files given
  image_list = []
  annotation_list = []
  image_count = 1
  object_count = 1

  for image_file in image_files:
    image_file = common.IMAGES_DIR + image_file
    # get annotation file from image file
    annotation_file = get_annotation_file_from_image_file(image_file)
    image_list.append(get_image_dict_from_image_file(image_file, image_count))
    annotations = get_annotations_from_file(annotation_file,
                                            image_count,
                                            object_count,
                                            category_list)
    annotation_list = annotation_list + annotations
    object_count += len(annotations)
    image_count += 1

  gt_dict = {}
  gt_dict['categories'] = get_categories_from_category_list(category_list)
  gt_dict['images'] = image_list
  gt_dict['annotations'] = annotation_list

  return gt_dict


def get_annotations_from_file(
    annotation_file,
    image_count,
    object_count,
    category_list
):
  """ Get the ground-truth annotations in COCO format from VisDrone annotation file

      Parameters
      ----------
      annotation_file: the annotation VisDrone file
      image_count: the image count id
      object_count: the general object detection count over all images
      category_list: the whitelist of categories

      Returns
      -------
      the list of gt annotations
  """
  lines = []
  annotations = []

  with open(annotation_file) as f:
    # read all the lines into list
    lines = [line.rstrip() for line in f.readlines()]

  # for each line of annotations transform to coco format
  for line in lines:
    annotation_dict = transform_line_to_coco(line,
                                            image_count,
                                            object_count,
                                            category_list)
    # do not increase the object count for None detections
    if annotation_dict:
      annotations.append(annotation_dict)
      object_count += 1

  return annotations


def transform_line_to_coco(
    line,
    image_count,
    object_count,
    category_list
):
  """ Transform a VisDrone annotation line to COCO format
      
      Parameters
      ----------
      line: the line from VisDrone annotation file
      image_count: the image count id
      object_count: the object count id
      category_list: the whitelist of categories

      Returns
      -------
      the annotation dict for the image
  """
  line = line.split(',')
  annotation_dict = {}
  annotation_dict['image_id'] = image_count
  annotation_dict['id'] = object_count
  category_id = get_coco_category_from_vd_category(int(line[5]))

  if category_id in category_list:
    annotation_dict['category_id'] = category_id
  else:
    # skip detections that are not in COCO or whitelisted in category_list
    return None
  
  annotation_dict['score'] = float(line[4])
  annotation_dict['bbox'] = get_bbox_from_vd_line(line)
  annotation_dict['area'] = dethelper.get_area_from_bbox(annotation_dict['bbox'])
  annotation_dict['iscrowd'] = 0

  return annotation_dict


def get_bbox_from_vd_line(line):
  """ Get the bounding box in COCO format from the given line in VisDrone format:
      <bbox_left>   [0]: The x coordinate of the top-left corner of the predicted bounding box
      <bbox_top>    [1]: The y coordinate of the top-left corner of the predicted object bounding box
      <bbox_width>  [2]: The width in pixels of the predicted object bounding box
      <bbox_height> [3]: The height in pixels of the predicted object bounding box

      Parameters
      ----------
      line: the line in VisDrone format

      Returns
      -------
      the bounding box in COCO format
  """
  x = int(line[0])
  y = int(line[1])
  width = int(line[2])
  height = int(line[3])

  return [x, y, width, height]


def get_coco_category_from_vd_category(category):
  """ Get the COCO category corresponding to the VisDrone category

      Parameters
      ----------
      category: the VisDrone category id

      Returns
      -------
      the COCO category, if matched, else return 0
  """
  if category == 1:
    # Person
    return 1
  elif category == 3:
    # Bycicle
    return 2
  elif category == 4:
    # Car
    return 3
  elif category == 6:
    # Truck
    return 8
  elif category == 9:
    # Bus
    return 6
  elif category == 10:
    # Motorcycle
    return 4
  else:
    # Ignored
    return 0

