import common
from dethelper import *
from figurehelper import *


def run_test(settings_dict, detect_fn, image_list):
  """ TODO

      Parameters
      ----------
      detect_fn: the detection function
      image_files: the list of image files

      Returns
      -------
      TODO
  """
  # create the subimages directory if not exists
  if not os.path.exists(common.SUBIMAGES_DIR):
    os.mkdir(common.SUBIMAGES_DIR)
  else:
    clear_directory_by_path(common.SUBIMAGES_DIR)
  # load the new settings
  common.load_settings(settings_dict)
  # get the image detections as a dict in COCO format
  det_dict = get_dict_from_detections(detect_fn,
                                      image_list,
                                      common.THRESHOLD_LOW,
                                      common.CATEGORIES_WHITELIST)

  return det_dict

def evaluate_test(det_file, gt_file):
  """
     Run the COCO evaluator to show the test results from
     the annotations json file

     Parameters
     ----------
     det_file: the json file of detections
     gt_file: the json file of ground-truth annotations
  """
  #TODO: meterlo en su propio metodo

  # initialize the COCO eval objects from the json files previously generated
  coco_gt=COCO(os.path.join(common.DETEVAL_DIR, gt_file))

  coco_det=COCO(os.path.join(common.DETEVAL_DIR, det_file))

  #TODO: get category list from CATEGORY WHITELIST
  categories = [3]

  # run the evaluation and show the results
  cocoEval = COCOeval(coco_gt,coco_det, 'bbox')
  img_ids = sorted(coco_gt.getImgIds())
  cocoEval.params.imgIds  = img_ids
  cocoEval.params.catIds = categories
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()


def show_test_results(image_file, ann_list, ann_list_gt, skip_scores, skip_labels):
  #TODO: add docstring
  if common.IMAGES_DIR not in image_file:
    image_file = os.path.join(common.IMAGES_DIR, image_file)

  image = get_image_from_file(image_file, False, False, 'uint8')

  visualize_results_on_image(image, ann_list, skip_scores, skip_labels, True)

  avg_det_score = get_average_detection_score(ann_list)
  print('Average score of detections: {}'.format(avg_det_score))

  count_det = len(ann_list)
  count_gt = len(ann_list_gt)
  ratio_count = 100*round(count_det/count_gt, 4)
  print('Numer of detections: {} / {} ({}%)'.format(count_det, count_gt, ratio_count))

  small_det, medium_det, large_det = get_det_count_by_size(ann_list)
  small_gt, medium_gt, large_gt = get_det_count_by_size(ann_list_gt)

  if small_gt != 0:
    ratio_small = 100*round(small_det/small_gt, 4)
    print('Small object detections: {} / {} ({}%)'.format(small_det, small_gt, ratio_small))
  else:
    print('Small object detections: {} / {}'.format(small_det, small_gt))

  if medium_gt != 0:
    ratio_medium = 100*round(medium_det/medium_gt, 4)
    print('Medium object detections: {} / {} ({}%)'.format(medium_det, medium_gt, ratio_medium))
  else:
    print('Medium object detections: {} / {}'.format(medium_det, medium_gt))
  if large_gt != 0:
    ratio_large = 100*round(large_det/large_gt, 4)
    print('Large object detections: {} / {} ({}%)'.format(large_det, large_gt, ratio_large))
  else:
    print('Large object detections: {} / {}'.format(large_det, large_gt))


def compare_two_results(image_file, ann_list_1, ann_list_2):
  #TODO: add docstring

  if common.IMAGES_DIR not in image_file:
    image_file = os.path.join(common.IMAGES_DIR, image_file)

  image = get_image_from_file(image_file, True, False, 'uint8')
  result_1 = visualize_results_on_image(image, ann_list_1, False, True, False)

  image = get_image_from_file(image_file, True, False, 'uint8')
  result_2 = visualize_results_on_image(image, ann_list_2, False, True, False)

  fig, axes = plt.subplots(nrows=1, ncols=2)
  fig.set_figheight(36)
  fig.set_figwidth(36)

  axes[0].imshow(result_1)
  axes[0].axis('off')
  axes[1].imshow(result_2)
  axes[1].axis('off')
  fig.tight_layout()

  plt.show()


def evaluate_det_model(
  detect_fn,
  threshold,
  images,
  categories
):
  """ Evaluate the detection model using the COCOeval API over a list of
      images with whitelisted categories and then print the mAP results

      Parameters
      ----------
      detect_fn
      threshold: the mininum score for a detection to be accepted
      images: the list of images from which perform the object detection
      categories: the whitelist of object categories
  """

  # if no images provided, use the images from GT json file
  if not images:
    images = get_image_list_from_json_file(common.DETEVAL_DIR, 'visdrone_gt')

  # get dict with gt annotations
  gt_dict = get_dict_from_gt(images, categories)

  # get dict from model detections
  det_dict = get_dict_from_detections(detect_fn, threshold, images, categories)
  write_annotations_to_json(common.DETEVAL_DIR, common.DET_MODEL_NAME, det_dict)

  # initialize the COCO eval objects from the json files previously generated
  coco_gt=COCO(os.path.join(common.DETEVAL_DIR, 'visdrone_gt.json'))
  coco_det=COCO(os.path.join(common.DETEVAL_DIR, common.DET_MODEL_NAME + '.json'))

  # run the evaluation and show the results
  cocoEval = COCOeval(coco_gt,coco_det,'bbox')
  img_ids = sorted(coco_gt.getImgIds())
  cocoEval.params.imgIds  = img_ids
  cocoEval.params.catIds = categories
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()
