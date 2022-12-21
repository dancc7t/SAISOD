import common
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from Cluster import *
from filehelper import *
from imagehelper import *
from srhelper import *
from tilinghelper import *
from vd2coco import *


def get_dict_from_detections(
    detect_fn,
    image_files,
    threshold,
    category_list
):
  """ Get the detection results given a detector function and a list of images
      
      Parameters
      ----------
      detect_fn: the detection function
      image_files: the list of image files
      threshold: the minimum threshold for detection score
      category_list: the category whitelist

      Returns
      -------
      the detection dictionary
  """
  det_dict = {}
  annotation_list = []
  image_count = 1
  object_count = 1

  for image_file in image_files:
    image_file = common.IMAGES_DIR + image_file
    # -----------------
    # initial detection
    # -----------------
    initial_annotation_list = []
    if common.INITIAL_DET:
      # get the annotations from the global image
      initial_annotation_list = get_annotations_from_image( image_file,
                                                            image_count, 
                                                            object_count,
                                                            detect_fn,
                                                            threshold,
                                                            category_list)
    # -----------------------
    # SR subimages detections
    # -----------------------
    si_ann_list = []
    if common.APPLY_SR:
      # get the annotations from all the SR subimages
      si_ann_list = get_si_annotations( image_file,
                                        image_count, 
                                        object_count,
                                        detect_fn,
                                        threshold,
                                        category_list,
                                        'SR')
    # -----------------------
    # LUT subimages detections
    # -----------------------
    lut_ann_list = []
    if common.APPLY_LUT:
      # get the annotations from the LUT-transformed SR subimages
      lut_ann_list = get_si_annotations(image_file,
                                        image_count, 
                                        object_count,
                                        detect_fn,
                                        threshold,
                                        category_list,
                                        'LUT')
    # ------------------------
    # Flip subimages detections
    # ------------------------
    flip_ann_list = []
    if common.APPLY_FLIP:
      # get the annotations from horizontally flipped SR subimages
      flip_ann_list = get_si_annotations(image_file,
                                        image_count, 
                                        object_count,
                                        detect_fn,
                                        threshold,
                                        category_list,
                                        'FLIP')

    # combine the annotation lists from the different stepts into one unique list
    all_annotations = initial_annotation_list + si_ann_list + lut_ann_list + flip_ann_list
    annotations = []
    # basic model evaluation does not need clustering
    if common.INITIAL_DET and not common.APPLY_SR and \
       not common.APPLY_LUT and not common.APPLY_FLIP:
      annotations = initial_annotation_list
    else:
      annotations = get_merged_annotations(all_annotations, image_count, object_count)

    # --------------
    # Object Zoom x4
    # --------------
    if common.APPLY_OZX4:
      ozx4_on_annotations(image_file, annotations, detect_fn)

    # clear the subimages folder
    # filename = get_filename_from_file(image_file, False)
    # remove_files_by_path(SUBIMAGES_DIR + filename)
    
    # save annotations and update count
    annotation_list = annotation_list + annotations
    object_count += len(annotations)
    image_count += 1

  # change annotations to COCO format
  change_annotations_to_coco_format(annotation_list)
  # save the annotations to dict of detections
  det_dict['annotations'] = annotation_list

  return det_dict


def get_si_annotations(
    image_file,
    image_count,
    object_count,
    detect_fn, 
    threshold, 
    category_list,
    operation
):
  """ Get the sub image annotations from a given image

      Parameters
      ----------
      image_file: the image file
      image_count: the image count id to pair with each detection in the image
      object_count: the general object detection count over all images
      detect_fn: the detection function
      threshold: the minimum threshold for detection score
      category_list: the category whitelist
      operation: the operation type (SR, LUT, FLIP)

      Returns
      -------
      the sub image annotations of detections
  """
  sisr_list = []
  tile_list = []

  if operation == 'SR':
    sisr_list, tile_list = get_sr_subimages_and_tiles(image_file, True)
  elif operation == 'LUT' or operation == 'FLIP':
    sisr_list, tile_list = get_sr_subimages_and_tiles(image_file, False)

  is_flipped = False
  si_ann_list = []
  si_count = 1
  si_object_count = object_count
  for sisr_file in sisr_list:
    if operation == 'LUT':
      # get the lut image saved to file
      sisr_file = get_lut_image(sisr_file, common.LUT_METHOD)
      # if no LUT image provided, then skip the current sub image
      if not sisr_file:
        si_count += 1
        break

    if operation == 'FLIP':
      # get the horizontally flipped image saved to file
      sisr_file = get_flip_image(sisr_file)
      is_flipped = True
    
    # get the annotations of each SR sub-image
    si_annotations = get_annotations_from_image(sisr_file,
                                                si_count, 
                                                si_object_count,
                                                detect_fn,
                                                threshold,
                                                category_list)
    # revert SR upscaling factor and add coord translation to each annotation
    real_annotations = get_real_annotations(si_annotations,
                                            tile_list,
                                            image_count,
                                            common.TILING_FACTOR,
                                            is_flipped)

    si_ann_list.extend(real_annotations)
    si_count += 1
  """
  # merge annotations obtained from all the sub images
  merged_annotations = get_merged_annotations(si_ann_list,
                                              image_count,
                                              object_count,
                                              threshold)
  """
  # return si_ann_list instead of invoking twice clustering method
  return si_ann_list


def get_annotations_from_image(
    image_file,
    image_count,
    object_count,
    detect_fn, 
    threshold, 
    category_list
):
  """ Get the list of annotations from a given image
      
      Parameters
      ----------
      image_file: the image file
      image_count: the image count id to pair with each detection in the image
      object_count: the general object detection count over all images
      detect_fn: the detection function
      threshold: the minimum threshold for detection score
      category_list: the category whitelist

      Returns
      -------
      the image annotations of detections
  """
  # get image as a tensor
  tensor_img = get_image_from_file(image_file, 
                                  as_rgb=True,
                                  as_tensor=True,
                                  as_type='uint8')
  height, width = get_image_shape(tensor_img)

  detections = detect_fn(tensor_img)
  # Convert output tensors to numpy arrays and remove the batch dimension
  # Keep only the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
  detections['num_detections'] = num_detections
  # get annotations in COCO format from detections
  annotations = get_annotations_from_detections(detections,
                                                threshold,
                                                category_list,
                                                height,
                                                width,
                                                image_count,
                                                object_count)
  return annotations


def get_annotations_from_detections(
    detections,
    threshold,
    category_list,
    height,
    width,
    image_id,
    object_count
):
  """ Get the annotations list from the image detections

      Parameters
      ----------
      detections: the detections dict
      threshold: the minimum threshold for detection score
      category_list: the whitelist of category ids
      height: the image height
      width: the image width
      image_id: the image count id
      object_id: the general object detection count over all images
    
      Returns
      ------- 
      the list of annotation dicts for the image
  """
  annotation_list = []
  for i in range(detections['num_detections']):
    category_id = int(detections['detection_classes'][i])
    score = float(detections['detection_scores'][i])
    # skip the current detection if not whitelisted or the score is too low
    if category_id not in category_list or score < threshold:
      continue

    ann_dict = {}
    ann_dict['image_id'] = image_id
    ann_dict['id'] = object_count
    ann_dict['category_id'] = category_id
    ann_dict['score'] = score
    # adjust detections to image scale before saving to bbox
    det_box = detections['detection_boxes'][i]
    ymin = int(det_box[0]*height)
    xmin = int(det_box[1]*width)
    ymax = int(det_box[2]*height)
    xmax = int(det_box[3]*width)
    # bbox in detection format
    ann_dict['bbox'] = [ymin, xmin, ymax, xmax]
    ann_dict['iscrowd'] = 0
    annotation_list.append(ann_dict)
    # only count detections that are whitelisted
    object_count += 1

  return annotation_list


def get_bbox_from_detection_box(det_box):
  """ Get the bbox from the detection box in COCO format
      detection box format: (ymin, xmin, ymax, xmax)
      bbox COCO format: (x-top left, y-top left, width, height)

      Parameters
      ----------
      det_box: the detection box in detection format

      Returns
      ------- 
      the detection box in COCO format
  """
  ymin = det_box[0]
  xmin = det_box[1]
  ymax = det_box[2]
  xmax = det_box[3]
  width = xmax-xmin
  height = ymax-ymin

  return [xmin, ymin, width, height]


def get_area_from_bbox(bbox):
  """ Get the area of the bbox in COCO format

      Parameters
      ----------
      bbox: the bounding box

      Returns
      ------- 
      the area of the bounding box
  """
  return bbox[2]*bbox[3]


def get_real_annotations(
  annotation_list, 
  tile_list, 
  image_count, 
  factor, 
  is_flipped
):
  """ Get the real image-wise locations of all detections in each of the sub images

      Parameters
      ----------
      annotation_list: the annotation list of sub images
      tile_list: the list of tiles
      image_count: the image count id to pair with each detection in the image
      factor: the upscaling factor used by the SR model
      is_flipped: If the image is flipped horizontally or not

      Returns
      -------
      the list of annotation with real coordinates for the image
  """
  real_annotation_list = []

  for ann in annotation_list:
    real_ann = {}
    real_ann['image_id'] = image_count
    real_ann['id'] = ann['id']
    real_ann['category_id'] = ann['category_id']
    real_ann['score'] = ann['score']

    # get the tile corresponding with sub image id of annotation 
    tile = tile_list[ann['image_id'] - 1]
    # top left corner coordinates of tile within global image
    xi = tile['xmin']
    yi = tile['ymin']
    old_bbox = ann['bbox']
    # undo horizontal flip before changing scale and translation
    if is_flipped:
      si_width = factor * (tile['xmax'] - tile['xmin'])
      # unflip the horizontal coords
      uf_xmin = si_width - old_bbox[3]
      uf_xmax = si_width - old_bbox[1]
      old_bbox[1] = uf_xmin
      old_bbox[3] = uf_xmax
    # undo SR upscaling and add tile translation to new bbox coords
    new_bbox = list(map(lambda x : int(x / factor), old_bbox))
    new_bbox[0] += yi
    new_bbox[1] += xi
    new_bbox[2] += yi
    new_bbox[3] += xi
    real_ann['bbox'] = new_bbox
    real_ann['iscrowd'] = 0
    # save new annotations to list
    real_annotation_list.append(real_ann)

  return real_annotation_list


def get_truple_list(annotation_list):
  """ Get the list of truples (bbox, class, score) from the list
      of annotations
      
      Parameters
      ----------
      annotation_list: the annotation list

      Returns
      -------
      the list of truples 
  """
  truple_list = []

  for annotation in annotation_list:
    bbox_list = []
    category_list = []
    score_list = []
    bbox_list.append(annotation['bbox'])
    category_list.append(annotation['category_id'])
    score_list.append(annotation['score'])
    truple = (np.array(bbox_list), np.array(category_list), np.array(score_list))
    truple_list.append(truple)

  return truple_list


def get_annotations_from_truples(clusted_annotations, image_count, object_count):
  """ Get the unified annotations from the lists of truples by selecting the element
      with the best score within the same truple

      Parameters
      ----------
      clusted_annotations: the list of clustered annotations, each line can contain multiple
                           elements, which indicates an object has been detected multiple times
                           in different subimages
      image_count: the general image count id to pair with each detection in the image
      object_count: the general object detection count over all images

      Returns
      -------
      the list of unified annotations for the image
  """
  annotations_list = []
  i = object_count
  for truple in clusted_annotations:
    annotation = {}
    # ge the list of truples from the same object
    det_box_list = truple[0].tolist()
    category_list = truple[1].tolist()
    score_list = truple[2].tolist()
    # get only the truple with the best score
    idx = np.argmax(score_list)
    bbox = det_box_list[idx]
    category = category_list[idx]
    score = score_list[idx]
    # save annotation to dict
    annotation['image_id'] = image_count
    annotation['id'] = i
    annotation['category_id'] = int(category)
    annotation['score'] = float(score)
    annotation['bbox'] = bbox
    annotation['iscrowd'] = 0
    annotations_list.append(annotation)
    i += 1

  return annotations_list


def get_merged_annotations(annotation_list, image_count, object_count):
  """ Get the unified list of annotations by merging them using clustering
  """
  # get the truple of lists for clustering
  truple_list = get_truple_list(annotation_list)
  clusted_annotations = create_clusters(truple_list, common.IOU_THRESHOLD)
  si_ann_list = get_annotations_from_truples( clusted_annotations,
                                              image_count,
                                              object_count)
  return si_ann_list


def ozx4_on_annotations(image_file, annotation_list, detect_fn):
  """
  """

  # create the image zoom directory
  folder_name = image_file.rsplit('/', 1)[1]
  folder_name = folder_name.rsplit('.', 1)[0]
  zoom_path = common.SUBIMAGES_DIR + folder_name + '/ozx4/' 
  if not os.path.exists(zoom_path):
    os.mkdir(zoom_path)

  # load the original image from file
  image = get_image_from_file(image_file, True, False, 'uint8')

  for annotation in annotation_list:
    score = annotation['score']
    bbox = annotation['bbox']
    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    # only try to improve those detections below the high threshold
    # and not below the minium area
    if score > common.THRESHOLD_HIGH or area < common.MIN_AREA_OZ:
      continue

    object_id = annotation['id']
    bbox = annotation['bbox']
    # generate the object focused image patch
    create_image_patch_from_object(image, zoom_path, object_id, bbox)

  # generate the x4-zoomed object images
  ozi_list = batch_image_superres(zoom_path, 'Real-ESRGAN', 4)

  for annotation in annotation_list:
    score = annotation['score']
    # only try to improve those detections below the high threshold
    if score > common.THRESHOLD_HIGH:
      continue
    
    # get the x4-zoomed image ref from the annotation object id
    object_id = annotation['id']
    ozi_file = zoom_path + str(object_id) + '_x4.jpg'

    # discard the detections that do not match the criteria and thus, were not created
    if not os.path.isfile(ozi_file):
      annotation_list.remove(annotation)
      continue 

    new_score = detect_ozx4_image(ozi_file, detect_fn)

    # only keep the detections that do make it past the high threshold after ozx4
    if new_score > common.THRESHOLD_HIGH:
      annotation['score'] = new_score
    else:
      annotation_list.remove(annotation)


def detect_ozx4_image(ozi_file, detect_fn):
  """ TODO
  """

  oz_img = get_image_from_file(ozi_file, as_rgb=False,as_tensor=False,as_type='uint8')

  # add padding to the object-zoomed image till it reaches the model input size
  full_ozi = add_padding_to_image(oz_img)
  # transform np image to tensor img
  tensor_img = tf.convert_to_tensor(np.array(full_ozi), dtype=np.uint8)
  tensor_img = tf.expand_dims(tensor_img, axis=0)
  height, width = get_image_shape(tensor_img)

  detections = detect_fn(tensor_img)
  # Convert output tensors to numpy arrays and remove the batch dimension
  # Keep only the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
  detections['num_detections'] = num_detections
  # get annotations in COCO format from detections
  annotations = get_annotations_from_detections(detections,
                                                common.THRESHOLD_LOW,
                                                common.CATEGORIES_WHITELIST,
                                                height, width, 1, 1)
  new_score = 0
  for i in range(detections['num_detections']):
    score = float(detections['detection_scores'][i])
    if score > new_score:
      new_score = score

  return new_score


def change_annotations_to_coco_format(annotation_list):
  """ Change the annotations to COCO format

      Parameters
      ----------
      annotation_list: the annotation list in detection format
  """
  for annotation in annotation_list:
    # get bbox in detection format
    det_box = annotation['bbox']
    # get bbox in COCO Format
    bbox = get_bbox_from_detection_box(det_box)
    annotation['bbox'] = bbox
    annotation['area'] = get_area_from_bbox(bbox)


def download_model(model_name, model_date):
  """ Download and extract the detection model from tensorflow

      Parameters
      ----------
      model_name: the detection model name
      model_date: the detection model date
      
      Returns
      -------
      the path to the downloaded detection model
  """
  base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(fname=model_name,
                                      origin=base_url + model_date + '/' + model_file,
                                      untar=True)
  return str(model_dir)


def get_detection_function(model_name, model_date):
  """ Get the detection function the downloaded detection model

      Parameters
      ----------
      model_name: the detection model name
      model_date: the detection model date
      
      Returns
      -------
      the saved model detection function
  """
  model_dir = download_model(model_name, model_date)

  return tf.saved_model.load(os.path.join(model_dir,'saved_model'))