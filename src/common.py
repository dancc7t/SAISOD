import json
import logging
import os
import random
import shutil
import subprocess
import warnings
import tensorflow as tf
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import ops as utils_ops

# -----------------#
# Global Variables #
# -----------------#

# size of testing sets
TEST_MAX_SIZE = 300
DETEVAL_SIZE = 300
SREVAL_SIZE = 10
# detection model image input size
DET_SIZE = 1024
# how many times each sub-image tile will be smaller than DET_SIZE
TILING_FACTOR = 2
# minimum area required to perform object zoom on low score detections
MIN_AREA_OZ = 512
# minimum threshold above which to improve via additional sr detection
THRESHOLD_LOW = 0.1
# maximun threshold below which to improve via additional sr detection
THRESHOLD_HIGH = 0.75
# ratio for how much to increase an image tile with padding for tile overlapping
PADDING_RATIO = 0.1
# minimum IoU to consider two near objects as identical
IOU_THRESHOLD = 0.5
# LUT param used in some methods, valid range: [20,60]
LUT_THRESHOLD = 40
# LUT method used to transform the image
LUT_METHOD = 'increase'
# apply detection from SR subimage tiles
APPLY_SR = True
# apply LUT automatically based on tile brightness
APPLY_LUT = False
# apply horizontal flip on sub images
APPLY_FLIP = False
# apply x4 zoom on object detections below THRESHOLD_HIGH
APPLY_OZX4 = False
# if True, the image will be expanded around the borders
# keeping the detection in the center of the image. Otherwise
# the image will increase from the main tile
FORCE_CENTER_PATCH = True
# allow initial detection of the original input image before tiling 
INITIAL_DET = True
# use the full image as input for the initial detection
# or use instead a centered crop image that fits the det model size
DET_MODEL_NAME = 'efficientdet_d4_coco17_tpu-32'
DET_MODEL_DATE = '20200711'
SR_MODEL = 'Real-ESRGAN'
RANDOM_SEED = 0
# car category whitelisted by default
CATEGORIES_WHITELIST = [3]
# directories
BASE_DIR = '/content/gdrive/MyDrive/TFG'
DATASET_DIR = BASE_DIR + '/VisDrone_2019-DET-test-dev'
IMAGES_DIR = DATASET_DIR + '/images/'
SUBIMAGES_DIR = '/content/subimages/'
UTILS_DIR = BASE_DIR + '/utils/'
LABELS_DIR = DATASET_DIR + '/annotations/'
DETEVAL_DIR = BASE_DIR + '/deteval/'
SREVAL_DIR = BASE_DIR + '/sreval/'
MODELS_DIR = BASE_DIR + '/models/'
SRMODELS_DIR = MODELS_DIR + '/SR/'
TEST_DIR = BASE_DIR + '/tests/'
FIGURES_DIR = BASE_DIR + '/figures/'
#IN_DIR = BASE_DIR + 'in'
#OUT_DIR = BASE_DIR + 'out'

COCO_CATEGORIES = {
  "categories": 
  [
    {
      "id": 1,
      "name": "person"
    },
    {
      "id": 2,
      "name": "bicycle"
    },
    {
      "id": 3,
      "name": "car"
    },
    {
      "id": 4,
      "name": "motorcycle"
    },
    {
      "id": 6,
      "name": "bus"
    },
    {
      "id": 8,
      "name": "truck"
    }
  ]
}

CATEGORY_INDEX = {
  1: {
    'name': 'person'
  },
  2: {
    'name': 'bicycle'
  },
  3: {
      'name': 'car'
  },
  4: {
    'name': 'motorcycle'
  },
  6: {
    'name': 'bus'
  },
  8: {
    'name': 'truck'
  }
}


def init():
  os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "False"
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
  tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
  warnings.filterwarnings('ignore')           # Suppress Matplotlib warnings
  logging.disable(logging.WARNING)
  
  # Disable TensorFlow warnings
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  # Enable GPU dynamic memory allocation
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

  # clone git repo for clustering
  get_clustering()

  # download the sr model
  download_sr_models()

  # install the rest of external utils
  install_utils()
  
  # set the random seed
  random.seed(RANDOM_SEED)


def load_settings(settings):
  data = settings
  if not settings:
    data = read_data_from_json(BASE_DIR, 'base_settings')

  # keep the default settings for global variables if no data provided
  if data:
    if 'test_max_size' in data:
      global TEST_MAX_SIZE
      TEST_MAX_SIZE = data['test_max_size']
    if 'deteval_size' in data:
      global DETEVAL_SIZE
      DETEVAL_SIZE = 300
    if 'sreval_size' in data:
      global SREVAL_SIZE
      SREVAL_SIZE = 10
    if 'det_size' in data:
      global DET_SIZE
      DET_SIZE = data['det_size']
    if 'min_area_sr' in data:
      global MIN_AREA_OZ
      MIN_AREA_OZ = data['min_area_oz']
    if 'tiling_factor' in data:
      global TILING_FACTOR
      TILING_FACTOR = data['tiling_factor']
    if 'threshold_low' in data:
      global THRESHOLD_LOW
      THRESHOLD_LOW = data['threshold_low']
    if 'threshold_high' in data:
      global THRESHOLD_HIGH
      THRESHOLD_HIGH = data['threshold_high']
    if 'padding_ratio' in data:
      global PADDING_RATIO
      PADDING_RATIO = data['padding_ratio']
    if 'iou_threshold' in data:
      global IOU_THRESHOLD
      IOU_THRESHOLD = data['iou_threshold']
    if 'initial_det' in data:
      global INITIAL_DET
      INITIAL_DET = data['initial_det']
    if 'lut_threshold' in data:
      global LUT_THRESHOLD
      LUT_THRESHOLD = data['lut_threshold']
    if 'lut_method' in data:
      global LUT_METHOD
      LUT_METHOD = data['lut_method']
    if 'apply_sr' in data:
      global APPLY_SR
      APPLY_SR = data['apply_sr']
    if 'apply_lut' in data:
      global APPLY_LUT
      APPLY_LUT = data['apply_lut']
    if 'apply_flip' in data:
      global APPLY_FLIP
      APPLY_FLIP = data['apply_flip']
    if 'apply_ozx4' in data:
      global APPLY_OZX4
      APPLY_OZX4 = data['apply_ozx4']
    if 'force_center_patch' in data:
      global FORCE_CENTER_PATCH
      FORCE_CENTER_PATCH = data['force_center_patch']
    if 'det_model_name' in data:
      global DET_MODEL_NAME
      DET_MODEL_NAME = data['det_model_name']
    if 'det_model_date' in data:
      global DET_MODEL_DATE
      DET_MODEL_DATE = data['det_model_date']
    if 'sr_model' in data:
      global SR_MODEL
      SR_MODEL = data['sr_model']
    if 'categories_whitelist' in data:
      global CATEGORIES_WHITELIST
      CATEGORIES_WHITELIST = data['categories_whitelist']


def print_settings():
  print('-------------------------------------')
  print('param_name \t\t param_value')
  print('-------------------------------------')
  print('{} \t\t {}'.format('INITIAL_DET', INITIAL_DET))
  print('{} \t\t {}'.format('APPLY_SR', APPLY_SR))
  print('{} \t\t {}'.format('APPLY_LUT', APPLY_LUT))
  print('{} \t\t {}'.format('APPLY_FLIP', APPLY_FLIP))
  print('{} \t\t {}'.format('APPLY_OZX4', APPLY_OZX4))
  print('{} \t\t {}'.format('THRESHOLD_LOW', THRESHOLD_LOW))
  print('{} \t\t {}'.format('THRESHOLD_HIGH', THRESHOLD_HIGH))
  print('{} \t\t {}'.format('PADDING_RATIO', PADDING_RATIO))
  print('{} \t\t {}'.format('IOU_THRESHOLD', IOU_THRESHOLD))
  print('{} \t\t {}'.format('LUT_THRESHOLD', LUT_THRESHOLD))
  print('{} \t\t {}'.format('LUT_METHOD', LUT_METHOD))
  print('{} \t\t {}'.format('MIN_AREA_OZ', MIN_AREA_OZ))
  print('{} \t\t {}'.format('DET_MODEL_NAME', DET_MODEL_NAME))
  print('{} \t\t {}'.format('SR_MODEL', SR_MODEL))
  print('')


def read_data_from_json(path, file_name):
  """ Read data from json file

      Parameters
      ----------
      path: the base path
      file_name: the file name

      Returns
      -------
      the json data as a dict
  """
  try:
    full_filename = os.path.join(path, file_name + '.json')
    with open(full_filename, 'r', encoding='utf-8') as json_file:
      return json.load(json_file)
  except FileNotFoundError:
    print("{}.json file not found, using default settings instead.".format(file_name))
    return None


def install_utils():
  p = subprocess.Popen(['python', '-m', 'pip', 'install', 'tensorflow-object-detection-api'])
  p.wait()


def download_sr_models():
  # Real-ESRGAN
  os.chdir('/content')
  os.system('git clone https://github.com/xinntao/Real-ESRGAN.git')
  os.chdir('/content/Real-ESRGAN')
  p = subprocess.Popen(['python', '-m', 'pip', 'install', 'basicsr'])
  p.wait()
  p = subprocess.Popen(['python', '-m', 'pip', 'install', '-r', 'requirements.txt'])
  p.wait()
  p = subprocess.Popen(['python', 'setup.py', 'develop', '-q'])
  p.wait()
  os.system('wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P /content/Real-ESRGAN/experiments/pretrained_models')
  os.system('wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /content/Real-ESRGAN/experiments/pretrained_models')

  # BSRGAN
  """
  os.chdir('/content')
  os.system('git clone https://github.com/cszn/BSRGAN.git')
  p = subprocess.Popen(['python', '/content/BSRGAN/main_download_pretrained_models'])
  p.wait()
  
  # overwrite default git repo file with some changes
  source = BASE_DIR + '/src/main_test_bsrgan.py'
  destination = '/content/BSRGAN/'

  if os.path.exists('/content/BSRGAN/main_test_bsrgan.py'):
    os.remove('/content/BSRGAN/main_test_bsrgan.py')
  
  shutil.move(source, destination)
  """


def get_clustering():
  os.chdir('/content')
  os.system('git clone https://github.com/IvanGarcia7/ALAF.git')
  source = '/content/ALAF/ALAF/Cluster.py'
  destination = '/content/gdrive/MyDrive/TFG/src/'
  if not os.path.isfile(destination + 'Cluster.py'): 
    shutil.move(source, destination) 