import common
import cv2 as cv
import glob
import math
import numpy as np
import shutil
import subprocess
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from filehelper import *
from imagehelper import *


def evaluate_sr_model(image_list, model_name, factor):
  """ Evaluates the SR model over a list of images by computing their PSNR and SSIM values.
      If no model name is given, it retrieves the PSNR and SSIM values from bicubic upscaling

      Parameters
      ----------
      image_list: the list of image files to evaluate
      model_name: the SR model name. If None is passed then it uses cv2 BICUBIC
      factor: the upscaling factor

      Returns
      -------
      A list with the average PSNR and SSIM values
  """
  sum_psnr = 0
  sum_ssim = 0

  for image in image_list:
    original_image = cv.imread(image)
    fake_image = get_hr_image_from_fmf(image, model_name, factor)
    # make sure both images have the same shape
    fake_image = equalize_shape_to_original_image(fake_image, original_image)

    sum_psnr += get_psnr(fake_image, original_image)
    sum_ssim += get_ssim(fake_image, original_image)

  mean_psnr = sum_psnr / len(image_list)
  mean_ssim = sum_ssim / len(image_list)

  return [mean_psnr, mean_ssim]


def generate_sr_images(image_list, model_name, factor):
  """ Generate HR images from LR images and save them to files

      Parameters
      ----------
      image_list: the list of LR images
      model_name: the SR model name. If model name is None, then it
         generates the HR images with bicubic interpolation instead
      factor: the upscaling factor
  """
  sr_model = None
  if model_name:
    sr_model = get_sr_model(model_name, factor)
  
  for image_file in image_list:
    hr_image = None
    if sr_model:
      # Generate HR image using SR model
      if model_name == 'ESRGAN':
        image = preprocess_image(image_file)
        hr_image = sr_model(image)
        hr_image = postprocess_image(hr_image)
      else:
        img = cv.imread(image_file)
        hr_image = sr_model.upsample(img)
    else:
      # Generate HR image using cv2 resize
      #hr_image = cv.resize(img, hr_dims, interpolation=cv.INTER_CUBIC)
      # Generate HR image using PIL resize
      img = Image.open(image_file)
      width, height = img.size
      hr_dims = [width*factor, height*factor]
      hr_image = img.resize(hr_dims)
      # convert Image to opencv format
      hr_image = cv.cvtColor(np.array(hr_image), cv.COLOR_RGB2BGR)
    
    #save the HR image to file
    save_hr_image(hr_image, image_file, model_name, factor)


def preprocess_image(image_file):
  """ Load image from path and preprocess it to make it model ready

      Parameters
      ----------
      image_file: path to the image file

      Returns
      -------
      the tensor image
  """
  img = tf.image.decode_image(tf.io.read_file(image_file))
  img = tf.cast(img, tf.float32)
  return tf.expand_dims(img, 0)


def postprocess_image(image):
  """ Convert the tensor image into a cv numpy image to be saved to file

      Parameters
      ----------
      image: the tensor HR image
    
      Returns
      -------
      the cv numpy image
  """
  image = tf.clip_by_value(image, 0, 255)
  image = tf.squeeze(image)
  image = tf.cast(image, tf.uint8).numpy()
  return cv.cvtColor(image, cv.COLOR_RGB2BGR)


def save_hr_image(hr_image, image_file, model_name, factor):
  """ Save the HR image to file

      Parameters
      ----------
      hr_image: the high-res image
      image_file: the image file name
      model_name: the SR model name. If None is passed, 
        then it saves the HR image to the BICUBIC subfolder instead
      factor: the upscaling factor
  """
  save_path = ''
  if model_name:
    save_path = os.path.join(common.SREVAL_DIR, model_name + '_x' + str(factor))
  else:
    save_path = os.path.join(common.SREVAL_DIR, 'BICUBIC_x' + str(factor))

  file_name = get_filename_from_file(image_file, True)
  write_image_to_file(save_path, file_name, hr_image)


def generate_lr_images(image_list, factor):
  """ Generate and save low-res images from a list 
      of GT images using bicubic interpolation
      Parameters
      ----------
      image_list: the list of images
      factor: the downscaling factor
  """
  for image in image_list:
    '''
    img = cv.imread(image)
    height, width, _ = img.shape
    lr_dims = [width//factor, height//factor]
    lr_image = cv.resize(img, lr_dims, cv.INTER_CUBIC)
    '''
    img = Image.open(image)
    width, height = img.size
    lr_dims = [width//factor, height//factor]
    lr_image = img.resize(lr_dims)
    # convert Image to opencv format
    lr_image = cv.cvtColor(np.array(lr_image), cv.COLOR_RGB2BGR)
    # save lr image to file
    file_name = get_filename_from_file(image, True)
    base_path = os.path.join(common.SREVAL_DIR, 'LR_x' + str(factor))
    write_image_to_file(base_path, file_name, lr_image)


def get_sr_model(model_name, factor):
  """ Get the SR model by model name and upscaling factor

      Parameters
      ----------
      model_name: the sr model name
      factor: the sr model upscaling factor

      Returns
      -------
      the SR model
  """
  sr = cv.dnn_superres.DnnSuperResImpl_create()
  if model_name == 'ESRGAN':
    return download_sr_model('ESRGAN', 4)

  model_filename = model_name + '_x' + str(factor) + '.pb'
  model_path = os.path.join(common.SRMODELS_DIR, model_filename)
  sr.readModel(model_path)
  sr.setModel(model_name.lower(), factor)
  return sr


def download_sr_model(model_name, factor):
  """ Download the super resolution model
      Allowed model names: EDSR, ESPCN, FSRCNN, LapSRN, ESRGAN
      
      Parameters
      ----------
      model_name: the sr model name
      factor: the sr model upscaling factor

      Returns
      -------
      the path to the downloaded super resolution model
  """
  base_url = ''
  if model_name == 'EDSR':
    base_url = 'https://github.com/Saafke/EDSR_Tensorflow/tree/master/models/'
  elif model_name == 'FSRCNN':
    base_url = 'https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models/'
  elif model_name == 'ESPCN':
    base_url = 'https://github.com/fannymonori/TF-ESPCN/tree/master/export/'
  elif model_name == 'LapSRN':
    base_url = 'https://github.com/fannymonori/TF-LapSRN/tree/master/export/'
  elif model_name == 'ESRGAN':
    return hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')

  origin_url = base_url + model_name + '_x' + str(factor) + '.pb'
  model_dir = tf.keras.utils.get_file(
                                      origin=origin_url,
                                      cache_subdir='models'
              )
  return str(model_dir)

def get_original_images_from_fake_images(file_list):
  """ Get the list of original images from a list of fake images
  
      Parameters
      ----------
      file_list: the list of paths to fake images

      Returns
      -------
      the list of paths to original images
  """
  original_images = []
  for file in file_list:
    original_images.append(get_original_image_from_fake_image(file))

  return original_images

def get_original_image_from_fake_image(file):
  """ Get the original image path from a fake image path

      Parameters
      ----------
      file: the path to fake image file

      Returns
      -------
      the path to original image file
  """
  return os.path.join(common.IMAGES_DIR, file.rsplit('/', 1)[1])

def get_hr_image_from_fmf(file, model_name, factor):
  """ Get the HR image given an image file, model name and factor

      Parameters
      ----------
      file: the image file path
      model_name: the SR model name. 
        If None is passed, it retrieves HR image from bicubic upscaling instead
      factor: the upscaling factor

      Returns
      -------
      the corresponding HR image
  """
  file_name = file.rsplit('/', 1)[1]
  subpath = ''
  if model_name:
    subpath = model_name + '_x' + str(factor)
  else:
    subpath = 'BICUBIC_x' + str(factor)
  
  full_path = os.path.join(common.SREVAL_DIR, subpath, file_name)

  return cv.imread(full_path)


def equalize_shape_to_original_image(fake, original):
  #TODO: add docstring
  fake_height, fake_width = get_image_shape(fake)
  original_height, original_width = get_image_shape(original)
  diff_width = original_width - fake_width
  diff_height = original_height - fake_height
  new_fake_img = None
  # adjustments along the x-axis
  if diff_width > 0:
    new_fake_img = cv.copyMakeBorder(fake, 0, 0, 0, diff_width, cv.BORDER_REPLICATE)
  elif diff_width < 0:
    new_fake_img = fake[0:-diff_width,:]
  else:
    new_fake_img = fake

  # adjustments along the y-axis
  if diff_height > 0:
    new_fake_img = cv.copyMakeBorder(new_fake_img, 0, diff_height, 0, 0, cv.BORDER_REPLICATE)
  elif diff_height < 0:
    new_fake_img = new_fake_img[:,0:-diff_height]
  
  return new_fake_img


def get_psnr(fake, original):
    """ Get the PSNR between the two images

        Parameters
        ----------
        fake: the fake image, upscaled from a LR image
        original: the original image

        Returns
        -------
        the PSNR
    """
    original = original.astype(float)
    fake = fake.astype(float)
    diff = (original - fake).flatten('C')
    MSE = np.mean(diff ** 2.)

    return 20 * math.log10(255.) - 10 * math.log10(MSE)


def get_ssim(fake, original):
    """ Get the SSIM between the two images

        Parameters
        ----------
        fake: the fake image, upscaled from a LR image
        original: the original image

        Returns
        -------
        the SSIM
    """
    # Add an outer batch to each image
    im1 = tf.expand_dims(fake, axis=0)
    im2 = tf.expand_dims(original, axis=0)
    # Compute SSIM over tf.uint8 Tensors
    return tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy()[0]


def batch_image_superres(image_folder, model_name, factor):
  """ Generate SR images given a sr model name and factor from a set
      of images contained in image_folder path

      Parameters
      ----------
      image_folder: the image folder path from which to obtain the input images
      model_name: the SR model name. Allowed values: Real-ESRGAN or BSRGAN 
      factor: the upscaling factor. Allowed values: 2 or 4

      Returns
      -------
      the list of new SR images as files
  """
  sr_images = []

  # make sure path ends with '/'
  if image_folder[-1] != '/':
    image_folder += '/'
  
  if model_name == 'Real-ESRGAN':
    # launch process for Real-ESRGAN inference
    p = subprocess.Popen(['python', '/content/Real-ESRGAN/inference_realesrgan.py',
                          '-n', 'RealESRGAN_x' + str(factor) + 'plus',
                          '-i', image_folder,
                          '-o', image_folder,
                          '-s', str(factor)])

    # wait for process to complete
    p.wait()

    # Rename the new sr subimages
    sisr_list = glob.glob(image_folder + '*_out.jpg')
    for sisr in sisr_list:
      new_image = sisr[:-7] + 'x' + str(factor) + '.jpg'
      sr_images.append(new_image)
      shutil.move(sisr, new_image)
    
  if model_name == 'BSRGAN':
    # launch process for BSRGAN inference
    p = subprocess.Popen(['python', '/content/BSRGAN/main_test_bsrgan.py',
                          image_folder,
                          str(factor)])
    # wait for process to complete
    p.wait()

    sr_images = glob.glob(image_folder + '*_x' + str(factor) + '.jpg')

  return sr_images