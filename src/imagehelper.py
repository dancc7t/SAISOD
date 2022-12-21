import common
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from filehelper import *


def write_image_to_file(path, file_name, image):
  """ Save image to file at specified path

      Parameters
      ----------
      path: the image base path
      file_name: the image filename
      image: the image to save
  """
  if not os.path.exists(path):
    os.mkdir(path)
  
  if isinstance(image, tf.Tensor):
    # convert tensor to numpy image
    image = tf.squeeze(image)
    image = tf.clip_by_value(image, 0, 255)
    if image.dtype != 'uint8':
      image = tf.cast(image, tf.uint8)
    image = image.numpy()
    # change the color space before saving to file
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

  cv.imwrite(os.path.join(path, file_name), image)


def write_subimages_to_file(image_name, sub_images):
  """ Saves the list of sub images to file

      Parameters
      ----------
      image_file: the image filename
      sub_images: the list of sub images to write to file

      Returns
      -------
      the folder path of subimages
  """
  folder_path = common.SUBIMAGES_DIR + image_name
  if not os.path.exists(folder_path):
    os.mkdir(folder_path)

  i = 1
  for sub_image in sub_images:
    write_image_to_file(folder_path, str(i) + '.jpg', sub_image)
    i += 1

  return folder_path + '/'


def get_image_from_file(
  image_file,  
  as_rgb=True, 
  as_tensor=False, 
  as_type='uint8'
):
  """ Get image object from image file

      Parameters
      ----------
      as_rgb: load the image using RGB color space, otherwise
              load it as BGR open cv default color space
      image_file: the path to image
      as_tensor: the image is returned as a tensor if True, 
                 else as a ndarray
      as_type: the dtype of the image


      Returns
      -------
      the image as a ndarray or as a tensor
  """
  img = cv.imread(image_file)
  if as_rgb:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  if as_tensor:
    if as_type == 'uint8' or as_type == 'int':
      tensor_img = tf.convert_to_tensor(np.array(img), dtype=np.uint8)
      return tf.expand_dims(tensor_img, axis=0)
    elif as_type == 'float32' or as_type == 'float':
      tensor_img = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
      return tf.expand_dims(tensor_img, axis=0)
  else:
    if as_type == 'uint8':
      return img
    elif as_type == 'float':
      return img.astype(float)


def get_image_shape(image):
  """ Get the image height and width sizes

      Parameters
      ----------
      image: the image

      Returns
      -------
      the height and width of the image
  """
  if isinstance(image, tf.Tensor):
    return image.shape.as_list()[1], image.shape.as_list()[2]
  else:
    return image.shape[0], image.shape[1]


def get_flip_image(image_file):
  """ Get the image flipped around the y-axis (horizontally)
  
      Parameters
      ----------
      image_file: the input image file

      Returns
      ------- 
      the image flipped horizontally
  """
  image = cv.imread(image_file)
  flip_image = cv.flip(image, 1)
  # save the lut image to file
  flip_image_file = image_file.replace('x2', 'flip')
  cv.imwrite(flip_image_file, flip_image)

  return flip_image_file


def get_lut_image(image_file, method):
  """ Get LUT transformed image given an input image and transform method
      
      Parameters
      ----------
      image_file: the input image file
      method: the transform method

      Returns
      ------- 
      the LUT transformed image
  """
  image = cv.imread(image_file)
  img_brightness = np.mean(image.flatten())

  # only process images with brightness below the threshold
  if img_brightness > common.LUT_THRESHOLD:
    return None

  if method == 'clahe':
    lut_img = get_clahe_img(image)
  elif method == 'gamma':
    lut = get_gamma_lut(image)
    lut_img = cv.LUT(image, lut)
  elif method == 'histcdf':
    lut = get_histcdf_lut(image)
    lut_img = cv.LUT(image, lut)
  elif method == 'increase':
    lut = get_increase_lut(image)
    lut_img = cv.LUT(image, lut)
  elif method == 'sqrt':
    lut = get_sqrt_lut()
    lut_img = cv.LUT(image, lut)
  else:
    # if no valid method then return None
    return None

  # save the lut image to file
  lut_image_file = image_file.replace('x2', 'lut')
  cv.imwrite(lut_image_file, lut_img)

  return lut_image_file


def get_clahe_img(image):
  """ Get image transformed with CLAHE

      Parameters
      ----------
      image: the input image

      Returns
      ------- 
      the transformed image
  """
  img = cv.cvtColor(image, cv.COLOR_RGB2Lab)
  # create clahe object with fixed params
  clahe = cv.createCLAHE(clipLimit=3,tileGridSize=(4,4))
  # apply clahe on the L channel
  img[:,:,0] = clahe.apply(img[:,:,0])
  # return clahe "lut" image 
  return cv.cvtColor(img, cv.COLOR_Lab2RGB)


def get_gamma_lut(image):
  """ Get LUT from gamma correction of input image

      Parameters
      ----------
      image: the input image

      Returns
      ------- 
      the LUT for the image
  """
  gamma = np.mean(image.flatten()) / common.LUT_THRESHOLD
  v_in = np.arange(256)
  v_out = (((v_in / 255.) ** gamma) * 255)
  return v_out.astype('uint8')


def get_histcdf_lut(image):
  """ Get LUT from cummulative histogram of image

      Parameters
      ----------
      image: the input image

      Returns
      ------- 
      the LUT for the image
  """
  hist, bins = np.histogram(image.flatten(),256,[0,256])
  cdf = hist.cumsum()
  v_out = 256 * (cdf / (2*cdf.max()))
  return v_out.astype('uint8')


def get_increase_lut(image):
  """ Get LUT from high contrast increase on the low in values

      Parameters
      ----------
      image: the input image

      Returns
      ------- 
      the LUT for the image
  """
  # interval of increase
  i = np.mean(image.flatten())

  # limit interval from below
  if i < common.LUT_THRESHOLD:
    i = common.LUT_THRESHOLD

  # slope of increase
  m = 255. / i
  v_out1 = m*np.arange(i)
  v_out2 = 255*np.ones(256-i)

  return np.concatenate((v_out1,v_out2))


def get_sqrt_lut():
  """ Get LUT from square root function over all in values

      Returns
      ------- 
      the LUT for the image
  """
  v_in = np.arange(256)
  v_out = np.sqrt(v_in)/np.sqrt(255)

  return np.floor(v_out*256)


def get_image_border_median(image):
  #TODO: add docstring
  height = image.shape[0]
  width = image.shape[1]

  top = image[0,:,:]
  bottom = image[height-1,:,:]
  left = image[:,0,:]
  right = image[:,width-1,:]

  border = np.vstack((top,bottom,left,right))

  median_b = np.median(border[:,0])
  median_g = np.median(border[:,1])
  median_r = np.median(border[:,2])

  return (median_b, median_g, median_r)


def add_padding_to_image(image):
  #TODO: add docstring
  
  height, width = get_image_shape(image)

  diff_height = common.DET_SIZE - height
  diff_width = common.DET_SIZE - width
  delta_y = int(0.5*diff_height)
  delta_x = int(0.5*diff_width)

  border_median = get_image_border_median(image)

  new_img = cv.copyMakeBorder(image,
                              delta_y, 
                              delta_y, 
                              delta_x, 
                              delta_x, 
                              cv.BORDER_CONSTANT, 
                              value=border_median)

  return new_img


def create_image_patch_from_object(image, zoom_path, object_id, bbox):
  #TODO: add docstring
  height, width = get_image_shape(image)
  ymin, xmin, ymax, xmax = enlarge_patch_from_bbox(height, width, bbox)
  # get the enlarged bbox image patch
  image_patch = image[ymin:ymax, xmin:xmax]
  # write image patch to file
  write_image_to_file(zoom_path, str(object_id) + '.jpg', image_patch)


def enlarge_patch_from_bbox(im_height, im_width, bbox):
  #TODO: add docstring
  # calculate the new rectangle points from the original ones 
  ymin = bbox[0]
  xmin = bbox[1]
  ymax = bbox[2]
  xmax = bbox[3]
  # increases on both axis
  deltax = int((xmax - xmin) * common.PADDING_RATIO)
  deltay = int((ymax - ymin) * common.PADDING_RATIO)

  new_xmin = xmin - deltax
  new_xmax = xmax + deltax
  new_ymin = ymin - deltay
  new_ymax = ymax + deltay

  # ensure new enlarged box is within the image limits
  if new_xmin < 0:
    new_xmin = 0
  if new_xmax > im_width:
    new_xmax = im_width
  if new_ymin < 0:
    new_ymin = 0
  if new_ymax > im_height:
    new_ymax = im_height

  return new_ymin, new_xmin, new_ymax, new_xmax

