import common
import numpy as np
from imagehelper import *
from srhelper import *


def get_sr_subimages_and_tiles(image_file, write_to_file):
  """ Get the list of SR subimages and tiles from image

      Parameters
      ----------
      image_file: the image file
      write_to_file: if True write the sr subimages to file, 
                     else get their filenames

      Returns
      -------
      the list of SR subimages
  """
  # get image from file
  image = get_image_from_file(image_file, as_rgb=False, as_tensor=False, as_type='uint8')
  height, width = get_image_shape(image)
  # get image tiling and sub images from the base image
  x_tiles, y_tiles = get_image_tiling(width, height)
  tiles = get_tiles(width, height, x_tiles, y_tiles)
  sub_images = get_subimages_from_tiles(image, tiles)
  # save sub images to image folder
  folder_name = image_file.rsplit('/', 1)[1]
  folder_name = folder_name.rsplit('.', 1)[0]
  sisr_list = []
  if write_to_file:
    subimage_path = write_subimages_to_file(folder_name, sub_images)
    # apply SR to each of the subimages
    sisr_list = batch_image_superres(subimage_path, common.SR_MODEL, common.TILING_FACTOR)
  else:
    sisr_list = get_files_by_expr_and_ext(common.SUBIMAGES_DIR + folder_name + '/', '*_x2', 'jpg')
  
  # make sure the list of sr subimages is sorted in the same order as the list of tiles
  sisr_list = sorted(sisr_list)
  
  return sisr_list, tiles


def get_image_tiling(width, height):
  """ Get the image tiling from its height and width sizes

      Parameters
      ----------
      width: the image width
      height: the image height

      Returns
      -------
      the number of horizontal and vertical tiles best-fitting
      for the image given the detection input size and tiling factor
  """
  num_tiles = np.arange(1,9)
  xvalues = abs(common.DET_SIZE / common.TILING_FACTOR - width/num_tiles)
  yvalues = abs(common.DET_SIZE / common.TILING_FACTOR - height/num_tiles)

  x_tiles = num_tiles[np.argmin(xvalues)]
  y_tiles = num_tiles[np.argmin(yvalues)]
    
  return x_tiles, y_tiles


def get_tiles(width, height, x_tiles, y_tiles):
  """ Get the list of tiles from image shape and tiling

      Parameters
      ----------
      width: the image width
      height: the image height
      x_tiles: the number of tiles along the x-axis
      y_tiles: the number of tiles along the y-axis

      Returns
      -------
      the list of tiles

  """
  tile_width = width / x_tiles
  tile_height = height / y_tiles
  tile_list = []

  num_tiles = x_tiles*y_tiles

  for i in range(num_tiles):
    tile = {}
    # compute the tile coords
    x = i % x_tiles
    y = int(i / x_tiles)
    
    xmin_padding = get_is_inner_point(x, x_tiles)*common.PADDING_RATIO*tile_width
    xmax_padding = get_is_inner_point(x+1, x_tiles)*common.PADDING_RATIO*tile_width
    ymin_padding = get_is_inner_point(y, y_tiles)*common.PADDING_RATIO*tile_height
    ymax_padding = get_is_inner_point(y+1, y_tiles)*common.PADDING_RATIO*tile_height

    xmin = int(x*tile_width - xmin_padding)
    xmax = int((x+1)*tile_width + xmax_padding)
    ymin = int(y*tile_height - ymin_padding)
    ymax = int((y+1)*tile_height + ymax_padding)

    # prevent out of bounds indexing
    """ TODO
    if xmax == width:
      xmax = xmax - 1
    if ymax == height:
      ymax = ymax - 1
    """
    #save tile to list of tiles
    tile['tile_id'] = i + 1
    tile['xmin'] = xmin
    tile['ymin'] = ymin
    tile['xmax'] = xmax
    tile['ymax'] = ymax
    tile_list.append(tile)

  return tile_list


def get_is_inner_point(axis_pos, axis_tiles):
  """ Get wether a point is inner or not. A point is inner
      if it is not on the edges (0 or max) along its axis
  """
  return 0 if axis_pos == 0 or axis_pos == axis_tiles else 1


def get_subimages_from_tiles(image, tile_list):
  """ Get a list of sub images given an input image and its tiles

      Parameters
      ----------
      image: the image
      tiles_list: the list of tiles

      Returns
      -------
      the list of sub images according to its tiling configuration
  """

  sub_images = []

  for tile in tile_list:
    sub_image = image[tile['ymin']:tile['ymax'],tile['xmin']:tile['xmax']]
    sub_images.append(sub_image)
  
  return sub_images