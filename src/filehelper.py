import common
import glob
import json
import os
import random
import shutil


def get_files_by_expr_and_ext(path, expr, ext):
  """ Get the files that match a given expression and extension 
      at specified path

      Parameters
      ----------
      path: the annotation/image path
      expr: the expression of files to match
      ext: the extension of files to match

      Returns
      -------
      the list of files
  """
  if path[-1] != '/':
    path += '/'

  return glob.glob(path + expr + '.' + ext)


def get_sample_files(file_list, sample_size):
  """ Get a sample list from a list of files
      
      Parameters
      ----------
      file_list: the list of files from which to get the samples
      sample_size: the sample size

      Returns
      -------
      the list of sample files
  """
  return random.sample(file_list, sample_size)


def get_filename_from_file(file_path, with_ext):
  """ Get the filename from the file full path

      Parameters
      ----------
      file: the path to file

      Returns
      -------
      the filename
  """
  if with_ext:
    return file_path.rsplit('/', 1)[1]
  else:
    filename = file_path.rsplit('/', 1)[1]
    return filename.rsplit('.', 1)[0]


def get_image_dict_from_image_file(image_file, image_count):
  """ Get the dict for the given image and id
      
      Parameters
      ----------
      image_file: the image filename
      image_count: the image id

      Returns
      -------
      the image dict

  """
  image_dict = {}
  image_dict['file_name'] = image_file.rsplit('/', 1)[1]
  image_dict['id'] = image_count

  return image_dict


def get_annotation_file_from_image_file(image_file):
  """ Get the corresponding annotation file for a given image file

      Parameters
      ----------
      image_file: the image file

      Returns
      -------
      the matching annotation file
  """
  file_name = image_file.rsplit('/',1)[1]
  file_name = file_name.rsplit('.', 1)[0] + '.txt'
  return common.LABELS_DIR + file_name


def get_image_list_from_json_file(path, file_name, sample_size):
  """ Get the list of image filenames from json file

      Parameters
      ----------
      path: the base path
      file_name: the json filename
      sample_size: the number of sample files to retrieve, limit defined by TEST_MAX_SIZE

      Returns
      -------
      the list of image filenames
  """
  images = read_annotations_from_json(path, file_name)['images']
  image_list = [image['file_name'] for image in images]
  if sample_size > common.TEST_MAX_SIZE:
    sample_size = common.TEST_MAX_SIZE

  return image_list[:sample_size]


def get_categories_from_category_list(category_list):
  """ Get the COCO categories whitelisted in category_list
      
      Parameters
      ----------
      category_list: the category whitelist

      Returns
      -------
      the whitelisted categories dict
  """
  return list(filter(lambda i: i['id'] in category_list, common.COCO_CATEGORIES['categories']))


def write_annotations_to_json(path, file_name, data):
  """ Save the annotations dict to json file

      Parameters
      ----------
      path: the base path
      file_name: the file name
      data: the dict data to be written to json file
  """
  full_filename = path + file_name + '.json'
  os.makedirs(os.path.dirname(full_filename), exist_ok=True)
  with open(full_filename, 'w', encoding='utf-8') as out_file:
    json.dump(data, out_file, indent=2)


def read_annotations_from_json(path, file_name):
  """ Read COCO format annotations from json file

      Parameters
      ----------
      path: the base path
      file_name: the file name

      Returns
      -------
      the json data as a dict
  """
  full_filename = path + file_name + '.json'
  with open(full_filename, 'r', encoding='utf-8') as json_file:
    return json.load(json_file)

def clear_directory_by_path(path):
  """ Clear the directory given by path

      Parameters
      ----------
      path: the directory path
  """
  if os.path.isdir(path):
    shutil.rmtree(path, ignore_errors=True)
  os.mkdir(path)