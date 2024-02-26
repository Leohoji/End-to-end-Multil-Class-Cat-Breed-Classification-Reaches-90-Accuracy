# file operation
import os
import copy
import shutil
import random
import zipfile
import imghdr
from pathlib import Path

# data processing
import numpy as np

# neural network model
import tensorflow as tf

# data visualization
import matplotlib.pyplot as plt


class DataProcessor:
  def __init__(self, filename):
    """
    Class can unzip zip file, read data, preprocess data and show images from data.

    Args:
      filename (str): a filepath to a target zip folder to be unzipped.
    """
    self.IMG_SIZE = 224
    self.BATCH_SIZE = 32
    self.filename = filename

    # home directory
    self.home_path = None

    # images class names
    self.class_names = ''

    # images datasets
    self.images_dict = ''
    self.images_list = ''

    # subdir paths
    self.train_dir = ''
    self.test_dir = ''

    # batch data
    self.train_data_batch = ''
    self.val_data_batch = ''
    self.test_data_batch = ''



  def unzip_file(self):
    """
    Unzips filename into the current working directory.
    """
    try:
      with zipfile.ZipFile(self.filename) as zip_ref:
        zip_ref.extractall()
        print(f"Extract zip file completely from {self.filename}.")
    except FileNotFoundError:
        print(f" {FileNotFoundError.__name__} --> The filepath is not found! ")


  def set_home_path(self, new_home_path):
    """
    Set home directory.
    """

    # home_path = input("Please set your home directory: ")
    while True:
      if os.path.isdir(new_home_path):
        self.home_path = new_home_path
        print(f"Set home directory successfully as {self.home_path}")
        break
      else:
        continue


  def walk_through_dir(self):
    """
    Walks through home directory returning its contents.

    Returns:
      Total number of images in dataset.
    """

    total_images = 0
    for dirpath, dirnames, filenames in os.walk(self.home_path):
      directory_number = len(dirnames)
      image_number = len(filenames)

      print(
          "Dirpath: [%s] | %s | %s"
          % (
              dirpath,
              "Dirnumber: " + str(directory_number) if directory_number > 0 else "No subdirectory",
              "Filenumber:" + str(image_number) if image_number > 0 else "No image"
              )
          )
      total_images += image_number

    return total_images


  def get_class_names(self, data_dir=None):
    """
    Get class names from dataset directory.
    """
    Home = data_dir if data_dir else Path(self.home_path)
    class_names = [p.name for p in Home.glob("*") if (p.name != 'train') & (p.name != 'test')]
    print(f"There are {len(class_names)} classes:\n {class_names}")

    self.class_names = class_names


  def read_acceptable_data(self, train_test_split=False):
    """
    Read acceptable images, filter non-datatype data and unaccepted by tensorflow data, only for all data.

    Return:
      images_dict (dict): A images dataset in dictionary datatype with label as key and image paths as values.
    """

    # full data collection
    images_dict = dict()

    if train_test_split:

      # create cat breed dictionary
      for cat_breed in self.class_names:
        images_dict[cat_breed] = list()

      # walk subdirectory and combine images
      for subdir in [self.train_dir, self.test_dir]:
        for image_label in os.listdir(subdir):
          cat_images = []
          data_dir = os.path.join(subdir, image_label)
          for image in Path(data_dir).rglob('*'):
            cat_images.append(str(image))
          images_dict[image_label].extend(cat_images)
    else:
      # check image extensions
      image_extensions = [".png", ".jpg"]
      img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]

      for class_dir in self.class_names:

        # cat breed images collection
        class_images = []
        data_dir = self.home_path + "/" + class_dir

        # unaccepted images counting
        not_image_count = 0
        not_accepted_by_tf = 0

        # Walk each image and check datatype (Non-datatype, unaccepted by TF, or acceptable)
        for filepath in Path(data_dir).rglob("*"):
          if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
              not_image_count += 1 # count non image
            elif img_type not in img_type_accepted_by_tf:
              not_accepted_by_tf += 1 # count unacceptable images
            else:
              class_images.append(str(filepath)) # if normal, collect it

        # Show numbers of unacceptable image(s)
        if not_image_count:
          print(f"Not images: {not_image_count} in {class_dir}")
        if not_accepted_by_tf:
          print(f"Not accepted by TensorFlow: {not_accepted_by_tf} in {class_dir}")

        # Create dataset
        images_dict[class_dir] = class_images
    print('images_dict has been created.')
    self.images_dict = images_dict
    return self.images_dict


  def create_train_test_folder(self, all_splitting=False, split_index=200):
    """
    Creating subdirectories named train and test respectively, and splitting partial images into train and test folder.

    Original:

      Main-directory\
      ......Class_a_subdir\
      ---------class_a_image_1
      ---------class_a_image_2
      ......Class_b_subdir\
      ---------class_b_image_1
      ---------class_b_image_2

    Splitting:
      Main-directory\
      ___train_subdir\
      ......Class_a_subdir\
      ---------class_a_image_1
      ---------class_a_image_2
      ......Class_b_subdir\
      ---------class_b_image_1
      ---------class_b_image_2
      ___test_subdir\
      ......Class_a_subdir\
      ---------class_a_image_1
      ---------class_a_image_2
      ......Class_b_subdir\
      ---------class_b_image_1
      ---------class_b_image_2

    Args:
      all_splitting (bool): Whether to split whole data.
      split_index (int): Number of images for splitting.
    """

    # train and test directory
    self.train_dir = os.path.join(self.home_path, 'train')
    self.test_dir = os.path.join(self.home_path, 'test')

    # create "train" and "test" folder
    for subdir in [self.train_dir, self.test_dir]:
      try:
        os.mkdir(subdir)
        print(f"{subdir} has been created.")
      except FileExistsError:
        print(f"{subdir} folder is exists, go next!")
      except Exception as e:
        print(f"Error: {e.__class__.__name__}")

    # make a copy
    images_dict_copy = copy.deepcopy(self.images_dict)

    # Create path of each cat-breed folder
    for cat_breed, cat_images in images_dict_copy.items():

      # image target directory
      cat_train_class_dir = os.path.join(self.train_dir, cat_breed) + '/'
      cat_test_class_dir = os.path.join(self.test_dir, cat_breed) + '/'

      # create cat breed subdirectory
      try:
        os.mkdir(cat_train_class_dir)
        os.mkdir(cat_test_class_dir)
      except FileExistsError:
        print("Splitting directly!")
      except Exception as e:
        print(f"Error: {e.__class__.__name__}")

      print(f"{cat_breed} images allocating...")

      # allocate images
      random.shuffle(cat_images)

      if all_splitting:
        # Set splitting index --> for all data
        # all_images_splitting_number = len(images_dict_copy[cat_breed])

        # train test splitting
        train_cat_images = cat_images[:-50]
        test_cat_images = cat_images[-50:]
      else:
        # Set splitting index --> for partial data
        partial_splitting_index = split_index

        # train test splitting
        train_cat_images = cat_images[:partial_splitting_index]
        test_cat_images = cat_images[partial_splitting_index:partial_splitting_index+50]

      # move train images
      for train_cat_image in train_cat_images:
        shutil.move(train_cat_image, cat_train_class_dir)

      # move test images
      for test_cat_image in test_cat_images:
        shutil.move(test_cat_image, cat_test_class_dir)

      print(f"{cat_breed} images have been allocated.")
      print('-'*50)


  def get_images_list(self):
    """
    Collect all image paths in a List.

    Return
      images_list (list): A images dataset in list datatype.
    """
    images = []
    for cat_images in list(self.images_dict.values()):
        images.extend(cat_images)
    self.images_list = images
    return self.images_list


  def show_25_images_from_dataset(self, data_list=None):
    """
    Display a plot of 25 images and labels from a data list.

    Args:
      data_list (list): Dataset passed into for visualization.
    """

    dataset = data_list if data_list else self.images_list

    # Set figure size
    plt.figure(figsize=(15, 15))  # Set figure size

    # Read and display an image -> Choose 25 images
    for i in range(25):
        cat_img = random.choice(dataset)
        label = cat_img.split('\\')[-2]
        plt.subplot(5, 5, i + 1)  # Create subplots (5 rows, 5 columns)
        img = plt.imread(cat_img)
        plt.imshow(img)
        plt.title(label, fontsize=14)  # Add the image label as the title
        plt.axis("off") # Turn the grid lines off


  def create_data_batches_from_directory(self, IMG_SIZE=0, validation_split=0.1):
    """
    Create data batches for training dataset, validation dataset, and testing dataset.

    Args:
      validation_split (float): The proportion for validation splitting.
    """
    # Set image size
    if IMG_SIZE:
      IMAGE_SIZE = (IMG_SIZE, IMG_SIZE)
    else:
      IMAGE_SIZE = (self.IMG_SIZE, self.IMG_SIZE)

    # training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=self.train_dir,
        image_size=IMAGE_SIZE,
        label_mode='categorical',
        batch_size=self.BATCH_SIZE,
        validation_split=validation_split,
        subset='training',
        seed=42
    )

    self.train_data_batch = train_dataset

    # validation dataset
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=self.train_dir,
        image_size=IMAGE_SIZE,
        label_mode='categorical',
        batch_size=self.BATCH_SIZE,
        validation_split=validation_split,
        subset='validation',
        seed=42
    )
    self.val_data_batch = val_dataset

    # testing dataset
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=self.test_dir,
        image_size=IMAGE_SIZE,
        label_mode='categorical',
        batch_size=self.BATCH_SIZE,
        shuffle=False
    )
    self.test_data_batch = test_dataset

    return self.train_data_batch, self.val_data_batch, self.test_data_batch


  def show_16_images_from_batch(self, batch_data):
    """
    Display a plot of 25 images and labels from a data batch.

    Args:
      batch_data: The dataset after data batching.
    """

    images_generation, labels_generation = next(batch_data.as_numpy_iterator())

    plt.figure(figsize=(13, 13))  # Set figure size
    for i in range(16):
        plt.subplot(4, 4, i + 1)  # Create subplots (5 rows, 5 columns)
        plt.imshow(images_generation[i])  # display an image
        plt.title(self.class_names[labels_generation[i].argmax()], fontsize=14) # Add the image label as the title
        plt.axis("off") # Turn the grid lines off


  def unbatchify(self, data):
    """
    Takes a batches dataset of (image, label) Tensors and returns seperate arrays
    of images and labels.

    Args:
      data: The dataset expected to unbatchified.
    """
    images = []
    labels = []
    # Loop through unbatched data
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(self.class_names[np.argmax(label)])
    return images, labels


def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.io.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img



__all__ = ['DataProcessor', 'load_and_prep_image']
