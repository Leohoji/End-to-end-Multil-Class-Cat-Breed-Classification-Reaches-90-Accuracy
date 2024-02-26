# file operation
import os
import pickle
from datetime import datetime as dt

# data processing
import pandas as pd

# neural network model
import tensorflow as tf

# data visualization
import matplotlib.pyplot as plt

class ModelTrainer:
  def __init__(self, IMG_SIZE=224, train_dataset=None, val_dataset=None, test_dataset=None, classes=20, handle_number=0):
    """
    ModelTrainer class object can build model, compile, train, evaluate, fine-tune, plotting results, and save model.

    Args:
      IMG_SIZE (int): image size for input.
      train_dataset: training dataset batched before.
      val_dataset: validation dataset batched before.
      test_dataset: testing dataset batched before.
      classes (int): label numbers of datasets, 20 default.
      handle_number (int): Number for naming name of each layer, format: "LayerName_{handle_number}"
    """

    # Set datasets
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.test_dataset = test_dataset

    # Set hyperparameters
    self.IMG_SIZE = IMG_SIZE
    self.INPUT_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)
    self.OUTPUT_SHAPE = classes
    print(f"Image size: {self.IMG_SIZE}")
    print(f"Input shape: {self.INPUT_SHAPE}")
    print(f"output shape: {self.OUTPUT_SHAPE}")

    # model
    self.model = ''

    # epochs
    self.epochs = 0
    self.initial_epochs = 0

    # callbacks
    self.checkpoint_path = None
    self.callbacks = []

    # model history
    self.model_history = ''
    self.model_fine_tune_history = ''
    self.evaluating_results = ''

    # model saving
    self.model_save_path = ''

    # Handle number
    self.handle_number = handle_number

  def _data_augmentation(self):
    """
    Build data augmentation layer.

    Return:
      model object for data augmentation.
    """
    data_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomHeight(0.2),
        tf.keras.layers.RandomWidth(0.2),
        tf.keras.layers.RandomContrast(0.5)
        ], name="data_augmentation_%s"%(str(self.handle_number)))

    return data_augmentation


  def build_model(self, base_model, preprocess_input, model_name):
    """
    Build a transfer learning model.

    Args:
      base_model: Transfer learning model passed into building structure.
      preprocess_input: Function for data preprocessing
    Return:
      model object
    """
    base_model = base_model
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=self.INPUT_SHAPE, name="input_layer_%s"%(str(self.handle_number)))
    x = preprocess_input(inputs)
    data_augmentation = self._data_augmentation()(x)
    x = base_model(data_augmentation, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_%s"%(str(self.handle_number)))(x)
    outputs = tf.keras.layers.Dense(self.OUTPUT_SHAPE, activation="softmax", name="output_layer_%s"%(str(self.handle_number)))(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)

    self.model = model
    print(model.summary())

    return self.model


  def set_checkpoint(self, save_path='full_data_model_checkpoints_weights/model_checkpoints_weights.h5'):
    """
    Set checkpoint path and callbacks

    Args:
      save_path: Path the checkpoint saved to.

    Return
      checkpoint path (str)
    """

    # Set checkpoint path
    checkpoint_path = save_path

    # Create ModelCheckpoint callback -> only save weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True, # only save the best weights
        verbose=1)

    self.checkpoint_path = checkpoint_path
    self.callbacks.append(checkpoint_callback)
    return self.checkpoint_path


  def compile(self, loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"]):
    """
    Compile the model built before.

    Args:
      loss: Loss function, categorical_crossentropy default.
      optimizer: Optimizer object, Adam default.
      metrics (list): Metrics, accuracy default.
    """
    if self.model:
      # Compile the model
      self.model.compile( loss=loss, optimizer=optimizer, metrics=metrics )
    else:
      return 'Build model first.'


  def __save_training_history(self, history, history_save_to):
    """
    Save history recording loss and accuracy in pkl format.

    Args:
      history: History object.
      history_save_to (str): The path history saved to
    """

    # Save history data and model score data
    with open(history_save_to, 'wb') as handle:
      pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


  def __load_training_history(self, save_path):
    """
    Load training history from saved path.

    Args:
      save_path (str): The path history saved at.
    """
    # Save history data and model score data
    try:
      with open(save_path, 'rb') as handle:
        target_history = pickle.load(handle)
      return target_history
    except Exception as e:
      print(e.__class__.__name__)


  def fit_model(self, epochs, fine_tune_epochs=0):
    """
    Fit built model with specific epochs.

    Args:
      epochs (int): Total epochs for model training.
      fine_tune_epochs (int): Epochs for fine tuning.

    Return:
      training history
    """
    # initial epochs
    self.epochs = epochs
    self.initial_epochs = self.epochs - fine_tune_epochs if fine_tune_epochs else 0

    # Fit the model
    model_history = self.model.fit(
        self.train_dataset,
        epochs = self.epochs,
        steps_per_epoch = len(self.train_dataset),
        validation_data = self.val_dataset,
        validation_steps = len(self.val_dataset),
        callbacks =  self.callbacks if self.callbacks else None,
        initial_epoch = self.initial_epochs
        )

    if fine_tune_epochs:
      self.model_fine_tune_history = model_history.history

      # save fine-tune training results
      self.__save_training_history(
        history=self.model_fine_tune_history, 
        history_save_to=os.path.join(os.getcwd(), 'fine_tune_training_history_' + str(self.handle_number)))
    else:
      self.model_history = model_history.history

      # save training results
      self.__save_training_history(
        history=self.model_history, 
        history_save_to=os.path.join(os.getcwd(), 'training_history_' + str(self.handle_number)))


  def evaluate_model(self):
    """
    Model evaluation.

    Return:
      evaluating results
    """
    model_results = self.model.evaluate(self.test_dataset)
    self.evaluating_results = model_results

    return self.evaluating_results


  def plot_loss_curves(self, save_path='', fine_tune=False, figsize=(12, 6)):
    """
    Returns separate loss curves for training and validation metrics.

    Args:
      save_path (str): Saved path of history for plotting.
      fine_tune_history (bool): plot original history or fine_tune history.
      figsize: Figure size of plot.
    """
    # Check which history -> Save path first
    if os.path.isfile(save_path):
      try:
        history = self.__load_training_history(save_path)
      except Exception as e:
        print(e.__class__.__name__)
    
    # Class variable second
    else: 
      if fine_tune: 
        history = self.model_fine_tune_history
      elif not fine_tune:
        history = self.model_history
      else:
        return "File Not Found."

    # losses and accuracy
    loss = history["loss"]
    val_loss = history["val_loss"]
    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    # epochs
    epochs = range(len(history["loss"]))

    # plot loss and accuracy
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();


  def compare_historys(self, original_save_path='', fine_tune_save_path='', initial_epochs=0):
    """
    Compares two TensorFlow model History objects.

    Args:
      original_save_path (str): Original path of training history.
      fine_tune_save_path (str): Fine-tuning path of training history.
      initial_epochs (int): Number of initial epoch, default is 0.
    """
    # Check which history -> Save path first
    if os.path.isfile(original_save_path) and os.path.isfile(fine_tune_save_path):
      try:
        original_history = self.__load_training_history(original_save_path)
        new_history = self.__load_training_history(fine_tune_save_path)
        initial_epochs = initial_epochs
      except Exception as e:
        print(e.__class__.__name__)
    
    # Class variable second
    else:
      if self.model_history and self.model_fine_tune_history:
        original_history = self.model_history
        new_history = self.model_fine_tune_history
        initial_epochs = self.initial_epochs
      else:
        return "File Not Found."

    # Get original history measurements
    acc = original_history["accuracy"]
    loss = original_history["loss"]

    val_acc = original_history["val_accuracy"]
    val_loss = original_history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history["accuracy"]
    total_loss = loss + new_history["loss"]

    total_val_acc = val_acc + new_history["val_accuracy"]
    total_val_loss = val_loss + new_history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2,1,1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs, initial_epochs], plt.ylim(), label='Model Freeze') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2,1,2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs, initial_epochs], plt.ylim(), label='Model Freeze') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


  def fine_tune_model_layers(self, fine_tune_ratio=None):
    """
    Set specific layers being trainable for further model training.

    Args:
      fine_tune_ratio (float): Proportion of layers estimated to be toggled.

    Return:
      model object
    """

    fine_tune_base_model = self.model.layers[2]
    layer_number = -int(len(fine_tune_base_model.layers) * fine_tune_ratio)

    # Unfreeze all of the layers in the base model
    fine_tune_base_model.trainable = True

    # Refreeze every layer except for the last 5
    for layer in fine_tune_base_model.layers[:layer_number]:
      layer.trainable = False

    print(f"Change last {-layer_number} layers of {fine_tune_base_model.name} successfully, please recompile model again.")

    return self.model


  def save_model(self, model_dir=None, suffix=None):
    """
    Saves a given model in a models directory and appends a suffix (string).

    Args:
      model_dir (str): Main directory for model saving, default path is root path + times stamp..
      suffix: Suffix for path name.

    Return:
      saving path (str)
    """

    # Create a model directory pathname with current time
    model_dir = model_dir if model_dir else os.path.join(os.getcwd(), dt.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_path = model_dir + '_' + suffix + '.h5' # save format model

    print(f'Saving model to: {model_path}...')
    self.model.save(model_path)
    self.model_save_path = model_path

    # Save labels in text file
    with open('labels.txt', 'w') as file:
      for class_name in self.test_dataset.class_names:
        file.write("%s\n"%(class_name))
    print(f'Saving labels to "labels.txt" ...')

    return self.model_save_path


  def load_model(self, model_path=None):
    """
    Loads a saved model from a specified path.

    Return:
      model object
    """        
    if model_path:
      model_path = model_path
    elif self.model_save_path:
      model_path = self.model_save_path
    elif self.checkpoint_path:
      model_path = self.checkpoint_path
    else:
      return "You do not save model before."

    if model_path:
      print(f'Loading saved model from: {model_path}')
      model = tf.keras.models.load_model(model_path) # custom_objects={'KerasLayer': hub.KerasLayer}
      self.model = model
      return self.model

def best_model_search(base_models_dict:dict, 
                      model_name:str, 
                      model_histories:dict, 
                      model_scores:pd.DataFrame, 
                      train_data_batch, 
                      val_data_batch, 
                      test_data_batch, 
                      input_shape, 
                      output_shape, 
                      epochs=5):
  """
  Training the models and recording the histories and test scores.

  Args:
    base_models_dict (dict): A dictionary contains model name as key and model object as values.
    train_data_batch: Training batch dataset
    val_data_batch: Validation batch dataset
    test_data_batch: Testing batch dataset
    model_histories (dict): Each model training history.
    model_scores (DataFrame: Each model test scores.
  """

  base_model, preprocess_input = base_models_dict[model_name]
  print(f"{model_name} training starts...")
  base_model.trainable = False

  # Setup model architecture with trainable top layers
  inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")

  # preprocess input image
  x = preprocess_input(inputs)

  # put the base model
  x = base_model(x)

  # pool the outputs of the base model
  x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

  # same number of outputs as classes
  outputs = tf.keras.layers.Dense(output_shape, activation="softmax", name="output_layer")(x)
  model = tf.keras.Model(inputs, outputs, name=model_name)

  print(model.summary())

  # Compile the model
  model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(), # use Adam with default settings
    metrics=["accuracy"])

  # Fit the model
  model_history = model.fit(
    train_data_batch,
    epochs=epochs,
    steps_per_epoch=len(train_data_batch),
    validation_data=val_data_batch,
    validation_steps=len(val_data_batch),
  )

  # Record the history
  model_histories[model_name] = model_history.history

  # Evaluate the model
  model_results = model.evaluate(test_data_batch)

  # Record the test scores
  model_scores[model_name] = model_results

def load_labels(file='labels.txt'):
  """
  Load class names from file saved before.

  Args:
    file (str): File path saving labels.
  
    Return:
      class_labels (list): Class names of list.
  """

  if os.path.isfile(file):

    # Load labels in list datatype
    class_labels = []
    print(f"Loading class names from {file}")
    with open('labels.txt', 'r') as f_labels:
      for line in f_labels:
        class_name = line[:-1]
        class_labels.append(class_name)
    print('OK')
    return class_labels
  else: 
    return "File of label names not found."

__all__ = ['ModelTrainer', 'best_model_search', 'load_labels']