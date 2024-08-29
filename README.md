# 🐈 End-to-End-Multi-Class-Cat-Breed-Classification-Reaches-93%-Accuracy

## Introduction
In the purpose of this project, I would like to train a model for various cat breed classification (13+) and try to predict my custom images.

<p align='center'>
  <img src="https://github.com/Leohoji/End-to-end-Multil-Class-Cat-Breed-Classification-Reaches-90-Accuracy/assets/101786128/7db1f918-19c2-4c94-bfd7-f7091400ee25" width=1000 height=800>
</p>

✨ If you could not render the notebook, copy `https://github.com/Leohoji/End-to-end-Multil-Class-Cat-Breed-Classification-Reaches-93-Percent-Accuracy/blob/main/End-to-end%20Multil-Class%20Cat-Breed%20Classification%20Reaches%2093%25%20Accuracy.ipynb` on [nbviewer](https://nbviewer.org/) to render it !

### 1. Problem expected to be solved
Identifying the breed of a cat given an image of a stranger cat.
When I sitting at anywhere and I take a picture of a lovely cat, I want to know what breed of cat it is. Furthermore, I also want the classifier to help me distinguish correct breed of a cat while crawling cat images from internet.

### 2. Source of Dataset

[**Source**]

The source of dataset in this project is from [Pop-Cats](https://www.kaggle.com/datasets/knucharat/pop-cats) of Kaggle Datasets. Moreover, the dataset consists of 20 popular cat breeds (Each class contains 500 images)。

[**Features**]

**Inputs**: We're dealing with images (unstructured data) so it's probably best we use deep learning/transfer learning.

**Outputs**: There are 20 breeds of cats (this means there are 20 different classes).

| Breeds |     Images        |   Breeds   |     Images     | Breeds |     Images        |   Breeds   |     Images     |
| :----: | :---------------: | :--------: | :------------: | :----: | :---------------: | :--------: | :------------: |
|  **Abyssinian** | 501 | **American Curl** | 501 | **American Shorthair** | 500 |  **Bengal** | 501 |
| **Birman** | 502 | **Bombay** | 502 |  **British Shorthair** | 500 | **Egyptian Mau** | 501 |
|  **Exotic Shorthair** | 501 |  **Himalayan** | 501 | **Maine Coon** | 501 | **Manx** | 501 |
|  **Munchkin** | 501 | **Norwegian Forest** | 500 | **Persian** | 501 |  **Regdoll**  | 501 |
| **Russian Blue** | 501 | **Scottish Fold** | 501 | **Siamese** | 501 | **Sphynx** | 501 |
| Total number | 10019 |

There are around 10,000+ images in the dataset (these images have no labels), but images are not divided into training set and testing set, we have to separate it by myself.

The data structure of dataset:

```
Main-directory
  |
  |___Class_a_subdir(500 images)
  |________class_a_image_1
  |________class_a_image_2
  |
  |___Class_b_subdir(500 images)
  |________class_b_image_1
  |________class_b_image_2
  |
  |... (other 18 breeds)
```

### 3. Evaluation for classification

The evaluation of the classifier is at least 90% accuracy.


### 🎁 How do I complete the project?

1. Training model in 20% of cat-breeds images dataset for transfer learning model selection.

2. Training model in all cat-breeds images dataset to select the best of for further modification.

3. Training a basic model and save the model with fine-tuning the model for further increasing accuracy.

4. Analyzing training results and modifying model until perform the 90% accuracy.

5. **Apply model**: 1️⃣ Trying to **predict my own images**; 2️⃣ **Filter crawled images** from google images search by selenium modules.

## Results

Finally choose `EfficientNetV2B0` belongs to the series of `EfficientNet` models as the final model to apply my own custom and crawled images. However, because of the similar characteristics causing difficult to be classified, deleting 6 breeds of cats including `American Curl`, `Birman`, `Maine Coon`, `Manx`, `Munchkin`, and `Bengal`, other breeds (14 breeds) are provided to be as output of final model and reaches 93% accuracy.

## How to use
If you would like to run the codes in this project, follow the steps below:

1. Open git cmd

Create a folder and open [git](https://git-scm.com/) cmd in it.

```
git clone https://github.com/Leohoji/End-to-end-Multil-Class-Cat-Breed-Classification-Reaches-93-Percent-Accuracy.git

Close the git cmd and go into the cloned folder: **End-to-end-Multil-Class-Cat-Breed-Classification-Reaches-93-Percent-Accuracy**

2. Open conda cmd

Create an [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) environment by command.

```
conda create --name myenv python=3.10.3
```

<ins>**Environments**</ins>
- conda: 23.1.0
- Python: 3.10.3

<ins>**Installation**</ins>

Run the code to install requiremental packages:

```pytphon
!pip install -r requirements.txt
```

<ins>**Requirement.txt**</ins>

Following modules are essential modules in `requirement.txt`
- TensorFlow: 2.8.0
- keras==2.8.0
- matplotlib==3.8.0
- numpy==1.26.0
- pandas==2.1.1
- scikit-learn==1.3.2
- seaborn==0.13.0
- selenium==4.15.1

3. Install `jupyter notebook` or `jupyter lab`

```
pip install jupyter
```
or

```
pip install jupyterlab
```

4. Download the images `zip` file from kaggle dataset:  [Pop-Cats](https://www.kaggle.com/datasets/knucharat/pop-cats) and put it into the project folder.
5. Open notebook and run codes
If you would like to run the selenium modules, you have to download the `webdriver`. Hence, check the version of browser by [**ChromedDriver versions**](https://chromedriver.chromium.org/downloads).

✨ Load pre-trained model to predict

```
import os
import tensorflow as tf
from my_packages.data_processing.data_processor import load_and_prep_image
from my_packages.model_training.model_trainer import load_labels

# Load model
model_dir = os.path.join(root_dir, '_cat_breed_fine_tune_classifier.h5')
final_model_loaded = tf.keras.models.load_model(model_dir)

# Load labels
label_names = load_labels()

# Preprocess image
custom_image = 'your\image\path'
custom_image = load_and_prep_image(custom_image, scale=False)
custom_image = tf.expand_dims(custom_image, axis=0)

# Model prediction
pred_prob = final_model_loaded.predict(custom_image)
pred_label = label_names[pred_prob.argmax()]

# Plot results ...
```

---
## Project Author
Lo Ho

*update for 2024/02/26*
