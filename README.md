# Agricultural Field Boundary Detection with Deep Learning and Hybrid Models

This repository provides a complete pipeline for agricultural field boundary detection using deep learning and hybrid models. The pipeline includes all stages of the process, from image cropping, data augmentation, model training, to post-processing. The goal is to detect agricultural field boundaries from satellite or aerial imagery, leveraging deep learning and hybrid models to improve prediction accuracy.

## Requirements

To run the scripts, you'll need the following Python libraries:

- tensorflow==2.12
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scikit-image
- opencv-python

You can install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Pipeline Overview

The pipeline consists of several scripts that together process the input data, train a model, and post-process the model’s predictions. Here's an overview of the pipeline, including the stages and the corresponding scripts.

1. **Image Preprocessing and Augmentation**  
   - **`crop_image.py`**: This script crops large satellite/aerial images into smaller, more manageable tiles (128x128 pixels), making them easier to process for model training.
   - **`augment_data.py`**: The cropped images are augmented by applying transformations such as rotations, flips, and scaling to improve model generalization.

2. **Model Definition and Training**  
   - **`main.py`**: This is the core script that integrates preprocessing, augmentation, and model training. It defines the deep learning model architecture, sets up training parameters (e.g., batch size, learning rate), and executes the training process.
   - The script uses mixed-precision training and applies a series of advanced loss functions and metrics (e.g., Tversky loss, dice coefficient) to improve model performance.

3. **Post-Processing Predictions**  
   - **`prepare_masks_for_post_process_predict.py`**: Once the model is trained, this script prepares the necessary binary masks from the predictions for further post-processing.
   - **`post_process_predict.py`**: This script refines the predictions through post-processing techniques like thresholding and morphological transformations to produce the final field boundary detection results.

## Image Sizes

- **Input Size**: 128x128 pixels (Each image is cropped into 128x128 tiles to make them manageable for model training and inference).
- **Output Size**: 128x128 pixels (The model predicts segmentation masks of the same size as the input images, where each pixel is classified as part of the boundary or not).

## Code Execution Order

1. **Prepare Data**:  
   - First, run `crop_image.py` to crop the large satellite images into smaller tiles of size 128x128 pixels. 
   - Then, run `augment_data.py` to apply data augmentation techniques on the cropped images.

2. **Train the Model**:  
   Run `main.py` to define and train the model. This script will handle data loading, model construction, training, and validation.

3. **Post-Process Model Predictions**:  
   Once the model is trained and saved, run `prepare_masks_for_post_process_predict.py` to prepare the masks based on the saved model.  
   Then, run `post_process_predict.py` to refine the predicted field boundaries.

## Code Overview

### 1. `crop_image.py`
The `crop_image.py` script is used to crop large images into smaller tiles (128x128 pixels), making them easier to process for model training. It takes raw images as input and outputs smaller, uniformly sized image tiles.

### 2. `augment_data.py`
Once the images are cropped, the `augment_data.py` script applies various augmentation techniques such as rotations, flips, and scaling to the cropped images. This increases the diversity of the dataset, which helps the model generalize better during training.

### 3. `models_script.py`
The `models_script.py` file defines the deep learning and hybrid models used for detecting agricultural field boundaries. This file contains the network architectures.

### 4. `main.py`
The `main.py` script integrates all the steps of the pipeline. It coordinates the image cropping, data augmentation, and model training. You can customize the hyperparameters such as batch size, learning rate, and number of epochs when training the models. Also, this file contains loss functions, optimizers, and training routines.

### 5. `prepare_masks_for_post_process_predict.py`
After obtaining predictions from the model, the `prepare_masks_for_post_process_predict.py` script prepares the necessary masks for post-processing. This might include generating binary masks or applying other transformations to format the model outputs for further processing.

### 6. `post_process_predict.py`
The `post_process_predict.py` script performs post-processing on the model’s predictions. This can include tasks such as thresholding, smoothing, or applying morphological transformations to refine and finalize the predicted agricultural field boundaries.

## Pipeline Steps in `main.py`

The following code block provides a summary of the key pipeline steps in `main.py`:

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
from os import listdir
from models_script import build_unet_model, build_DenseNets, ResUNet, DenseUnet, unet_densenet_resunet

# Enable mixed precision training for efficiency
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Set random seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Dataset configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 1
EPOCHS = 50

# Define model and dataset paths
image_dir = "./All_images/"
masks_dir = "./All_masks/"
img_folders = sorted(listdir(image_dir))
mask_folders = sorted(listdir(masks_dir))

# Load the dataset
train_images_path, train_masks_path, val_images_path, val_masks_path, test_images_path, test_masks_path = make_dataset(image_dir, masks_dir)

# Create TensorFlow datasets for training, validation, and testing
train_dataset = create_tf_dataset(train_images_path, train_masks_path)
val_dataset = create_tf_dataset(val_images_path, val_masks_path)
test_dataset = create_tf_dataset(test_images_path, test_masks_path)

# Build and compile the model
model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Define training callbacks
callbacks = [
    ModelCheckpoint(f'./models/model.h5', verbose=1, save_best_only=True)
]

# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Evaluate the model on test data
model.evaluate(test_dataset)
```

## How the Pipeline Works

1. **Data Preprocessing**:
   - Images are loaded, resized, and normalized.
   - The dataset is split into training, validation, and test sets.
   - Augmentation is applied to increase dataset diversity.

2. **Model Training**:
   - The model is trained with a custom loss function (e.g., Tversky loss) and metrics (e.g., F1-score).
   - Mixed precision training is enabled to speed up the process on supported GPUs.

3. **Post-Processing**:
   - Predictions are refined using thresholding and other post-processing techniques to improve accuracy.

By running the pipeline in sequence, the model can be trained on the processed dataset and make predictions that are post-processed to yield accurate agricultural field boundary detection results.

