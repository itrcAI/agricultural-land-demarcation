# Agricultural Field Boundary Detection with Deep Learning and Hybrid Models

This repository provides a complete pipeline for agricultural field boundary detection using deep learning and hybrid models. 
The code covers all stages of the pipeline, including image cropping, data augmentation, model training, and post-processing. 
The goal is to detect agricultural field boundaries from satellite or aerial imagery, leveraging deep learning and hybrid models to improve prediction accuracy.

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

## Code Overview

### 1. `crop_image.py`
The `crop_image.py` script is used to crop large images into smaller tiles, making them easier to process for model training. It takes raw images as input and outputs smaller, uniformly sized image tiles.

### 2. `augment_data.py`
Once the images are cropped, the `augment_data.py` script applies various augmentation techniques such as rotations, flips, and scaling to the cropped images. This increases the diversity of the dataset, which helps the model generalize better during training.

### 3. `models_script.py`
The `models_script.py` file defines the deep learning and hybrid models used for detecting agricultural field boundaries. This file contains the network architectures.

### 4. `main.py`
The `main.py` script integrates all the steps of the pipeline. It coordinates the image cropping, data augmentation, and model training. You can customize the hyperparameters such as batch size, learning rate, and number of epochs when training the models. Also This file contains loss functions, optimizers, and training routines.

### 5. `prepare_masks_for_post_process_predict.py`
After obtaining predictions from the model, the `prepare_masks_for_post_process_predict.py` script prepares the necessary masks for post-processing. This might include generating binary masks or applying other transformations to format the model outputs for further processing.

### 6. `post_process_predict.py`
The `post_process_predict.py` script performs post-processing on the model’s predictions. This can include tasks such as thresholding, smoothing, or applying morphological transformations to refine and finalize the predicted agricultural field boundaries.

## Code Order
First run `crop_image.py` then `augment_data.py` to augment croped images.
Then run `main.py` to train the model.
After saved model prepared, run `prepare_masks_for_post_process_predict.py` based on saved model name (change inside the code) then run `post_process_predict.py`  
