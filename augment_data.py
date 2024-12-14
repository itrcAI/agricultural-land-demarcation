# Import necessary libraries
import numpy as np
from os import listdir, makedirs
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Define the directories for images and masks
image_dir = "./data_copies/All_images/"
masks_dir = "./data_copies/All_masks/"
dest_image = "./All_images"
dest_mask = "./All_masks"

# List of subfolders containing the images and masks
img_folders = sorted(listdir(image_dir))  # Get sorted list of image folders
mask_folders = sorted(listdir(masks_dir))  # Get sorted list of mask folders

# Function to apply zoom to an image
def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

# Function to apply various augmentation techniques to an image
def aug_data(image, path, format):
    # Save the original image
    cv2.imwrite(f"{path[:-4]}.{format}", image)
    
    # Flip image horizontally and vertically, and both
    hf = cv2.flip(image, 0)
    cv2.imwrite(f"{path[:-4]}hf.{format}", hf)
    
    vf = cv2.flip(image, 1)
    cv2.imwrite(f"{path[:-4]}vf.{format}", vf)
    
    bothf = cv2.flip(image, -1)
    cv2.imwrite(f"{path[:-4]}bothf.{format}", bothf)
    
    # Rotate image 90 degrees clockwise and counterclockwise
    rotc = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"{path[:-4]}rotc.{format}", rotc)
    
    rota = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f"{path[:-4]}rota.{format}", rota)

    # Zoom and brightness adjustment can be added here, but are commented out for now
    # zom = zoom(image)
    # cv2.imwrite(f"{path[:-4]}zoom.{format}", zom)
    
    # Adjust brightness (commented out)
    # bright_img = cv2.convertScaleAbs(image, alpha=1, beta=21)
    # cv2.imwrite(f"{path[:-4]}brit.{format}", bright_img)

    # Darken the image (commented out)
    # dark_img = cv2.convertScaleAbs(image, alpha=1, beta=-21)
    # cv2.imwrite(f"{path[:-4]}dark.{format}", dark_img)

# Function to apply augmentation to masks (same transformations as for images)
def aug_mask(image, path, format):
    # Save the original mask
    cv2.imwrite(f"{path[:-4]}.{format}", image)
    
    # Flip the mask horizontally, vertically, and both
    hf = cv2.flip(image, 0)
    cv2.imwrite(f"{path[:-4]}hf.{format}", hf)
    
    vf = cv2.flip(image, 1)
    cv2.imwrite(f"{path[:-4]}vf.{format}", vf)
    
    bothf = cv2.flip(image, -1)
    cv2.imwrite(f"{path[:-4]}bothf.{format}", bothf)
    
    # Rotate the mask 90 degrees clockwise and counterclockwise
    rotc = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"{path[:-4]}rotc.{format}", rotc)
    
    rota = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f"{path[:-4]}rota.{format}", rota)

    # Zoom and brightness adjustments for masks can be added, but are commented out
    # zom = zoom(image)
    # cv2.imwrite(f"{path[:-4]}zoom.{format}", zom)
    
    # Adjust brightness (commented out)
    # bright_img = image
    # cv2.imwrite(f"{path[:-4]}brit.{format}", bright_img)

    # Darken the mask (commented out)
    # dark_img = image
    # cv2.imwrite(f"{path[:-4]}dark.{format}", dark_img)

# Loop through each pair of image and mask folders
for fol, mfol in zip(img_folders, mask_folders):
    # Get the list of images and masks in the respective folders
    images_list = listdir(image_dir + fol)
    masks_list = listdir(masks_dir + mfol)

    # Create lists of file names (without extensions) for images and masks
    img_list = [i.split('.jpg')[0] for i in images_list]
    msk_list = [i.split('.png')[0] for i in masks_list]

    # Find common files between images and masks (i.e., files that exist in both)
    commons = list(set(img_list) & set(msk_list))
    print(len(commons))  # Print the number of common files

    # Create paths for images and masks based on the common file names
    ip = [image_dir + fol + "/" + i + ".jpg"  for i in commons]
    mp = [masks_dir + mfol + "/" + i + ".png"  for i in commons]

    # Loop through each image path and apply augmentation
    for path in ip:
        tempim = cv2.imread(path)  # Read the image
        aug_data(tempim, "." + path.split('./data_copies')[-1], "jpg")  # Apply augmentations

    # Loop through each mask path and apply augmentation
    for path in mp:
        tempms = cv2.imread(path)  # Read the mask
        aug_mask(tempms, "." + path.split('./data_copies')[-1], "png")  # Apply augmentations
