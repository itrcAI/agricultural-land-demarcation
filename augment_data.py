# import os; os.environ['CUDA_VISIBLE_DEVICES'] = ""
# from google.colab import drive
# drive.mount('/content/drive')
## importing required packages
import numpy as np
from os import listdir, makedirs
# import os
import cv2
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
# image_dir = "/content/drive/My Drive/3d printer/codes/datasets/cropped-datasets/"
# masks_dir = "/content/drive/My Drive/3d printer/codes/datasets/ground-truth/"

image_dir = "./data_copies/All_images/"
masks_dir = "./data_copies/All_masks/"

dest_image = "./All_images"
dest_mask = "./All_masks"

#image_dir = "./org_ir_images/"
#masks_dir = "./org_ir_masks/"

#image_dir = "./out_images/image/"
#masks_dir = "./out_images/mask/"

#dest_image = "./aug_out_images"
#dest_mask = "./aug_out_images"

# choose on of the pairs of data and mask for dataset

img_folders = sorted(listdir(image_dir))
mask_folders = sorted(listdir(masks_dir))

# img_folders = ['abyek_images', 'Khorasan_part1_images', 'Khorasan_part2_images', 'bound_images', 'France_images', 'orthophoto_images']
# mask_folders = ['abyek_masks', 'Khorasan_part1_masks', 'Khorasan_part2_masks', 'bound_masks', 'France_masks', 'orthophoto_masks']

## out
#img_folders = ['bound_images', 'orthophoto_images']
#mask_folders = ['bound_masks', 'orthophoto_masks']

## Iran data
#img_folders = ['Abyek_images', 'Khorasan_images', 'Ghazvin_images', 'Hamedan_images', 'Ardebil_images', 'Golestan_images', 'Markazi_images']
#mask_folders = ['Abyek_masks', 'Khorasan_masks', 'Ghazvin_masks', 'Hamedan_masks', 'Ardebil_masks', 'Golestan_masks', 'Markazi_masks']

# img_folders = ['Abyek_images', 'Ardebil_images', 'Golestan_images', 'Markazi_images', 'Ghazvin_images', 'Hamedan_images']
# mask_folders = ['Abyek_masks', 'Ardebil_masks', 'Golestan_masks', 'Markazi_masks', 'Ghazvin_masks', 'Hamedan_masks']


## Iran + foreign  
# img_folders = ['abyek_images', 'Khorasan_part1_images', 'Khorasan_part2_images', 'bound_images', 'orthophoto_images']
# mask_folders = ['abyek_masks', 'Khorasan_part1_masks', 'Khorasan_part2_masks', 'bound_masks', 'orthophoto_masks']

# img_folders = ['abyek_images', 'Khorasan_part1_images',  'deliniation_images', 'bound_images']
# mask_folders = ['abyek_masks', 'Khorasan_part1_masks', 'deliniation_masks', 'bound_masks']

#img_folders = ['img']
#mask_folders = ['img']

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def aug_data(image, path, format):
    cv2.imwrite(f"{path[:-4]}.{format}", image)
    hf = cv2.flip(image, 0)
    cv2.imwrite(f"{path[:-4]}hf.{format}", hf)
    vf = cv2.flip(image, 1)
    cv2.imwrite(f"{path[:-4]}vf.{format}", vf)
    bothf = cv2.flip(image, -1)
    cv2.imwrite(f"{path[:-4]}bothf.{format}", bothf)
    rotc = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"{path[:-4]}rotc.{format}", rotc)
    rota = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f"{path[:-4]}rota.{format}", rota)
    # zom = zoom(image)
    # cv2.imwrite(f"{path[:-4]}zoom.{format}", zom)
    #bright_img = cv2.convertScaleAbs(image, alpha = 1, beta = 21)
    #cv2.imwrite(f"{path[:-4]}brit.{format}", bright_img)
    #dark_img = cv2.convertScaleAbs(image, alpha = 1, beta = -21)
    #cv2.imwrite(f"{path[:-4]}dark.{format}", dark_img)

def aug_mask(image, path, format):
    cv2.imwrite(f"{path[:-4]}.{format}", image)    
    hf = cv2.flip(image, 0)
    cv2.imwrite(f"{path[:-4]}hf.{format}", hf)
    vf = cv2.flip(image, 1)
    cv2.imwrite(f"{path[:-4]}vf.{format}", vf)
    bothf = cv2.flip(image, -1)
    cv2.imwrite(f"{path[:-4]}bothf.{format}", bothf)
    rotc = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"{path[:-4]}rotc.{format}", rotc)
    rota = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f"{path[:-4]}rota.{format}", rota)
    # zom = zoom(image)
    # cv2.imwrite(f"{path[:-4]}zoom.{format}", zom)
    #bright_img = image
    #cv2.imwrite(f"{path[:-4]}brit.{format}", bright_img)
    #dark_img = image
    #cv2.imwrite(f"{path[:-4]}dark.{format}", dark_img)

# makedirs(path, exist_ok=True)

# images_path, masks_path = [], []
# loop in pair of data and mask to make list of images and masks based on folder image
for fol, mfol in zip(img_folders, mask_folders):
    images_list = listdir(image_dir + fol)
    masks_list = listdir(masks_dir + mfol)

    img_list = [i.split('.jpg')[0] for i in images_list]
    msk_list = [i.split('.png')[0] for i in masks_list]

    commons = list(set(img_list) & set(msk_list))

    # shuffle(commons)

    # commons = commons[:1200]

    print(len(commons))

    ip =  [image_dir + fol + "/" + i + ".jpg"  for i in commons]
    mp = [masks_dir + mfol + "/" + i + ".png"  for i in commons]
    # print(len(ip), len(mp))

    # images_path.extend(ip)
    # masks_path.extend(mp)
    # print(len(images_path), len(masks_path))

    for path in ip:
        # tempim = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        tempim = cv2.imread(path)
        aug_data(tempim, "." + path.split('./data_copies')[-1], "jpg")

    for path in mp:
        # tempms = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        tempms = cv2.imread(path)
        #print(tempms.shape)
        aug_mask(tempms, "." + path.split('./data_copies')[-1], "png")

