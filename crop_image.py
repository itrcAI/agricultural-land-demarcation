## importing required packages
import numpy as np
import os 
import cv2
import shutil

import matplotlib.pyplot as plt

def make_mask(img_path):
    img = cv2.imread(img_path, 0)
    # print(np.unique(img, return_counts=True))
    img[img > 50] = 255
    img[img <= 50] = 0
    print(np.unique(img, return_counts=True))
    # img[(img >= 0) & (img <= 234)] = 0
    
    return 255 - img

# img = cv2.imread(orgpath + org_masks[-1])
# plt.imshow(img)
# plt.show()
# temp = make_mask(orgpath + org_masks[-1])
# plt.imshow(temp, cmap='gray')


# image_coords = []
def sliding_window(image, stepSize, windowSize):    
    images, image_coords = [], []
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            img = image[y:y + windowSize[0], x:x + windowSize[1]]
            if img.shape[0] >= windowSize[0] and img.shape[1] >= windowSize[1]:
                images.append(img)
                image_coords.append([y, y + windowSize[0], x, x + windowSize[1]])
            if x + stepSize > image.shape[1]:
                # print(x)
                img = image[y:y + windowSize[0], -windowSize[1]:]
                if img.shape[0] >= windowSize[0] and img.shape[1] >= windowSize[1]:
                    images.append(img)
                    image_coords.append([y, y + windowSize[0], image.shape[1] - windowSize[1], image.shape[1]])
            if y + stepSize > image.shape[0]:
                # print(y)
                img = image[-windowSize[0]:, x:x + windowSize[1]]
                if img.shape[0] >= windowSize[0] and img.shape[1] >= windowSize[1]:
                    images.append(img)
                    image_coords.append([image.shape[0] - windowSize[0], image.shape[0], x, x + windowSize[1]])
                if x + stepSize > image.shape[1]:
                    img = image[-windowSize[0]:, -windowSize[1]:]
                    images.append(img)
                    image_coords.append([image.shape[0] - windowSize[0], image.shape[0], image.shape[1] - windowSize[1], image.shape[1]])
                    # break
    return images, image_coords

def make_window(img, path, region, name, form):
    images, image_coords = sliding_window(img, 64, (128, 128))
    for i, window in enumerate(images):
        # print(window.shape, i)
        cv2.imwrite(f"{path}{i}{name}{region}{form}", window)
        # features.append(featureVector)

# region = 'Khorasan_part3'
# region_part = "kh-p3"

# region = 'abyek'
# region_part = "abyk"

# region = 'Markazi'
# region_part = "mark"

#regions = ['Abyek', 'Ardebil', 'Golestan', 'Markazi', 'Ghazvin', 'Hamedan']
#regions_part = ['abyk', 'ardb', 'gols', 'mark', 'ghaz', 'hamd']

regions = ['Abyek', 'Khorasan', 'Ghazvin', 'Hamedan', 'Ardebil', 'Golestan', 'Markazi']
regions_part = ['abyk', 'khor', 'ghaz', 'hamd', 'ardb', 'gols', 'mark']

for j, region in enumerate(regions):
    imgpath = f'./All_images/{region}_images/' 
    maskpath = f'./All_masks/{region}_masks/' 

    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    if not os.path.exists(maskpath):
        os.makedirs(maskpath)

    orgpath = f"./original_dataset/{region}/"
    org_list = os.listdir(orgpath)

    org_imgs ,org_masks = [] ,[]
    for path in org_list:
        if path.split('.')[-1] == "jpg":
            org_imgs.append(path)
        if path.split('.')[-1] == "png":
            org_masks.append(path)

    org_imgs.sort()
    org_masks.sort()
    i = 0
    for img, msk in zip(org_imgs, org_masks):
        image = cv2.cvtColor(cv2.imread(orgpath + img), cv2.COLOR_BGR2RGB)
        # mask = 255 - cv2.cvtColor(cv2.imread(orgpath + msk), cv2.COLOR_BGR2RGB)
        mask = make_mask(orgpath + msk)
        make_window(image, imgpath, regions_part[j], i, ".jpg")
        make_window(mask, maskpath, regions_part[j], i, ".png")
        i += 1    

# shutil.copytree('All_images/', 'data_copies/', dirs_exist_ok = True)
# shutil.copytree('All_masks/', 'data_copies/', dirs_exist_ok = True)

if not os.path.exists('data_copies/'):
    os.makedirs('data_copies/')
        
os.system('cp -r All_images/ data_copies/')
os.system('cp -r All_masks/ data_copies/')