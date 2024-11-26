# Import necessary libraries
import numpy as np
import os
import cv2
import shutil
import matplotlib.pyplot as plt

# Function to create a binary mask from an image
def make_mask(img_path):
    # Read the image as grayscale
    img = cv2.imread(img_path, 0)
    
    # Threshold the image: pixels greater than 50 become 255 (white), others become 0 (black)
    img[img > 50] = 255
    img[img <= 50] = 0
    
    # Output the unique values in the image and their counts
    print(np.unique(img, return_counts=True))
    
    # Return the inverted mask (255 becomes 0, and 0 becomes 255)
    return 255 - img

# Function to extract sliding windows from an image
def sliding_window(image, stepSize, windowSize):
    images, image_coords = [], []  # List to store extracted windows and their coordinates
    for y in range(0, image.shape[0], stepSize):  # Loop through the image rows with step size
        for x in range(0, image.shape[1], stepSize):  # Loop through the image columns with step size
            # Extract the window from the image
            img = image[y:y + windowSize[0], x:x + windowSize[1]]
            
            # Check if the window is valid (within image bounds)
            if img.shape[0] >= windowSize[0] and img.shape[1] >= windowSize[1]:
                images.append(img)  # Add window to the list
                image_coords.append([y, y + windowSize[0], x, x + windowSize[1]])  # Save the coordinates
            
            # Handle cases where the window goes out of bounds horizontally
            if x + stepSize > image.shape[1]:
                img = image[y:y + windowSize[0], -windowSize[1]:]  # Take the right edge portion
                if img.shape[0] >= windowSize[0] and img.shape[1] >= windowSize[1]:
                    images.append(img)
                    image_coords.append([y, y + windowSize[0], image.shape[1] - windowSize[1], image.shape[1]])

            # Handle cases where the window goes out of bounds vertically
            if y + stepSize > image.shape[0]:
                img = image[-windowSize[0]:, x:x + windowSize[1]]  # Take the bottom edge portion
                if img.shape[0] >= windowSize[0] and img.shape[1] >= windowSize[1]:
                    images.append(img)
                    image_coords.append([image.shape[0] - windowSize[0], image.shape[0], x, x + windowSize[1]])

                # Handle bottom-right corner case
                if x + stepSize > image.shape[1]:
                    img = image[-windowSize[0]:, -windowSize[1]:]  # Take the bottom-right portion
                    images.append(img)
                    image_coords.append([image.shape[0] - windowSize[0], image.shape[0], image.shape[1] - windowSize[1], image.shape[1]])

    return images, image_coords  # Return the extracted windows and their coordinates

# Function to save windows from an image and its mask
def make_window(img, path, region, name, form):
    # Extract windows from the image
    images, image_coords = sliding_window(img, 64, (128, 128))
    
    # Save each window to the specified path with unique names
    for i, window in enumerate(images):
        cv2.imwrite(f"{path}{i}{name}{region}{form}", window)  # Save window as a file

# List of regions and their corresponding part codes
regions = ['Abyek', 'Khorasan', 'Ghazvin', 'Hamedan', 'Ardebil', 'Golestan', 'Markazi']
regions_part = ['abyk', 'khor', 'ghaz', 'hamd', 'ardb', 'gols', 'mark']

# Loop through each region to process the images and masks
for j, region in enumerate(regions):
    # Define the paths for images and masks
    imgpath = f'./All_images/{region}_images/' 
    maskpath = f'./All_masks/{region}_masks/' 

    # Create directories if they do not exist
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    if not os.path.exists(maskpath):
        os.makedirs(maskpath)

    # Define the original dataset path for the region
    orgpath = f"./original_dataset/{region}/"
    org_list = os.listdir(orgpath)  # List all files in the original dataset

    # Separate images and masks based on file extensions
    org_imgs ,org_masks = [] ,[]
    for path in org_list:
        if path.split('.')[-1] == "jpg":
            org_imgs.append(path)
        if path.split('.')[-1] == "png":
            org_masks.append(path)

    # Sort the images and masks alphabetically
    org_imgs.sort()
    org_masks.sort()
    
    # Loop through each image-mask pair
    i = 0
    for img, msk in zip(org_imgs, org_masks):
        # Read and process the image
        image = cv2.cvtColor(cv2.imread(orgpath + img), cv2.COLOR_BGR2RGB)
        
        # Create the mask using the `make_mask` function
        mask = make_mask(orgpath + msk)
        
        # Extract sliding windows and save them
        make_window(image, imgpath, regions_part[j], i, ".jpg")
        make_window(mask, maskpath, regions_part[j], i, ".png")
        
        i += 1    

# If 'data_copies' folder doesn't exist, create it
if not os.path.exists('data_copies/'):
    os.makedirs('data_copies/')
        
# Copy the images and masks to the 'data_copies' directory
os.system('cp -r All_images/ data_copies/')
os.system('cp -r All_masks/ data_copies/')
