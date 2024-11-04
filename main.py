import tensorflow as tf

# Get the list of GPUs available
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Set the GPU memory limit to 30 GB (30 * 1024 MB = 30720 MB)
        for gpu in gpus:
            memory_limit=41000
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        print(f"GPU memory limit set to {memory_limit} GB.")
    except RuntimeError as e:
        print(f"Error: {e}")
else:
    print("No GPUs found.")

from tensorflow.keras import mixed_precision

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# from google.colab import drive
# drive.mount('/content/drive')

## importing required packages
import numpy as np
from os import listdir
import keras
from  keras import layers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from sklearn.utils import shuffle
from models_script import build_unet_model, build_DenseNets, ResUNet, DenseUnet, unet_densenet_resunet, unetplus_resunet_attentionunet, bpat_unet, unetpp_resunet, unetpp_res_transformer, transformer_segmentation_model

import matplotlib.pyplot as plt
import random
from tensorflow.keras.optimizers.legacy import Adam
import keras.backend as K

# Set the seed for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

## CONFIGURATION 
EPOCHS = 50
BUFFER_SIZE = 5000
BATCH_SIZE = 8

IMG_HEIGHT = 128
IMG_WIDTH = 128

#tf.random.set_seed(seed)
#np.random.seed(seed)
run_model = unetpp_res_transformer
model_name = 'unetpp-res-transformer-tverfbeta-64-2-ir-ed'

# image_dir = "/content/drive/My Drive/3d printer/codes/datasets/cropped-datasets/"
# masks_dir = "/content/drive/My Drive/3d printer/codes/datasets/ground-truth/"

# ir directory
image_dir = "./All_images/"
masks_dir = "./All_masks/"

org_image_dir = "./data_copies/All_images/"
org_masks_dir = "./data_copies/All_masks/"

# out directory
#image_dir = "./aug_out_images/image/"
#masks_dir = "./aug_out_images/mask/"

#org_image_dir = "./out_images/image/"
#org_masks_dir = "./out_images/mask/"

# choose on of the pairs of data and mask for dataset

## out
#img_folders = ['bound_images']
#mask_folders = ['bound_masks']

## Iran data
img_folders = sorted(listdir(image_dir))
mask_folders = sorted(listdir(masks_dir))

#img_folders = ['Ardebil_images', 'Ghazvin_images']
#mask_folders = ['Ardebil_masks', 'Ghazvin_masks']

# img_folders = ['Ardebil_images', 'Golestan_images', 'Markazi_images', 'Ghazvin_images']
# mask_folders = ['Ardebil_masks', 'Golestan_masks', 'Markazi_masks', 'Ghazvin_masks']

#img_folders = ['Khorasan_images']
#mask_folders = ['Khorasan_masks']



def make_dataset(image_dir, masks_dir, val_split=0.15, test_split=0.15, seed=42):
    train_images_path, train_masks_path = [], []
    val_images_path, val_masks_path = [], []
    test_images_path, test_masks_path = [], []
    
    # Loop through each pair of image and mask folders
    for fol, mfol in zip(img_folders, mask_folders):
        images_list = listdir(image_dir + fol)
        masks_list = listdir(masks_dir + mfol)

        img_list = [i.split('.jpg')[0] for i in images_list]
        msk_list = [i.split('.png')[0] for i in masks_list]

        commons = list(set(img_list) & set(msk_list))
        # Shuffle the list with a static random seed
        random.Random(seed).shuffle(commons)

        # Calculate split sizes
        val_size = int(len(commons) * val_split)
        test_size = int(len(commons) * test_split)
        train_size = len(commons) - val_size - test_size
        
        # Perform splits
        val_commons = commons[:val_size]
        test_commons = commons[val_size:val_size + test_size]
        train_commons = commons[val_size + test_size:]

        # Create paths for training data
        ip_train = [image_dir + fol + "/" + i + ".jpg" for i in train_commons]
        mp_train = [masks_dir + mfol + "/" + i + ".png" for i in train_commons]

        # Create paths for validation data
        ip_val = [image_dir + fol + "/" + i + ".jpg" for i in val_commons]
        mp_val = [masks_dir + mfol + "/" + i + ".png" for i in val_commons]

        # Create paths for test data
        ip_test = [image_dir + fol + "/" + i + ".jpg" for i in test_commons]
        mp_test = [masks_dir + mfol + "/" + i + ".png" for i in test_commons]

        # Extend lists with current folder's data
        train_images_path.extend(ip_train)
        train_masks_path.extend(mp_train)
        val_images_path.extend(ip_val)
        val_masks_path.extend(mp_val)
        test_images_path.extend(ip_test)
        test_masks_path.extend(mp_test)
    
    return train_images_path, train_masks_path, val_images_path, val_masks_path, test_images_path, test_masks_path


# Call the function with your directories
train_images_path, train_masks_path, val_images_path, val_masks_path, test_images_path, test_masks_path = make_dataset(image_dir, masks_dir)
# original dataset path
org_train_images_path, org_train_masks_path, org_val_images_path, org_val_masks_path, org_test_images_path, org_test_masks_path = make_dataset(org_image_dir, org_masks_dir)

print(f"Number of training samples: {len(train_images_path)}")
print(f"Number of validation samples: {len(val_images_path)}")
print(f"Number of test samples: {len(test_images_path)}")

# Create TensorFlow datasets
def load_images(image_path, mask_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    # Load and preprocess the images and masks
    img_data = tf.io.read_file(image_path)
    img = tf.io.decode_png(img_data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    segm_data = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(segm_data)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = mask / 255

    img = tf.image.resize(img, img_size, method='nearest')
    mask = tf.image.resize(mask, img_size, method='nearest')

    return img, mask

# Training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images_path, train_masks_path))
train_dataset = train_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_images_path, val_masks_path))
val_dataset = val_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_images_path, test_masks_path))
test_dataset = test_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Training dataset
org_train_dataset = tf.data.Dataset.from_tensor_slices((org_train_images_path, org_train_masks_path))
org_train_dataset = org_train_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
org_train_dataset = org_train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Validation dataset
org_val_dataset = tf.data.Dataset.from_tensor_slices((org_val_images_path, org_val_masks_path))
org_val_dataset = org_val_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
org_val_dataset = org_val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Test dataset
org_test_dataset = tf.data.Dataset.from_tensor_slices((org_test_images_path, org_test_masks_path))
org_test_dataset = org_test_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
org_test_dataset = org_test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

## gaussian filter 
def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img):
    blur = _gaussian_kernel(3, 2, 3, img.dtype)
    # print(img.shape)
    img = tf.nn.depthwise_conv2d(img, blur, [1,1,1,1], 'SAME')
    # print(img.shape)
    return img

def compute_sample_weights(image, mask):
	''' Compute sample weights for the image given class. '''
	# Compute relative weight of class
	class_weights = tf.constant([1., 8.0])
	class_weights = class_weights/tf.reduce_sum(class_weights)

  	# Compute same-shaped Tensor as mask with sample weights per
  	# mask element. 
	sample_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))

	return image, mask, sample_weights

train_dataset.cardinality().numpy()

# plot images
def display(display_list, num_plot, flag=False):
  plt.figure(figsize=(12,4))
  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  if flag:
    plt.savefig(f"./Result_plots/img_{model_name}_{num_plot}.png")
  plt.close()
  
# N = 1
# for image,mask in dataset.take(N):
#   sample_image, sample_mask = image[0],mask[0]
#   display([sample_image,sample_mask], False)
  

# build the model
model = run_model(input_shape=(IMG_HEIGHT,IMG_WIDTH,3))

model.summary()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# calculate iou metric
iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def weighted_binary_crossentropy(y_true, y_pred):
	zero_weight=1
	one_weight=3
	b_ce = K.binary_crossentropy(y_true, y_pred)
	weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
	weighted_b_ce = weight_vector * b_ce
	return K.mean(weighted_b_ce)

def dice_binary_crossentropy(y_true, y_pred):
    zero_weight=1
    one_weight=2
    b_ce = K.binary_crossentropy(y_true, y_pred)
    dicloss = 1 - dice_coeff(y_true, y_pred)
    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce * dicloss
    return K.mean(weighted_b_ce)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    cross_entropy = tf.keras.backend.binary_crossentropy(y_true_f, y_pred_f)
    weight = alpha * tf.keras.backend.pow((1 - y_pred_f), gamma)
    loss = weight * cross_entropy
    return tf.keras.backend.mean(loss)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_f * y_pred_f)
    false_neg = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))
    false_pos = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

def tversky_index(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    false_neg = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    false_pos = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return tversky_index

def fbeta_score(y_true, y_pred, beta=1, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    precision = true_pos / (tf.reduce_sum(y_pred_f) + smooth)
    recall = true_pos / (tf.reduce_sum(y_true_f) + smooth)
    fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + smooth)
    return fbeta

def tversky_fbeta_loss(y_true, y_pred, alpha=0.6, beta=0.4, gamma=2, smooth=1e-6):
    tversky = tversky_index(y_true, y_pred, alpha, beta, smooth)
    fbeta = fbeta_score(y_true, y_pred, beta=gamma, smooth=smooth)
    return 1 - (0.5 * tversky + 0.5 * fbeta)

# compile the model 
model.compile(
     optimizer=Adam(learning_rate=0.001),
        #loss="binary_crossentropy",
        #loss=focal_loss,
        #loss=tversky_loss,
        loss=tversky_fbeta_loss,
        #loss=weighted_binary_crossentropy,
        #loss=dice_binary_crossentropy,
        metrics=["accuracy", "Precision", "Recall", f1_score],
        # sample_weight_mode="temporal",
  )

# make checkpoint to save model in each epoch
callbacks = [
    # ReduceLROnPlateau(patience=3, verbose=1),
    # EarlyStopping(patience=12, verbose=1),
    ModelCheckpoint(f'./models/{model_name}-model.h5', verbose=0, save_freq="epoch", save_best_only=True, ave_weights_only=False)
    ]

# fiting the model
history = model.fit(
      train_dataset,
      epochs=EPOCHS,
      validation_data = val_dataset,
      # validation_split = 0.1,
      # validation_steps = 8,
      # batch_size=BATCH_SIZE,
      callbacks = callbacks,
      # class_weight=class_weights,
      # sample_weight = [1.0, 2.0], 
    )

#print("Train evaluate: ", model.evaluate(train_dataset))
print("Test evaluate: ", model.evaluate(org_test_dataset))

# plot the accuracy and loss plots for train and test data
plt.plot(history.history["accuracy"], label='train acc')
plt.plot(history.history["val_accuracy"], label='val acc')
# plt.ylim([0.93, 1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"./Result_plots/{model_name}_acc.png")
# plt.show()

plt.plot(history.history["loss"], label='train loss')
plt.plot(history.history["val_loss"], label='val loss')
# plt.ylim([0.93, 1])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(f"./Result_plots/{model_name}_loss.png")
# plt.show()

# plot accuracy, precision and IOU  
fig , ax = plt.subplots(1, 3, figsize=(20, 4))

# ax[0].plot(history.history["accuracy"], label='train acc')
# ax[0].plot(history.history["val_accuracy"], label='val acc')
# ax[0].set_title('Accuracy')
# ax[0].legend()

ax[0].plot(history.history["precision"], label='train pr')
ax[0].plot(history.history["val_precision"], label='val pr')
ax[0].set_title('Precision')
ax[0].legend()

ax[1].plot(history.history["recall"], label='train re')
ax[1].plot(history.history["val_recall"], label='val re')
ax[1].set_title('Recall')
ax[1].legend()

ax[2].plot(history.history["f1_score"], label='train f1')
ax[2].plot(history.history["val_f1_score"], label='val f1')
ax[2].set_title('F1_score')
ax[2].legend()

plt.savefig(f"./Result_plots/{model_name}_final.png")
# plt.show()

# create mask from predicted mask 
def create_mask(pred_mask):
    # pred_mask = tf.argmax(pred_mask, axis=-1)
    pred = pred_mask[0]
    # threshold of 0.5 to differ labels from predicted mask
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    # pred_mask = pred_mask[..., tf.newaxis]
    return pred

def show_predictions(dataset=None, num=1):
    """
    Displays the first image of each of the num batches
    """
    if dataset:
        counter = 0
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image, verbose=0)
            # print(pred_mask[0][pred_mask[0]>0.5])
            display([image[0], mask[0], create_mask(pred_mask)], counter, True)
            counter += 1
    else:
        display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

## displaying our result
show_predictions(val_dataset,num=10)

print(f"Number of training samples: {len(train_images_path)}")
print(f"Number of validation samples: {len(val_images_path)}")
print(f"Number of test samples: {len(test_images_path)}")
print(f"Number of all samples: {len(train_images_path) + len(val_images_path) + len(test_images_path)}")

print(f"Number of org_ training samples: {len(org_train_images_path)}")
print(f"Number of org_ validation samples: {len(org_val_images_path)}")
print(f"Number of org_ test samples: {len(org_test_images_path)}")
print(f"Number of all samples: {len(org_train_images_path) + len(org_val_images_path) + len(org_test_images_path)}")

# Function to evaluate dataset based on folder and calculate metrics
def evaluate_folder(image_dir, masks_dir, folder_name, mask_folder_name, model):
    # Create paths for the folder
    images_list = listdir(image_dir + folder_name)
    masks_list = listdir(masks_dir + mask_folder_name)

    img_list = [i.split('.jpg')[0] for i in images_list]
    msk_list = [i.split('.png')[0] for i in masks_list]

    commons = list(set(img_list) & set(msk_list))

    # Create paths for test data
    test_images_path = [image_dir + folder_name + "/" + i + ".jpg" for i in commons]
    test_masks_path = [masks_dir + mask_folder_name + "/" + i + ".png" for i in commons]

    # Create TensorFlow dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images_path, test_masks_path))
    test_dataset = test_dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Evaluate the model on the test dataset
    evaluation = model.evaluate(test_dataset)
    
    metrics = {
        # "accuracy": evaluation[1],
        "precision": evaluation[2],
        "recall": evaluation[3],
        "f1_score": evaluation[4]
    }

    return metrics

# Loop through each folder and evaluate the model
folder_metrics = {}
for fol, mfol in zip(img_folders, mask_folders):
    print(f"Evaluating folder: {fol}")
    metrics = evaluate_folder(org_image_dir, org_masks_dir, fol, mfol, model)
    folder_metrics[fol] = metrics
    print(f"Metrics for {fol}: {metrics}")
    
'''
# Display the metrics for each folder
for folder, metrics in folder_metrics.items():
    print(f"Folder: {folder}")
    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("----------------------------")
'''