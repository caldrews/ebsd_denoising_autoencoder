import numpy as np
import pandas as pd
import matplotlib
import pathlib
import os
import cv2

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input, Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from progressbar import ProgressBar
pbar = ProgressBar()
##If you've configured CUDA on your system change this.
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

## Paths for input noisy data, the folder containing the Keras model/weights, and the output directory
noise_data_path = pathlib.Path('D:\Kikuchi Pattern ML - Code and Base Imgs\Input_test')
#save_img_folder = pathlib.Path('G:\Kikuchi Pattern ML\Dump_Real_Imgs_Cleaned_higherCLAHE(cliplim75_tilegrid10x10)_S11R10_p8_4x_800x')
model_out_path = pathlib.Path('D:\Kikuchi Pattern ML - Code and Base Imgs\Model_Out_600k_p0022loss')
img_out_path = pathlib.Path('D:\Kikuchi Pattern ML - Code and Base Imgs\Output_test')
trained_noise = []
##Contrast adaptive enhancement object, depending on the quality and composition of your patterns,
##this might be useful to normalize contrast and highten some features prior to or following denoising.
##the clip limits and everything need to be dialed in, and will depend on your patterns. It can make
##things worse, but depending on the dataset, also make things better.
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))

## Load the patterns image-by-image, reshapes them to the model param, then does forward prediction to denoise them, returns one at a time to 'load_dataset'
def decode_image(img):
    
    img = cv2.imread(img)
    img = cv2.resize(img, (236,236))
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ##Uncomment this to apply CLAHE pre-ML process.
    #img = clahe.apply(img)
    img = img.astype('float32') / 255
    img = img.reshape(1,236,236,1)
    img = np.asarray(img)
    
    cleaned_img = model.predict(img)
    
    cleaned_img = np.asarray(cleaned_img)

    return cleaned_img

#Calls decode_image to turn images to np arrays, basically parses input images then outputs them.
def load_dataset(noise_data):
    #decoded_img_stack = []
    for i in pbar(sorted(os.listdir(noise_data))):
        f_name = i
        noise_path = os.path.join(noise_data_path,f_name)
        decoded_img = (decode_image(noise_path))
        decoded_img = decoded_img[0,:,:,0]
        #240x240 is our 2x2 binning setting, change to match your desired output resolution.
        decoded_img = cv2.resize(decoded_img, (240,240))
        decoded_img = decoded_img * 255
        decoded_img = np.array(decoded_img, dtype = np.uint16)
        ##Uncomment this to apply CLAHE post-ML process.
        ##I think it works better pre-ML process, but YMMV.
        #decoded_img = clahe.apply(decoded_img)

        filena=str(i)
        full_path = os.path.join(img_out_path,filena)
        cv2.imwrite(full_path, decoded_img)


##Display original and denoised images
##Generally useful to compare datasets quickly, but isn't in use here.
def display_images(array1, array2):
    n = 5
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(5, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(236, 236))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(236, 236))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



model = tf.keras.models.load_model(model_out_path)
model.summary()
cleaned_img = load_dataset(noise_data_path)