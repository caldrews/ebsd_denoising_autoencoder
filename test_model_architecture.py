import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
import cv2

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input, Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from progressbar import ProgressBar
pbar = ProgressBar()

input = layers.Input(shape=(236, 236, 1))

# Encoder Layers
x = layers.Conv2D(64, (1, 1), activation="relu", padding="same")(input)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(128, (5, 5), activation="relu", padding="same")(x)
#x = layers.Conv2D(64, (5, 5), activation="relu", padding="same")(x)
#x = layers.Conv2D(128, (7, 7), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder Layers
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (1, 1), strides=2, activation="relu", padding="same")(x)
#x = layers.Conv2D(32, (3, 3), activation="sigmoid", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")
autoencoder.summary()