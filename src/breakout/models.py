#https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb

import numpy as np
import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt

#input shape: 4D tensor with shape: (batch, rows, cols, channels)

models = {}
#large kernal size should improve performance by lowering image size. 
#8 pixels is half the size of the bar, should recognise that easily
#stride of 4 should still allow the ball (size 4x2) to be recognized in the darkness
#and reduce image size
models["deepmind_paper"] = keras.Sequential([
    Conv2D(16, kernel_size=(8,8), strides=4, activation='relu', input_shape=(80,72,4)), 
    Conv2D(32, kernel_size=(4,4), strides=2, activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(4, activation="linear")
])


def crop(state: np.ndarray) -> np.ndarray:
    state = state[32:192, 8:152, :]
    return state

def to_grayscale(state: np.ndarray) -> np.ndarray:
    r = state[:,:,0]
    g = state[:,:,1]
    b = state[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def cropped_grayscale(state: np.ndarray) -> np.ndarray:
    cropped_state = crop(state)
    gray_state = to_grayscale(cropped_state)
    #plt.imshow(gray_state, cmap=plt.cm.gray)
    #plt.show()
    return gray_state

#cant use, scipy.misc.imresize deprecated may not use scikit-image
def cropped_scaled_grayscale(state: np.ndarray) -> np.ndarray:
    cropped_state = crop(state)
    gray_state = to_grayscale(cropped_state)
    resized = gray_state[::2, ::2]
    #print(f"size: {resized.shape}")
    #plt.imshow(resized, cmap=plt.cm.gray)
    #plt.show()
    return resized