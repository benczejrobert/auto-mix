import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt  # FIXME
import pywt  # FIXME
from scipy.io import wavfile
from inspect import currentframe, getframeinfo, stack

import os
import shutil

import time
import re
from shutil import rmtree, copy
import tensorflow as tf
# python interp uses and runs C:/Users/FlRE_DEATH/anaconda3/envs/py310tg210gpu/lib/site-packages/keras/api/_v2/keras/
# but inspector uses C:\Users\FlRE_DEATH\anaconda3\envs\py310tg210gpu\Lib\site-packages\tensorflow\python
# from tensorflow.keras import Model, Sequential
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization, Dropout, Conv2D, Conv1D, Flatten, \
#     MaxPooling2D, Softmax, AveragePooling2D, LeakyReLU, MaxPool2D, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import SGD
from keras.api._v2.keras import Model, Sequential
from keras.api._v2.keras.callbacks import ModelCheckpoint, TensorBoard
from keras.api._v2.keras.layers import Dense, Input, Activation, BatchNormalization, Dropout, Conv2D, Conv1D, Flatten, \
    MaxPooling2D, Softmax, AveragePooling2D, LeakyReLU, MaxPool2D, GlobalAveragePooling2D
from keras.api._v2.keras.optimizers import SGD

import datetime
# from tensorflow.keras.utils import to_categorical
from keras.api._v2.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
import librosa
import scipy.stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import shuffle
import sklearn.metrics
from yodel import filter
import soundfile as sf
import taglib
from typing import Any
import pyloudnorm as pyln
from scipy.io import wavfile
from itertools import product
import json
