from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, merge, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf

class fire_modul(tf.Module):
    def __init__(self, input_sizes, sizes, name=None):
        
