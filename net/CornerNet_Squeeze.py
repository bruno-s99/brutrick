from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, merge, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf

def forward_convolution(x,something_that_I_dont_understand,kernel_size,inp_dim,stride = 1,bias = True):
    
    '''questions:
        padding, in pytorch amount of implicit zero-paddings on both sides, in tf _valid_ or _same_??
        
        input and output dimension??
        should i use tf.nn layers??
        '''
    conv=Conv2D(something_that_I_dont_understand,kernel_size=kernel_size,
                strides = (stride,stride), use_bias = not bias )(x)
    #conv = tf.nn.convolution()
    batch=BatchNormalization(axis=1)(conv)
    relu = tf.nn.relu(batch)
    return relu

def create_fire_module(x, nb_squeeze_filter, name, use_bypass=False):
    """
    Creates a fire module
    
    Arguments:
        x                 : input
        nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
        use_bypass        : if True then a bypass will be added
        name              : name of module e.g. fire123
    
    Returns:
        x                 : returns a fire module
    
    Not quite sure if BatchNormalization works with axis=1 works like nn.BatchNorm2d(out_dim // 2)
    Now a function maybe it is easier to implement it as a class like heilaw did it in pytorch
    """
    
    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    normalized = BatchNormalization(axis=1)(squeeze)
    expand_1x1 = Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(normalized)
    expand_3x3 = Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(normalized)
    
    axis = get_axis()
    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    output = BatchNormalization(axis=1)(x_ret)

    if use_bypass:
        output = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return output


def get_axis():
    axis = -1 if K.image_data_format() == 'channels_last' else 1
    return axis

def make_pool_layer(dim):
    return tf.keras.Sequential()

def make_unpool_layer(dim):
    #Possible error to look at if it doesnt work: 
    #cant specify kernel_size?
    # do I  need to specify filters
    return tf.nn.conv2d_transpose(dim, output_shape=dim,n strides=2,padding='SAME')
