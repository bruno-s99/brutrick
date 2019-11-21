from keras.models_squeeze import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, merge, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf
'''
idee: CornerNet_Squeeze.py ist wie network funktion corner_net_squeeze ist der Durchlauf des Netzwerks,
      create_fire_module gehört in model_squeeze.py. model_squeeze.py ist wie model mit den Änderungen in den verschiedenen Blöcken.
      Ich hab create_fire_module trotzdem hier drin gelassen, weil ich dir nicht so sehr reinpfuschen wollte.
'''

class Squeeze_NetWork():
    def __init__(slef, pullweight=0.1, push_weight=0.1, offset_weight=1):
        self.n_deep  = 5
        self.n_dims  = [256, 256, 384, 384, 384, 512]
        self.n_res   = [2, 2, 2, 2, 2, 4]
        self.out_dim = 80
        self.model=Model()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.offset_weight = offset_weight
        self.focal_loss  = focal_loss
        self.tag_loss     = tag_loss
        self.offset_loss   = offset_loss



    
    def corner_net_squeeze(self,img,gt_tag_tl=None,gt_tag_br=None,is_training=True,scope='CornerNet'):
        with tf.compat.v1.variable_scope(scope):    
            outs=[]
            test_outs=[]
            
            #hier wurde ein residual block benutzt, ab jetzt fire module 
            start_layer=self.model.start_conv(img,is_training=is_training,k=1)#[b,128,128,256]
            
            with tf.compat.v1.variable_scope('das versteh ich noch nicht'):
                hourglass_1=self.model.hourglass(start_layer,self.n_deep,self.n_res,self.n_dims,is_training=is_training)#[b,128,128,256]
                hinge_is=self.model.hinge(hourglass_1,256,256,is_training=is_training)
                #changed k to 1
                top_left_is,bottom_right_is=self.model.corner_pooling(hinge_is,256,256,is_training=is_training,k=1)
                
#TODO:
#könnte man löschen steht jetzt in model_squeeze drin
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
