import tensorflow as tf
from .module.corner_pooling import TopPool,BottomPool, LeftPool, RightPool
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate
# from tensorflow.nn import depthwise_conv2d as Depth2D




class Model():
    def conv_bn_re(self,inputs,input_dim,out_dim,strides=(1,1),use_relu=True,use_bn=True,k=3,training=True,scope='conv_bn_re'):
        with tf.compat.v1.variable_scope(scope):
            #x=tf.contrib.layers.conv2d(inputs,out_dim,k,stride=strides,activation_fn=None)
            x=Conv2D(out_dim,k,strides=strides,padding='same')(inputs)
            if use_bn:
                x=BatchNormalization()(x,training=training)
            if use_relu:
                x=tf.nn.relu(x)
            return x
    
    def residual(self,inputs,input_dim,out_dim,k=3,strides=(1,1),training=True,scope='residual'):
        with tf.compat.v1.variable_scope(scope):
            #assert inputs.get_shape().as_list()[3]==input_dim
            #low layer 3*3>3*3
            x=self.conv_bn_re(inputs,input_dim,out_dim,strides=strides,training=training,scope='up_1')
            x=self.conv_bn_re(x,out_dim,out_dim,use_relu=False,training=training,scope='up_2')
            #skip,up layer 1*1
            skip=self.conv_bn_re(inputs,input_dim,out_dim,strides=strides,use_relu=False,k=1,training=training,scope='low')
            #skip+x
            res=tf.nn.relu(tf.add(skip,x))
            return res
        
        
    def res_block(self,inputs,input_dim,out_dim,n,k=3,training=True,scope='res_block'):
        with tf.compat.v1.variable_scope(scope):
            x=self.create_fire_module(x,nb_squeeze_filter,'fire1')
            for i in range(1,n):
                x=self.create_fire_module(x,nb_squeeze_filter,'fire1')
            return x

    def get_axis():
        axis = -1 if K.image_data_format() == 'channels_last' else 1
        return axis

    def fire_module(self,inputs,input_dim,  out_dim,strides=1,k=3, sr=2,skip=False,relu=True,training=True,scope="fire_block"):
        with tf.compat.v1.variable_scope(scope):
            conv1   = Conv2D(out_dim//sr,(1,1), padding='same', use_bias=False)(inputs)
            normalized = BatchNormalization(axis=1)(conv1)
            expand_1x1 = Conv2D(out_dim//sr, kernel_size=(1,1), padding='same',strides=strides)(normalized)
            #depth2d performt auf input + gegebenen filter
            expand_3x3 = Conv2D(out_dim//sr, kernel_size=(3,3), padding='same',strides=strides)(normalized)
            #FIXME: depthwise
            #depth_expand_3x3 = tf.nn.depthwise_conv2d(normalized,tf.Variable([3,3,out_dim//sr,sr],dtype=tf.float32), [1,1,1,1], 'SAME')
        
            #axis = get_axis()
            x_ret = tf.concat([expand_1x1, expand_3x3],3)
            output = BatchNormalization(axis=1)(x_ret)
            return tf.keras.activations.relu(output)
    

    def fire_residual(self,inputs,input_dim,out_dim,k=3,strides=(1,1),training=True,scope='residual'):
        with tf.compat.v1.variable_scope(scope):
            x=self.fire_module(inputs,input_dim,out_dim,strides=strides,scope='up_1')
            x=self.fire_module(x,out_dim,out_dim,scope='up_2')
            skip=self.fire_module(inputs, input_dim, out_dim, strides=strides,k=1,skip=True,scope='low')
            res=tf.nn.relu(tf.add(skip,x))
            return res
    
    
    def fire_block(self,inputs,input_dim,out_dim,n,k=3,training=True,scope='fire_block'):
        with tf.compat.v1.variable_scope(scope):
            x=self.fire_residual(inputs,input_dim,out_dim,k=k,scope='residual_0')
            for i in range(1,n):
                x=self.fire_residual(x,out_dim,out_dim,k=k,scope='residual_%d'%i)
            return x


    def hourglass(self,inputs,n_deep,n_res,n_dims,training=True,scope='hourglass_5'):
        #TODO:
        #
        
        with tf.compat.v1.variable_scope(scope):
            curr_res=n_res[0]
            next_res=n_res[1]
            curr_dim=n_dims[0]
            next_dim=n_dims[1]

            up_1=self.fire_block(inputs,curr_dim,curr_dim,curr_res,training=training,scope='up_1')

            #half=tf.nn.max_pool2d(input=inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            #remove downsample layer. -> replace half mit up_1
            low_1=self.fire_block(up_1,curr_dim,next_dim,curr_res,training=training,scope='low_1')
            if n_deep>1:
                low_2=self.hourglass(low_1,n_deep-1,n_res[1:],n_dims[1:],training=training,scope='hourglass_%d'%(n_deep-1))
            else:
                low_2=self.fire_block(low_1,next_dim,next_dim,next_res,training=training,scope='low_2')
            low_3=self.fire_block(low_2,next_dim,curr_dim,curr_res,training=training,scope='low_3')
          #muss vielleicht noch ersetzt werde. Tf Lite unterstützt resize_nearest_ neighbor && [1:3]*2 ersetzt durch [1:3]
            up_2=tf.image.resize(low_3,tf.shape(input=low_3)[1:3],name='up_2', method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            merge=tf.add(up_1,up_2)

            return merge

            
    def start_conv(self,img,training=True,scope='start',k=3):
        with tf.compat.v1.variable_scope(scope):
            x=tf.keras.layers.Conv2D(128,7,2)(img)
            x=tf.keras.layers.BatchNormalization()(x,training=training)
            x=tf.keras.activations.relu(x)
            #changed 256 to 128 and added a second residual 
            x=self.residual(x,128,256,strides=(2,2),scope='residual_start')
            x=self.residual(x,256,256,strides=(2,2),scope='residual_start1')

            return x

    def corner_pooling(self,inputs,input_dim,out_dim,k=3,training=True,scope='corner_pooling'):
        with tf.compat.v1.variable_scope(scope):
            with tf.compat.v1.variable_scope('top_left'):
                top=self.conv_bn_re(inputs,input_dim,128,training=training,scope='top')
                top_pool=TopPool(top)

                left=self.conv_bn_re(inputs,input_dim,128,training=training,scope='left')
                left_pool=LeftPool(left)


                #top_left=tf.add(top,left)

                top_left=tf.add(top_pool,left_pool)
                top_left=self.conv_bn_re(top_left,128,out_dim,use_relu=False,training=training,scope='top_left')

                skip_tl=self.conv_bn_re(inputs,input_dim,out_dim,use_relu=False,k=1,training=training,scope='skip')

                merge_tl=tf.nn.relu(tf.add(skip_tl,top_left))
                merge_tl=self.conv_bn_re(merge_tl,out_dim,out_dim,training=training,scope='merge_tl')
            with tf.compat.v1.variable_scope('bottom_right'):
                bottom=self.conv_bn_re(inputs,input_dim,128,training=training,scope='bottom')
                bottom_pool=BottomPool(bottom)

                right=self.conv_bn_re(inputs,input_dim,128,training=training,scope='right')
                right_pool=RightPool(right)

                #bottom_right=tf.add(bottom,right)

                bottom_right=tf.add(bottom_pool,right_pool)
                bottom_right=self.conv_bn_re(bottom_right,128,out_dim,use_relu=False,training=training,scope='bottom_right')

                skip_br=self.conv_bn_re(inputs,input_dim,out_dim,use_relu=False,k=1,training=training,scope='skip')

                merge_br=tf.nn.relu(tf.add(skip_br,bottom_right))
                merge_br=self.conv_bn_re(merge_br,out_dim,out_dim,training=training,scope='merge_br')
            return merge_tl,merge_br
    
    def heat(self,inputs,input_dim,out_dim,scope='heat'):
        #out_dim=80
        #replace 3x2 kernels with 1x1 in prediction module
        with tf.compat.v1.variable_scope(scope):
            x=self.conv_bn_re(inputs,input_dim,input_dim,use_bn=False)
            x=Conv2d(out_dim,1)(x)
            return x
    def tag(self,inputs,input_dim,out_dim,scope='tag'):
        #out_dim=1
        with tf.compat.v1.variable_scope(scope):
            x=self.conv_bn_re(inputs,input_dim,input_dim,use_bn=False)
            x=Conv2D(out_dim,1)(x)
            return x
    def offset(self,inputs,input_dim,out_dim,scope='offset'):
        #out_dim=2
        with tf.compat.v1.variable_scope(scope):
            x=self.conv_bn_re(inputs,input_dim,input_dim,use_bn=False)
            x=Conv2D(out_dim,1)(x)
            return x
    def hinge(self,inputs,input_dim,out_dim,training=True,scope='hinge'):
        with tf.compat.v1.variable_scope(scope):
            x=self.conv_bn_re(inputs,input_dim,out_dim,training=training)
            return x
    def inter(self,input_1,input_2,out_dim,training=True,scope='inter'):
        with tf.compat.v1.variable_scope(scope):
            x_1=self.conv_bn_re(input_1,tf.shape(input=input_1)[3],out_dim,use_relu=False,k=1,training=training,scope='branch_start')
            x_2=self.conv_bn_re(input_2,tf.shape(input=input_2)[3],out_dim,use_relu=False,k=1,training=training,scope='branch_hourglass1')
            x=tf.nn.relu(tf.add(x_1,x_2))
            x=self.residual(x,out_dim,out_dim,training=training)
            return x
    
    '''ab hier neues zeug was vorher nicht da war'''
  
    def pred_mod(self,input,inp_dim=256,out_dim=256,dim=80, kernel=1,scope='pred'):
        x=Conv2D(out_dim,kernel,padding='SAME',use_bias=False)(input)
        x=BatchNormalization()(x)
        x=tf.nn.relu(x)
        return Conv2D(dim,kernel_size=(1,1))(x)
