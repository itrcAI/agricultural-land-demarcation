import keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, ReLU, Add
from tensorflow.keras import Input, layers, models, regularizers
from tensorflow.keras.models import *
from tensorflow.keras.layers import Activation,Conv2D,MaxPooling2D,UpSampling2D,Dense,BatchNormalization,Input,Reshape,multiply,add,Dropout,AveragePooling2D,GlobalAveragePooling2D,concatenate
from keras.layers.convolutional import Conv2DTranspose
from keras.regularizers import l2



def build_unet_model(input_shape):
    def double_conv_block(x, n_filters):
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        return x

    def downsample_block(x, n_filters):
        f = double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.)(p)
        return f, p

    def upsample_block(x, conv_features, n_filters):
        # upsample
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        # concatenate
        x = layers.concatenate([x, conv_features])
        # dropout
        x = layers.Dropout(0.)(x)
        # Conv2D twice with ReLU activation
        x = double_conv_block(x, n_filters)
        return x
    
    # inputs
    inputs = layers.Input(shape=input_shape)
    l1 = 4
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, l1*2)
    # 2 - downsample
    f2, p2 = downsample_block(p1, l1*4)
    # 3 - downsample
    f3, p3 = downsample_block(p2, l1*8)
    # 4 - downsample
    f4, p4 = downsample_block(p3,l1*16)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, l1*32)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, l1*16)
    # 7 - upsample
    u7 = upsample_block(u6, f3, l1*8)
    # 8 - upsample
    u8 = upsample_block(u7, f2, l1*4)
    # 9 - upsample
    u9 = upsample_block(u8, f1, l1*2)

    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


def build_DenseNets(input_shape=(None,None,2),
                       n_classes=1, 
                       n_filters_first_conv=48, 
                       n_pool=3, 
                       growth_rate=16,
                       n_layers_per_block= [4, 5, 7, 10, 7, 5, 4], #[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
                       dropout_p=0.2):
    
    def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
        '''
        Apply BatchNormalization, ReLU nonlinearity, Convolution and Dropout successively
        Dense block layers are composed of BN, followed by ReLU, a 3x3 same convolution (no resolution loss)
        and dropoutwith probability p=0.2
        '''
        l = BatchNormalization()(inputs)
        l = Activation('relu')(l)
        l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
        if dropout_p != 0.0: 
            l = Dropout(dropout_p)(l)
        return l 

    def TransitionDown(inputs, n_filters, dropout_p=0.2): 
        '''Apply a BN_ReLU_Conv layer with filter size = 1, and a max pooling with a factor of 2'''
        l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
        l = MaxPooling2D((2,2))(l)
        return l 

    def TransitionUp(skip_connection, block_to_upsample, n_filters_keep): 
        '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
        l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
        l = concatenate([l, skip_connection], axis=-1)
        return l

    def SoftmaxLayer(inputs, n_classes): 
        '''
        Performs 1x1 convolution followed by softmax nonlinearity 
        The output will have the shape (batch_size x n_rows x n_cols, n_classes)
        '''
        l = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform', activation='sigmoid')(inputs)
        # l = Reshape((-1, n_classes))(l)
        # l = Activation('softmax') # 'sigmoid' for binary classes
        return l

    if type(n_layers_per_block) == list:
            print(len(n_layers_per_block))
    elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
    
    #####################
    # First Convolution #
    #####################        
    inputs = Input(shape=input_shape)
    stack = Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inputs)
    n_filters = n_filters_first_conv

    #####################
    # Downsampling path #
    #####################  
    skip_connection_list = []
    for i in range(n_pool): # each iteration creates a dense block in the down path
        for j in range(n_layers_per_block[i]): # each iteration composes layers for a dense block
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)      
            stack = concatenate([stack, l])
            n_filters += growth_rate   

        skip_connection_list.append(stack)
        stack = TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1] # reverse the skip_connection_list for upsampling

    #####################
    #    Bottleneck     #
    #####################     
    block_to_upsample = []

    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l) 
        stack = concatenate([stack, l])
    block_to_upsample = concatenate(block_to_upsample)

    #####################
    #  Upsampling path  #
    #####################
    for i in range(n_pool): 
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]): 
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])
        block_to_upsample = concatenate(block_to_upsample)

    #####################
    #  Softmax          #
    #####################
    output = SoftmaxLayer(stack, n_classes)
    model = Model(inputs = inputs, outputs = output)
    # model.summary()

    return model


def DenseUnet(input_shape):
    def Conv_Block(input_tensor, filters, bottleneck=False, weight_decay=1e-4):

        concat_axis = 1 if K.image_data_format() == 'channel_first' else -1 

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor)
        x = Activation('relu')(x)

        # if bottleneck:
        #     inter_channel = filters
        #     x = Conv2D(inter_channel, (1, 1),
        #                kernel_initializer='he_normal',
        #                padding='same', use_bias=False,
        #                kernel_regularizer=l2(weight_decay))(x)
        #     x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        #     x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)

        return x
    def dens_block(input_tensor, nb_filter):
        x1 = Conv_Block(input_tensor,nb_filter)
        add1 = concatenate([x1, input_tensor], axis=-1)
        x2 = Conv_Block(add1,nb_filter)
        add2 = concatenate([x1, input_tensor,x2], axis=-1)
        x3 = Conv_Block(add2,nb_filter)
        return x3

    l1 = 4
    inputs = Input(shape=input_shape)
    # x  = Conv2D(32, 1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = Conv2D(32, 7, kernel_initializer='he_normal', padding='same', strides=1,use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    #down first
    down1 = dens_block(x,nb_filter=l1*2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)#256
    #down second
    down2 = dens_block(pool1,nb_filter=l1*4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down2)#128
    #down third
    down3 = dens_block(pool2,nb_filter=l1*8)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)#64
    #down four
    down4 = dens_block(pool3,nb_filter=l1*16)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)#32
    #center
    conv5 = Conv2D(l1*32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(l1*32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # up first
    up6 = UpSampling2D(size=(2, 2))(drop5)
    # up6 = UpSampling2D(size=(2, 2))(drop5)
    add6 = concatenate([down4, up6], axis=3)
    up6 = dens_block(add6,nb_filter=l1*16)
    # up second
    up7 = UpSampling2D(size=(2, 2))(up6)
    #up7 = UpSampling2D(size=(2, 2))(conv6)
    add7 = concatenate([down3, up7], axis=3)
    up7 = dens_block(add7,nb_filter=l1*8)
    # up third
    up8 = UpSampling2D(size=(2, 2))(up7)
    #up8 = UpSampling2D(size=(2, 2))(conv7)
    add8 = concatenate([down2, up8], axis=-1)
    up8 = dens_block(add8,nb_filter=l1*4)
    #up four
    up9 =UpSampling2D(size=(2, 2))(up8)
    add9 = concatenate([down1, up9], axis=-1)
    up9 = dens_block(add9,nb_filter=l1*2)
    # output
    conv10 = Conv2D(32, 7, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv10)
    model = Model(inputs, conv10)
    # print(model.summary())
    return model

def ResUNet(input_shape):
    def bn_act(x, act=True):
        x = tf.keras.layers.BatchNormalization()(x)
        if act == True:
            x = tf.keras.layers.Activation("relu")(x)
        return x

    def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = bn_act(x)
        conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)
        
        output = tf.keras.layers.Add()([conv, shortcut])
        return output

    def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
        
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)
        
        output = tf.keras.layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(x, xskip):
        u = tf.keras.layers.UpSampling2D((2, 2))(x)
        c = tf.keras.layers.Concatenate()([u, xskip])
        return c


    f = [8, 16, 32, 64, 128]
    inputs = tf.keras.layers.Input(input_shape)
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = tf.keras.models.Model(inputs, outputs)
    return model

def SegNet(input_shape):
    l1 = 16
    # Encoding layer
    img_input = Input(shape=input_shape)
    x = Conv2D(l1*4, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)
    x = Activation('relu')(x)
    x = Conv2D(l1*4, (3, 3), padding='same', name='conv2')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(l1*8, (3, 3), padding='same', name='conv3')(x)
    x = Activation('relu')(x)
    x = Conv2D(l1*8, (3, 3), padding='same', name='conv4')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(l1*16, (3, 3), padding='same', name='conv5')(x)
    x = Activation('relu')(x)
    x = Conv2D(l1*16, (3, 3), padding='same', name='conv6')(x)
    x = Activation('relu')(x)
    x = Conv2D(l1*16, (3, 3), padding='same', name='conv7')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(l1*32, (3, 3), padding='same', name='conv8')(x)
    x = Activation('relu')(x)
    x = Conv2D(l1*32, (3, 3), padding='same', name='conv9')(x)
    x = Activation('relu')(x)
    x = Conv2D(l1*32, (3, 3), padding='same', name='conv10')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(l1*32, (3, 3), padding='same', name='conv11')(x)
    x = Activation('relu')(x)
    x = Conv2D(l1*32, (3, 3), padding='same', name='conv12')(x)
    x = Activation('relu')(x)
    x = Conv2D(l1*32, (3, 3), padding='same', name='conv13')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D()(x)

    x = Dense(l1*64, activation = 'relu', name='fc1')(x)
    x = Dense(l1*64, activation = 'relu', name='fc2')(x)

    # Decoding Layer 
    x = UpSampling2D()(x)
    x = Conv2DTranspose(l1*32, (3, 3), padding='same', name='deconv1')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(l1*32, (3, 3), padding='same', name='deconv2')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(l1*32, (3, 3), padding='same', name='deconv3')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = UpSampling2D()(x)
    x = Conv2DTranspose(l1*32, (3, 3), padding='same', name='deconv4')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(l1*32, (3, 3), padding='same', name='deconv5')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(l1*16, (3, 3), padding='same', name='deconv6')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(l1*16, (3, 3), padding='same', name='deconv7')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(l1*16, (3, 3), padding='same', name='deconv8')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(l1*8, (3, 3), padding='same', name='deconv9')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(l1*8, (3, 3), padding='same', name='deconv10')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(l1*4, (3, 3), padding='same', name='deconv11')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = UpSampling2D()(x)
    x = Conv2DTranspose(l1*4, (3, 3), padding='same', name='deconv12')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
    x = Activation('softmax')(x)
    pred = Reshape((input_shape[0], input_shape[1], 1))(x)
    
    model = Model(inputs=img_input, outputs=pred)
    
    return model

def DeeplabV3(input_shape, num_classe=1):
    def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
            )(block_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x

    def DilatedSpatialPyramidPooling(dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
        )(x)

        out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = convolution_block(x, kernel_size=1)
        return output


    model_input = keras.Input(shape=input_shape)
    resnet50 = keras.applications.resnet.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(input_shape[0] // 4 // x.shape[1], input_shape[0] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(input_shape[0] // x.shape[1], input_shape[0] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation = "sigmoid")(x)
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    
    return model



# UNet-DenseNet-ResUNet Hybrid Model
def unet_densenet_resunet(input_shape):


    # Dense Block
    def dense_block(x, blocks, growth_rate, name):
        for i in range(blocks):
            x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
        return x

    def conv_block(x, growth_rate, name):
        x1 = layers.BatchNormalization(name=name + '_bn')(x)
        x1 = layers.Activation('relu', name=name + '_relu')(x1)
        x1 = layers.Conv2D(growth_rate, 3, padding='same', name=name + '_conv')(x1)
        x = layers.Concatenate(name=name + '_concat')([x, x1])
        return x

    # Residual Block with filter matching
    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        if x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
        
        y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(filters, kernel_size, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Add()([shortcut, y])
        y = layers.Activation('relu')(y)
        return y

    # Encoder
    def encoder_block(x, filters, growth_rate, name):
        x = dense_block(x, 4, growth_rate, name=name + '_dense')
        p = layers.MaxPooling2D((2, 2), name=name + '_pool')(x)
        return x, p

    # Decoder
    def decoder_block(x, skip, filters, name):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=name + '_upconv')(x)
        x = layers.Concatenate(name=name + '_concat')([x, skip])
        x = residual_block(x, filters)
        return x

    L = 8
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, L, L, 'enc1')
    s2, p2 = encoder_block(p1, 2 * L, L, 'enc2')
    s3, p3 = encoder_block(p2, 4 * L, L, 'enc3')
    s4, p4 = encoder_block(p3, 8 * L, L, 'enc4')

    # Bottleneck
    b = dense_block(p4, 4, L, 'bottleneck')
    
    # Decoder
    d1 = decoder_block(b, s4, 8 * L, 'dec1')
    d2 = decoder_block(d1, s3, 4 * L, 'dec2')
    d3 = decoder_block(d2, s2, 2 * L, 'dec3')
    d4 = decoder_block(d3, s1, L, 'dec4')

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs, outputs, name='UNet_DenseNet_ResUNet')
    return model


def unetplus_resunet_attentionunet(input_shape):
    
    def attention_block(x, g, inter_channel):
        theta_x = layers.Conv2D(inter_channel, kernel_size=1, strides=1, padding='same')(x)
        phi_g = layers.Conv2D(inter_channel, kernel_size=1, strides=1, padding='same')(g)
        phi_g = layers.UpSampling2D(size=(x.shape[1] // g.shape[1], x.shape[2] // g.shape[2]))(phi_g)  # Upsample phi_g to match theta_x shape
        f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
        psi_f = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(f)
        rate = layers.Activation('sigmoid')(psi_f)
        att_x = layers.multiply([x, rate])
        return att_x

    def resunet_block(x, filters, kernel_size=3):
        conv = layers.Conv2D(filters, kernel_size, padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters, kernel_size, padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Dropout(0.3)(conv)  # Add dropout
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
        res_path = layers.add([shortcut, conv])
        res_path = layers.Activation('relu')(res_path)
        return res_path

    def unetpp_block(x, filters, kernel_size=3):
        conv = layers.Conv2D(filters, kernel_size, padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters, kernel_size, padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Dropout(0.3)(conv)  # Add dropout
        return conv

    inputs = tf.keras.Input(input_shape)
    L = 16

    # Encoder path
    e1 = resunet_block(inputs, L)
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = unetpp_block(p1, L * 2)
    p2 = layers.MaxPooling2D((2, 2))(e2)
    
    e3 = resunet_block(p2, L * 4)
    p3 = layers.MaxPooling2D((2, 2))(e3)

    # Bridge
    bridge = unetpp_block(p3, L * 8)

    # Decoder path
    d3 = layers.Conv2DTranspose(L * 4, (2, 2), strides=(2, 2), padding='same')(bridge)
    d3 = layers.concatenate([d3, e3])
    d3 = attention_block(d3, bridge, L * 4)
    d3 = resunet_block(d3, L * 4)
    
    d2 = layers.Conv2DTranspose(L * 2, (2, 2), strides=(2, 2), padding='same')(d3)
    d2 = layers.concatenate([d2, e2])
    d2 = attention_block(d2, d3, L * 2)
    d2 = unetpp_block(d2, L * 2)
    
    d1 = layers.Conv2DTranspose(L, (2, 2), strides=(2, 2), padding='same')(d2)
    d1 = layers.concatenate([d1, e1])
    d1 = attention_block(d1, d2, L)
    d1 = resunet_block(d1, L)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model




def unetpp_resunet(input_shape):
    def resunet_block(x, filters, kernel_size=3):
        conv = layers.Conv2D(filters, kernel_size, padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters, kernel_size, padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Dropout(0.3)(conv)  # Add dropout
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
        res_path = layers.add([shortcut, conv])
        res_path = layers.Activation('relu')(res_path)
        return res_path

    def unetpp_block(x, filters, kernel_size=3):
        conv = layers.Conv2D(filters, kernel_size, padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters, kernel_size, padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Dropout(0.3)(conv)  # Add dropout
        return conv

    inputs = tf.keras.Input(input_shape)
    L = 16

    # Encoder path
    e1 = resunet_block(inputs, L)
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = unetpp_block(p1, L * 2)
    p2 = layers.MaxPooling2D((2, 2))(e2)
    
    e3 = resunet_block(p2, L * 4)
    p3 = layers.MaxPooling2D((2, 2))(e3)

    # Bridge
    bridge = unetpp_block(p3, L * 8)

    # Decoder path
    d3 = layers.Conv2DTranspose(L * 4, (2, 2), strides=(2, 2), padding='same')(bridge)
    d3 = layers.concatenate([d3, e3])
    d3 = resunet_block(d3, L * 4)
    
    d2 = layers.Conv2DTranspose(L * 2, (2, 2), strides=(2, 2), padding='same')(d3)
    d2 = layers.concatenate([d2, e2])
    d2 = unetpp_block(d2, L * 2)
    
    d1 = layers.Conv2DTranspose(L, (2, 2), strides=(2, 2), padding='same')(d2)
    d1 = layers.concatenate([d1, e1])
    d1 = resunet_block(d1, L)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model


def bpat_unet(input_shape):


    def boundary_points_supervision_module(x, num_channels):
        def stripe_pooling(x):
            shape = tf.shape(x)
            stripe_horizontal = tf.reshape(x, (shape[0], shape[1], -1, shape[3]))
            stripe_vertical = tf.reshape(x, (shape[0], -1, shape[2], shape[3]))
            return stripe_horizontal, stripe_vertical
        
        def spsa(query, key, value):
            score = tf.matmul(query, key, transpose_b=True)
            score = score / tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))
            attention_weights = tf.nn.softmax(score, axis=-1)
            output = tf.matmul(attention_weights, value)
            return output

        def ppsa(x):
            pooled = layers.GlobalAveragePooling2D()(x)
            pooled = layers.Reshape((1, 1, num_channels))(pooled)
            return pooled
        
        stripe_h, stripe_v = stripe_pooling(x)
        query = layers.Conv2D(num_channels, 1)(stripe_h)
        key = layers.Conv2D(num_channels, 1)(stripe_v)
        value = layers.Conv2D(num_channels, 1)(stripe_h)
        
        spsa_output = spsa(query, key, value)
        
        ppsa_output = ppsa(x)
        
        combined = layers.Add()([spsa_output, ppsa_output])
        combined = layers.Conv2D(num_channels, 1, activation='relu')(combined)
        combined = layers.Add()([combined, x])
        
        return combined

    def adaptive_multi_scale_feature_fusion_module(x_low, x_high, num_channels):
        def deformable_conv(x, filters):
            offset = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            return x
        
        def deformable_attention(x):
            shape = tf.shape(x)
            num_heads = 8
            depth = num_channels // num_heads
            
            query = layers.Conv2D(num_channels, 1)(x)
            key = layers.Conv2D(num_channels, 1)(x)
            value = layers.Conv2D(num_channels, 1)(x)
            
            query = tf.reshape(query, (shape[0], shape[1] * shape[2], num_heads, depth))
            key = tf.reshape(key, (shape[0], shape[1] * shape[2], num_heads, depth))
            value = tf.reshape(value, (shape[0], shape[1] * shape[2], num_heads, depth))
            
            score = tf.matmul(query, key, transpose_b=True)
            score = score / tf.math.sqrt(tf.cast(depth, tf.float32))
            attention_weights = tf.nn.softmax(score, axis=-1)
            output = tf.matmul(attention_weights, value)
            output = tf.reshape(output, shape)
            
            return output
        
        x_low = deformable_conv(x_low, num_channels)
        x_high = deformable_attention(x_high)
        
        combined = layers.Add()([x_low, x_high])
        combined = layers.Conv2D(num_channels, 3, padding='same', activation='relu')(combined)
        
        return combined

    def assembled_transformer_module(x, num_channels):
        def windowed_deformable_attention(x):
            shape = tf.shape(x)
            window_size = 4
            num_heads = 8
            depth = num_channels // num_heads
            
            query = layers.Conv2D(num_channels, 1)(x)
            key = layers.Conv2D(num_channels, 1)(x)
            value = layers.Conv2D(num_channels, 1)(x)
            
            query = tf.reshape(query, (shape[0], shape[1] // window_size, shape[2] // window_size, window_size * window_size, num_heads, depth))
            key = tf.reshape(key, (shape[0], shape[1] // window_size, shape[2] // window_size, window_size * window_size, num_heads, depth))
            value = tf.reshape(value, (shape[0], shape[1] // window_size, shape[2] // window_size, window_size * window_size, num_heads, depth))
            
            score = tf.matmul(query, key, transpose_b=True)
            score = score / tf.math.sqrt(tf.cast(depth, tf.float32))
            attention_weights = tf.nn.softmax(score, axis=-1)
            output = tf.matmul(attention_weights, value)
            output = tf.reshape(output, shape)
            
            return output
        
        def external_attention(x):
            ea = layers.GlobalAveragePooling2D()(x)
            ea = layers.Reshape((1, 1, num_channels))(ea)
            return ea
        
        local_features = windowed_deformable_attention(x)
        global_features = external_attention(x)
        
        combined = layers.Add()([local_features, global_features])
        combined = layers.Conv2D(num_channels, 1, activation='relu')(combined)
        
        return combined

    inputs = tf.keras.Input(shape=input_shape)
    L=16
    
    # Encoder
    conv1 = layers.Conv2D(L, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(L, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(2 * L, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(2 * L, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(4 * L, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(4 * L, 3, activation='relu', padding='same')(conv3)
    bpsm3 = boundary_points_supervision_module(conv3, 4 * L)  # Add BPSM
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bpsm3)
    
    conv4 = layers.Conv2D(8 * L, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(8 * L, 3, activation='relu', padding='same')(conv4)
    bpsm4 = boundary_points_supervision_module(conv4, 8 * L)  # Add BPSM
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bpsm4)
    
    conv5 = layers.Conv2D(16 * L, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(16 * L, 3, activation='relu', padding='same')(conv5)
    bpsm5 = boundary_points_supervision_module(conv5, 16 * L)  # Add BPSM
    
    # Bottleneck
    bottleneck = assembled_transformer_module(bpsm5, 16 * L)
    
    # Decoder
    upconv4 = layers.Conv2DTranspose(8 * L, 2, strides=(2, 2), padding='same')(bottleneck)
    amffm4 = adaptive_multi_scale_feature_fusion_module(upconv4, bpsm4, 8 * L)  # Add AMFFM
    upconv4 = layers.concatenate([amffm4, conv4], axis=-1)
    upconv4 = layers.Conv2D(8 * L, 3, activation='relu', padding='same')(upconv4)
    upconv4 = layers.Conv2D(8 * L, 3, activation='relu', padding='same')(upconv4)
    
    upconv3 = layers.Conv2DTranspose(4 * L, 2, strides=(2, 2), padding='same')(upconv4)
    amffm3 = adaptive_multi_scale_feature_fusion_module(upconv3, bpsm3, 4 * L)  # Add AMFFM
    upconv3 = layers.concatenate([amffm3, conv3], axis=-1)
    upconv3 = layers.Conv2D(4 * L, 3, activation='relu', padding='same')(upconv3)
    upconv3 = layers.Conv2D(4 * L, 3, activation='relu', padding='same')(upconv3)
    
    upconv2 = layers.Conv2DTranspose(2 * L, 2, strides=(2, 2), padding='same')(upconv3)
    upconv2 = layers.concatenate([upconv2, conv2], axis=-1)
    upconv2 = layers.Conv2D(2 * L, 3, activation='relu', padding='same')(upconv2)
    upconv2 = layers.Conv2D(2 * L, 3, activation='relu', padding='same')(upconv2)
    
    upconv1 = layers.Conv2DTranspose(L, 2, strides=(2, 2), padding='same')(upconv2)
    upconv1 = layers.concatenate([upconv1, conv1], axis=-1)
    upconv1 = layers.Conv2D(L, 3, activation='relu', padding='same')(upconv1)
    upconv1 = layers.Conv2D(L, 3, activation='relu', padding='same')(upconv1)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(upconv1)
    
    model = Model(inputs, outputs)
    return model



def unetplus_res_attgate(input_shape):

    class ConvBlock(tf.keras.layers.Layer):
        def __init__(self, filters):
            super(ConvBlock, self).__init__()
            self.conv1 = layers.Conv2D(filters, kernel_size=3, padding='same')
            self.conv2 = layers.Conv2D(filters, kernel_size=3, padding='same')
            self.relu = layers.ReLU()
        
        def call(self, inputs):
            x = self.relu(self.conv1(inputs))
            x = self.relu(self.conv2(x))
            return x

    class ResidualBlock(tf.keras.layers.Layer):
        def __init__(self, filters):
            super(ResidualBlock, self).__init__()
            self.conv_block = ConvBlock(filters)
            
        def call(self, inputs):
            return inputs + self.conv_block(inputs)

    class AttentionGate(tf.keras.layers.Layer):
        def __init__(self, F_g, F_l, F_int):
            super(AttentionGate, self).__init__()
            self.W_g = models.Sequential([
                layers.Conv2D(F_int, kernel_size=1, strides=1, padding='same'),
                layers.BatchNormalization()
            ])
            
            self.W_x = models.Sequential([
                layers.Conv2D(F_int, kernel_size=1, strides=1, padding='same'),
                layers.BatchNormalization()
            ])
            
            self.psi = models.Sequential([
                layers.Conv2D(1, kernel_size=1, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.Activation('sigmoid')
            ])
            
            self.relu = layers.ReLU()
            
        def call(self, g, x):
            # Ensure the spatial dimensions of g and x are the same
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            
            # Resize g1 to match the dimensions of x1
            g1_resized = tf.image.resize(g1, size=(x1.shape[1], x1.shape[2]))
            
            psi = self.relu(g1_resized + x1)
            psi = self.psi(psi)
            return x * psi

    inputs = layers.Input(shape=input_shape)
    L=16
    filters = [L, 2*L, 4*L, 8*L]
    
    # Downsampling path
    x0_0 = ConvBlock(filters[0])(inputs)
    x1_0 = layers.MaxPooling2D((2, 2))(x0_0)
    x1_0 = ConvBlock(filters[1])(x1_0)
    x2_0 = layers.MaxPooling2D((2, 2))(x1_0)
    x2_0 = ConvBlock(filters[2])(x2_0)
    x3_0 = layers.MaxPooling2D((2, 2))(x2_0)
    x3_0 = ConvBlock(filters[3])(x3_0)
    
    # Nested skip connections and residuals
    x0_1 = ConvBlock(filters[0])(layers.Concatenate()([x0_0, layers.UpSampling2D(size=(2, 2))(x1_0)]))
    x0_1 = ResidualBlock(filters[0])(x0_1)
    
    x1_1 = ConvBlock(filters[1])(layers.Concatenate()([x1_0, layers.UpSampling2D(size=(2, 2))(x2_0)]))
    x1_1 = ResidualBlock(filters[1])(x1_1)
    x0_2 = ConvBlock(filters[0])(layers.Concatenate()([x0_0, x0_1, layers.UpSampling2D(size=(2, 2))(x1_1)]))
    x0_2 = ResidualBlock(filters[0])(x0_2)
    
    x2_1 = ConvBlock(filters[2])(layers.Concatenate()([x2_0, layers.UpSampling2D(size=(2, 2))(x3_0)]))
    x2_1 = ResidualBlock(filters[2])(x2_1)
    x1_2 = ConvBlock(filters[1])(layers.Concatenate()([x1_0, x1_1, layers.UpSampling2D(size=(2, 2))(x2_1)]))
    x1_2 = ResidualBlock(filters[1])(x1_2)
    x0_3 = ConvBlock(filters[0])(layers.Concatenate()([x0_0, x0_1, x0_2, layers.UpSampling2D(size=(2, 2))(x1_2)]))
    x0_3 = ResidualBlock(filters[0])(x0_3)
    
    # Attention gates
    g3 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[2])(x3_0, x3_0)
    g2 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1])(g3, x2_0)
    g1 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0])(g2, x1_0)
    g0 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0])(g1, x0_0)
    
    outputs = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(x0_3)
    
    return models.Model(inputs, outputs)



def unetp_res_att_vit(input_shape):
    filters = 16  # Set filters to 16

    def residual_block(x, filters, kernel_size=3):
        res = layers.Conv2D(filters, kernel_size, padding='same')(x)
        res = layers.BatchNormalization()(res)
        res = layers.ReLU()(res)
        res = layers.Conv2D(filters, kernel_size, padding='same')(res)
        res = layers.BatchNormalization()(res)
        return layers.add([res, x])

    def attention_block(x, g, filters):
        theta_x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        phi_g = layers.Conv2D(filters, (1, 1), padding='same')(g)
        upsample_g = layers.UpSampling2D(size=(theta_x.shape[1] // phi_g.shape[1], theta_x.shape[2] // phi_g.shape[2]))(phi_g)

        concat_xg = layers.add([theta_x, upsample_g])
        act_xg = layers.Activation('relu')(concat_xg)
        psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)

        upsample_psi = layers.UpSampling2D(size=(x.shape[1] // sigmoid_xg.shape[1], x.shape[2] // sigmoid_xg.shape[2]))(sigmoid_xg)
        upsample_psi = layers.Reshape((x.shape[1], x.shape[2], 1))(upsample_psi)

        y = layers.multiply([upsample_psi, x])
        result = layers.Conv2D(filters, (1, 1), padding='same')(y)
        result_bn = layers.BatchNormalization()(result)
        return result_bn

    def unetpp_block(x, filters):
        c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(c5)

        u4 = layers.Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
        u4 = layers.concatenate([u4, c4])
        c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(u4)
        c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c6)

        u3 = layers.Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
        u3 = layers.concatenate([u3, c3])
        c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(u3)
        c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c7)

        u2 = layers.Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
        u2 = layers.concatenate([u2, c2])
        c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(u2)
        c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c8)

        u1 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u1 = layers.concatenate([u1, c1])
        c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(u1)
        c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c9)

        return c9

    def vit_block(x, filters, output_shape):
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        # Calculate the size for reshaping based on the original input shape
        size = (output_shape[0] // 16) * (output_shape[1] // 16) * filters
        x = layers.Dense(size, activation='relu')(x)
        x = layers.Reshape((output_shape[0] // 16, output_shape[1] // 16, filters))(x)
        x = layers.UpSampling2D((16, 16))(x)
        return x
    
    inputs = layers.Input(input_shape)
    
    # U-Net++ Block
    unetpp = unetpp_block(inputs, filters)
    
    # Residual Block
    res_block = residual_block(unetpp, filters)
    
    # Attention Block
    g = layers.Conv2D(filters, (1, 1), padding='same')(res_block)
    att_block = attention_block(res_block, g, filters)
    
    # Vision Transformer Block
    vit = vit_block(att_block, filters, input_shape)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(vit)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
    

def unetpp_att_vit(input_shape):
    filters = 16

    def attention_block(x, g, filters):
        theta_x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        phi_g = layers.Conv2D(filters, (1, 1), padding='same')(g)
        upsample_g = layers.UpSampling2D(size=(theta_x.shape[1] // phi_g.shape[1], theta_x.shape[2] // phi_g.shape[2]))(phi_g)
        
        concat_xg = layers.add([theta_x, upsample_g])
        act_xg = layers.Activation('relu')(concat_xg)
        psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        
        upsample_psi = layers.UpSampling2D(size=(x.shape[1] // sigmoid_xg.shape[1], x.shape[2] // sigmoid_xg.shape[2]))(sigmoid_xg)
        upsample_psi = layers.Reshape((x.shape[1], x.shape[2], 1))(upsample_psi)
        
        y = layers.multiply([upsample_psi, x])
        result = layers.Conv2D(filters, (1, 1), padding='same')(y)
        result_bn = layers.BatchNormalization()(result)
        return result_bn

    def unetpp_block(x, filters):
        c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(c5)

        u4 = layers.Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
        u4 = layers.concatenate([u4, c4])
        c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(u4)
        c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c6)

        u3 = layers.Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
        u3 = layers.concatenate([u3, c3])
        c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(u3)
        c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c7)

        u2 = layers.Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
        u2 = layers.concatenate([u2, c2])
        c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(u2)
        c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c8)

        u1 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u1 = layers.concatenate([u1, c1])
        c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(u1)
        c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c9)

        return c9

    def vit_block(x, filters, output_shape):
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        size = (output_shape[0] // 16) * (output_shape[1] // 16) * filters
        x = layers.Dense(size, activation='relu')(x)
        x = layers.Reshape((output_shape[0] // 16, output_shape[1] // 16, filters))(x)
        x = layers.UpSampling2D((16, 16))(x)
        return x
    
    inputs = layers.Input(input_shape)
    
    # U-Net++ Block
    unetpp = unetpp_block(inputs, filters)
    
    # Attention Block
    g = layers.Conv2D(filters, (1, 1), padding='same')(unetpp)
    att_block = attention_block(unetpp, g, filters)
    
    # Vision Transformer Block
    vit = vit_block(att_block, filters, input_shape)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(vit)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
    
    
def transunet_att_res(input_shape=(256, 256, 3)):
    filters = 16

    def residual_block(x, filters):
        res = x
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, res])
        x = layers.Activation('relu')(x)
        return x

    def attention_block(x, g, filters):
        theta_x = layers.Conv2D(filters, (2, 2), strides=(2, 2), padding='same')(x)
        phi_g = layers.Conv2D(filters, (1, 1), padding='same')(g)
        upsample_g = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(phi_g)
        
        # Adjust the shape of upsample_g to match theta_x
        upsample_g = tf.image.resize(upsample_g, size=(tf.shape(theta_x)[1], tf.shape(theta_x)[2]))
        
        concat = layers.Add()([theta_x, upsample_g])
        act = layers.Activation('relu')(concat)
        psi = layers.Conv2D(1, (1, 1), padding='same')(act)
        sigmoid = layers.Activation('sigmoid')(psi)
        upsample_psi = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(sigmoid)
        upsample_psi = tf.image.resize(upsample_psi, size=(tf.shape(x)[1], tf.shape(x)[2]))
        y = layers.Multiply()([x, upsample_psi])
        result = layers.Conv2D(filters, (1, 1), padding='same')(y)
        result = layers.BatchNormalization()(result)
        return result

    inputs = layers.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = residual_block(conv1, filters)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(filters*2, (3, 3), padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = residual_block(conv2, filters*2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(filters*4, (3, 3), padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = residual_block(conv3, filters*4)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(filters*8, (3, 3), padding='same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = residual_block(conv4, filters*8)

    # Decoder
    upconv3 = layers.Conv2DTranspose(filters*4, (3, 3), strides=(2, 2), padding='same')(conv4)
    upconv3 = tf.image.resize(upconv3, size=(tf.shape(conv3)[1], tf.shape(conv3)[2]))  # Adjusting the shape
    att3 = attention_block(conv3, upconv3, filters*4)
    concat3 = layers.Concatenate()([upconv3, att3])
    conv5 = layers.Conv2D(filters*4, (3, 3), padding='same')(concat3)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    conv5 = residual_block(conv5, filters*4)

    upconv2 = layers.Conv2DTranspose(filters*2, (3, 3), strides=(2, 2), padding='same')(conv5)
    upconv2 = tf.image.resize(upconv2, size=(tf.shape(conv2)[1], tf.shape(conv2)[2]))  # Adjusting the shape
    att2 = attention_block(conv2, upconv2, filters*2)
    concat2 = layers.Concatenate()([upconv2, att2])
    conv6 = layers.Conv2D(filters*2, (3, 3), padding='same')(concat2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)
    conv6 = residual_block(conv6, filters*2)

    upconv1 = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(conv6)
    upconv1 = tf.image.resize(upconv1, size=(tf.shape(conv1)[1], tf.shape(conv1)[2]))  # Adjusting the shape
    att1 = attention_block(conv1, upconv1, filters)
    concat1 = layers.Concatenate()([upconv1, att1])
    conv7 = layers.Conv2D(filters, (3, 3), padding='same')(concat1)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)
    conv7 = residual_block(conv7, filters)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    model = Model(inputs, outputs)
    return model


def unetpp_res_transformer(input_shape, num_classes=1):

    def transformer_block(x, num_heads, ff_dim, dropout_rate=0.1):
        # Normalization and Attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x1, x1)
        x2 = layers.Add()([x, attn_output])
        
        # Feed Forward Network
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ff_output = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(x.shape[-1])
        ])(x3)
        return layers.Add()([x2, ff_output])
    
    inputs = tf.keras.Input(input_shape)
    L = 16

    # Encoder path with Transformer blocks
    e1 = layers.Conv2D(L, (3, 3), padding='same')(inputs)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.Activation('relu')(e1)
    e1 = layers.MaxPooling2D((2, 2))(e1)  # Output shape: (None, height/2, width/2, L)

    e2 = layers.Conv2D(L * 2, (3, 3), padding='same')(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.Activation('relu')(e2)
    e2 = layers.MaxPooling2D((2, 2))(e2)  # Output shape: (None, height/4, width/4, 2L)

    e3 = layers.Conv2D(L * 4, (3, 3), padding='same')(e2)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.Activation('relu')(e3)
    e3 = layers.MaxPooling2D((2, 2))(e3)  # Output shape: (None, height/8, width/8, 4L)

    # Bridge
    bridge = layers.Conv2D(L * 8, (3, 3), padding='same')(e3)
    bridge = layers.BatchNormalization()(bridge)
    bridge = layers.Activation('relu')(bridge)

    # Apply transformer block on bridge output
    batch_size = tf.shape(bridge)[0]
    height, width, channels = bridge.shape[1:4]  # Get height, width, and channels of the feature map
    bridge_reshaped = tf.reshape(bridge, (batch_size, height * width, channels))  # Reshape for transformer
    bridge_transformed = transformer_block(bridge_reshaped, num_heads=2, ff_dim=channels * 2)
    bridge = tf.reshape(bridge_transformed, (batch_size, height, width, channels))  # Reshape back to original shape

    # Decoder path
    d3 = layers.Conv2DTranspose(L * 4, (2, 2), strides=(2, 2), padding='same')(bridge)

    # Resize e3 before concatenation
    e3_resized = layers.Conv2D(L * 4, (1, 1), padding='same')(e3)  # Match channels
    e3_resized = layers.Conv2DTranspose(L * 4, (2, 2), strides=(2, 2), padding='same')(e3_resized)  # Resize to match d3
    d3 = layers.concatenate([d3, e3_resized])  # Now they should match

    d3 = transformer_block(d3, num_heads=4, ff_dim=L * 8)
    d3 = layers.Conv2D(L * 4, (3, 3), padding='same')(d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Activation('relu')(d3)

    d2 = layers.Conv2DTranspose(L * 2, (2, 2), strides=(2, 2), padding='same')(d3)

    # Resize e2 before concatenation
    e2_resized = layers.Conv2D(L * 2, (1, 1), padding='same')(e2)  # Match channels
    e2_resized = layers.Conv2DTranspose(L * 2, (2, 2), strides=(2, 2), padding='same')(e2_resized)  # Resize to match d2
    d2 = layers.concatenate([d2, e2_resized])  # Now they should match

    d2 = transformer_block(d2, num_heads=4, ff_dim=L * 4)
    d2 = layers.Conv2D(L * 2, (3, 3), padding='same')(d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Activation('relu')(d2)

    d1 = layers.Conv2DTranspose(L, (2, 2), strides=(2, 2), padding='same')(d2)

    # Resize e1 before concatenation
    e1_resized = layers.Conv2D(L, (1, 1), padding='same')(e1)  # Match channels
    e1_resized = layers.Conv2DTranspose(L, (2, 2), strides=(2, 2), padding='same')(e1_resized)  # Resize to match d1
    d1 = layers.concatenate([d1, e1_resized])  # Now they should match

    d1 = transformer_block(d1, num_heads=4, ff_dim=L * 2)
    d1 = layers.Conv2D(L, (3, 3), padding='same')(d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Activation('relu')(d1)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(d1)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def transformer_segmentation_model(input_shape, num_classes=1, num_heads=4, ff_dim=128, num_transformer_blocks=4, patch_size=16):
    def transformer_block(x, num_heads, ff_dim, dropout_rate=0.1):
        # Layer normalization and multi-head self-attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x1, x1)
        x2 = layers.Add()([x, attn_output])
        
        # Feed forward network
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ff_output = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(x.shape[-1])
        ])(x3)
        return layers.Add()([x2, ff_output])

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Extract patches
    projection_dim = 64  # Dimension for projection of patches
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    
    # Calculate output dimensions for the patches
    output_height = input_shape[0] // patch_size
    output_width = input_shape[1] // patch_size
    num_patches = output_height * output_width

    # Reshape to (batch_size, num_patches, projection_dim)
    patches = layers.Reshape((-1, projection_dim))(patches)  # Ensure it matches (None, num_patches, projection_dim)

    # Transformer blocks
    x = patches
    for _ in range(num_transformer_blocks):  # Stack transformer blocks
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)

    # Reshape back for processing
    x = layers.Reshape((output_height * output_width, projection_dim))(x)  # Correct shape: (None, num_patches, projection_dim)

    # Project back to original patch size
    x = layers.Dense(output_height * output_width * projection_dim, activation='relu')(x)  # Ensure it matches total elements

    # Reshape to (None, output_height, output_width, projection_dim)
    x = layers.Reshape((output_height, output_width, projection_dim))(x)

    # Final upsampling
    x = layers.Conv2DTranspose(num_classes, kernel_size=2, strides=2, padding='same')(x)  # Final upsampling
    x = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid')(x)  # Final segmentation layer

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model