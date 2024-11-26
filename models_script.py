import keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, ReLU, Add
from tensorflow.keras import Input, layers, models, regularizers
from tensorflow.keras.models import *
from tensorflow.keras.layers import Activation,Conv2D,MaxPooling2D,UpSampling2D,Dense,BatchNormalization,Input,Reshape,multiply,add,Dropout,AveragePooling2D,GlobalAveragePooling2D,concatenate
from keras.layers.convolutional import Conv2DTranspose
from keras.regularizers import l2



# **************** U-Net Model ****************
def build_unet_model(input_shape):
    """
    U-Net model for image segmentation.
    Takes an input shape and returns a U-Net model with convolutional blocks, 
    downsampling, bottleneck, and upsampling layers.
    """
    
    def double_conv_block(x, n_filters):
        """
        A block with two convolutional layers with ReLU activations.
        """
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        return x

    def downsample_block(x, n_filters):
        """
        Downsampling block with convolutional layers followed by MaxPooling and Dropout.
        """
        f = double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.)(p)
        return f, p

    def upsample_block(x, conv_features, n_filters):
        """
        Upsampling block with transpose convolution, concatenation with skip connections,
        dropout, and convolution.
        """
        # upsample
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        # concatenate
        x = layers.concatenate([x, conv_features])
        # dropout
        x = layers.Dropout(0.)(x)
        # Conv2D twice with ReLU activation
        x = double_conv_block(x, n_filters)
        return x

    # Model Inputs
    inputs = layers.Input(shape=input_shape)
    l1 = 4
    
    # Encoder: Contracting path (downsample)
    f1, p1 = downsample_block(inputs, l1*2)
    f2, p2 = downsample_block(p1, l1*4)
    f3, p3 = downsample_block(p2, l1*8)
    f4, p4 = downsample_block(p3, l1*16)

    # Bottleneck
    bottleneck = double_conv_block(p4, l1*32)

    # Decoder: Expanding path (upsample)
    u6 = upsample_block(bottleneck, f4, l1*16)
    u7 = upsample_block(u6, f3, l1*8)
    u8 = upsample_block(u7, f2, l1*4)
    u9 = upsample_block(u8, f1, l1*2)

    # Model Outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)

    # Final U-Net Model
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


# **************** DenseNet Model ****************
def build_DenseNets(input_shape=(None, None, 2), n_classes=1, n_filters_first_conv=48, n_pool=3, 
                    growth_rate=16, n_layers_per_block=[4, 5, 7, 10, 7, 5, 4], dropout_p=0.2):
    """
    DenseNet model for image segmentation.
    Takes an input shape and returns a DenseNet model.
    """

    def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
        """
        Applies BatchNormalization, ReLU activation, Convolution, and Dropout.
        """
        l = BatchNormalization()(inputs)
        l = Activation('relu')(l)
        l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
        if dropout_p != 0.0: 
            l = Dropout(dropout_p)(l)
        return l 

    def TransitionDown(inputs, n_filters, dropout_p=0.2): 
        """
        Downsampling block with 1x1 convolution followed by max pooling.
        """
        l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
        l = MaxPooling2D((2, 2))(l)
        return l 

    def TransitionUp(skip_connection, block_to_upsample, n_filters_keep): 
        """
        Upsampling block that concatenates the skip connection with the upsampled feature map.
        """
        l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
        l = concatenate([l, skip_connection], axis=-1)
        return l

    def SoftmaxLayer(inputs, n_classes): 
        """
        Final 1x1 convolution followed by a sigmoid activation for binary classification.
        """
        l = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform', activation='sigmoid')(inputs)
        return l

    # Handle layer configuration
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
    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)      
            stack = concatenate([stack, l])
            n_filters += growth_rate   

        skip_connection_list.append(stack)
        stack = TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]

    #####################
    # Bottleneck        #
    #####################     
    block_to_upsample = []
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l) 
        stack = concatenate([stack, l])
    block_to_upsample = concatenate(block_to_upsample)

    #####################
    # Upsampling path   #
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
    # Softmax Output    #
    #####################
    output = SoftmaxLayer(stack, n_classes)
    model = Model(inputs=inputs, outputs=output)

    return model


# **************** Dense-U-Net Model ****************
def DenseUnet(input_shape):
    """
    Dense U-Net model combining DenseNet and U-Net architecture.
    Takes an input shape and returns a DenseU-Net model.
    """
    
    def Conv_Block(input_tensor, filters, bottleneck=False, weight_decay=1e-4):
        """
        Convolution block with BatchNormalization, ReLU activation, and 3x3 convolution.
        """
        concat_axis = 1 if K.image_data_format() == 'channel_first' else -1 
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        return x

    def dens_block(input_tensor, nb_filter):
        """
        Dense block with 3 convolutional layers concatenated with the input.
        """
        x1 = Conv_Block(input_tensor, nb_filter)
        add1 = concatenate([x1, input_tensor], axis=-1)
        x2 = Conv_Block(add1, nb_filter)
        add2 = concatenate([x1, input_tensor, x2], axis=-1)
        x3 = Conv_Block(add2, nb_filter)
        return x3

    l1 = 4
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 7, kernel_initializer='he_normal', padding='same', strides=1, use_bias=False, kernel_regularizer=l2(1e-4))(inputs)

    # Downsampling
    down1 = dens_block(x, nb_filter=l1*2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)
    down2 = dens_block(pool1, nb_filter=l1*4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down2)
    down3 = dens_block(pool2, nb_filter=l1*8)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)
    down4 = dens_block(pool3, nb_filter=l1*16)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)

    # Bottleneck
    conv5 = Conv2D(l1*32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(l1*32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Upsampling
    up6 = UpSampling2D(size=(2, 2))(drop5)
    add6 = concatenate([down4, up6], axis=3)
    up6 = dens_block(add6, nb_filter=l1*16)
    
    up7 = UpSampling2D(size=(2, 2))(up6)
    add7 = concatenate([down3, up7], axis=3)
    up7 = dens_block(add7, nb_filter=l1*8)

    up8 = UpSampling2D(size=(2, 2))(up7)
    add8 = concatenate([down2, up8], axis=-1)
    up8 = dens_block(add8, nb_filter=l1*4)

    up9 = UpSampling2D(size=(2, 2))(up8)
    add9 = concatenate([down1, up9], axis=-1)
    up9 = dens_block(add9, nb_filter=l1*2)

    # Output layer
    conv10 = Conv2D(32, 7, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv10)
    model = Model(inputs, conv10)

    return model


# **************** ResUNet Model ****************
def ResUNet(input_shape):
    """
    ResUNet model combining residual blocks with U-Net architecture.
    Takes an input shape and returns a ResUNet model.
    """
    
    def bn_act(x, act=True):
        """
        BatchNormalization followed by ReLU activation.
        """
        x = tf.keras.layers.BatchNormalization()(x)
        if act:
            x = tf.keras.layers.Activation("relu")(x)
        return x

    def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        """
        A block with BatchNormalization, ReLU activation, and Convolution.
        """
        conv = bn_act(x)
        conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        """
        The initial convolution followed by residual addition.
        """
        conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)
        
        output = tf.keras.layers.Add()([conv, shortcut])
        return output

    def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        """
        A residual block that performs two convolutions followed by skip connections.
        """
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
        
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)
        
        output = tf.keras.layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(x, xskip):
        """
        Upsampling the input and concatenating with the skip connection.
        """
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

# ******************** SegNet Model ********************
def SegNet(input_shape):
    l1 = 16  # Base number of filters for the convolution layers

    # Encoding Layer (Downsampling Path)
    img_input = Input(shape=input_shape)  # Define the input shape for the image
    x = Conv2D(l1*4, (3, 3), padding='same', name='conv1', strides=(1,1))(img_input)
    x = Activation('relu')(x)  # Apply ReLU activation after the first convolution
    x = Conv2D(l1*4, (3, 3), padding='same', name='conv2')(x)
    x = Activation('relu')(x)  # Apply ReLU activation
    x = Dropout(0.5)(x)  # Add dropout layer to prevent overfitting
    x = MaxPooling2D()(x)  # MaxPooling layer for downsampling

    # Further convolutional blocks with increasing filter sizes
    x = Conv2D(l1*8, (3, 3), padding='same', name='conv3')(x)
    x = Activation('relu')(x)  # ReLU activation
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

    # Additional convolutional layers to further process the features
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

    # Fully connected layers
    x = Dense(l1*64, activation='relu', name='fc1')(x)
    x = Dense(l1*64, activation='relu', name='fc2')(x)

    # ******************** Decoding Layer (Upsampling Path) ********************
    # Using Conv2DTranspose and UpSampling2D to upsample and reconstruct the output
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
    x = Activation('softmax')(x)  # Output layer with softmax activation
    pred = Reshape((input_shape[0], input_shape[1], 1))(x)  # Reshape to match input shape
    
    model = Model(inputs=img_input, outputs=pred)  # Create model
    return model

# ******************** DeeplabV3 Model ********************
def DeeplabV3(input_shape, num_classes=1):
    # ******************** Convolutional Block ********************
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

    # ******************** Dilated Spatial Pyramid Pooling (DSPP) ********************
    def DilatedSpatialPyramidPooling(dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
        )(x)

        # Multiple convolutional outputs with different dilation rates
        out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])  # Concatenate outputs
        output = convolution_block(x, kernel_size=1)  # Final convolution
        return output

    # ******************** DeeplabV3 Model ********************
    model_input = keras.Input(shape=input_shape)
    resnet50 = keras.applications.resnet.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output  # Get feature map from a specific layer of ResNet50
    x = DilatedSpatialPyramidPooling(x)  # Apply Dilated Spatial Pyramid Pooling

    input_a = layers.UpSampling2D(
        size=(input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]), interpolation="bilinear"
    )(x)

    # Final output layer
    x = layers.Conv2D(
        num_classes, kernel_size=1, padding="same", kernel_initializer=keras.initializers.HeNormal()
    )(input_a)
    x = layers.Activation("softmax")(x)

    model = Model(model_input, x)
    return model

    # ******************** UNet Model ********************
    def UNet(input_shape):
        # Encoder part (downsampling)
        inputs = Input(input_shape)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bottleneck
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

        # Decoder part (upsampling)
        up1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv3)
        up1 = concatenate([up1, conv2], axis=3)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

        up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv4)
        up2 = concatenate([up2, conv1], axis=3)
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)  # Final output layer for binary segmentation

        model = Model(inputs=inputs, outputs=outputs)
        return model



# ******** unetpp_resunet ********
def unetpp_resunet(input_shape):
    # ******** resunet_block ********
    def resunet_block(x, filters, kernel_size=3):
        # Convolutional block with residual connection
        conv = layers.Conv2D(filters, kernel_size, padding='same')(x)  # First convolution
        conv = layers.BatchNormalization()(conv)  # Batch normalization
        conv = layers.Activation('relu')(conv)  # ReLU activation
        conv = layers.Conv2D(filters, kernel_size, padding='same')(conv)  # Second convolution
        conv = layers.BatchNormalization()(conv)  # Batch normalization
        conv = layers.Dropout(0.3)(conv)  # Dropout for regularization
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same')(x)  # Shortcut connection
        shortcut = layers.BatchNormalization()(shortcut)  # Batch normalization for shortcut
        res_path = layers.add([shortcut, conv])  # Residual connection
        res_path = layers.Activation('relu')(res_path)  # Apply ReLU activation after residual
        return res_path

    # ******** unetpp_block ********
    def unetpp_block(x, filters, kernel_size=3):
        # U-Net++ block without residual connections
        conv = layers.Conv2D(filters, kernel_size, padding='same')(x)  # Convolutional layer
        conv = layers.BatchNormalization()(conv)  # Batch normalization
        conv = layers.Activation('relu')(conv)  # ReLU activation
        conv = layers.Conv2D(filters, kernel_size, padding='same')(conv)  # Another convolution
        conv = layers.BatchNormalization()(conv)  # Batch normalization
        conv = layers.Activation('relu')(conv)  # ReLU activation
        conv = layers.Dropout(0.3)(conv)  # Dropout for regularization
        return conv

    inputs = tf.keras.Input(input_shape)
    L = 16  # Initial filter size
    
    # Encoder path
    e1 = resunet_block(inputs, L)  # First encoder block with residual connection
    p1 = layers.MaxPooling2D((2, 2))(e1)  # Max pooling
    
    e2 = unetpp_block(p1, L * 2)  # Second encoder block
    p2 = layers.MaxPooling2D((2, 2))(e2)  # Max pooling
    
    e3 = resunet_block(p2, L * 4)  # Third encoder block with residual connection
    p3 = layers.MaxPooling2D((2, 2))(e3)  # Max pooling
    
    # Bridge
    bridge = unetpp_block(p3, L * 8)  # U-Net++ block acting as bridge

    # Decoder path
    d3 = layers.Conv2DTranspose(L * 4, (2, 2), strides=(2, 2), padding='same')(bridge)  # Deconvolution (Upsampling)
    d3 = layers.concatenate([d3, e3])  # Skip connection from encoder
    d3 = resunet_block(d3, L * 4)  # Apply residual block

    d2 = layers.Conv2DTranspose(L * 2, (2, 2), strides=(2, 2), padding='same')(d3)  # Deconvolution
    d2 = layers.concatenate([d2, e2])  # Skip connection from encoder
    d2 = unetpp_block(d2, L * 2)  # Apply U-Net++ block
    
    d1 = layers.Conv2DTranspose(L, (2, 2), strides=(2, 2), padding='same')(d2)  # Deconvolution
    d1 = layers.concatenate([d1, e1])  # Skip connection from encoder
    d1 = resunet_block(d1, L)  # Apply residual block
    
    # Output layer for segmentation
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)  # Final segmentation output

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# ******** bpat_unet ********
def bpat_unet(input_shape):

    # ******** boundary_points_supervision_module ********
    def boundary_points_supervision_module(x, num_channels):
        # Boundary points supervision module (BPSM) with stripe pooling and attention
        def stripe_pooling(x):
            shape = tf.shape(x)
            stripe_horizontal = tf.reshape(x, (shape[0], shape[1], -1, shape[3]))  # Horizontal stripes
            stripe_vertical = tf.reshape(x, (shape[0], -1, shape[2], shape[3]))  # Vertical stripes
            return stripe_horizontal, stripe_vertical
        
        def spsa(query, key, value):
            # Scaled dot-product attention for stripe pooling
            score = tf.matmul(query, key, transpose_b=True)  # Compute dot product
            score = score / tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))  # Scale the score
            attention_weights = tf.nn.softmax(score, axis=-1)  # Softmax for attention weights
            output = tf.matmul(attention_weights, value)  # Compute weighted sum
            return output

        def ppsa(x):
            # Pooling-based self-attention
            pooled = layers.GlobalAveragePooling2D()(x)  # Global average pooling
            pooled = layers.Reshape((1, 1, num_channels))(pooled)  # Reshape for attention
            return pooled
        
        stripe_h, stripe_v = stripe_pooling(x)  # Apply stripe pooling
        query = layers.Conv2D(num_channels, 1)(stripe_h)  # Learnable query for attention
        key = layers.Conv2D(num_channels, 1)(stripe_v)  # Learnable key for attention
        value = layers.Conv2D(num_channels, 1)(stripe_h)  # Learnable value for attention
        
        spsa_output = spsa(query, key, value)  # Apply scaled dot-product attention
        
        ppsa_output = ppsa(x)  # Apply pooling-based self-attention
        
        combined = layers.Add()([spsa_output, ppsa_output])  # Combine the attention outputs
        combined = layers.Conv2D(num_channels, 1, activation='relu')(combined)  # Convolution after combination
        combined = layers.Add()([combined, x])  # Add residual connection
        
        return combined

    # ******** adaptive_multi_scale_feature_fusion_module ********
    def adaptive_multi_scale_feature_fusion_module(x_low, x_high, num_channels):
        # Multi-scale feature fusion module using deformable convolutions and attention
        def deformable_conv(x, filters):
            # Deformable convolution layer
            offset = layers.Conv2D(filters, 3, padding='same')(x)  # Learnable offsets
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)  # Convolution with ReLU activation
            return x
        
        def deformable_attention(x):
            # Deformable attention layer
            shape = tf.shape(x)
            num_heads = 8  # Number of attention heads
            depth = num_channels // num_heads  # Depth per attention head
            
            query = layers.Conv2D(num_channels, 1)(x)  # Query for attention
            key = layers.Conv2D(num_channels, 1)(x)  # Key for attention
            value = layers.Conv2D(num_channels, 1)(x)  # Value for attention
            
            query = tf.reshape(query, (shape[0], shape[1] * shape[2], num_heads, depth))  # Reshape query
            key = tf.reshape(key, (shape[0], shape[1] * shape[2], num_heads, depth))  # Reshape key
            value = tf.reshape(value, (shape[0], shape[1] * shape[2], num_heads, depth))  # Reshape value
            
            score = tf.matmul(query, key, transpose_b=True)  # Attention score
            score = score / tf.math.sqrt(tf.cast(depth, tf.float32))  # Scaling
            attention_weights = tf.nn.softmax(score, axis=-1)  # Attention weights
            output = tf.matmul(attention_weights, value)  # Attention output
            output = tf.reshape(output, shape)  # Reshape back to original dimensions
            
            return output
        
        x_low = deformable_conv(x_low, num_channels)  # Apply deformable convolution
        x_high = deformable_attention(x_high)  # Apply deformable attention
        
        combined = layers.Add()([x_low, x_high])  # Combine low-level and high-level features
        combined = layers.Conv2D(num_channels, 3, padding='same', activation='relu')(combined)  # Final convolution
        
        return combined

    # ******** assembled_transformer_module ********
    def assembled_transformer_module(x, num_channels):
        # Assembled transformer module with windowed deformable attention
        def windowed_deformable_attention(x):
            # Windowed deformable attention
            shape = tf.shape(x)
            window_size = 4  # Window size for attention
            num_heads = 8  # Number of attention heads
            depth = num_channels // num_heads  # Depth per attention head
            
            query = layers.Conv2D(num_channels, 1)(x)  # Query for attention
            key = layers.Conv2D(num_channels, 1)(x)  # Key for attention
            value = layers.Conv2D(num_channels, 1)(x)  # Value for attention
            
            query = tf.reshape(query, (shape[0], shape[1] // window_size, shape[2] // window_size, window_size * window_size, num_heads, depth))  # Reshape query
            key = tf.reshape(key, (shape[0], shape[1] // window_size, shape[2] // window_size, window_size * window_size, num_heads, depth))  # Reshape key
            value = tf.reshape(value, (shape[0], shape[1] // window_size, shape[2] // window_size, window_size * window_size, num_heads, depth))  # Reshape value
            
            score = tf.matmul(query, key, transpose_b=True)  # Attention score
            score = score / tf.math.sqrt(tf.cast(depth, tf.float32))  # Scale score
            attention_weights = tf.nn.softmax(score, axis=-1)  # Attention weights
            output = tf.matmul(attention_weights, value)  # Attention output
            output = tf.reshape(output, shape)  # Reshape back to original dimensions
            
            return output
        
        def external_attention(x):
            # External attention based on global pooling
            ea = layers.GlobalAveragePooling2D()(x)  # Global average pooling
            ea = layers.Reshape((1, 1, num_channels))(ea)  # Reshape to match input dimensions
            return ea
        
        local_features = windowed_deformable_attention(x)  # Apply local deformable attention
        global_features = external_attention(x)  # Apply external attention
        
        combined = layers.Add()([local_features, global_features])  # Combine local and global features
        combined = layers.Conv2D(num_channels, 1, activation='relu')(combined)  # Final convolution
        
        return combined

    inputs = tf.keras.Input(shape=input_shape)
    L = 16  # Initial filter size
    
    # Encoder
    conv1 = layers.Conv2D(L, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(L, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(2 * L, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(2 * L, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(4 * L, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(4 * L, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    bottleneck = layers.Conv2D(8 * L, 3, activation='relu', padding='same')(pool3)
    bottleneck = layers.Conv2D(8 * L, 3, activation='relu', padding='same')(bottleneck)
    
    # Decoder
    upconv3 = layers.Conv2DTranspose(4 * L, 3, strides=(2, 2), padding='same')(bottleneck)
    dec3 = layers.concatenate([upconv3, conv3])
    dec3 = bpat_unet(dec3, 4 * L)
    
    upconv2 = layers.Conv2DTranspose(2 * L, 3, strides=(2, 2), padding='same')(dec3)
    dec2 = layers.concatenate([upconv2, conv2])
    dec2 = bpat_unet(dec2, 2 * L)
    
    upconv1 = layers.Conv2DTranspose(L, 3, strides=(2, 2), padding='same')(dec2)
    dec1 = layers.concatenate([upconv1, conv1])
    dec1 = bpat_unet(dec1, L)

    # Output layer for segmentation
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(dec1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# ******** unetpp_att_vit ********
def unetpp_att_vit(input_shape):
    filters = 16  # Set the number of filters for the convolutional layers

    # ******** attention_block ********
    def attention_block(x, g, filters):
        # Apply 1x1 convolution to x and g to match the filter size
        theta_x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        phi_g = layers.Conv2D(filters, (1, 1), padding='same')(g)
        
        # Upsample g to match the spatial dimensions of theta_x
        upsample_g = layers.UpSampling2D(size=(theta_x.shape[1] // phi_g.shape[1], theta_x.shape[2] // phi_g.shape[2]))(phi_g)
        
        # Concatenate theta_x and the upsampled g
        concat_xg = layers.add([theta_x, upsample_g])
        act_xg = layers.Activation('relu')(concat_xg)  # Apply ReLU activation

        # Apply a 1x1 convolution and sigmoid to get the attention mask
        psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        
        # Upsample the attention mask to match the input x shape
        upsample_psi = layers.UpSampling2D(size=(x.shape[1] // sigmoid_xg.shape[1], x.shape[2] // sigmoid_xg.shape[2]))(sigmoid_xg)
        upsample_psi = layers.Reshape((x.shape[1], x.shape[2], 1))(upsample_psi)
        
        # Multiply the attention mask with x
        y = layers.multiply([upsample_psi, x])
        result = layers.Conv2D(filters, (1, 1), padding='same')(y)  # Convolve again to get the final output
        result_bn = layers.BatchNormalization()(result)  # Apply batch normalization
        return result_bn

    # ******** unetpp_block ********
    def unetpp_block(x, filters):
        # Encoder path: apply convolutions, ReLU activations, and max-pooling layers
        c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)  # Max pool to downsample

        c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)  # Max pool to downsample

        c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)  # Max pool to downsample

        c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)  # Max pool to downsample

        c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(c5)

        # Decoder path: apply transpose convolutions and concatenate with encoder outputs
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

        return c9  # Final output of the U-Net++ block

    # ******** vit_block ********
    def vit_block(x, filters, output_shape):
        # Apply convolution layers followed by a flattening layer
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.Flatten()(x)  # Flatten for transformer input

        # Calculate the size for the transformer input
        size = (output_shape[0] // 16) * (output_shape[1] // 16) * filters
        x = layers.Dense(size, activation='relu')(x)
        x = layers.Reshape((output_shape[0] // 16, output_shape[1] // 16, filters))(x)  # Reshape to 2D feature map
        x = layers.UpSampling2D((16, 16))(x)  # Upsample back to original size
        return x  # Return the processed features after ViT block
    
    # Main input layer
    inputs = layers.Input(input_shape)
    
    # U-Net++ Block
    unetpp = unetpp_block(inputs, filters)
    
    # Attention Block
    g = layers.Conv2D(filters, (1, 1), padding='same')(unetpp)  # Create the gating signal for attention
    att_block = attention_block(unetpp, g, filters)
    
    # Vision Transformer Block
    vit = vit_block(att_block, filters, input_shape)
    
    # Output layer with sigmoid activation for binary classification (segmentation)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(vit)

    # Create and return the final model
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# ******** transunet_att_res ********
def transunet_att_res(input_shape=(256, 256, 3)):
    filters = 16

    # ******** residual_block ********
    def residual_block(x, filters):
        res = x  # Store the input for the residual connection
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)  # Apply first convolution
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)  # Apply ReLU activation
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)  # Apply second convolution
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, res])  # Add the input to the output (residual connection)
        x = layers.Activation('relu')(x)  # Apply ReLU activation again
        return x

    # ******** attention_block ********
    def attention_block(x, g, filters):
        # Apply 2x2 convolution to x and 1x1 convolution to g to match filter size
        theta_x = layers.Conv2D(filters, (2, 2), strides=(2, 2), padding='same')(x)
        phi_g = layers.Conv2D(filters, (1, 1), padding='same')(g)
        
        # Apply 2x2 transposed convolution to phi_g to upsample
        upsample_g = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(phi_g)
        
        # Perform the attention mechanism
        concat_xg = layers.add([theta_x, upsample_g])
        act_xg = layers.Activation('relu')(concat_xg)
        psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        
        # Apply the attention mask to x
        upsample_psi = layers.UpSampling2D(size=(x.shape[1] // sigmoid_xg.shape[1], x.shape[2] // sigmoid_xg.shape[2]))(sigmoid_xg)
        upsample_psi = layers.Reshape((x.shape[1], x.shape[2], 1))(upsample_psi)
        y = layers.multiply([upsample_psi, x])
        result = layers.Conv2D(filters, (1, 1), padding='same')(y)
        result_bn = layers.BatchNormalization()(result)
        return result_bn

    # ******** transunet_att_res_block ********
    def transunet_att_res_block(x, filters):
        x = residual_block(x, filters)  # Apply residual block
        g = layers.Conv2D(filters, (1, 1), padding='same')(x)  # Create a gating signal for attention
        x = attention_block(x, g, filters)  # Apply attention block
        return x
    
    # Main input layer
    inputs = layers.Input(input_shape)

    # Apply the transunet-att-res block
    x = transunet_att_res_block(inputs, filters)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    # Create and return the final model
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# ******** transformer_segmentation_model ********
def transformer_segmentation_model(input_shape, num_classes=1, num_heads=4, ff_dim=128, num_transformer_blocks=4, patch_size=16):
    # ******** transformer_block ********
    def transformer_block(x, num_heads, ff_dim, dropout_rate=0.1):
        # Layer normalization and multi-head self-attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)  # Apply layer normalization to the input
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x1, x1)  # Multi-head attention mechanism
        x2 = layers.Add()([x, attn_output])  # Skip connection (Residual connection) for attention output
        
        # Feed forward network
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)  # Apply normalization after the attention block
        ff_output = tf.keras.Sequential([  # Feed-forward network applied after attention
            layers.Dense(ff_dim, activation='relu'),  # First fully connected layer with ReLU
            layers.Dense(x.shape[-1])  # Second fully connected layer to restore the dimensionality
        ])(x3)
        
        # Skip connection for the feed-forward network output
        return layers.Add()([x2, ff_output])  # Add the feed-forward output to the residual connection

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Extract patches using a convolutional layer (patching technique)
    projection_dim = 64  # Set the dimensionality for projecting patches
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)  # Apply a convolution to extract patches
    
    # Calculate output dimensions for the patches
    output_height = input_shape[0] // patch_size  # Calculate the height of the patches after division
    output_width = input_shape[1] // patch_size   # Calculate the width of the patches after division
    num_patches = output_height * output_width    # Total number of patches

    # Reshape the patches for transformer input: (batch_size, num_patches, projection_dim)
    patches = layers.Reshape((-1, projection_dim))(patches)  # Flatten the patches to match transformer input

    # Transformer blocks stack
    x = patches  # Start with the reshaped patches
    for _ in range(num_transformer_blocks):  # Iterate to stack multiple transformer blocks
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)  # Apply transformer block

    # Reshape the output of transformer blocks
    x = layers.Reshape((output_height * output_width, projection_dim))(x)  # Correct shape: (batch_size, num_patches, projection_dim)

    # Project back to original patch size
    x = layers.Dense(output_height * output_width * projection_dim, activation='relu')(x)  # Use a Dense layer to project back to the original patch size

    # Reshape to (None, output_height, output_width, projection_dim)
    x = layers.Reshape((output_height, output_width, projection_dim))(x)  # Reshape into the final feature map

    # Final upsampling to match input size and generate segmentation map
    x = layers.Conv2DTranspose(num_classes, kernel_size=2, strides=2, padding='same')(x)  # Upsample the features to the original input size
    x = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid')(x)  # Final convolution to produce the segmentation map

    # Create and return the final model
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
