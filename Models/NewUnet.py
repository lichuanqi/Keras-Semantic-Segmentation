#coding=utf-8
import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.layers import LeakyReLU


def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    # out = Activation('relu')(out)
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = LeakyReLU(0.2)(out)
    # out = Activation('relu')(out)
    return out


def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = LeakyReLU(0.2)(out)
    # out = Activation('relu')(out)
    return out


def channel_attention(input_feature, ratio=8):
    '''
    通道注意力机制
    '''
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    # 折扣率，用来节省开销
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    '''
    空间注意力机制
    '''
    kernel_size = 7
    if K.image_data_format() == "channels_first":
    	channel = input_feature._keras_shape[1]
    	cbam_feature = Permute((2,3,1))(input_feature)
    else:
    	channel = input_feature._keras_shape[-1]
    	cbam_feature = input_feature
      
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    activation = 'hard_sigmoid',
                    strides=1,
                    padding='same',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1
	
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature,ratio=8):
	"""
    功能：
        Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	    As described in CBAM: Convolutional Block Attention Module.
	paper: 
        https://arxiv.org/abs/1807.06521v2
    code : 
        https://github.com/Jongchan/attention-module
    """
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature, )

	return cbam_feature


def NewUnet(nClasses, input_height=224, input_width=224):
    """
    NewUnet based UNet - Basic Implementation
    修改：
        * 激活函数
    增加：
        * 通道注意力和空间注意力
    """
    inputs = Input(shape=(input_height, input_width, 3))
    
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
    conv1 = conv_block(inputs, n1)

    conv2 = MaxPooling2D(strides=2)(conv1)
    conv2 = conv_block(conv2, filters[1])

    conv3 = MaxPooling2D(strides=2)(conv2)
    conv3 = conv_block(conv3, filters[2])

    conv4 = MaxPooling2D(strides=2)(conv3)
    conv4 = conv_block(conv4, filters[3])

    conv5 = MaxPooling2D(strides=2)(conv4)
    conv5 = conv_block(conv5, filters[4])

    d5 = up_conv(conv5, filters[3])
    conv4 = cbam_block(conv4)
    d5 = Concatenate()([conv4, d5])

    d4 = up_conv(d5, filters[2])
    conv3 = cbam_block(conv3)
    d4 = Concatenate()([conv3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    conv2 = cbam_block(conv2)
    d3 = Concatenate()([conv2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    conv1 = cbam_block(conv1)
    d2 = Concatenate()([conv1, d2])
    d2 = conv_block(d2, filters[0])

    o = Conv2D(nClasses, (3, 3), padding='same')(d2)

    outputHeight = Model(inputs, o).output_shape[1]
    outputWidth = Model(inputs, o).output_shape[2]

    out = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    out = Activation('softmax')(out)

    model = Model(inputs=inputs, outputs=out)
    model.outputHeight = outputHeight
    model.outputWidth = outputWidth

    return model