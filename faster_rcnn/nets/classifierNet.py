from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,TimeDistributed,Add,BatchNormalization
from tensorflow.keras.layers import Activation,Flatten

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2,2), trainable=True):
	nb_filter1, nb_filter2, nb_filter3 = filters

	bn_axis = 3

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	#共享多锚框的参数
	x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
	x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(nb_filter2, (kernel_size,kernel_size), trainable=trainable, kernel_initializer='normal', padding='same'), name=conv_name_base + '2b')(x)
	x = TimeDistributed(BatchNormalization(name=bn_name_base+'2b', axis=bn_axis))(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
	x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

	shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
	shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

	x = Add()([x, shortcut])
	x = Activation('relu')(x)
	return x

def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
	nb_filter1, nb_filter2, nb_filter3 = filters
	bn_axis = 3

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
	x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
	x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
	x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)

	return x
def classifier_layers(x, input_shape, trainable=False):
	x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
	x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
	x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
	x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

	return x

