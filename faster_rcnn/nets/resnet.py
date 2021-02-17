from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,TimeDistributed,Add,BatchNormalization
from tensorflow.keras.layers import Activation,Flatten

#https://blog.csdn.net/strive_for_future/article/details/108375682


def conv_block(input_tensor, kernel_size, filters, stage,block,strides=(2, 2)):
	'''
	input_tensor: 输入的张量
	kernel_size: 卷积核大小
	filters: [],3个卷机通道
	block: 卷积命名
	stage: 阶段命名
	strides: 卷积步长
	'''
	filter1,filter2,filter3 = filters

	conv_name_base = "res"+str(stage)+block+'_branch'
	bn_name_base = 'bn' + str(stage)+block+'_branch'

	x = Conv2D(filter1, (1,1), strides=strides, name=conv_name_base+"2a")(input_tensor)
	x = BatchNormalization(name=bn_name_base+"2a")(x)
	x = Activation('relu')(x)

	x = Conv2D(filter2, kernel_size, padding='same',name=conv_name_base+"2b")(x)
	x = BatchNormalization(name=bn_name_base+"2b")(x)
	x = Activation('relu')(x)

	x = Conv2D(filter3, (1,1),name=conv_name_base+"2c")(x)
	x = BatchNormalization(name=bn_name_base+"2c")(x)

	shortcut = Conv2D(filter3, (1,1), strides=strides, name=conv_name_base+"2d")(input_tensor)
	shortcut = BatchNormalization(name=bn_name_base+"2d")(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
	'''
	input_tensor: 输入的张量
	kernel_size: 卷积核大小
	filters: [],3个卷机通道
	block: 卷积命名
	stage: 阶段命名
	'''
	filter1,filter2,filter3 = filters

	conv_name_base = "res"+str(stage)+block+'_branch'
	bn_name_base = 'bn' + str(stage)+block+'_branch'

	x = Conv2D(filter1, (1,1), name=conv_name_base+"2a")(input_tensor)
	x = BatchNormalization(name=bn_name_base+"2a")(x)
	x = Activation('relu')(x)

	x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base+"2b")(x)
	x = BatchNormalization(name=bn_name_base+"2b")(x)
	x = Activation('relu')(x)	

	x = Conv2D(filter3, (1,1), name=conv_name_base+"2c")(x)
	x = BatchNormalization(name=bn_name_base+"2c")(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)
	return x

def resNet50(inputs):
	'''
	inputs:(None, None, None, 3) 3通道图片
	'''
	img = inputs
	#上下左右填充3个0
	x = ZeroPadding2D((3,3))(img)
	#卷积
	x = Conv2D(64, (7,7), strides=(2, 2), name='conv1')(x)
	#BN
	x = BatchNormalization(name="bn_conv1")(x)
	#激活函数
	x = Activation("relu")(x)

	x = MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

	x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1,1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')


	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

	return x



if __name__ == '__main__':
	input_tensor = Input(shape=(600, 600, 3))
	out_put = resNet50(input_tensor)
	model = keras.Model(inputs=input_tensor, outputs=out_put)
	model.summary()

	

