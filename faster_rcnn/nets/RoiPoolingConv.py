from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf

class RoiPoolingConv(Layer):
	def __init__(self, pool_size, num_rois, **kwargs):		
		'''
		https://www.sohu.com/a/414474326_823210
		pool_size: roi框池化后的大小
		num_rois：定义一个batch size中 最大锚框数量
		'''
		self.pool_size = pool_size
		self.num_rois = num_rois

		super(RoiPoolingConv, self).__init__(**kwargs)

	#一般用于初始化层内的参数和变量。在调用call()方法前，类会自动调用该方法。	
	# 在该方法末尾需要设置self.built = True，保证build()方法只被调用一次	
	def build(self, input_shape):
		#feature map的通道数，这里是1024
		self.nb_channels = input_shape[0][3]


	def compute_output_shape(self, input_shape):
		return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

	def call(self, x):
		assert(len(x)==2)
		#feature map
		img = x[0]
		#锚框坐标
		rois = x[1]

		outputs = []

		for roi_idx in range(self.num_rois):
			x = rois[0, roi_idx, 0]
			y = rois[0, roi_idx, 1]
			w = rois[0, roi_idx, 2]
			h = rois[0, roi_idx, 3]

			x = K.cast(x, 'int32')
			y = K.cast(y, 'int32')
			w = K.cast(w, 'int32')
			h = K.cast(h, 'int32')

			#在feature map中，取出roi, 并缩放到14*14
			rs = tf.compat.v1.image.resize_images(img[:,y:y+h,x:x+w,:], (self.pool_size, self.pool_size))
			outputs.append(rs)
		#类似，tf.concat()，这里转化为tensor
		final_output = K.concatenate(outputs, axis=0)
		final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

		#permute:置换、dimensions：尺寸
		final_output = K.permute_dimensions(final_output, (0,1,2,3,4))
		return final_output


