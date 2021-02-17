import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,TimeDistributed,Add,BatchNormalization
from tensorflow.keras.layers import Activation,Flatten,Reshape
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from nets.RoiPoolingConv import RoiPoolingConv
from nets.classifierNet import classifier_layers
from nets.resnet import resNet50

# from RoiPoolingConv import RoiPoolingConv
# from classifierNet import classifier_layers
# from resnet import resNet50


def rpnnet(feature_map, num_anchors):
	'''
	feature_map: resnet50后的网络层
	num_anchors: 锚框数量
	'''
	x = Conv2D(512, (3,3), padding="same", activation='relu', 
		kernel_initializer='normal', name='rpn_conv1')(feature_map)

	#分类
	x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
	#锚框回归
	x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

	x_class = Reshape((-1, 1), name="classification")(x_class)
	x_regr = Reshape((-1, 4), name="regression")(x_regr)

	return [x_class, x_regr, feature_map]


def get_classifier(feature_map, rois_input, rois_num, nb_class, trainable=False):
	'''
	feature_map: resnet50后的网络层
	rois_input: 锚框输入(None, None, 4) 标示多个图片中的多个锚框
	rois_num： 锚框数量
	nb_class：分类数量
	'''
	pooling_regions = 14
	input_shape = (rois_num, 14, 14, 1024)

	#roipool 特征提取
	out_roi_pool = RoiPoolingConv(pooling_regions, rois_num)([feature_map, rois_input])

	#最终分类层
	out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
	out = TimeDistributed(Flatten())(out)
	#分类层
	out_class = TimeDistributed(Dense(nb_class, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_class))(out)
	#回归层
	out_regr = TimeDistributed(Dense(4 * (nb_class-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_class))(out)

	return [out_class, out_regr]

def get_model(num_classes):
	'''
	num_classes：锚框标出的分类总数
	'''
	#原始图片输入
	inputs = Input(shape=(None, None, 3))
	#图片锚框输入
	roi_input = Input(shape=(None, 4))
	#restnet50 得到特征图
	feature_map = resNet50(inputs)
	#每个特征点锚框数量
	num_anchors = 9
	#rpn网络
	rpn = rpnnet(feature_map, num_anchors)
	model_rpn =Model(inputs, rpn[:2])
	#锚框数量
	num_rois = 128
	#分类层
	classifier = get_classifier(feature_map, roi_input, num_rois, nb_class=num_classes, trainable=True)
	model_classifier = Model([inputs, roi_input], classifier)

	#汇总模型
	model_all = Model([inputs, roi_input], rpn[:2]+classifier)
	return model_rpn,model_classifier,model_all

#锚框分类（正负）损失函数
def cls_loss(ratio=3):
	def _cls_loss(y_true, y_pred):

		labels= y_true
		anchor_state   = y_true[:,:,-1] 
		classification = y_pred

		#正向锚框
		indices_for_object = tf.where(keras.backend.equal(anchor_state, 1))
		labels_for_object = tf.gather_nd(labels, indices_for_object)
		classification_for_object = tf.gather_nd(classification, indices_for_object)

		#二进制交叉熵损失函数，用户解决二分类问题
		cls_loss_for_object = keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

		#负向锚框
		indices_for_back= tf.where(keras.backend.equal(anchor_state, 0))
		labels_for_back = tf.gather_nd(labels, indices_for_back)
		classification_for_back = tf.gather_nd(classification, indices_for_back)

		cls_loss_for_back = keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

		normalizer_pos = tf.where(keras.backend.equal(anchor_state, 1))
		normalizer_pos = keras.backend.cast(keras.backend.shape(normalizer_pos)[0], keras.backend.floatx())
		normalizer_pos = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_pos)

		normalizer_neg = tf.where(keras.backend.equal(anchor_state, 0))
		normalizer_neg = keras.backend.cast(keras.backend.shape(normalizer_neg)[0], keras.backend.floatx())
		normalizer_neg = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_neg)

		cls_loss_for_object = keras.backend.sum(cls_loss_for_object)/normalizer_pos
		cls_loss_for_back = ratio*keras.backend.sum(cls_loss_for_back)/normalizer_neg


		loss = cls_loss_for_object + cls_loss_for_back

		return loss
	return _cls_loss

#l1 = 0.5x^2, |x|<1
#    = |x| - 0.5, |x|>1
#回归损失函数
def smooth_l1(sigma=1.0):
	sigma_squared = sigma ** 2

	def _smooth_l1(y_true, y_pred):

		regression= y_pred
		regression_target = y_true[:, :, :-1]
		anchor_state = y_true[:, :, -1]

		#正向锚框
		indices = tf.where(keras.backend.equal(anchor_state, 1))
		regression = tf.gather_nd(regression, indices)
		regression_target = tf.gather_nd(regression_target, indices)

		regression_diff = regression - regression_target
		regression_diff = keras.backend.abs(regression_diff)
		regression_loss = tf.where(
			keras.backend.less(regression_diff, 1.0 / sigma_squared),
			0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
			regression_diff - 0.5 / sigma_squared
		)

		normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
		normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
		loss = keras.backend.sum(regression_loss) / normalizer

		return loss

	return _smooth_l1

#锚框边损失函数
def class_loss_regr(num_classes):
	epsilon = 1e-4
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		loss = 4*K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
		return loss
	return class_loss_regr_fixed_num

#分类损失函数
def class_loss_cls(y_true, y_pred):
	return K.mean(K.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

if __name__ == '__main__':
	model_rpn,model_classifier,model_all = get_model(21)
	model_all.summary()



