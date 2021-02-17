from tensorflow import keras
import numpy as np

from nets.allnet import get_model,cls_loss,smooth_l1,class_loss_cls,class_loss_regr
from util.generator import Generator
from util.anchors import get_anchors
from util.boxutil import BBoxUtility
from util.roi_helpers import calc_iou

if __name__ == "__main__":
	#图片分类
	num_classes = 21
	epoch = 100
	Learning_rate=1e-3
	num_rois = 128

	#获取模型
	model_rpn, model_classifier, model_all = get_model(num_classes)
	model_all.summary()

	#迁移学习，加载已经训练好的模型数据
	# model_rpn.load_weights("model_data/model_weights.h5", by_name=True)
	# model_classifier.load_weights("model_data/model_weights.h5",by_name=True)

	model_rpn.compile(loss={"regression": smooth_l1(), 
		"classification":cls_loss()},optimizer=keras.optimizers.Adam(lr=Learning_rate))


	model_classifier.compile(loss=[
		class_loss_cls, 
		class_loss_regr(num_classes-1)
		], 
		metrics={'dense_class_{}'.format(num_classes): 'accuracy'},optimizer=keras.optimizers.Adam(lr=Learning_rate)
	)
	model_all.compile(optimizer='sgd', loss='mae')


	with open('./train.txt') as f:
		lines = f.readlines()

	gen = Generator()
	rpn_train = gen.gen(lines, (36,36))

	bbox_util = BBoxUtility()

	rpn_accuracy_rpn_monitor = []
	rpn_accuracy_for_epoch = [] 
	for i in range(epoch):
		for iteration, batch in enumerate(rpn_train):
			X, Y, boxes = batch[0],batch[1],batch[2]

			loss_rpn = model_rpn.train_on_batch(X,Y)

			P_rpn = model_rpn.predict_on_batch(X)


			height,width,_ = np.shape(X[0])
			anchors = get_anchors((38,38), width, height)

			results = bbox_util.detection_out(P_rpn, anchors, 1, confidence_threshold=0)

			R = results[0][:, 2:]

			X2, Y1, Y2, IouS = calc_iou(R, boxes[0], width, height, num_classes)

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue
			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []

			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			if len(neg_samples)==0:
				continue

			if len(pos_samples) < num_rois//2:
				selected_pos_samples = pos_samples.tolist()
			else:
				selected_pos_samples = np.random.choice(pos_samples, num_rois//2, replace=False).tolist()
			try:
				selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples), replace=False).tolist()
			except:
				selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples), replace=True).tolist()
			sel_samples = selected_pos_samples + selected_neg_samples
			#训练分类框
			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])


