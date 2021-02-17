import numpy as np
import tensorflow as tf

class BBoxUtility(object):
	def __init__(self):
		self.overlap_threshold = 0.7
		self.ignore_threshold = 0.3
		self._nms_thresh = 0.7
		self._top_k = 300

	def iou(self, box, priors):
		'''
		box: 真实框
		priors: 先验锚框
		'''
		#获取真实宽的内部框
		inter_upleft = np.maximum(priors[:, :2], box[:2])
		inter_botright = np.minimum(priors[:, 2:], box[2:])

		inter_wh = inter_botright - inter_upleft
		inter_wh = np.maximum(inter_wh, 0)

		#公共区域的面积
		inter = inter_wh[:, 0] * inter_wh[:, 1]
		#真实框面积
		area_true = (box[2] - box[0]) * (box[3] - box[1])
		#锚框面积
		area_gt = (priors[:, 2] - priors[:, 0])*(priors[:, 3] - priors[:, 1])
		#真实宽和锚框总面积
		union = area_true + area_gt - inter
		iou = inter/union
		return iou

	def ignore_box(self, box, priors):
		'''
		过滤无效锚框
		box 真实框
		priors 锚框
		'''
		iou = self.iou(box, priors)

		ignored_box = np.zeros((len(priors), 1))

		#求交集>0.3 <0.7 
		assign_mask = (iou > self.ignore_threshold)&(iou<self.overlap_threshold)

		#如果没有任何满足求交集的条件
		if not assign_mask.any():
			assign_mask[iou.argmax()] = True

		#取过滤的锚框
		ignored_box[:, 0][assign_mask] = iou[assign_mask]

		#全部张开
		return ignored_box.ravel()

	def encode_box(self, box, priors, return_iou=True):
		iou = self.iou(box, priors)

		encoded_box = np.zeros((len(priors), 4 + return_iou))

		#证类标记
		assign_mask = iou > self.overlap_threshold
		if not assign_mask.any():
			assign_mask[iou.argmax()] = True

		if return_iou:
			encoded_box[:, -1][assign_mask] = iou[assign_mask]

		assigned_priors = priors[assign_mask]

		box_center = 0.5 * (box[:2] + box[2:])
		box_wh = box[2:] - box[:2]

		assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:4])

		assigned_priors_wh = (assigned_priors[:, 2:4] - assigned_priors[:, :2])

		encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
		encoded_box[:, :2][assign_mask] /= assigned_priors_wh
		encoded_box[:, :2][assign_mask] *= 4

		encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
		encoded_box[:, 2:4][assign_mask] *= 4
		return encoded_box.ravel()		



	def assign_boxes(self, boxes, anchors):
		'''
		boxes: 真实框
		anchors: 生成的锚框
		'''
		num_priors = len(anchors)
		priors = anchors

		assignment = np.zeros((num_priors, 4+1))
		assignment[:, 4] = 0.0
		if len(boxes) == 0:
			return assignment

		#需要过滤的锚框
		ingored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4], priors)
		ingored_boxes = ingored_boxes.reshape(-1, num_priors, 1)
		ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
		ignore_iou_mask = ignore_iou > 0
		#标记需要过滤的锚框为-1
		assignment[:, 4][ignore_iou_mask] = -1

		encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4], priors)
		encoded_boxes = encoded_boxes.reshape(-1, num_priors, 5)
		best_iou = encoded_boxes[:, :, -1].max(axis=0)
		best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
		best_iou_mask = best_iou > 0
		best_iou_idx = best_iou_idx[best_iou_mask]
		assign_num = len(best_iou_idx)
		encoded_boxes = encoded_boxes[:, best_iou_mask, :]
		assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
		#正向锚框标记为1
		assignment[:, 4][best_iou_mask] = 1
		return assignment


	def decode_boxes(self, mbox_loc, mbox_priorbox):
		'''
		获取锚框
		mbox_loc：预测框
		mbox_priorbox: 原始锚框
		'''
		prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
		prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
		prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
		prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

		decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
		decode_bbox_center_x += prior_center_x
		decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
		decode_bbox_center_y += prior_center_y
        
		decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
		decode_bbox_width *= prior_width
		decode_bbox_height = np.exp(mbox_loc[:, 3] /4)
		decode_bbox_height *= prior_height

		decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
		decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
		decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
		decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

		decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
									  decode_bbox_ymin[:, None],
									  decode_bbox_xmax[:, None],
									  decode_bbox_ymax[:, None]), axis=-1)
		decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
		return decode_bbox

	def detection_out(self, predictions, mbox_priorbox, num_classes,confidence_threshold=0.5):
		'''
		获取锚框
		predictions：rpn网络预测值
		mbox_priorbox: 原始锚框
		num_classes： 锚框分类
		confidence_threshold：
		'''
		mbox_conf = predictions[0]	#（-1，1）框分类
		mbox_loc = predictions[1]	#（-1，4）框坐标

		mbox_priorbox = mbox_priorbox
		results = []
		for i in range(len(mbox_loc)):
			results.append([])
			#真实框上的锚框
			decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)
			for c in range(num_classes):
				c_confs = mbox_conf[i, :, c]
				#正向框
				c_confs_m = c_confs > confidence_threshold

				if len(c_confs[c_confs_m]) > 0:
					boxes_to_process = decode_bbox[c_confs_m]
					confs_to_process = c_confs[c_confs_m]
					idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process,self._top_k, iou_threshold=self._nms_thresh).numpy()

					good_boxes = boxes_to_process[idx]
					confs = confs_to_process[idx][:, None]

					labels = c * np.ones((len(idx), 1))
					c_pred = np.concatenate((labels, confs, good_boxes),axis=1)

					results[-1].extend(c_pred)

			if len(results[-1]) > 0:
				results[-1] = np.array(results[-1])
				argsort = np.argsort(results[-1][:, 1])[::-1]
				results[-1] = results[-1][argsort]
		return results





