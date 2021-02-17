from PIL import Image,ImageDraw
import numpy as np
import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from random import shuffle
import random

from util.anchors import get_anchors
from util.boxutil import BBoxUtility


def rand(a=0, b=1):
	return np.random.rand()*(b-a) + a

class Generator(object):
	'''
	数据获取类
	'''
	def __init__(self, solid_shape=[600,600]):
		self.solid = True
		self.solid_shape=solid_shape
		self.boxutil = BBoxUtility()

	def imgShow(self, image, boxes):
		'''
		打印图片
		'''
		image = Image.fromarray(np.uint8(image))
		draw = ImageDraw.Draw(image)
		for row in boxes:
			draw.rectangle((row[0],row[1],row[2],row[3]), None, "red")
		image.show()

	def get_random_data(self, annotation_line,jitter=.3, hue=.1, sat=1.5, val=1.5):
		'''
		annotation_line: 图片及框数据
		'''
		line = annotation_line.split()
		image = Image.open(line[0])
		iw, ih = image.size
		if self.solid:
			w,h = self.solid_shape
		else:
			w, h = get_new_img_size(iw, ih)
		box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

		new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
		scale = rand(.25, 2)
		if new_ar < 1:
			nh = int(scale*h)
			nw = int(nh*new_ar)
		else:
			nw = int(scale*w)
			nh = int(nw/new_ar)
		image = image.resize((nw,nh), Image.BICUBIC)

		dx = int(rand(0, w-nw))
		dy = int(rand(0, h-nh))
		new_image = Image.new('RGB', (w,h), (128,128,128))
		new_image.paste(image, (dx, dy))
		image = new_image

		flip = rand()<.5
		if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

		# distort image
		hue = rand(-hue, hue)
		sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
		val = rand(1, val) if rand()<.5 else 1/rand(1, val)
		x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
		x[..., 0] += hue*360
		x[..., 0][x[..., 0]>1] -= 1
		x[..., 0][x[..., 0]<0] += 1
		x[..., 1] *= sat
		x[..., 2] *= val
		x[x[:,:, 0]>360, 0] = 360
		x[:, :, 1:][x[:, :, 1:]>1] = 1
		x[x<0] = 0
		image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
		box_data = np.zeros((len(box),5))
		if len(box)>0:
			np.random.shuffle(box)
			box[:, [0,2]] = box[:, [0,2]]*nw/iw+dx
			box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
			if flip: box[:, [0,2]] = w - box[:, [2,0]]
			box[:, 0:2][box[:, 0:2]<0] = 0
			box[:, 2][box[:, 2]>w] = w
			box[:, 3][box[:, 3]>h] = h
			box_w = box[:, 2] - box[:, 0]
			box_h = box[:, 3] - box[:, 1]
			box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
			box_data = np.zeros((len(box),5))
			box_data[:len(box)] = box
		if len(box) == 0:
			return image_data, []

		if (box_data[:,:4]>0).any():
			return image_data, box_data
		else:
			return image_data, []

	def gen(self, lines, feature_map_shape):
		'''
		获取图片数据
		lines: 图片数据
		feature_map_shape: 特征图大小
		'''
		while True:
			shuffle(lines)
			for annotation_line in lines:
				img,y = self.get_random_data(annotation_line)

				height, width, _ = np.shape(img)

				#无锚框
				if len(y) == 0:
					continue
				#缩放box数据
				boxes = np.array(y[:,:4],dtype=np.float32)
				boxes[:,0] = boxes[:,0]/width
				boxes[:,1] = boxes[:,1]/height
				boxes[:,2] = boxes[:,2]/width
				boxes[:,3] = boxes[:,3]/height

				#过滤无效框
				box_heights = boxes[:,3] - boxes[:,1]
				box_widths = boxes[:,2] - boxes[:,0]
				if (box_heights<=0).any() or (box_widths<=0).any():
					continue

				y[:, :4] = boxes[:, :4]

				#获取锚框
				anchors = get_anchors(feature_map_shape, width, height)
				#锚框标记
				assignment = self.boxutil.assign_boxes(y,anchors)

				num_regions = 256
				#分类
				classification = assignment[:,4]
				#锚框回归
				regression = assignment[:,:]

				mask_pos = classification[:]>0
				num_pos = len(classification[mask_pos])

				#正锚框数量大于128，则减少到最多128
				if num_pos > num_regions/2:
					val_locs = random.sample(range(num_pos), int(num_pos - num_regions/2))
					temp_classification = classification[mask_pos]
					temp_regression = regression[mask_pos]
					temp_classification[val_locs] = -1
					temp_regression[val_locs,-1] = -1
					classification[mask_pos] = temp_classification
					regression[mask_pos] = temp_regression

				#负向锚框，为0
				mask_neg = classification[:]==0
				num_neg = len(classification[mask_neg])
				mask_pos = classification[:]>0
				num_pos = len(classification[mask_pos])
				if len(classification[mask_neg]) + num_pos > num_regions:
					val_locs = random.sample(range(num_neg), int(num_neg + num_pos - num_regions))
					temp_classification = classification[mask_neg]
					temp_classification[val_locs] = -1
					classification[mask_neg] = temp_classification
				classification = np.reshape(classification,[-1,1])
				regression = np.reshape(regression,[-1,5])
				tmp_inp = np.array(img)
				tmp_targets = [np.expand_dims(np.array(classification,dtype=np.float32),0),np.expand_dims(np.array(regression,dtype=np.float32),0)]

				yield preprocess_input(np.expand_dims(tmp_inp,0)), tmp_targets, np.expand_dims(y,0)
if __name__ == '__main__':
	obj = Generator()
	annotation_line = "../../VOCdevkit/VOC2007/JPEGImages/000007.jpg 141,50,500,330,6"
	img,box = obj.get_random_data(annotation_line)
	obj.imgShow(img, box)



