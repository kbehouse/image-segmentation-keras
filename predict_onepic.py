import argparse

import Models
from Models import VGGUnet, VGGSegnet, FCN8, FCN32

from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os



import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
'''
python predict_onepic.py --input=test.png --output=test_label.png
python predict_onepic.py --input=3obj.png --output=3obj_label.png --want_class=2  
python predict_onepic.py --input=3obj_93.png --output=3obj_93_label.png --want_class=3  
# class:1 -> CHO, class:1 -> CHO, class:2 -> fu, class:3 -> iPhone 

python predict_onepic.py --input=test/022.jpg --output=test/022_label.jpg --want_class=1
'''


class SegnetLabel:
	def __init__(self, n_classes, input_height, input_width, save_weights_path, epoch_number):

		self.limit_gpu_memory()

		modelFN = Models.VGGSegnet.VGGSegnet

		self.m = modelFN( n_classes , input_height=input_height, input_width=input_width , trainning = False  )
		weight_path = save_weights_path + "." + str(  epoch_number )
		print('Load Segnet weight_path: ' + weight_path)
		self.m.load_weights(weight_path )
		self.m.compile(loss='categorical_crossentropy',
			optimizer= 'adadelta' ,
			metrics=['accuracy'])

		self.n_classes = n_classes
		self.input_height = input_height
		self.input_width = input_width
		self.want_class = 1


	def set_want_class(self, want_class):
		self.want_class = want_class 

	def limit_gpu_memory(self):
		# ---- limit  GPU memory resource-----#
		import tensorflow as tf
		from keras.backend.tensorflow_backend import set_session
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.4
		set_session(tf.Session(config=config))

	def predict(self, inName):
		X = self.getImageArr(inName , self.input_width  , self.input_height , ordering='None' )
		pr = self.m.predict( np.array([X]) )[0]

		
		pr = pr.reshape(( self.m.outputHeight ,  self.m.outputWidth , self.n_classes ) ).argmax( axis=2 )
		# print('pr.shape = ', pr.shape)
		pr = (pr[:,:] == self.want_class).astype('uint8')
		# seg_img = np.zeros( ( self.m.outputHeight  , self.m.outputHeight  , 3  ) )
		# (pr[:,: ] >= 2
		
		return pr

	def getImageArr(self, path , width , height , imgNorm="sub_mean" , ordering='channels_first' ):

		try:
			img = cv2.imread(path, 1)

			if imgNorm == "sub_and_divide":
				img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
			elif imgNorm == "sub_mean":
				img = cv2.resize(img, ( width , height ))
				img = img.astype(np.float32)
				img[:,:,0] -= 103.939
				img[:,:,1] -= 116.779
				img[:,:,2] -= 123.68
			elif imgNorm == "divide":
				img = cv2.resize(img, ( width , height ))
				img = img.astype(np.float32)
				img = img/255.0

			if ordering == 'channels_first':
				img = np.rollaxis(img, 2, 0)

			# print('img -> ', img.shape)
			return img
		except Exception as e:
			print(path , e)
			img = np.zeros((  height , width  , 3 ))
			if ordering == 'channels_first':
				img = np.rollaxis(img, 2, 0)
			return img

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_weights_path", type = str, default='weights/3obj'  ) #default='weights/ex1'  )
	parser.add_argument("--epoch_number", type = int, default = 5 )
	parser.add_argument("--input", type = str , default = "")
	parser.add_argument("--output", type = str , default = "")
	parser.add_argument("--input_height", type=int , default = 224  )
	parser.add_argument("--input_width", type=int , default = 224 )
	parser.add_argument("--n_classes", type=int , default = 4)
	parser.add_argument("--want_class", type=int , default = 1)

	args = parser.parse_args()


	s = SegnetLabel(args.n_classes, args.input_height , args.input_width, args.save_weights_path, args.epoch_number )
	s.set_want_class(args.want_class)
	seg_img = s.predict(args.input)

	print('output seg_img.shape = ', seg_img.shape)

	cv2.imwrite(args.output , seg_img)
	cv2.imwrite(args.output +'_x255.jpg', seg_img*255.0)
	print('Save to ' +  args.output)
	print('Save to ' + args.output +'_x255.jpg  ---> for SHOW')