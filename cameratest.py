import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf


mod = load_model('plastic_model.h5')

myvid = cv2.VideoCapture(0)

while True:
	_,frame = myvid.read()

	if frame is not None:
		im = Image.fromarray(frame,'RGB')
		im = cv2.resize(frame,(150,150))

	else:
		break

	img_array = np.array(im)
	img_array = np.expand_dims(img_array, axis=0)

	prediction = mod.predict(img_array, batch_size = 100)

	if prediction[0][0]==1:
		name = 'HDPE'
	if prediction[0][3]==1:
		name = 'PP'
	if prediction[0][2]==1:
		name = 'PET'
	if prediction[0][1]==1:
		name = 'LDPE'
	else:
		name = 'Unknown'

	print(name)
	print('test')
	videotoshow = cv2.resize(frame,(800,400))
	cv2.imshow('Vid',videotoshow)
	key = cv2.waitKey(1)
	if key == ord('q'):
		break

myvid.release()
cv2.destroyAllWindows()