from mss import mss
from PIL import Image
import cv2 as cv
import numpy as np
import time
import os


def capture_screenshot():
    with mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')


# EMOTION_DICT = {1: 'not confused', 0:'confused'}
base_path='/Users/TraceyTan/Documents/W251-Scalingup/FINAL_TEST_T_M_L2/'


for i in range(100):
	time.sleep(1)
	imgid = 0
	
	screen_img = capture_screenshot()
	screen_img.save(base_path +'screen.jpg')
	ts = str(time.time())


	# Load XML classifier
	print('Load Face Classifer')
	face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

	n_img = cv.imread(base_path +'screen.jpg')
	faces = face_cascade.detectMultiScale(n_img, scaleFactor=1.3, minNeighbors=5)
	face_count = len(faces)
	print("Found " + str(face_count) + " in this frame.")
	imgid += 1

	face_crop = []
	for f in faces:
	    x, y, w, h = [ v for v in f ]
	    cv.rectangle(n_img, (x,y), (x+w, y+h), (255,0,0), 3)
	    # Define the region of interest in the image  
	    face_crop.append(n_img[y:y+h, x:x+w])

	for face in face_crop:
	    name = base_path+ "face_" + ts + "_" + str(imgid) + ".jpg"
	    cv.waitKey(0)
	    cv.resize(face, (224,224))
	    cv.imwrite(name,face)
	    imgid +=1



# emotionCount =[]
# model_top = load_model('2clsmodel_highres_conf_happy_addimg.h5')

# for file in os.listdir(base_path):
#     read_image = cv.imread(base_path+file)
#     read_image = read_image.reshape(1,read_image.shape[0], read_image.shape[1], 3)
#     top_pred = model_top.predict(read_image)
#     emotion_label = top_pred[0].argmax()
#     print(file, "   ", emotion_label, EMOTION_DICT[emotion_label], "   ", top_pred)
#     emotionCount.append(EMOTION_DICT[emotion_label])

# print(Counter(emotionCount))