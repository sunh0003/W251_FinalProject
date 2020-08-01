import numpy as np
import cv2

index = 0
path = '/home/lingyao/w251/final_project/confused/test/'

# Use the web camera to capture frame.
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame.
    ret, frame = cap.read()
    # Throw out the color information and get the gray frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection.
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        name = 'lm_%s.png'%str(index)
        cv2.imwrite(path+name, face)
   
        index += 1

        cv2.waitKey(1)
        break
        
cap.release()
cv2.destroyAllWindows()
