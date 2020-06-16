# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:23:23 2020

@author: shalom
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
face_cascade=cv2.CascadeClassifier('C:/Users/shalo/Desktop/ML stuffs/DL/haarcascade_frontalface_default.xml')
#loading the model
classifier = load_model('C:/Users/shalo/Desktop/ML stuffs/projects/corona/maskNOmask_classifier_checkpoint95%.h5')


vid = cv2.VideoCapture(0)# To capture video from webcam.

iter1=0
nam  = input('Enter Project Name : ')
path23='C:/Users/shalo/Desktop/ML stuffs/DL/os check/%s'%(nam)
os.mkdir(path23)
print(path23)
print("Enter 'q' to turn the camera off")
from tensorflow.keras.preprocessing import image
import numpy as np
while True:
    
    r,frame = vid.read();#reading the input from webcam
    frame = cv2.resize(frame,(800,600))#resizing the image
    im1 = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)#converting to grey scale
    face=face_cascade.detectMultiScale(im1,1.1,4)#It takes 3 arguments — the input image, scaleFactor and minNeighbours
    # Draw the rectangle around each face
    for x,y,w,h in (face):
        cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],4)#arguments respectively input image,start_point, end_point, color, thickness
        iter1=iter1+1
        im_f = im1[y:y+h,x:x+w]#to capture from each faces 
        
        path2 = '%s/%d.png'%(path23,iter1)
        cv2.imwrite(path2,im_f)#saving the image
        
        im_f=path2#reading the path of saved image
        img2 = image.load_img(im_f, target_size=(64, 64))#loading the captured image
        img = image.img_to_array(img2)#converting iamge to array
        img = img/255
        img = np.expand_dims(img, axis=0)
        prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
        prediction1=np.ceil(prediction[:,:]*100)#rounding off the probability
        
        prediction2='pred='+str(prediction1)+("%")

        
        if(prediction[:,:]>0.80):
            cv2.putText(frame,'NOT WEARING MASK'+str(prediction2),(x,y), cv2.FONT_ITALIC, 1,
                    (255,0,255),2,cv2.LINE_AA)


        else:
            cv2.putText(frame,'WEARING MASK'+str(prediction2),(x,y), cv2.FONT_ITALIC, 1,
                    (255,0,255),2,cv2.LINE_AA)
    
    cv2.imshow('Video',frame) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.


vid.release()# Release the VideoCapture object
cv2.destroyAllWindows()