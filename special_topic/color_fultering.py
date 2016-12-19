# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:02:52 2016

@author: user
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)


import numpy as np 
import argparse
#import imutils
import time
import cv2
from imutils.object_detection import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",default="D://senior/CCL/video/April30/April30_2sentence1.mpg", help="Path to the file")
ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#load the features
row = np.load('D://senior/CCL/{April30_2sentence1.mpg}_out_features.npy')
[width, length] = row.shape
width = int(width)
length = int(length)


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# Take first frame and find corners in it
(ret, old_frame) = camera.read()
r = 400.0 / old_frame.shape[1]
dim = (400, int(old_frame.shape[0] * r))
resized_old_frame = cv2.resize(old_frame, dim, interpolation = cv2.INTER_AREA)
resized_old_gray = cv2.cvtColor(resized_old_frame, cv2.COLOR_BGR2GRAY)
(rects_old, weights_old) = hog.detectMultiScale(resized_old_frame, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

#mask1 = np.zeros((resized_old_frame.shape[0],resized_old_frame.shape[1],resized_old_frame.shape[2]))


#plot first frame to check rectangle
while(camera.isOpened()):
    hsv = cv2.cvtColor(resized_old_frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,45])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    res = cv2.bitwise_and(resized_old_frame,resized_old_frame, mask= mask)
    cv2.imshow('frame',resized_old_frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
#    for i in range(resized_old_frame.shape[2]):
#        for j in range(mask.shape[0]):
#            for k in range(mask.shape[1]):
#                if mask[j,k] ==0:
#                    mask1[j,k,i] = resized_old_frame[j,k,i] * mask[j,k]
#                else:
#                    mask1[j,k,i] = resized_old_frame[j,k,i]
#    cv2.imshow('mask1',mask1)
    res1 = np.zeros((res.shape[0],res.shape[1],res.shape[2]))
    res1 = res
    res1 = res[res==0]=255
    cv2.imshow('res1',np.uint8(res1))
                    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

#while(1):
#    _, frame = cap.read()
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    
#    lower_red = np.array([0,0,0])
#    upper_red = np.array([0,0,0])
#    
#    mask = cv2.inRange(hsv, lower_red, upper_red)
#    res = cv2.bitwise_and(frame,frame, mask= mask)
#
#    cv2.imshow('frame',frame)
#    cv2.imshow('mask',mask)
#    cv2.imshow('res',res)
#    
#    k = cv2.waitKey(5) & 0xFF
#    if k == ord("q"):
#        break
#
#cv2.destroyAllWindows()
#cap.release()