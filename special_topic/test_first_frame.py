# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:40:25 2016

@author: user
"""

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

 # params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors to draw line
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
(ret, old_frame) = camera.read()
r = 400.0 / old_frame.shape[1]
dim = (400, int(old_frame.shape[0] * r))
resized_old_frame = cv2.resize(old_frame, dim, interpolation = cv2.INTER_AREA)
resized_old_gray = cv2.cvtColor(resized_old_frame, cv2.COLOR_BGR2GRAY)
(rects_old, weights_old) = hog.detectMultiScale(resized_old_frame, winStride=(4, 4),
		padding=(8, 8), scale=1.05)



#plot first frame to check rectangle
while(camera.isOpened()):
    cv2.rectangle(resized_old_frame,(148,55),(234,227),(0,255,0),2)
    cv2.rectangle(resized_old_frame,(151,90),(218,223),(0,0,255),2)
    cv2.rectangle(resized_old_frame,(193,69),(270,223),(255,0,0),2)
    cv2.imshow('test',resized_old_frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()