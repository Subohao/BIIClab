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

#person a
a_left = rects_old[0,0]
a_top = rects_old[0,1]
a_right = rects_old[0,0] + rects_old[0,2]
a_bottom = rects_old[0,1] + rects_old[0,3]
#find good features in the person a box
p0_a = cv2.goodFeaturesToTrack(resized_old_gray[a_left:a_right+1,a_top:a_bottom+1], mask= None , **feature_params)
[p0_a_dim,p0_a_row,p0_a_col] = p0_a.shape
for i in range(p0_a_dim):
    p0_a[i,0,0] = p0_a[i,0,0]+a_left
    p0_a[i,0,1] = p0_a[i,0,1]+a_top
#person b
b_left = rects_old[1,0]
b_top = rects_old[1,1]
b_right = rects_old[1,0] + rects_old[1,2]
b_bottom = rects_old[1,1] + rects_old[1,3]
#find good features in the person b box
p0_b = cv2.goodFeaturesToTrack(resized_old_gray[b_left:b_right+1,b_top:b_bottom+1], mask= None , **feature_params)
[p0_b_dim,p0_b_row,p0_b_col] = p0_b.shape
for i in range(p0_b_dim):
    p0_b[i,0,0] = p0_b[i,0,0]+b_left
    p0_b[i,0,1] = p0_b[i,0,1]+b_top

#plot first frame to check rectangle
while(camera.isOpened()):
    cv2.rectangle(resized_old_frame,(a_left,a_top),(a_right,a_bottom),(0,255,0),2)
    cv2.rectangle(resized_old_frame,(b_left,b_top),(b_right,b_bottom),(0,0,255),2)
    cv2.imshow('test',resized_old_frame[a_top:a_bottom+1,a_left:a_right+1])
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()