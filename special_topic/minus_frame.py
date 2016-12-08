# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:24:08 2016

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
 
# Take first frame 
(ret, old_frame) = camera.read()
#r = 400.0 / old_frame.shape[1]
#dim = (400, int(old_frame.shape[0] * r))
#resized_old_frame = cv2.resize(old_frame, dim, interpolation = cv2.INTER_AREA)
#resized_old_gray = cv2.cvtColor(resized_old_frame, cv2.COLOR_BGR2GRAY)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

while(camera.isOpened()):
    # grab the current frame
    (grabbed, frame) = camera.read()
    
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
    if args.get("video") and not grabbed:
		break
#    #frame = imutils.resize(frame, width=500)
#    r = 400.0 / frame.shape[1]
#    dim = (400, int(frame.shape[0] * r))
#
#    # perform the actual resizing of the image and show it
#    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#    frame_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#    comparison_frame = frame_gray - resized_old_gray
#    cv2.imshow("old_gray",resized_old_gray)
    cv2.imshow("old_gray",old_gray)
    cv2.imshow("current_gray",frame_gray)
    comparison_frame = frame_gray - old_gray
    cv2.imshow("comparison",comparison_frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


#i=1
#while (i==1):
#    # grab the current frame
#    (grabbed, frame) = camera.read()
#    
#	# if we are viewing a video and we did not grab a frame,
#	# then we have reached the end of the video
#    if args.get("video") and not grabbed:
#		break
#    #frame = imutils.resize(frame, width=500)
#    r = 400.0 / frame.shape[1]
#    dim = (400, int(frame.shape[0] * r))
#
#    # perform the actual resizing of the image and show it
#    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#    frame_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#    
#    comparison_frame = frame_gray - resized_old_gray
#    cv2.imshow("comparison",comparison_frame)
#    
#    
#    key = cv2.waitKey(1) & 0xFF
#    if key == ord("q"):
#        break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()