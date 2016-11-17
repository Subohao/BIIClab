# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:35:33 2016

@author: user
"""

from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import cv2
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="D:/April8_2sentence1.mpg", help="the path for video")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])
 
## initialize the first frame in the video stream
#firstFrame = None

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break
	frame = imutils.resize(frame, width=min(400, frame.shape[1]))
#	r = 400.0 / frame.shape[1]
#	dim = (400, int(frame.shape[0] * r))
#	frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
	orig_frame = frame.copy()
	(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
	for (x, y, w, h) in rects:
		cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	cv2.imshow("Before NMS", orig_frame)
	cv2.imshow("After NMS", frame)
	#cv2.waitKey(0)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
camera.release()
cv2.destroyAllWindows()