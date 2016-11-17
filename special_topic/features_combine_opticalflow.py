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
p0_a = cv2.goodFeaturesToTrack(resized_old_gray[a_top:a_bottom+1,a_left:a_right+1], mask= None , **feature_params)
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
p0_b = cv2.goodFeaturesToTrack(resized_old_gray[b_top:b_bottom+1,b_left:b_right+1], mask= None , **feature_params)
[p0_b_dim,p0_b_row,p0_b_col] = p0_b.shape
for i in range(p0_b_dim):
    p0_b[i,0,0] = p0_b[i,0,0]+b_left
    p0_b[i,0,1] = p0_b[i,0,1]+b_top

    

# Create a mask image for drawing purposes
mask = np.zeros_like(resized_old_frame)



count = 0
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
    if args.get("video") and not grabbed:
		break
    #frame = imutils.resize(frame, width=500)
    r = 400.0 / frame.shape[1]
    dim = (400, int(frame.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    frame_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    orig_resized = resized.copy()
    (rects, weights) = hog.detectMultiScale(resized, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
    for (x_rect, y_rect, w_rect, h_rect) in rects:
                cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)

    
    #plot the trajectory features in the rect
    for e in range(width):
        if row[e,0] == count:
            (x,y) = (row[e,1], row[e,2])
            for (x_rect, y_rect, w_rect, h_rect) in rects:
                if (x_rect < int(x*r)) and (int(x*r) < (x_rect+w_rect)) and (y_rect < int(y*r)) and (int(y*r) < (y_rect+h_rect)):
                    cv2.circle(orig_resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)                    
            rects_nms = np.array([[x_rect, y_rect, x_rect + w_rect, y_rect + h_rect] for (x_rect, y_rect, w_rect, h_rect) in rects])
            pick = non_max_suppression(rects_nms, probs=None, overlapThresh=0.65)
            for (xA, yA, xB, yB) in pick:
                if (xA < int(x*r)) and (int(x*r) < (xB)) and (yA < int(y*r)) and (int(y*r) < (yB)):
                    cv2.circle(resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)
    #plot rect after NMS
    for (xA, yA, xB, yB) in pick:
                cv2.rectangle(resized, (xA, yA), (xB, yB), (0, 255, 0), 2)
#    [pick_row,pick_col] = pick.shape
#    pick_row = int(pick_row)
#    pick_col = int(pick_col)
#    if pick_old:
#        for  in range(pick_row):
#            distance = 
#            
#    else:
#        break
                
                
    p1, st, err = cv2.calcOpticalFlowPyrLK(resized_old_gray, frame_gray, p0_b, None, **lk_params)                
    # Select good points
    good_new = p1[st==1]
    good_old = p0_b[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)#problem found!
        cv2.circle(resized,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(resized,mask)
         #show the frame to our screen
    cv2.imshow("After_NMS_Frame", img)
    cv2.imshow("Before_NMS_Fram", orig_resized)
    
    
    #store old rect coordinate    
    (rects_old, weights_old) = (rects, weights)
    pick_old = pick

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    count +=1

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()