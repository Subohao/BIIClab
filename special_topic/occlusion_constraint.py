# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:33:58 2016

@author: user
"""

import numpy as np
import cv2
#people detection 
#from __future__ import print_function
from imutils.object_detection import non_max_suppression
import argparse
#import imutils
import time

camera = cv2.VideoCapture('D://senior/CCL/video/April30/April30_2sentence1.mpg')

# take first frame of the video
(grabbed,frame_old) = camera.read()
#rame = frame_old[:,:,:]
frame = frame_old[:,:,:]
r = 400.0 / frame.shape[1]
dim = (400, int(frame.shape[0] * r))
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
frame_num = 0

while(camera.isOpened()):
    ret, frame = camera.read()
    r = 400.0 / frame.shape[1]
    dim = (400, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    if frame_num == AnswerA[index_A+1,0]:
        cv2.rectangle(frame,(AnswerA[index_A,1],AnswerA[index_A,2]),(AnswerA[index_A,3], AnswerA[index_A,4]), (0, 255, 0), 2)
        index_A = index_A + 1        
    else:
        AnswerA_temp = AnswerA[index_A,0:5]
        AnswerA_temp[0] = frame_num
        AnswerA = np.insert(AnswerA,index_A,AnswerA_temp,axis = 0)
        if frame_num >= first_AnswerA:
            cv2.rectangle(frame,(AnswerA[index_A,1],AnswerA[index_A,2]),(AnswerA[index_A,3], AnswerA[index_A,4]), (0, 255, 0), 2)
        index_A = index_A + 1
        
    if frame_num == AnswerB[index_B+1,0]:
        cv2.rectangle(frame,(AnswerB[index_B,1],AnswerB[index_B,2]),(AnswerB[index_B,3], AnswerB[index_B,4]), (0, 0, 255), 2)
        index_B = index_B + 1        
    else:
        AnswerB_temp = AnswerB[index_B,0:5]
        AnswerB_temp[0] = frame_num
        AnswerB = np.insert(AnswerB,index_B,AnswerB_temp,axis = 0)
        if frame_num >= first_AnswerA:
            cv2.rectangle(frame,(AnswerB[index_B,1],AnswerB[index_B,2]),(AnswerB[index_B,3], AnswerB[index_B,4]), (0, 0, 255), 2)
        index_B = index_B + 1
#    for i in range(AnswerA.shape[0]) :
#        if frame_num == AnswerA[i,0]:
#            cv2.rectangle(frame, (AnswerA[i,1],AnswerA[i,2]), (AnswerA[i,3], AnswerA[i,4]), (0, 255, 0), 2)
#            index_A = index_A+1
#        elif frame_num != AnswerA[i,0] and index_A != 0:
#            cv2.rectangle(frame,(AnswerA[index_A-1,1],AnswerA[index_A-1,2]), (AnswerA[index_A-1,3], AnswerA[index_A-1,4]), (0, 255, 0), 2)
#    for i in range(AnswerB.shape[0]) :
#        if frame_num == AnswerB[i,0]:
#            cv2.rectangle(frame, (AnswerB[i,1],AnswerB[i,2]), (AnswerB[i,3], AnswerB[i,4]), (0, 0, 255), 2)
#            index_B = index_B +1;
#        elif frame_num != AnswerB[i,0] and index_B !=0:
#            cv2.rectangle(frame,(AnswerB[index_B-1,1],AnswerB[index_B-1,2]), (AnswerB[index_B-1,3], AnswerB[index_B-1,4]), (0, 0, 255), 2)            
    cv2.putText(frame, "Frame_index: {}".format(frame_num), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_num +=1

camera.release()
cv2.destroyAllWindows() 