# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 13:44:26 2016

@author: user
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('D://senior/CCL/video/April30/April30_2sentence1.mpg')

fgbg = cv2.BackgroundSubtractorMOG2()
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

ret,old_frame = cap.read()

#old_frame[130:385,180:255] = 0;
#old_frame[130:400,365:470] = 0;
ret, first_frame = cap.read()
#first_frame[130:385,180:255] = (255,255,255)
#first_frame[130:400,365:470] = (255,255,255)
#
#gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

#while(cap.isOpened()):
#    cv2.rectangle(first_frame,(180,130),(255,385),(0,255,0),2)
#    cv2.rectangle(first_frame,(420,130),(470,400),(0,0,255),2)
#    cv2.imshow('test1',first_frame)
##    gray_first_frame[130:385,180:255] = 0
##    gray_first_frame[130:400,365:470] = 0
##    cv2.imshow('test',gray_first_frame)
#    if cv2.waitKey(0) & 0xFF == ord('q'):
#        break
index = 0;
history = 1000;
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame,learningRate = 1.0/history)
    #fgmask[130:385,180:255] = 0;
    #fgmask[130:400,365:470] = 0;
#    if index == 1000:
#        for i  in range(130,385,1):
#            for j in range(180,255,1):
#                if fgmask[i,j] != 0:
#                    fgmask[i,j] = 0
    cv2.imshow('frame',fgmask)
    index = index+1
    print(index)
    #old_frame = frame
    k = cv2.waitKey(30) & 0xff
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()