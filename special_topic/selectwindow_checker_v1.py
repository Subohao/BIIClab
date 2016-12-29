#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:32:12 2016

@author: DennisLin
"""
import numpy as np
import cv2
#people detection 
#from __future__ import print_function
from imutils.object_detection import non_max_suppression
import argparse
#import imutils
import time
#A = np.array([[1,2,3],[4,5,6],[7,8,9]])
#
#B = np.array([[1,1,1],[2,2,2],[3,3,3]])
#
#A = np.insert(A,0,[0,0,0],axis=0)
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def selectwindow(record_info_clear):

    checkpoint = 0 #if 0 then a, if 1 then b
    person_A = []
    person_B = []
#    if record_info_clear[0,1] < 130: #initialize
    person_A_temp = record_info_clear[0]
    person_A.append(person_A_temp)
    person_A_info = np.array(person_A)
    person_B_info = np.array([])
    checkpoint = 0
#    elif record_info_clear[0,1] > 130:
#        person_B_temp = record_info_clear[0]
#        person_B.append(person_B_temp)
#        person_B_info = np.array(person_B)
#        person_A_info = np.array([])
#        checkpoint = 1
    [row_record_info_clear,col_record_info_clear] = np.shape(record_info_clear)

   
    
    for i in range(row_record_info_clear-1):
        if (record_info_clear[i+1,0] != record_info_clear[i,0]):
            if checkpoint == 0:
                center_dist = np.linalg.norm((record_info_clear[i+1,1]+record_info_clear[i+1,3])-(record_info_clear[i,1]+record_info_clear[i,3]))
                if center_dist <= 5:
                    person_A_temp = record_info_clear[i+1]
                    person_A.append(person_A_temp)
                    person_A_info = np.array(person_A)
                    checkpoint = 0
                elif (center_dist > 5) and (center_dist <= 40):
                    record_info_clear[i+1,1:5]  = record_info_clear[i,1:5]                    
                else:
                    person_B_temp = record_info_clear[i+1]
                    person_B.append(person_B_temp)
                    person_B_info = np.array(person_B)
                    checkpoint = 1 
            else: 
                center_dist = np.linalg.norm((record_info_clear[i+1,1]+record_info_clear[i+1,3])-(record_info_clear[i,1]+record_info_clear[i,3]))
                if  center_dist <= 5:
                    person_B_temp = record_info_clear[i+1]
                    person_B.append(person_B_temp)
                    person_B_info = np.array(person_B)
                    checkpoint = 1 
                elif (center_dist > 5) and (center_dist <= 40):
                    record_info_clear[i+1,1:5]  = record_info_clear[i,1:5]                     
                else:
                    person_A_temp = record_info_clear[i+1]
                    person_A.append(person_A_temp)
                    person_A_info = np.array(person_A)
                    checkpoint = 0           
        else:# same frame
            if checkpoint == 0:
                person_B_temp = record_info_clear[i+1]
                person_B.append(person_B_temp)
                person_B_info = np.array(person_B) 
                checkpoint = 1
            else:
                person_A_temp = record_info_clear[i+1]
                person_A.append(person_A_temp)
                person_A_info = np.array(person_A) 
                checkpoint = 0            
    return (person_A_info, person_B_info)
AnswerA = np.array([])
AnswerB = np.array([])
#record_info_clear = unique_rows(record_info)
#record_info_clear = np.load('D://senior/CCL/record_info_clear_mask_NMS.npy')
(AnswerA,AnswerB) = selectwindow(record_info_clear)      
#record_info = np.load('D://senior/CCL/special_topic/record_info.npy')
#record_info_clear = unique_rows(record_info)

camera = cv2.VideoCapture('D://senior/CCL/video/April30/April30_2sentence1.mpg')

#load the features
#row = np.load('D://senior/CCL/special_topic/{April30_2sentence1.mpg}_out_features.npy')
#[width, length] = row.shape
#width = int(width)
#length = int(length)

## initialize the HOG descriptor/person detector
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# take first frame of the video
#(grabbed,frame_old) = camera.read()
##rame = frame_old[:,:,:]
#frame = frame_old[:,:,:]
#r = 400.0 / frame.shape[1]
#dim = (400, int(frame.shape[0] * r))
#frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#(rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32,32), scale=1.05)
#rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
frame_num = 0
index_A = 0
index_B = 0
first_AnswerA = AnswerA[0,0]
first_AnswerB = AnswerB[0,0]
#[row_A,col_A] = AnswerA.shape()
#[row_B,col_B] = AnswerB.shape()


while(camera.isOpened()):
    ret, frame = camera.read()
    r = 400.0 / frame.shape[1]
    dim = (400, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    print(index_A,frame_num,AnswerA[index_A,0])
    if frame_num == AnswerA[index_A,0]:
        cv2.rectangle(frame,(AnswerA[index_A,1],AnswerA[index_A,2]),(AnswerA[index_A,3], AnswerA[index_A,4]), (0, 255, 0), 2)
        index_A = index_A + 1
        [row_A,col_A] = np.shape(AnswerA)
        if index_A > row_A-1:
            AnswerA_temp1 = AnswerA[index_A-1,:].copy()
            AnswerA_temp1[0] = frame_num
            AnswerA = np.insert(AnswerA,index_A,AnswerA_temp1,axis = 0)   
    else:
        AnswerA_temp = AnswerA[index_A,:].copy()
        AnswerA_temp[0] = frame_num
        AnswerA = np.insert(AnswerA,index_A,AnswerA_temp,axis = 0)
        if frame_num > first_AnswerA:#600
            cv2.rectangle(frame,(AnswerA[index_A,1],AnswerA[index_A,2]),(AnswerA[index_A,3], AnswerA[index_A,4]), (0, 255, 0), 2)
        index_A = index_A + 1
        
    if frame_num == AnswerB[index_B,0]:
        cv2.rectangle(frame,(AnswerB[index_B,1],AnswerB[index_B,2]),(AnswerB[index_B,3], AnswerB[index_B,4]), (0, 0, 255), 2)
        index_B = index_B + 1
        [row_B,col_B] = np.shape(AnswerB)
        if index_B > row_B-1:
            AnswerB_temp1 = AnswerB[index_B-1,:].copy()
            AnswerB_temp1[0] = frame_num
            AnswerB = np.insert(AnswerB,index_B,AnswerB_temp1,axis = 0)        
    else:
        AnswerB_temp = AnswerB[index_B,:].copy()
        AnswerB_temp[0] = frame_num
        AnswerB = np.insert(AnswerB,index_B,AnswerB_temp,axis = 0)
        
        if frame_num > first_AnswerB:
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
#    elif frame_num == 8776:
#        break
    frame_num +=1
    if frame_num == 8784:
        break
camera.release()
cv2.destroyAllWindows()
np.save('AnswerA.npy',AnswerA)
np.save('AnswerB.npy',AnswerB)      