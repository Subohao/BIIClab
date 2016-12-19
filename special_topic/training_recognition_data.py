# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 21:23:57 2016

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

def selectwindow(record_info_clear):

    checkpoint = 0 #if 0 then a, if 1 then b
    person_A = []
    person_B = []
    if record_info_clear[0,1] < 130: #initialize
        person_A_temp = record_info_clear[0]
        person_A.append(person_A_temp)
        person_A_info = np.array(person_A)
        person_B_info = np.array([])
        checkpoint = 0
    elif record_info_clear[0,1] > 130:
        person_B_temp = record_info_clear[0]
        person_B.append(person_B_temp)
        person_B_info = np.array(person_B)
        person_A_info = np.array([])
        checkpoint = 1
    [row_record_info_clear,col_record_info_clear] = np.shape(record_info_clear)

#    index_A = 0
#    index_B = 0 
   
    
    for i in range(row_record_info_clear-1):
        if (record_info_clear[i+1,0] != record_info_clear[i,0]):
            if checkpoint == 0:
                if np.linalg.norm(record_info_clear[i+1,1:5]-record_info_clear[i,1:5]) <= 30:
                    person_A_temp = record_info_clear[i+1]
                    person_A.append(person_A_temp)
                    person_A_info = np.array(person_A)
#                    index_A = index_A + 1
                    checkpoint = 0
                else:
                    person_B_temp = record_info_clear[i+1]
                    person_B.append(person_B_temp)
                    person_B_info = np.array(person_B)
                    checkpoint = 1 
            else: 
                if np.linalg.norm(record_info_clear[i+1,1:5]-record_info_clear[i,1:5]) <= 30:
                    person_B_temp = record_info_clear[i+1]
                    person_B.append(person_B_temp)
                    person_B_info = np.array(person_B)
                    checkpoint = 1 
                else:
                    person_A_temp = record_info_clear[i+1]
                    person_A.append(person_A_temp)
                    person_A_info = np.array(person_A)
#                    index_A = index_A + 1
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
record_info_clear = np.load('D://senior/CCL/record_info_clear.npy')
(AnswerA,AnswerB) = selectwindow(record_info_clear)

camera = cv2.VideoCapture('D://senior/CCL/video/April30/April30_2sentence1.mpg')
# take first frame of the video
(grabbed,frame_old) = camera.read()
#rame = frame_old[:,:,:]
frame = frame_old[:,:,:]
r = 400.0 / frame.shape[1]
dim = (400, int(frame.shape[0] * r))

#load the features
row = np.load('D://senior/CCL/{April30_2sentence1.mpg}_out_features.npy')
[width, length] = row.shape
width = int(width)
length = int(length)
[row_A,col_A] = AnswerA.shape
[row_B,col_B] = AnswerA.shape
Answer_A_array = []
Answer_A_feat = np.array(Answer_A_array)
Answer_B_array = []
Answer_B_feat = np.array(Answer_B_array)

for i in range(width):
    for j in range(row_A):
        if AnswerA[j,0] == row[i,0] and AnswerA[j,1] < row[i,2]*r and row[i,2]*r < AnswerA[j,3] and AnswerA[j,2] < row[i,1]*r and row[i,1]*r < AnswerA[j,4]:
            Answer_A_temp = row[i]
            Answer_A_array.append(Answer_A_temp)
            Answer_A_feat = np.array(Answer_A_array)
        elif AnswerA[j,0] >= 1200:
            break
    if row[i,0] == 1200:
        break
np.save('personA_feat_0_1200.npy',Answer_A_feat)

for k in range(width):
    for m in range(row_A):
        if  AnswerB[m,0] == row[k,0] and AnswerB[m,1] < row[k,2]*r and row[k,2]*r < AnswerB[m,3] and AnswerB[m,2] < row[k,1]*r and row[k,1]*r < AnswerB[m,4]:
            Answer_B_temp = row[k]
            Answer_B_array.append(Answer_B_temp)
            Answer_B_feat = np.array(Answer_B_array)
        elif AnswerB[m,0] >= 1200:
            break
    if row[k,0] == 1200:
        break
np.save('personB_feat_0_1200.npy',Answer_B_feat)