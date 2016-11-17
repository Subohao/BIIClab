# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:50:47 2016

@author: user
"""


import re
import numpy as np

f = open('D:/senior/CCL/person01_boxing_d1/person01_boxing_d1','rb')
s = f.read()
line = re.split('\n', s)

data_num = len(line)
feat_num = len(line[0].split('\t'))-1

#create an matrix to store
A2 = np.zeros([data_num, feat_num], float)

for i in range(data_num):
  current_line = line[i].split('\t')
  for j in range(feat_num):
    A2[i,j]=float(current_line[j])

np.save('feat.npy', A2)
    
#file_content = f.readline().split("\t")
#print file_content
#
#i =0;
#while i<436:
#    A2[0,i] = float(file_content[i])
#    i += 1

#f = open('test_read.txt','rb')
#file_content = f.readline()
#print file_content

#import numpy as np
#A1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
#A2 = np.zeros((100,437),float)
#A1 = np.array([np.zeros(len(5)),np.zeros(len(5))])