# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:50:47 2016

@author: user
"""


import re
import numpy as np

f = open('D://senior/CCL/{April23_2sentence1.mpg}_out_features/{April23_2sentence1.mpg}_out_features','rb')
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

np.save('{April23_2sentence1.mpg}_out_features.npy', A2)