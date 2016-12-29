from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-A", "--A_feature", help="input A_person feature")
parser.add_argument("-B", "--B_feature", help="input B_person feature")
parser.add_argument("-start", "--start", default="10", help="select the start feature range")
parser.add_argument("-end", "--end", default="244", help="select the end feature range")
parser.add_argument("-K", "--K_fold", default="10", help="choose K for cross-validation")
args = vars(parser.parse_args())

A_feature_ori = np.load(args["A_feature"])
B_feature_ori = np.load(args["B_feature"])
feature_start = int(args["start"])
feature_end = int(args["end"])
if(feature_end > 437 or feature_start < 0):
	print "[ValueError] Invalid feature range !"
	sys.exit()

A_dim = A_feature_ori.shape[0]
B_dim = B_feature_ori.shape[0]
pdb.set_trace()
# label append on the last column(person A will be 0, person B will be 1)
# that means the dimension of each row will be 
# 1*(feature_end-feature_start +1)
y_A = [0] * (A_dim)
y_B = [1] * (B_dim)
A_feature = A_feature_ori[0:A_dim, feature_start:feature_end]
B_feature = B_feature_ori[0:B_dim, feature_start:feature_end]
A_feature = np.c_[A_feature, y_A]
B_feature = np.c_[B_feature, y_B]
X = np.append(A_feature, B_feature, axis=0)
# shuffle the training data
np.random.shuffle(X)

# Then, we use K-fold cross-validation to estimate our accuracy
K = int(args["K_fold"])
each_part = X.shape[0]/K
tmp_accu = 0
total_accu = 0
for fold in range(K):
	if(fold != K-1):
		valid_data = X[each_part*fold : each_part*(fold+1) , :]
		training_data = np.append(X[0 : each_part*fold, :], X[each_part*(fold+1) : X.shape[0], :], axis=0)
	else:
		valid_data = X[each_part*fold : , :]
		training_data = X[0 : each_part*fold, :]

	row_dim = training_data.shape[1]
	feature = training_data[:, 0:row_dim-1]
	label = training_data[:, row_dim-1]
	classifier = svm.SVC(kernel='linear', C = 1.0)
	classifier.fit(feature, label)

	w = classifier.coef_[0]
	correct = 0
	valid_data_num = valid_data.shape[0]
	for valid in range(valid_data_num):
		pre = classifier.predict(valid_data[valid, 0:row_dim-1])
		if int(pre) == int(valid_data[valid][row_dim-1]):
			correct = correct + 1
	tmp_accu = (correct/float(valid_data_num))*100
	total_accu = total_accu + tmp_accu
	print "[Fold " + str(fold+1) + "/" + str(K) + "] Accuracy = " + str(tmp_accu) + "%"

print "Total Accuracy = " + str(total_accu/K) + "%"
