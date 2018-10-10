import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

from sklearn import datasets 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time


data_file = 'data.csv'
Location = r'%s' % data_file
df = pd.read_csv(Location, header=None)
digits = datasets.load_digits()
#X = digits.images
#y = digits.target

#n_samples = len(digits.images)
#X = X.reshape((n_samples, -1))

#data = digits.images.reshape((n_samples, -1))

#X = import_data.data()
#y = import_data.target
##print('Class labels:', np.unique(y))


num_instance = int(sys.argv[1])
num_feature = int(sys.argv[2])

extra_set = sys.argv[3]
extra_start =  int(sys.argv[4])
extra_label =  int(sys.argv[5])

if num_instance == 0:
   num_instance  = ''

if num_feature == 0:
   num_feature =  ''



if extra_set == '0' :
   loop_count = 2
else:
	loop_count = 1

print(extra_set)

for count in range(loop_count):
	if count == 0 and extra_set != '0':
		X = df.iloc[:num_instance, extra_start:num_feature].values
		y = df.iloc[:num_instance, extra_label].values
		data_name = 'Your Result'
	elif count == 0 and extra_set == '0' :
		X = digits.images
		y = digits.target  
		n_samples = len(digits.images)
		X = X.reshape((n_samples, -1))
		data_name = 'The digits dataset result'
	else:
		X = df.iloc[:, 1:119].values
		y = df.iloc[:, 119].values
		data_name = 'REALDISP Activity Recognition Dataset Data Set result'
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

	##########  stanardize the data  ############
	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)

	###############  Perceptron  #####################################
	start = time.perf_counter()
	ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
	ppn.fit(X_train_std, y_train)

	y_pred = ppn.predict(X_test_std)
	end = time.perf_counter()
	runtime = end - start

	print('\n\n %s' % data_name)
	print('\n\nPerceptron:')
	print('Misclassified samples: %d' % (y_test != y_pred).sum())
	print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
	print('Running Time: %.4f seconds\n\n' % runtime)
	############################################################

	##############   SVC Linear   ################################
	start = time.perf_counter()
	svm = SVC(kernel='linear', random_state=1, gamma='scale', C=10)
	svm.fit(X_train_std, y_train)
	y_pred_svm = svm.predict(X_test_std)
	end = time.perf_counter()
	runtime = end - start

	print('SVC Linear:')
	print('Misclassified samples: %d' % (y_test != y_pred_svm).sum())
	print('Accuracy: %.4f' % accuracy_score(y_test, y_pred_svm))
	print('Running Time: %.4f seconds\n\n' % runtime)
	################################################################


	##################   SVC rbf kernal  ####################################
	start = time.perf_counter()
	svm = SVC(kernel='rbf', random_state=1, gamma='scale', C=10)
	svm.fit(X_train_std, y_train)
	y_pred_svm_kernel = svm.predict(X_test_std)
	end = time.perf_counter()
	runtime = end - start

	print('SVC rbf kernel:')
	print('Misclassified samples: %d' % (y_test != y_pred_svm_kernel).sum())
	print('Accuracy: %.4f' % accuracy_score(y_test, y_pred_svm_kernel)) 
	print('Running Time: %.4f seconds\n\n' % runtime) 
	#########################################################################

	################### Decision Tree  #######################################
	start = time.perf_counter()
	tree = DecisionTreeClassifier ( criterion='gini',max_depth=4,random_state=1, min_samples_split = 4, min_impurity_decrease = 0.005)
	
    ##  This decision Tree use Pre-Pruning
    ##  min_samples_split to set threshold to stop at when number of instances less than threshold
    ##
    ##  min_impurity_decrease is used to stop when the current node doesn't improve more than this threshold 
	tree.fit(X_train ,y_train)
	y_pred_tree = tree.predict(X_test)
	end = time.perf_counter()
	runtime = end - start

	print('Decision Tree:')
	print('Misclassified samples: %d' % (y_test != y_pred_tree).sum())
	#accuracy = (y_test==y_pred_tree).sum()/len(y_test)
	print('Accuracy: %.4f' % accuracy_score(y_test, y_pred_tree))
	print('Running Time: %.4f seconds\n\n' % runtime) 
	##########################################################################

	#####################   KNN  ###########################################
	start = time.perf_counter()
	knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
	knn.fit(X_train_std , y_train)
	y_pred_knn = knn.predict(X_test_std)
	end = time.perf_counter()
	runtime = end - start

	print('KNN:')
	print('Misclassified samples: %d' % (y_test != y_pred_knn).sum())
	print('Accuracy: %.4f' % accuracy_score(y_test, y_pred_knn)) 
	print('Running Time: %.4f seconds\n\n' % runtime)  
	###########################################################################

	#####################   Logistic Regression    ###############################
	start = time.perf_counter()
	lr = LogisticRegression(C=1.0, random_state=1, multi_class ='auto')
	lr.fit (X_train_std , y_train)
	y_pred_lr = lr.predict(X_test_std)
	end = time.perf_counter()
	runtime = end - start

	#y_pred_lr = lr.predict(X_test_std)
	print('Logistic Regression:')
	print('Misclassified samples: %d' % (y_test != y_pred_lr).sum())
	print('Accuracy: %.4f' % accuracy_score(y_test, y_pred_lr)) 
	print('Running Time: %.4f seconds\n\n' % runtime)  
	###########################################################################














