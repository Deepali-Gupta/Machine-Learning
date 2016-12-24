from svmutil import *
import csv
import matplotlib.pyplot as plt
import numpy as np
import math

datafile = open('part1.csv')
data = csv.reader(datafile)
features = [[] for x in xrange(25)]
labels = []
for row in data:
		labels.append(int(row[25]))
		i=0
		for col in row:
			if i <= 9:
				features[i].append((float(col)))
			i=i+1	
#scale the features
for i in range(10):
	mini = min(features[i])
	maxi = max(features[i])
	for j in range(len(features[0])):
		pt = features[i][j]
		pt = 2*(pt-mini)/(maxi-mini)-1
		features[i][j] = pt

#convert into svm matrix form
instances = []
for i in range(len(features[0])):
	inst = []
	for j in range(10):
		inst.append(features[j][i])
	instances.append(inst)
#form training and test sets
trdat = instances[:2500]
tedat = instances[2500:]
trlb = labels[:2500]
telb = labels[2500:]
#list of C values
C = []
arr = []

for i in range(-5,10):
	C.append(pow(2,i))
	arr.append(i)

#cross validation
acc = 0
C_opt = 0
g_opt = 0
acclist = []
tracc = []
tsacc = []
for c in C:
	#~ new_acc = svm_train(labels,instances,'-q -t 0 -v 10 -c %f '%c)
	#~ acclist.append(new_acc)
	model = svm_train(trlb,trdat,'-q -t 0 -c %f '%c)
	p_labels1,p_acc1,p_vals1 = svm_predict(trlb,trdat,model)
	p_labels2,p_acc2,p_vals2 = svm_predict(telb,tedat,model)	
	tracc.append(p_acc1[0])
	tsacc.append(p_acc2[0])
	#~ if new_acc>acc:
		#~ acc=new_acc
		#~ C_opt = c
#~ 
#~ model = svm_train(labels,instances,'-q -t 0 -c %f'%C_opt)
#~ p_labels,p_acc,p_vals = svm_predict(labels,instances,model)
plt.plot(arr,tracc,label="Training")
plt.plot(arr,tsacc,label="Test")
plt.xlabel('log2 of C')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")
plt.show()
	
import pdb;pdb.set_trace()	
	


