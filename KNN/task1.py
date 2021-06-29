import matplotlib
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# define distances
def Euclidean(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def Manhattan(a,b):
    return np.sum(np.abs(a-b))

def Chebyshev(a,b):
    return np.max(np.abs(a-b))

# read datasets
trainset_read = np.array(pd.read_csv('train_data.csv',header=0))
testset_read = np.array(pd.read_csv('test_data.csv',header=0))
evalset_read = np.array(pd.read_csv('val_data.csv',header=0))
trainset_without_label = trainset_read[:,:4]
train_labels = trainset_read[:,4]
evalset_without_label = evalset_read[:,:4]
eval_labels = evalset_read[:,4]
t_max = np.array([trainset_without_label.max(axis=0), testset_read.max(axis=0), evalset_without_label.max(axis=0)]).max(axis=0)
t_min = np.array([trainset_without_label.min(axis=0), testset_read.min(axis=0), evalset_without_label.min(axis=0)]).min(axis=0)
trainset_without_label = (trainset_without_label - t_min) / (t_max - t_min)
evalset_without_label = (evalset_without_label - t_min) / (t_max - t_min)
testset = (testset_read - t_min) / (t_max - t_min)

def calAllDistance(x, disFunc):
    return [disFunc(x,i) for i in trainset_without_label]
def nearest_k_label(x, disFunc, k):
    sort_args = np.argsort(calAllDistance(x, disFunc))
    return train_labels[sort_args[:k]]
nearest_k_label(testset[0], Chebyshev, 8)

def classify(X, disFunc, k):
    predict = np.zeros(len(X))
    for i in range(len(X)):
        label_k = nearest_k_label(X[i], disFunc, k)
        sum = np.sum(label_k)
        predict[i] = 1 if sum*np.sum(train_labels)>(len(label_k)-sum)*(len(train_labels)-np.sum(train_labels)) else 0
    return predict

def cal_ACC(disFunc):
    allACC = np.zeros(40)
    for k in range(40):
        addon = classify(evalset_without_label, disFunc, k) + eval_labels
        acc = np.sum(np.where(addon==1,0,1))/len(eval_labels)
        allACC[k] = acc
    return allACC
ACC_Eu, ACC_Man, ACC_Che = cal_ACC(Euclidean), cal_ACC(Manhattan), cal_ACC(Chebyshev)
print("ACC_Eu", ACC_Eu)
print("ACC_Man", ACC_Man)
print("ACC_Che", ACC_Che)
plt.plot(ACC_Eu,c='r')
plt.plot(ACC_Man,c='g')
plt.plot(ACC_Che,c='b')
plt.show()

k_Eu, k_Man, k_Che = np.argmax(ACC_Eu), np.argmax(ACC_Man), np.argmax(ACC_Che)
np.savetxt('task1_test_prediction_Euclidean.csv', np.c_[testset_read, classify(testset, Euclidean, k_Eu)], delimiter=',')
np.savetxt('task1_test_prediction_Manhattan.csv', np.c_[testset_read, classify(testset, Manhattan, k_Man)], delimiter=',')
np.savetxt('task1_test_prediction_Chebyshev.csv', np.c_[testset_read, classify(testset, Chebyshev, k_Che)], delimiter=',')