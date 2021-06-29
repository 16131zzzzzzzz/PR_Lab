import matplotlib
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math

# define distances
def Mahalanobis_distance(xi, A, xj):
    temp = np.dot(A,(xi-xj).T)
    return np.dot(temp.T,temp)

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

def f_gradient(A):
    data_divide = [trainset_without_label[np.where(train_labels==0)], trainset_without_label[np.where(train_labels==1)]]
    len_train = len(trainset_without_label)
    exps = [[math.exp(-Mahalanobis_distance(trainset_without_label[i], A, trainset_without_label[j])) for j in range (len_train)] for i in range(len_train)]
    exps_sum = np.sum(exps,axis=1)
    p_ij = [exps[i]/exps_sum[i] for i in range(len(exps_sum))]
    for i in range(len(p_ij)):
        p_ij[i][i] = 0
    p_i = [np.sum(p_ij[i][np.where(train_labels==train_labels[i])]) for i in range(len(train_labels))]
    sum = np.zeros((4,4))
    for i in range(len(train_labels)):
        sum1 = np.multiply(p_i[i],np.sum([np.multiply(p_ij[i][k],np.outer((trainset_without_label[i]-trainset_without_label[k]).T,(trainset_without_label[i]-trainset_without_label[k]))) for k in range(len(train_labels))], axis=0))
        sum2 = np.sum([np.multiply(p_ij[i][k],np.outer((trainset_without_label[i]-trainset_without_label[k]).T,(trainset_without_label[i]-trainset_without_label[k]))) for k in np.array(np.where(train_labels==train_labels[i]))[0]], axis=0)
        sum = sum + (sum1-sum2)
    return np.sum(p_i), np.dot(np.multiply(2,A),sum)

def Gredient_Descent_batch(A, lr = 1 ):
    print("graient begin")
    epoch = 100
    histroy = []
    history_f = []
    for j in range(epoch):
        f, gd = f_gradient(A)
        sum = np.sum(np.square(gd))
        # lr /= ((j+1)**0.5)
        if sum >= 0.00001:
            A += lr * gd
            histroy.append(sum)
            history_f.append(f)
        else:
            break
        print(j+1," batch")
        print("A=", A)
    print("graient finish")
    return A, histroy, history_f
A = 0.1*np.ones([2, 4])
A_better, history, history_f = Gredient_Descent_batch(A)
plt.plot(history_f, '*')
plt.show()
plt.plot(history,'*')
plt.show()

def Mahalanobis_distance_better(xi,xj):
    # [[-9.67501079,3.81199418,3.81199418,-3.26855729],[-9.67501079,3.81199418,3.81199418,-3.26855729]]
    # ACC=0.945
    A = A_better
    temp = np.dot(A,(xi-xj).T)
    return np.dot(temp.T,temp)

def calAllDistance(x, disFunc):
    return [disFunc(x,i) for i in trainset_without_label]
def nearest_k_label(x, disFunc, k):
    sort_args = np.argsort(calAllDistance(x, disFunc))
    return train_labels[sort_args[:k]]

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
ACC_Mah = cal_ACC(Mahalanobis_distance_better)
print(ACC_Mah)
plt.plot(ACC_Mah)
plt.show()

k_Mah = np.argmax(ACC_Mah)
np.savetxt('task2_test_prediction_Mahalanobis.csv', np.c_[testset_read, classify(testset_read, Mahalanobis_distance_better, k_Mah)].astype(np.uint8), delimiter=',')