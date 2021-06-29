import numpy as np
import csv
import math

# load dataset
p = r'./data/train_data.csv'
with open(p,encoding = 'utf-8') as f:
    dataset = np.loadtxt(f,delimiter = ",")
k = r'./data/test_data.csv'
with open(k,encoding = 'utf-8') as f:
    testset = np.loadtxt(f,delimiter = ",")

# calculate mean and std
mean_train = [np.mean(dataset[np.where(dataset[:,0]==i)], axis=0)[1:] for i in range(1,4)]
std_train = [np.std(dataset[np.where(dataset[:,0]==i)], axis=0, ddof=1)[1:] for i in range(1,4)]

# gus function and p(x|w)
def gus(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)
def prod13(x):
    return [np.prod([gus(x[i],mean_train[k][i],std_train[k][i]) for i in range(13)]) for k in range(3)]

# calculate p(w)
p_w = [np.sum(np.where(dataset[:,0]==i,1,0))/len(dataset) for i in range(1,4)]

# test and save file
predict = np.zeros(len(testset))
output = np.zeros((len(testset),4))
sum = 0
for i in range(len(testset)):
    label = testset[i][0]
    x = testset[i][1:]
    p = [p_w[i]*prod13(x)[i] for i in range(3)]
    output[i,0] = predict[i] = np.argmax(p)+1
    output[i,1:] = p/np.sum(p)
    if predict[i] == label:
        sum = sum + 1
print('ACC:',sum/len(testset))
np.savetxt('task.csv', output,delimiter=',')