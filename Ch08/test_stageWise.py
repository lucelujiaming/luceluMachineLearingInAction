# coding: utf-8

import sys
import regression
from numpy import *

xArr, yArr = regression.loadDataSet('abalone.txt')
stageWiseWeights = regression.stageWise(xArr, yArr, 0.01, 200)

print("stageWiseWeights  : ", stageWiseWeights )

xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regression.regularize(xMat)
yM = mean(yMat, 0)
yMat = yMat - yM

weights = regression.standRegres(xMat, yMat.T)
print("weights  : ", weights )

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(weights)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(stageWiseWeights)
plt.show()
