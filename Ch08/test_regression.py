# coding: utf-8

import sys
import regression
from numpy import *

xArr, yArr = regression.loadDataSet('ex0.txt')
print("xArr : ", xArr)
print("yArr : ", yArr)

ws = regression.standRegres(xArr, yArr)
print("ws : ", ws)

xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws
corrcoefRet = corrcoef(yHat.T, yMat)
print("corrcoefRet : ", corrcoefRet)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], \
	yMat.T[:,0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:,1], yHat)
plt.show()
