# coding: utf-8
import kMeans
from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

myDat = kMeans.loadDataSet('testSet.txt')
myMat = mat(myDat)

centroid0 = mean(myMat, axis=0).tolist()[0]
print("centroid0 : \n", centroid0)

datMat3 = mat(kMeans.loadDataSet('testSet2.txt'))
centList, myNewAssments = kMeans.biKmeans(datMat3, 7)
print("centList : ", centList)

# 开始绘图。
fig = plt.figure()
ax = fig.add_subplot(111)
# 绘制两个坐标点list。
ax.scatter(datMat3[:,0].A, datMat3[:,1].A, s=30, c='red', marker='s')
ax.scatter(centList[:,0].A, centList[:,1].A, s=30, c='green')
plt.xlabel('X1'); plt.ylabel('X2');
plt.show()
