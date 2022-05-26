# coding: utf-8
import kMeans
from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

myDat = kMeans.loadDataSet('testSet.txt')
myMat = mat(myDat)
# print("myMat : \n", myMat)
# print("shape(myMat) : \n", shape(myMat))
# print("shape(myMat)[1] : \n", shape(myMat)[1])
n = shape(myMat)[1]
# print("mat(zeros((k,n))) : \n", mat(zeros((5,n))))

#   for j in range(n):
#   	minJ = min(myMat[:,j]) 
#   	print("min(myMat[:,j]) : \n", min(myMat[:,j]))
#   	print("max(myMat[:,j]) : \n", max(myMat[:,j]))
#   	rangeJ = float(max(myMat[:,j]) - minJ)
#   	print("rangeJ : \n", rangeJ)

centroidsRet = kMeans.randCent(myMat, 2)
# print("centroidsRet : \n", centroidsRet)

myCentroids, myClusterAssment = kMeans.kMeans(myMat, 4)
# print("myCentroids : \n", myCentroids)
# print("myClusterAssment : \n", myClusterAssment)

# 开始绘图。
fig = plt.figure()
ax = fig.add_subplot(111)
# 绘制两个坐标点list。
ax.scatter(myMat[:,0].A, myMat[:,1].A, s=30, c='red', marker='s')
ax.scatter(myCentroids[:,0].A, myCentroids[:,1].A, s=30, c='green')
#		# 给出X / Y坐标范围。
#		x = arange(-3.0, 3.0, 0.1)
#		# 决策边界是: w0 + w1*x1 + w2*x2 = 0，所以x2 = (-w0-w1*x1)/w2
#		# y = (- 4.12414349 - 0.48007329 * x) / -0.6168482
#		#   = 0.778268 x + 6.6858320
#		y = (-weights[0]-weights[1]*x)/weights[2]
#		ax.plot(x, y)
plt.xlabel('X1'); plt.ylabel('X2');
plt.show()
