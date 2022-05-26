'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

# 将数据转换成前N个主成分的伪码大致如下：
#   去除平均值
#   计算协方差矩阵
#   计算协方差矩阵的特征值和特征向量
#   将特征值从大到小排序
#   保留最上面的N个特征向量
#   将数据转换到上述N个特征向量构建的新空间中
# 有两个参数：
#       第一个参数是用于进行PCA操作的数据集，
#       第二个参数topNfeat则是一个可选参数，即应用的N个特征。
#。     如果不指定topNfeat的值，那么函数就会返回前9 999 999个特征，
#       或者原始数据中全部的特征。
def pca(dataMat, topNfeat=9999999):
    # 1. 计算原始数据集的平均值
    meanVals = mean(list(dataMat), axis=0)
    # 减去原始数据集的平均值
    meanRemoved = dataMat - meanVals #remove mean
    # 计算协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值，
    eigVals,eigVects = linalg.eig(mat(covMat))
    # 对特征值进行从小到大的排序，到特征值排序结果的逆序
    # sort, sort goes smallest to largest
    eigValInd = argsort(eigVals)            
    # 2. 根据逆序得到topNfeat个最大的特征向量。
    #cut off unwanted dimensions
    eigValInd = eigValInd[:-(topNfeat+1):-1]  
    # #reorganize eig vects largest to smallest
    # 从大到小排序。
    redEigVects = eigVects[:,eigValInd]       
    # 3. 利用N个特征将原始数据转换到新空间中
    # transform data into new dimensions
    lowDDataMat = meanRemoved * redEigVects 
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # 原始数据被重构后返回用于调试，同时降维之后的数据集也被返回了。
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 计算出那些非NaN值的平均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        # 将所有NaN替换为该平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
