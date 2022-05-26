# coding: utf-8

from numpy import *

a = arange(24).reshape(4, 6)
print("a : ", a)

dataMat = mat(a)
print("dataMat.shape : ", shape(dataMat))
print("dataMat.T : ", dataMat.T)
print("a.transpose : ", a.transpose())
print("dataMat.transpose : ", dataMat.transpose())



