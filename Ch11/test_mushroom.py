# coding: utf-8
import apriori
from numpy import *

mushroomDatSet = [line.split() for line in 
    open('mushroom.dat').readlines()]
L, suppData = apriori.apriori(mushroomDatSet, minSupport = 0.3)
# print("L : ", L)

# 搜索包含有毒特征值2的频繁项集：
for item in L[1]:
    if item.intersection('2'):
        print("L[1] : ", item)

# 搜索包含有毒特征值2的频繁项集：
for item in L[3]:
    if item.intersection('2'):
        print("L[3] : ", item)
