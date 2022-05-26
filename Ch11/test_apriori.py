# coding: utf-8
import apriori
from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dataSet = apriori.loadDataSet()
# print("dataSet : \n", dataSet)

C1 = apriori.createC1(dataSet)
print("C1 : \n", list(C1))

D = map(set, dataSet)
DD = list(D)
print("lucelu DD : \n", DD)

L1, supportData0 = apriori.scanD(DD, C1, 0.5)
print("L1 : \n", L1)
print("supportData0 : \n", supportData0)
print("supportData0 : \n", list(supportData0))

L, suppData = apriori.apriori(dataSet, minSupport = 0.5)
print("L : \n", L)
print("suppData : \n", suppData)

itemSet = apriori.aprioriGen(L[0], 2)
# 1235的两两组合。
print("itemSet : \n", itemSet)

rules = apriori.generateRules(L, suppData,  minConf=0.7)
print("rules : \n", rules)

rules50 = apriori.generateRules(L, suppData,  minConf=0.5)
print("rules50 : \n", rules50)

L1_70, supportData0_70 = apriori.apriori(dataSet, minSupport = 0.7)
print("L1_70 : \n", L1_70)
print("supportData0_70 : \n", supportData0_70)

itemSet70 = apriori.aprioriGen(L1_70[0], 2)
print("itemSet70 : \n", itemSet70)

rules70 = apriori.generateRules(L1_70, supportData0_70,  minConf=0.7)
print("rules70 : \n", rules70)

rules7050 = apriori.generateRules(L1_70, supportData0_70,  minConf=0.5)
print("rules7050 : \n", rules7050)


