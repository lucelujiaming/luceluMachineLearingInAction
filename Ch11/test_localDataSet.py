# coding: utf-8
import apriori
from numpy import *

def loadLocalDataSet():
    return [[1, 3, 4, 6, 7], [2, 3, 5, 6, 7], [1, 2, 3, 5, 6, 7], [2, 5, 6, 7]]

dataSet = loadLocalDataSet()
print("dataSet : \n", dataSet)

print("-----------------------apriori--------------------------------- \n")
L, suppData = apriori.apriori(dataSet, minSupport = 0.5)
print("L : \n", L)
print("suppData : \n", suppData)

print("-----------------------aprioriGen--------------------------------- \n")
itemSet = apriori.aprioriGen(L[0], 2)
print("itemSet : \n", itemSet)

print("-----------------------generateRules--------------------------------- \n")
# rules = apriori.generateRules(L, suppData,  minConf=0.7)
rules = apriori.generateRules(L, suppData,  minConf=0.5)
print("rules : \n", rules)
