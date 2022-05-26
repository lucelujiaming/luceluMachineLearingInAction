# coding: utf-8
import fpGrowth
from numpy import *

parseDat = [line.split() for line in open('kosarak.dat').readlines()]
initSet = fpGrowth.createInitSet(parseDat)
myFPTree, myHeaderTab = fpGrowth.createTree(initSet, 100000)

myFreqList = []
fpGrowth.mineTree(myFPTree, myHeaderTab, 100000, set([]), myFreqList)
print("myFreqList : ", myFreqList)

