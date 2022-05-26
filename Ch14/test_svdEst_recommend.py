# coding: utf-8
import svdRec
from numpy import *

initData = svdRec.loadExData()
myMat = mat(initData)

# Modify some data to make it interest
myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
myMat[3,3] = 2
print("myMat : ", myMat)
recommendCosTwo = svdRec.recommend(myMat, 1, estMethod = svdRec.svdEst)
print("recommendCosTwo : ", recommendCosTwo)

recommendEcludTwo = svdRec.recommend(myMat, 2, simMeas=svdRec.ecludSim, estMethod = svdRec.svdEst)
print("recommendEcludTwo : ", recommendEcludTwo)

recommendPearTwo = svdRec.recommend(myMat, 2, simMeas=svdRec.pearsSim, estMethod = svdRec.svdEst)
print("recommendPearTwo : ", recommendPearTwo)

