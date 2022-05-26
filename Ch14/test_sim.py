# coding: utf-8
import svdRec
from numpy import *

initData = svdRec.loadExData()
myMat = mat(initData)
ecludSimVal = svdRec.ecludSim(myMat[:,0], myMat[:,4])
print("ecludSimVal : ", ecludSimVal)
ecludSimVal = svdRec.ecludSim(myMat[:,0], myMat[:,1])
print("ecludSimVal : ", ecludSimVal)

cosSimVal = svdRec.cosSim(myMat[:,0], myMat[:,4])
print("cosSimVal : ", cosSimVal)
cosSimVal = svdRec.cosSim(myMat[:,0], myMat[:,1])
print("cosSimVal : ", cosSimVal)

pearsSimVal = svdRec.pearsSim(myMat[:,0], myMat[:,4])
print("pearsSimVal : ", pearsSimVal)
pearsSimVal = svdRec.pearsSim(myMat[:,0], myMat[:,1])
print("pearsSimVal : ", pearsSimVal)




