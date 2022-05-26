# coding: utf-8
import svdRec
from numpy import *

initData = svdRec.loadExData2()
myMat = mat(initData)

U, Sigma, VT = linalg.svd(initData)
print("U : ", U)
print("Sigma : ", Sigma)
print("VT : ", VT)

Sig2 = Sigma ** 2
print("sum(Sig2) : ", sum(Sig2))
print("sum(Sig2) * 0.9 : ", sum(Sig2) * 0.9)
print("sum(Sig2[:2]) : ", sum(Sig2[:2]))
print("sum(Sig2[:3]) : ", sum(Sig2[:3]))
minorSum = abs(sum(Sig2) * 0.9 - sum(Sig2[:3]))
print("minorSum : ", minorSum)
print("1 - abs(minorSum/sum(Sig2) * 0.9) : ", 
	1 - abs(minorSum/sum(Sig2) * 0.9))
