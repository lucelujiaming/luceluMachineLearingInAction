# coding: utf-8
import svdRec
from numpy import *

#  U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])
#  print("U : ", U)
#  print("Sigma : ", Sigma)
#  print("VT : ", VT)

initData = svdRec.loadExData()
print("Data : ", mat(initData))
U, Sigma, VT = linalg.svd(initData)
print("U : ", U)
print("Sigma : ", Sigma)
print("VT : ", VT)

Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
# 3 is a super parameter from the heuristic way.
newData = U[:,:3] * Sig3 * VT[:3,:]

print("initData : ", mat(initData))
print("newData : ", newData.astype(int))

