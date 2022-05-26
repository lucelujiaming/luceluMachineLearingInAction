# coding: utf-8

import sys
import regression
from numpy import *

abX, abY = regression.loadDataSet('abalone.txt')
yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1  = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

rssError01 = regression.rssError(abY[0:99], yHat01.T)
rssError1  = regression.rssError(abY[0:99], yHat1.T)
rssError10 = regression.rssError(abY[0:99], yHat10.T)

print("abY[0:99] : ", abY[0:99])
print("yHat01.T : ", yHat01.T)
print("abY[0:99] - yHat01.T : ", abY[0:99] - yHat01.T) 
print("rssError01 : ", rssError01)

print("abY[0:99] : ", abY[0:99])
print("yHat1.T : ", yHat1.T)
print("abY[0:99] - yHat1.T : ", abY[0:99] - yHat1.T) 
print("rssError1  : ", rssError1 )

print("abY[0:99] : ", abY[0:99])
print("yHat10.T  : ", yHat10.T)
print("abY[0:99] - yHat10.T : ", abY[0:99] - yHat10.T) 
print("rssError10 : ", rssError10)
