# coding: utf-8
import fpGrowth
from numpy import *

simpDat = fpGrowth.loadSimpDat()
print("simpDat : ", simpDat)

# 使用dataSet更好理解一点。
dataSet = fpGrowth.createInitSet(simpDat)
print("dataSet : ", dataSet)

# 例如：对于 {'r': 3, 'z': 5, 'h': 1, 'p': 2, 'j': 1, 'x': 4, 'u': 1, 
# 'v': 1, 'y': 3, 'w': 1, 't': 3, 's': 3, 'o': 1, 
# 'n': 1, 'q': 2, 'm': 1, 'e': 1} 
# 只剩下{'r', 'z', 'x', 'y', 's', 't'}

print("----------createTree----------")
myFPTree, myHeaderTab = fpGrowth.createTree(dataSet, 3)
print("myHeaderTab : ", myHeaderTab)
myFPTree.disp()

print("----------findPrefixPath('x', myHeaderTab['x'][1])----------")
x1Item = fpGrowth.findPrefixPath('x', myHeaderTab['x'][1])
print("x1Item : ", x1Item)
print("----------findPrefixPath('z', myHeaderTab['z'][1])----------")
z1Item = fpGrowth.findPrefixPath('z', myHeaderTab['z'][1])
print("z1Item : ", z1Item)
print("----------findPrefixPath('r, myHeaderTab['r'][1])----------")
r1Item = fpGrowth.findPrefixPath('r', myHeaderTab['r'][1])
print("r1Item : ", r1Item)

print("----------mineTree----------")
freqItems = []
condTree = fpGrowth.mineTree(myFPTree, myHeaderTab, 3, set([]), freqItems)
print("freqItems : ", freqItems)
print("condTree : ", condTree)


