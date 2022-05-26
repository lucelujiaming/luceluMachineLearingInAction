# coding: utf-8
import fpGrowth
from numpy import *

rootNode = fpGrowth.treeNode('pyramid', 9, None)
rootNode.children['eye'] = fpGrowth.treeNode('eye', 13, None)
rootNode.disp()

rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix', 3, None)
rootNode.disp()



