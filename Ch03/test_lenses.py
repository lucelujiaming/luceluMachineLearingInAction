# coding: utf-8

import trees
import treePlotter

import matplotlib
from matplotlib import pyplot as plt

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astimatic', 'tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print("lensesTree : ", lensesTree)
treePlotter.createPlot(lensesTree)
