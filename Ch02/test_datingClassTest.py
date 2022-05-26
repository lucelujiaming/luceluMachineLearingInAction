# coding: utf-8

import kNN
import matplotlib
from matplotlib import pyplot as plt

# 支持中文
plt.rcParams['font.family'] = ['Songti SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


kNN.datingClassTest()

