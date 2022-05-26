from numpy import *

# from Tkinter import *
from tkinter import *
import regTrees

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# 主要目的是把树绘制出来。
def reDraw(tolS,tolN):
    # 清空之前的图像，使得前后两个图像不会重叠。
    reDraw.f.clf()        # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    # 检查复选框是否被选中。
    # 选中就构建模型树。
    if chkBtnVar.get():
        if tolN < 2: 
            tolN = 2
        myTree=regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,\
                                   regTrees.modelErr, (tolS,tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, \
                                       regTrees.modelTreeEval)
    # 否则就构建回归树。
    else:
        myTree=regTrees.createTree(reDraw.rawDat, ops=(tolS,tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    # Use scatter for data set
    # reDraw.a.scatter(reDraw.rawDat[:,0], reDraw.rawDat[:,1], s=5)
    # 真实值采用scatter()方法绘制离散型散点图，
    reDraw.a.scatter(array(reDraw.rawDat[:,0]), array(reDraw.rawDat[:,1]), s=5)
    # 而预测值则采用plot()方法绘制连续曲线。
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0) #use plot for yHat
    # reDraw.canvas.show()
    reDraw.canvas.draw()
# 试图理解用户的输入并防止程序崩溃。
def getInputs():
    # tolN期望的输入是整数。
    try: tolN = int(tolNentry.get())
    except: 
        tolN = 10 
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0,'10')
    # tolS期望的输入是浮点数，
    try: tolS = float(tolSentry.get())
    except: 
        tolS = 1.0 
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS
# 有人点击ReDraw按钮时就会调用该函数。
def drawNewTree():
    # 第一，调用getInputs()方法得到输入框的值；
    tolN,tolS = getInputs()#get values from Entry boxes
    # 第二，利用该值调用reDraw()方法生成一个漂亮的图。
    reDraw(tolS,tolN)
    
root=Tk()
# 整个界面分为三个部分。第一部分是绘图部分。
reDraw.f = Figure(figsize=(5,4), dpi=100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
# reDraw.canvas.show()
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

# 第二个部分是控制部分。包括两个输入框。一个按钮，一个CheckBox。
# 一个tolN的输入框。
Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'10')
# 一个tolS的输入框。
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')
# 一个按钮。
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
# 一个CheckBox。
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)
# 加载用于绘制的数据文件。
reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0, 10)
               
root.mainloop()
