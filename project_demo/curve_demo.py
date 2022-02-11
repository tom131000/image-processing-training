import numpy as np
from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt

## 求二维圆周长，半径为1，采用参数形式
def circle_2d(dt=0.001,plot=True):
    dt = dt # 变化率
    t = np.arange(0,2*np.pi, dt)
    x = np.cos(t)
    y = np.sin(t)

    # print(len(t))
    area_list = [] # 存储每一微小步长的曲线长度

    for i in range(1,len(t)):
        # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
        dl_i = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 )
        # 将计算结果存储起来
        area_list.append(dl_i)

    area = sum(area_list)# 求和计算曲线在t:[0,2*pi]的长度

    print("二维圆周长：{:.4f}".format(area))
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y)
        plt.title("circle")
        plt.show()


## 二维空间曲线，采用参数形式
def curve_param_2d(dt=0.0001,plot=True):
    dt = dt # 变化率
    t = np.arange(0,2*np.pi, dt)
    x = t*np.cos(t)
    y = t*np.sin(t)

    # print(len(t))
    area_list = [] # 存储每一微小步长的曲线长度

    # 下面的方式是循环实现
    # for i in range(1,len(t)):
    #     # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
    #     dl_i = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 )
    #     # 将计算结果存储起来
    #     area_list.append(dl_i)

    # 更加pythonic的写法
    area_list = [np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) for i in range(1,len(t))]

    area = sum(area_list)# 求和计算曲线在t:[0,2*pi]的长度

    print("二维参数曲线长度：{:.4f}".format(area))

    if plot:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y)
        plt.title("2-D Parameter Curve")
        plt.show()

## 二维空间曲线
def curve_2d(dt=0.0001,plot=True):
    dt = dt # 变化率
    t = np.arange(-6,10, dt)
    x = t
    y = x**3/8 - 4*x + np.sin(3*x)

    # print(len(t))
    area_list = [] # 存储每一微小步长的曲线长度

    # for i in range(1,len(t)):
    #     # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
    #     dl_i = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 )
    #     # 将计算结果存储起来
    #     area_list.append(dl_i)

    area_list = [np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) for i in range(1,len(t))]

    area = sum(area_list)# 求和计算曲线在t:[0,2*pi]的长度

    print("二维曲线长度：{:.4f}".format(area))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y)
        plt.title("2-D Curve")
        plt.show()

## 三维空间曲线，采用参数形式
def curve_3d(dt=0.001,plot=True):
    dt = dt # 变化率
    t = np.arange(0,2*np.pi, dt)
    x = t*np.cos(t)
    y = t*np.sin(t)
    z = 2*t

    # print(len(t))
    area_list = [] # 存储每一微小步长的曲线长度

    for i in range(1,len(t)):
        # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
        dl_i = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2 )
        # 将计算结果存储起来
        area_list.append(dl_i)

    area = sum(area_list)# 求和计算曲线在t:[0,2*pi]的长度

    print("三维空间曲线长度：{:.4f}".format(area))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(x,y,z)
        plt.title("3-D Curve")
        plt.show()

x = np.arange(-1, 1, 0.02)
y = ((x * x - 1) ** 3 + 1) * (np.cos(x * 2) + 0.6 * np.sin(x * 1.3))

y1 = y + (np.random.rand(len(x)) - 0.5)

##################################
### 核心程序
# 使用函数y=ax^3+bx^2+cx+d对离散点进行拟合，最高次方需要便于修改，所以不能全部列举，需要使用循环
# A矩阵
m = []
for i in range(7):  # 这里选的最高次为x^7的多项式
    a = x ** (i)
    m.append(a)
A = np.array(m).T
b = y1.reshape(y1.shape[0], 1)


##################################

def projection(A, b):
    AA = A.T.dot(A)  # A乘以A转置
    w = np.linalg.inv(AA).dot(A.T).dot(b)
    print(w)  # w=[[-0.03027851][ 0.1995869 ] [ 2.43887827] [ 1.28426472][-5.60888682] [-0.98754851][ 2.78427031]]
    return A.dot(w)


yw = projection(A, b)
yw.shape = (yw.shape[0],)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
plt.plot(x, y, color='g', linestyle='-', marker='', label=u"理想曲线")
plt.plot(x, y1, color='m', linestyle='', marker='o', label=u"已知数据点")
plt.plot(x, yw, color='r', linestyle='', marker='.', label=u"拟合曲线")
plt.legend(loc='upper left')
plt.show()