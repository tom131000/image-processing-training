#! usr/bin/env python
# coding:utf-8

import cv2
import matplotlib.pyplot as plt


img = cv2.imread('D:/code/video/picOutput2/0001.jpg',0)
# 其中，0表示将图片以灰度读出来。


#### 图像边缘处理sobel细节
sobelx = cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=3)
# 利用Sobel方法可以进行sobel边缘检测
# img表示源图像，即进行边缘检测的图像
# cv2.CV_64F表示64位浮点数即64float。
# 这里不使用numpy.float64，因为可能会发生溢出现象。用cv的数据则会自动
# 第三和第四个参数分别是对X和Y方向的导数（即dx,dy），对于图像来说就是差分，这里1表示对X求偏导（差分），0表示不对Y求导（差分）。其中，X还可以求2次导。
# 注意：对X求导就是检测X方向上是否有边缘。
# 第五个参数ksize是指核的大小。

# 这里说明一下，这个参数的前四个参数都没有给谁赋值，而ksize则是被赋值的对象
# 实际上，这时可省略的参数，而前四个是不可省的参数。注意其中的不同点
# 还有其他参数，有需要的话可以去看，也可留言。

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# 与上面不同的是对y方向进行边缘检测

sobelXY = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
# 这里对两个方向同时进行检测，则会过滤掉仅仅只是x或者y方向上的边缘


##### 图像展示
# 展示上面处理的图片，包括源图像。
# 注意使用subplot和title方法
plt.subplot(2,2,1)
plt.imshow(img,'gray')
# 其中gray表示将图片用灰度的方式显示，注意需要使用引号表示这是string类型。
# 可以用本行命令显示'gray'的类型：print(type('gray'))
plt.title('src')
plt.subplot(2,2,2)
plt.imshow(sobelx,'gray')
plt.title('sobelX')
plt.subplot(2,2,3)
plt.imshow(sobely,'gray')
plt.title('sobelY')
plt.subplot(2,2,4)
plt.imshow(sobelXY,'gray')
plt.title('sobelXY')
plt.show()
