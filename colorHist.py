#API实现
#计算直方图函数：cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]])
#imaes:输入的图像
# channels:选择图像的通道
# mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
# histSize:使用多少个bin(柱子)，一般为256
# ranges:像素值的范围，一般为[0,255]表示0~255
# 后面两个参数基本不用管。
# 注意，除了mask，其他四个参数都要带[]号
import cv2
import numpy as np


def ImageHist(image,type):
    color = (255,255,255)
    windowName = 'Gray'
    if type == 31:
        color = (255,0,0)
        windowName = 'B Hist'
    elif type == 32:
        color = (0,255,0)
        windowName = 'G Hist'
    elif type == 33:
        color = (0,0,255)
        windowName = 'R Hist'
    hist = cv2.calcHist([image],[0],None,[256],[0.0,255.0])
    minV, maxV, minL, maxL = cv2.minMaxLoc(hist)#返回矩阵的最小值，最大值，并得到最大值，最小值的索引
    histImg = np.zeros([256,256,3],np.uint8)
    for h in range(0,256):
        intenNormal = int(hist[h]*256/maxV)#归一化
        cv2.line(histImg,(h,256),(h,256-intenNormal),color)
    cv2.imshow(windowName,histImg)
    return  histImg


img = cv2.imread("D:/code/Single-Underwater-Image-Enhancement-and-Color-Restoration-master"
"/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark/UIQS/B/B_001.jpg")
# img = cv2.imread('D:/code/Single-Underwater-Image-Enhancement-and-Color-Restoration-master'
#                 '/Underwater Image Enhancement/OutputImages/CLAHE/B/B_001_CLAHE.jpg')
channels = cv2.split(img)#分解通道：RGB——>R，G，B
for i in range(0,3):
    ImageHist(channels[i],31+i)
cv2.waitKey(0)