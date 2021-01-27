import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def reduce_gray_level(img,level):
    resolved_img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    num = 256 / level
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            resolved_img[i, j] = np.fix((img[i, j] / num)) * 255 / (level - 1)
    return resolved_img

def neighborhood(img, factor):
    resolved_img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            resolved_img[i, j] = np.nanmean(img[i:i+factor, j:j+factor])
    return resolved_img

def resolution(img, factor):
    resolved_img = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    for i in range(0, img.shape[0],factor):
        for j in range(0, img.shape[1],factor):
            resolved_img[i:i+factor, j:j+factor] = np.nanmean(img[i:i+factor, j:j+factor])
    return resolved_img



# img = cv.imread("rose.jpg", cv.IMREAD_GRAYSCALE)
# result = ndimage.generic_filter(img, np.nanmean, size=20, mode='constant', cval=np.NaN)
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.show()


img2 = cv.imread("../rose.jpg", cv.IMREAD_GRAYSCALE)
(h, w) = img2.shape[:2]
center = (w / 2, h / 2)

# rotate the image by 180 degrees
M = cv.getRotationMatrix2D(center, 180, 1.0)
rotated = cv.warpAffine(img2, M, (w, h))

result = reduce_gray_level(img2, 2)
result2 = neighborhood(img2, 30)
result3 = resolution(img2, 30)
#原图
plt.imshow(img2, cmap='gray',interpolation='bicubic')
plt.show()

#改变灰度
plt.imshow(result, cmap='gray',interpolation='bicubic')
plt.show()

#邻接求均值替换
plt.imshow(result2, cmap='gray',interpolation='bicubic')
plt.show()

#降低分辨率
plt.imshow(result3, cmap='gray',interpolation='bicubic')
plt.show()

#翻转
plt.imshow(rotated, cmap='gray',interpolation='bicubic')
plt.show()