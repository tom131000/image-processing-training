import numpy as np
import cv2
import os

# rawImg = np.fromfile('./raw/Image_520-4.raw', dtype=np.uint8)
# img = cv2.imread("./raw/Image_520-5.jpg")
# ouput_dir = "output/"
# print(np.max(rawImg))
# rawImg = rawImg.reshape(1024, int(rawImg.size / 1024 /3), 3)
# if not os.path.exists(ouput_dir):
#     os.makedirs(ouput_dir)
# cv2.imwrite(ouput_dir + 'img_test.jpg', rawImg)


img1 = cv2.imread("0001.jpg")
img2 = cv2.imread("0001_FM.jpg")

# python里数组传递传递的是数组的引用
# 对图像进行缩放操作，方便观察
img1_resize = cv2.resize(img1, None, fx=0.5, fy=0.5)
img2_resize = cv2.resize(img2, None, fx=0.5, fy=0.5)
# 合并两张图像为一张
print(img1_resize.size == img2_resize.size)
half = int(np.shape(img1_resize)[1]/2)
diff = np.hstack((img1_resize[:, :half, :], img2_resize[:, half:, :]))
cv2.imshow('jpg', diff)
#cv2.imwrite("test.jpg", img1_resize)
cv2.waitKey()