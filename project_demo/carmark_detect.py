import numpy as np
import cv2
from skimage import morphology
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def show_coler_space(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(img)

    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    fig = plt.figure()

    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def show_color(lower, upper):
    low_square = np.full((10, 10, 3), lower, dtype=np.uint8) / 255.0
    up_square = np.full((10, 10, 3), upper, dtype=np.uint8) / 255.0

    plt.subplot(1, 2, 1)
    plt.imshow(hsv_to_rgb(low_square))
    plt.subplot(1, 2, 2)
    plt.imshow(hsv_to_rgb(up_square))
    plt.show()

def get_morphology_image(mask_image):
    """
    对二值化图像进行形态学处理-消除噪音影响
      1) 先进行膨胀处理
         膨胀就是使用算法，来将图像的边缘扩大些。作用就是将目标的边缘或者是内部的坑填掉,去掉较小的孔洞
      2) 再进行腐蚀处理
         腐蚀：腐蚀会把物体的边界腐蚀掉，主要应用在去除白噪声，也可以断开连在一起的物体
    """

    kernel = np.ones((5, 5), np.uint8)
    dialationIamge = cv2.dilate(mask_image, kernel, iterations=1)  # 膨胀处理
    eroded_iamge = cv2.erode(dialationIamge, kernel, iterations=3)  # 腐蚀处理

    cv2.imshow("Eroded Iamge ", eroded_iamge)  # 显示形态学处理过后的图片
    return eroded_iamge


def rgb2hsv(r, g, b):
    Cmax = max(r, g, b)
    Cmin = min(r, g, b)
    H = 0
    delta = Cmax - Cmin
    if delta != 0:
        if r == Cmax:
            H = (g - b) / delta
        if g == Cmax:
            H = 2 + (b - r) / delta
        if b == Cmax:
            H = 4 + (r - g) / delta

    H = H * 60
    if H < 0:
        H = H + 360

    V = Cmax
    if Cmax != 0:
        S = delta / Cmax
    else:
        S = 0

    return H, S, V


img_path = r'D:\code\carmark\mark (23).jpg'
kernel = np.ones((8, 8), np.uint8)
img = cv2.imread(img_path)
h, w, _ = img.shape
crop_h = int(h / 3)
crop_w = int(w / 3)
lower = np.array([20, 21, 42])
upper = np.array([63,63, 99])
# 取图片的角落
# img = img[:crop_h, :crop_w]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




ero = cv2.erode(img, kernel)
gray_ero = cv2.cvtColor(ero, cv2.COLOR_BGR2GRAY)
# gray_ero = np.float32(gray_ero)
ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|
                         cv2.THRESH_OTSU)

se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6), (-1, -1))
thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, se)
thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, se2)
# thr = cv2.bitwise_not(thr)
# gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(thr,100, 100)
# plt.imshow(ero),plt.show()

'''
connectivity：4或者8， 判断连通的像素点，周围4像素或者8像素，默认为8；
labels：图像标记；
stats：[[x1, y1, width1, height1, area1], ...[xi, yi, widthi, heighti, areai]]，存放外接矩形和连通区域的面积信息；
centroids:[cen_x, cen_y]，质心的点坐标，浮点类型
'''
# _, labels, stats, centroid = cv2.connectedComponentsWithStats(erosion)
# # 需要考虑背景，最大连通区域是整个图像
# max_area = sorted(stats, key = lambda s : s[-1], reverse = False)[-2]
# # 绘制连通区域
# cv2.rectangle(img, (max_area[0], max_area[1]), (max_area[0] + max_area[2], max_area[1] + max_area[3]), (25, 25, 255), 3)
# # 按照连通区域的索引来打上标签
# cv2.putText(img, str(1), (max_area[0], max_area[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)

# cnt, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# img2 = cv2.drawContours(img.copy(), cnt,-1, (0, 0, 255), thickness=5)
# skeleton = morphology.skeletonize(mask_b)
# skeleton = np.array(skeleton * 255, dtype='uint8')
# area = 0
# for i in cnt:
#     area += cv2.contourArea(i)
# print(area)
# rows = [max(row) for row in edge[:,]]
# y_top = np.nonzero([max(row) for row in edge[:,]])
# x_top = int(np.mean(np.nonzero(edge[y_top])))
# print(x_top ,y_top)
# print(rgb2hsv())
show_color(lower, upper)
show_coler_space(ero)
cv2.imshow('mask', cv2.resize(thr, None, fx=0.5, fy=0.5))
cv2.imshow('color', cv2.resize(gray, None, fx=0.5, fy=0.5))

cv2.waitKey(0)
cv2.destroyAllWindows()
