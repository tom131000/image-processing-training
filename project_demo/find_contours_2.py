import cv2
import numpy as np


def read_resize_image(path, size=1.0):
    """
    读取原始图片文件并从新设置大小
     1) 使用imread读取文件
     2）使用resize从新设置文件大小
     3) 函数返回的是图片的原始数据，可理解为一个三维数组，每个维度分别表示长、宽、通道数，数组中的值表示图片每个像素点不同通道的取值
    """

    original_image = cv2.imread(path)
    if size != 1.0:
        height, width = original_image.shape[:2]
        size = (int(width * size), int(height * size))
        original_image = cv2.resize(original_image, size, interpolation=cv2.INTER_AREA)

    # print(original_image)
    # print(original_image.shape)

    return original_image


def get_hsv_image(original_image):
    """
      获取hsv色彩空间图片
       1) 先将BGR色彩转化为HSV色彩
      """

    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)  # 将BGR色彩转化为HSV色彩

    cv2.imshow("HSV Image", hsv_image)
    print(hsv_image.shape)

    return hsv_image


def get_mask_image(hsv_image, lower, upper):
    """
    获取二值化图片
     1）通过阀值获取二值化图像
    """
    mask_image = cv2.inRange(hsv_image, lower, upper)  # 获取二值化图片

    cv2.imshow("Mask Image", mask_image)
    print(mask_image)
    print(mask_image.shape)

    return mask_image


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
    eroded_iamge = cv2.erode(dialationIamge, kernel, iterations=1)  # 腐蚀处理

    cv2.imshow("Eroded Iamge ", eroded_iamge)  # 显示形态学处理过后的图片
    return eroded_iamge


def get_contours(eroded_iamge):
    """
    获取光伏组件串图片的轮廓并在原始图像上绘制轮廓
       输入为形态学处理后的二值图像，黑色为背景，白色为目标
       cv2.RETR_EXTERNAL 只检测外轮廓
       cv2.CHAIN_APPROX_SIMPLE
    """
    # contours是轮廓本身，hierarchy 每条轮廓对应的属性（后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号）
    contours, hierarchy = cv2.findContours(eroded_iamge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取组件区域轮廓

    return contours


def get_and_save_areas(contours, original_image):
    """
       在原始图片中切割并保存轮廓区域
    """
    min_area = 300  # 过滤那些小面积的噪音区域
    index = 0  # 用来标识组件区域的数量
    for numcontours, contour in enumerate(contours):
        iamge_area = cv2.contourArea(contour)
        if iamge_area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            module_image = original_image[y:y + h, x: x + w]
            index = index + 1
            cv2.imwrite(
                'Output/' + str(x) + '_' + str(y) + '_' + str(w) + '_' + str(h) + '_' + str(iamge_area) + '_' + str(
                    index) + '.jpg', module_image)  # 文件名以坐标（X、Y）,图片的高度与宽度,图片面积,文件序号

    return index


def execute():
    """
       组合执行每个一个处理步骤
    """

    path = 'D:/code/Optical Components/A7/12L.jpg'
    original_image = read_resize_image(path, 0.5)  # 原始图像为3000X40000为方便观察，先将图片缩小5倍 既0.2

    hsv_image = get_hsv_image(original_image)  # 将rgb原始图像转化为hsv图像
    #hsv_image = cv2.bitwise_not(hsv_image)
    #cv2.imshow('2', hsv_image)
    lower = np.array([44, 57, 130])
    upper = np.array([120, 191, 249])
    mask_image = get_mask_image(hsv_image, lower, upper)  # 获取二值化图像 通过HSV的高低阈值，提取图像部分区域

    eroded_iamge = get_morphology_image(mask_image)  # 对二值化的图像进行形态学处理

    contours = get_contours(eroded_iamge)  # 获取光伏组件轮廓区域
    get_and_save_areas(contours, original_image)  # 将获取的轮廓区域截取图片并保存到文件

    cv2.drawContours(original_image, contours, -1, (0, 0, 255), 1)  # 在原图上绘制轮廓
    cv2.imshow("Original Image", original_image)  # 显示原始带轮廓图像


execute()  # 执行入口函数

while True:
    cv2.waitKey(0)
    cv2.destroyAllWindows()