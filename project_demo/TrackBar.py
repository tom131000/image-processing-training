import cv2
import numpy as np
import cv2 as cv

def ManyImgs(scale, imgarray):
    rows = len(imgarray)         # 元组或者列表的长度
    cols = len(imgarray[0])      # 如果imgarray是列表，返回列表里第一幅图像的通道数，如果是元组，返回元组里包含的第一个列表的长度
    # print("rows=", rows, "cols=", cols)

    # 判断imgarray[0]的类型是否是list
    # 是list，表明imgarray是一个元组，需要垂直显示
    rowsAvailable = isinstance(imgarray[0], list)

    # 第一张图片的宽高
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    # print("width=", width, "height=", height)

    # 如果传入的是一个元组
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # 遍历元组，如果是第一幅图像，不做变换
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # 将其他矩阵变换为与第一幅图像相同大小，缩放比例为scale
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
                # 如果图像是灰度图，将其转换成彩色显示
                if  len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # 创建一个空白画布，与第一张图片大小相同
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows   # 与第一张图片大小相同，与元组包含列表数相同的水平空白图像
        for x in range(0, rows):
            # 将元组里第x个列表水平排列
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)   # 将不同列表垂直拼接
    # 如果传入的是一个列表
    else:
        # 变换操作，与前面相同
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # 将列表水平排列
        hor = np.hstack(imgarray)
        ver = hor
    return ver

def draw_min_rect_circle(im, cnts):  # conts = contours
    im = np.copy(im)
    tmp_circumference = 0
    rect_box = None
    max_rect = None
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        rect_circumference = rect[1][0] + rect[1][1]
        if rect_circumference > 0:
            tmp_circumference = rect_circumference
            max_rect = rect
    #print(f"width:{max_rect[1][0]},height:{max_rect[1][1]}")
        rect_box = np.int0(cv2.boxPoints(max_rect))
        draw_rect = rect_box
        cv2.drawContours(im, [draw_rect], 0, (0, 255, 0), 2)  # green
    return im, rect_box

def nothing(x):
    pass
# 创建一个黑色的图像，一个窗口
img = np.zeros((300,512,3), np.uint8)
# img = cv.imread(r'D:\code\yuanjian_2021_11_26\defect\masks\3_201908302_7_1_mask.jpg')
show = img.copy()
# edges = cv.Canny(img, 0, 255)
cv.namedWindow('image')
# cv.resizeWindow('image', 500,500)
# 创建颜色变化的轨迹栏
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
# cv.createTrackbar('Gray','image',0,255,nothing)
# 为 ON/OFF 功能创建开关
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)
while(1):
    cv.imshow('image', cv2.resize(show, None,fy=0.5,fx=0.5))
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # 得到四条轨迹的当前位置
    r_th = cv.getTrackbarPos('R','image')
    g_th = cv.getTrackbarPos('G','image')
    b_th = cv.getTrackbarPos('B','image')
    # gray_th = cv2.getTrackbarPos('Gray', 'image')
    s = cv.getTrackbarPos(switch,'image')
    if s == 0:
        show = img
    else:
        #edges = cv.Canny(img, th1, th2)
        #contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # blur_img = cv2.GaussianBlur(img.copy(), (5, 5), 10)
        # b,g,r = cv2.split(img.copy())
        # gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
        # _, thresh_b = cv2.threshold(b, b_th, 255, cv2.THRESH_BINARY)
        # _, thresh_g = cv2.threshold(g, g_th, 255, cv2.THRESH_BINARY)
        # _, thresh_r = cv2.threshold(r, r_th, 255, cv2.THRESH_BINARY)
        # _, thresh_gray = cv2.threshold(gray, gray_th, 255, cv2.THRESH_BINARY)
        # thresh_gray = cv2.bitwise_not(thresh_gray)
        # contours_b, hierarchy = cv2.findContours(thresh_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_g, hierarchy = cv2.findContours(thresh_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_r, hierarchy = cv2.findContours(thresh_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_gray, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # rect_b,box = draw_min_rect_circle(img.copy(),contours_b)
        # rect_g,box = draw_min_rect_circle(img.copy(),contours_g)
        # rect_r,box = draw_min_rect_circle(img.copy(),contours_r)
        # rect_gray,box = draw_min_rect_circle(img.copy(),contours_gray)
        # show = ManyImgs(0.2, ([img, thresh_b, rect_b],
        #                       [img, thresh_g, rect_g],
        #                       [img, thresh_r, rect_r],
        #                 [img, thresh_gray, rect_gray]) )
        # thresh_gray = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2BGR )
        # show,_= draw_min_rect_circle(thresh_gray, contours_gray)
        #show = edges
        show[:] = [b_th, g_th, r_th]
cv.destroyAllWindows()

