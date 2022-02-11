import numpy as np
import cv2
from skimage import morphology

def draw_approx_hull_polygon(img, cnts):
    # img = np.copy(img)
    img = np.zeros(img.shape, dtype=np.uint8)

    cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue

    min_side_len = img.shape[0] / 32  # 多边形边长的最小值 the minimum side length of polygon
    min_poly_len = img.shape[0] / 16  # 多边形周长的最小值 the minimum round length of polygon
    min_side_num = 3  # 多边形边数的最小值
    approxs = [cv2.approxPolyDP(cnt, min_side_len, True) for cnt in cnts]  # 以最小边长为限制画出多边形
    approxs = [approx for approx in approxs if cv2.arcLength(approx, True) > min_poly_len]  # 筛选出周长大于 min_poly_len 的多边形
    approxs = [approx for approx in approxs if len(approx) > min_side_num]  # 筛选出边长数大于 min_side_num 的多边形
    # Above codes are written separately for the convenience of presentation.
    cv2.polylines(img, approxs, True, (0, 255, 0), 2)  # green

    hulls = [cv2.convexHull(cnt) for cnt in cnts]
    cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red

    # for cnt in cnts:
    #     cv2.drawContours(img, [cnt, ], -1, (255, 0, 0), 2)  # blue
    #
    #     epsilon = 0.02 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     cv2.polylines(img, [approx, ], True, (0, 255, 0), 2)  # green
    #
    #     hull = cv2.convexHull(cnt)
    #     cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red
    return img


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def perspective_crop(img, rotated_box, box):
    width = int(rotated_box[1][0])
    height = int(rotated_box[1][1])

    box = order_points(box)

    print(width, height)

    if width > height:
        w = width
        h = height
    else:
        w = height
        h = width

    src_pts = box.astype("float32")
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst_pts = np.array([[0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1],
                        [0, h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped


# 检测符合标准的麻点以及划痕，并返回其最小外接矩阵
def defect_detect(img, cnts):  # conts = contours
    img = np.copy(img)
    s_r_boxs = []
    dot_r_boxs = []
    s_boxs = []
    dot_boxs = []
    for i, cnt in enumerate(cnts):
        rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        if rect[1][0] < rect[1][1]:
            rect_w, rect_h = rect[1][0], rect[1][1]
        else:
            rect_w, rect_h = rect[1][1], rect[1][0]
        # 画出划痕并记录矩阵
        if rect_w > 1024 * (2 / 800) and rect_h > 1024 * (30 / 800) and rect_h / rect_w > 5:
            draw_rect = np.int0(cv2.boxPoints(rect))
            s_boxs.append(draw_rect)
            s_r_boxs.append(rect)
            cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)
            # cv2.drawContours(img, [draw_rect], 0, (0, 255, 0), 2)  # green
            continue
        # 画出麻点并记录矩阵
        if (rect_w + rect_h) / 2 >= 1024 * (10 / 800):
            draw_rect = np.int0(cv2.boxPoints(rect))
            dot_boxs.append(draw_rect)
            dot_r_boxs.append(rect)
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
            # cv2.drawContours(img, [draw_rect], 0, (255, 125, 0), 2)
    return img, s_r_boxs, s_boxs, dot_boxs, dot_r_boxs


def scratch_count(mask, r_boxs, boxs):
    for i, rotated_box in enumerate(r_boxs):
        warped = perspective_crop(mask, rotated_box, boxs[i])
        skel = morphology.skeletonize(warped / 255)
        skel = np.array(skel, dtype='uint8')
        if warped.shape[0] > 1024 * (23 / 800):
            s_defect_length[0]+=np.sum(skel)*800/1024
        elif 1024 * (16 / 800)< warped.shape[0] <= 1024 * (23 / 800):
            s_defect_length[1]+=np.sum(skel)*800/1024
        elif 1024 * (8 / 800)< warped.shape[0] <= 1024 * (16 / 800):
            s_defect_length[2]+=np.sum(skel)*800/1024
        elif 1024 * (2 / 800)<= warped.shape[0] <= 1024 * (8 / 800):
            s_defect_length[3]+=np.sum(skel)*800/1024
    pass

def dot_count(r_boxs):
    for r_box in r_boxs:
        if (r_box[1][0] + r_box[1][1]) / 2 > 1024 * (68 / 800):
            d_defect_count[0]+=1
        elif 1024 * (40 / 800) < (r_box[1][0] + r_box[1][1]) / 2 <= 1024 * (68 / 800):
            d_defect_count[1]+=1
        elif 1024 * (20 / 800) < (r_box[1][0] + r_box[1][1]) / 2 <= 1024 * (40 / 800):
            d_defect_count[2]+=1
        elif 1024 * (10 / 800) <= (r_box[1][0] + r_box[1][1]) / 2 <= 1024 * (20 / 800):
            d_defect_count[3]+=1
    pass


s_std = [0, 300, 500, 1000]
d_std = [0, 3, 5, 10]
img_path = './images/1_bengkou_unnamed_11.jpg'
mask_path = './masks/podian_20190826_14_0.png'
kernel = np.ones((9,9), np.uint8)
img = cv2.imread(img_path)
h,w,_ = img.shape
img = img[:int(h/3), int((w/3)*2):]
# mask = cv2.imread(mask_path, 0)
# mask_b = mask / 255
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# mask_cl = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 10)
dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
dst = cv2.dilate(dst, None)
gray = cv2.erode(gray, kernel)
# img[dst>0.01*dst.max()]=[0,0,255]
ret, thr = cv2.threshold(gray, 128, 255,cv2.THRESH_BINARY)
erosion = cv2.erode(img, kernel)
gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray,100, 100)
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

cnt, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img2 = cv2.drawContours(img.copy(), cnt,-1, (0, 0, 255), thickness=5)
s_defect_length = np.zeros((4), np.float32)
d_defect_count = np.zeros((4), np.float32)
# skeleton = morphology.skeletonize(mask_b)
# skeleton = np.array(skeleton * 255, dtype='uint8')
img2 = draw_approx_hull_polygon(img2.copy(), cnt)
area = 0
for i in cnt:
    area += cv2.contourArea(i)
print(area)
rows = [max(row) for row in edge[:,]]
y_top = np.nonzero([max(row) for row in edge[:,]])
x_top = int(np.mean(np.nonzero(edge[y_top])))
print(x_top ,y_top)
cv2.imshow('skeletion', cv2.resize(edge ,None, fx=1, fy=1))

# show = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
# show2, s_r_boxs, s_boxs, dot_r_box, dot_box = defect_detect(img, contours)
# scratch_count(mask, s_r_boxs, s_boxs)
# dot_count(dot_r_box)
# s_defect_length = s_defect_length - s_std
# d_defect_count = d_defect_count - d_std
# print(s_defect_length)
# print(d_defect_count)
# cv2.imshow('1', cv2.resize(show, None, fx=0.5, fy=0.5))
# cv2.imshow('2', cv2.resize(show2, None, fx=0.5, fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
