import cv2
import numpy as np

"""
REFER: https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
2018-06-30 Yonv1943
2018-07-01 comment to test.png
2018-07-01 gray in threshold, hierarchy
2018-07-01 draw_approx_hull_polygon() no [for loop]
2018-11-24 
"""




def perspective_crop(img, rotated_box, box):
    width = int(rotated_box[1][0])
    height = int(rotated_box[1][1])

    print(width, height)

    if width > height:
        w = width
        h = height
    else:
        w = height
        h = width

    src_pts = box.astype("float32")
    dst_pts = np.array([[w - 1, h - 1],
                        [0, h - 1],
                        [0, 0],
                        [w - 1, 0]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))

    #cv2.imshow('1', warped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('yinzi_unnamed_30_crop.jpg', warped)
    return warped

def draw_contours(img, cnts):  # conts = contours
    img = np.copy(img)
    img = cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    return img


def draw_min_rect_circle(img, cnts):  # conts = contours
    img = np.copy(img)
    tmp_circumference = 0
    max_rect = None
    rotated_boxs = []
    boxs = []
    for cnt in cnts:
        #x, y, w, h = cv2.boundingRect(cnt)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

        rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        rect_circumference = rect[1][0] + rect[1][1]
        if rect_circumference > 1000:
            #tmp_circumference = rect_circumference
            rotated_boxs.append(rect)
            #(x, y), radius = cv2.minEnclosingCircle(cnt)
            #center, radius = (int(x), int(y)), int(radius) # center and radius of minimum enclosing circle
            #img = cv2.circle(img, center, radius, (0, 0, 255), 2)# red
            draw_rect = np.int0(cv2.boxPoints(rect))
            boxs.append(draw_rect)
            #cv2.drawContours(img, [draw_rect], 0, (0, 255, 0), 2)  # green
    return img


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


def get_bbox(box):
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    bbox = np.array([left_point_x, top_point_y, right_point_x, bottom_point_y])

    return bbox

def run():
    image = cv2.imread(r'D:\code\Optical_Components\original_image\A1_12M.jpg')  # a black objects on white image is better
    blur_img = cv2.GaussianBlur(image.copy(), (5, 5), 10)
    gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 63, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    #thresh = cv2.Canny(image, 127, 256)
    #thresh = cv2.bitwise_not(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy, ":hierarchy")
    """
    [[[-1 -1 -1 -1]]] :hierarchy  # cv2.Canny()

    [[[ 1 -1 -1 -1]
      [ 2  0 -1 -1]
      [ 3  1 -1 -1]
      [-1  2 -1 -1]]] :hierarchy  # cv2.threshold()
    """
    # image, rotated_boxs, boxs = draw_min_rect_circle(image.copy(), contours)
    # cv2.imshow('1',cv2.resize(image, None,fx=0.5,fy=0.5))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for i, rotated_box in enumerate(rotated_boxs):
    #     warped = perspective_crop(image, rotated_box, boxs[i])
    #     cv2.imwrite(f'corp_{i}.png', warped)





    imgs = [
        image, thresh,
        cv2.drawContours(image.copy(), contours,-1, (0, 0, 255), 3),
        draw_min_rect_circle(image, contours),
        draw_approx_hull_polygon(image, contours),
    ]

    for img in imgs:
        # cv2.imwrite("%s.jpg" % id(img), img)
        img = cv2.resize(img,None,fx=0.5,fy=0.5)
        cv2.imshow("contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
pass