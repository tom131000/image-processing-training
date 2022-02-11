from __future__ import print_function, absolute_import

import json
from glob import glob
from os.path import join

import numpy as np
import cv2
import os
import natsort



def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]                     # pred bbox top_x
    y1 = dets[:, 1]                     # pred bbox top_y
    x2 = dets[:, 2]                     # pred bbox bottom_x
    y2 = dets[:, 3]                     # pred bbox bottom_y
    scores = dets[:, 4]              # pred bbox cls score

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)    # pred bbox areas
    order = scores.argsort()[::-1]              # 对pred bbox按score做降序排序，对应step-2

    keep = []    # NMS后，保留的pred bbox
    while order.size > 0:
        i = order[0]          # top-1 score bbox
        keep.append(i)   # top-1 score的话，自然就保留了
        xx1 = np.maximum(x1[i], x1[order[1:]])   # top-1 bbox（score最大）与order中剩余bbox计算NMS
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)      # 无处不在的IoU计算~~~

        inds = np.where(ovr <= thresh)[0]     # 这个操作可以对代码断点调试理解下，结合step-3，我们希望剔除所有与当前top-1 bbox IoU > thresh的冗余bbox，那么保留下来的bbox，自然就是ovr <= thresh的非冗余bbox，其inds保留下来，作进一步筛选
        order = order[inds + 1]   # 保留有效bbox，就是这轮NMS未被抑制掉的幸运儿，为什么 + 1？因为ind = 0就是这轮NMS的top-1，剩余有效bbox在IoU计算中与top-1做的计算，inds对应回原数组，自然要做 +1 的映射，接下来就是step-4的循环

    return keep    # 最终NMS结果返回


def draw_min_rect_circle(img, cnts):  # conts = contours
    img = np.copy(img)
    tmp_circumference = 0
    max_rect = ((0,0), (0,0), (0.0),)
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        rect_circumference = rect[1][0] + rect[1][1]
        if rect_circumference > tmp_circumference:
            tmp_circumference = rect_circumference
            max_rect = rect
    rect_box = np.int0(cv2.boxPoints(max_rect))
    draw_rect = rect_box
    cv2.drawContours(img, [draw_rect], 0, (0, 255, 0), 2)  # green
    return img, rect_box, max_rect[1][0], max_rect[1][1]


def get_bbox(box):
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    bbox = np.array([left_point_x, top_point_y, right_point_x, bottom_point_y])

    return bbox


def get_IoU(pred_bbox, gt_bbox):
    """
    return iou score between pred / gt bboxes
    :param pred_bbox: predict bbox coordinate
    :param gt_bbox: ground truth bbox coordinate
    :return: iou score
    """

    x1min, y1min, x1max, y1max = pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]
    x2min, y2min, x2max, y2max = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # 计算iou
    iou = intersection / union
    return iou



def get_max_IoU(pred_bboxes, gt_bbox):
    """
    given 1 gt bbox, >1 pred bboxes, return max iou score for the given gt bbox and pred_bboxes
    :param pred_bbox: predict bboxes coordinates, we need to find the max iou score with gt bbox for these pred bboxes
    :param gt_bbox: ground truth bbox coordinate
    :return: max iou score
    """

    # bbox should be valid, actually we should add more judgements, just ignore here...
    # assert ((abs(gt_bbox[2] - gt_bbox[0]) > 0) and
    #         (abs(gt_bbox[3] - gt_bbox[1]) > 0))

    if pred_bboxes.shape[0] > 0:
        # -----0---- get coordinates of inters, but with multiple predict bboxes
        ixmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
        iymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
        ixmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
        iymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        # -----1----- intersection
        inters = iw * ih

        # -----2----- union, uni = S1 + S2 - inters
        uni = ((gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) +
               (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.) -
               inters)

        # -----3----- iou, get max score and max iou index
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    return overlaps, ovmax, jmax


if __name__ == "__main__":
    # test1
    Input = 'D:/code/Optical Components/'
    folders = os.listdir(Input)
    folders = natsort.natsorted(folders)
    scores = []
    pred_list = []
    for folder in folders:
        dir_names = Input + folder + '/'
        img_paths = sorted(glob(join(dir_names, "*.jpg")))
        json_paths = sorted(glob(join(dir_names, "*.json")))
        for i, img_path in enumerate(img_paths):
            image = cv2.imread(img_path)  # a black objects on white image is better
            blur_img = cv2.GaussianBlur(image.copy(), (5, 5), 10)
            gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
            # thresh = cv2.Canny(image,127,256)
            pred_list = []
            scores = []
            for thr in range(127):
                ret, thresh = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
                thresh = cv2.bitwise_not(thresh)
                contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                draw, rect_box,width,height = draw_min_rect_circle(
                    image, contours)
                if -3 <= (width-height) <= 3 and width != 0:
                    pred_bbox = get_bbox(rect_box)
                    with open(json_paths[i], 'r', encoding='utf8') as fp:
                        json_file = json.load(fp)
                        gt_box = np.array(json_file['shapes'][0]['points'])
                        gt_bbox = get_bbox(gt_box)
                        score = get_IoU(pred_bbox, gt_bbox)
                        pred_list.append(pred_bbox)
                        scores.append(score)
            pred_array = np.array(pred_list)
            scores_array = np.array(scores)
            inds = np.where(scores_array==np.max(scores_array))[0]
            c_bboxes = pred_array[inds]
            c_scores = scores_array[inds]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            c_dets = np.unique(c_dets,axis=0)
            #keep = py_cpu_nms(c_dets,0.45)
            #c_dets = c_dets[keep, :]
            # print(c_dets[:, 0:4])
            #print(c_dets)
            print(f"{img_path}  bbox:{c_dets[:, 0:4]}" + ' score: %.4f' % c_dets[:,4])

            #print(img_path + ':' +str(score))

    print('mean: ' + str(np.mean(scores)))

    # pred_bbox = np.array([50, 50, 90, 100])  # top-left: <50, 50>, bottom-down: <90, 100>, <x-axis, y-axis>
    # gt_bbox = np.array([70, 80, 120, 150])
    # print(get_IoU(pred_bbox, gt_bbox))
    #
    # # test2
    # pred_bboxes = np.array([[15, 18, 47, 60],
    #                         [50, 50, 90, 100],
    #                         [70, 80, 120, 145],
    #                         [130, 160, 250, 280],
    #                         [25.6, 66.1, 113.3, 147.8]])
    # gt_bbox = np.array([70, 80, 120, 150])
    # print(get_max_IoU(pred_bboxes, gt_bbox))
