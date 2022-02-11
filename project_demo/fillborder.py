import cv2
import numpy as np
import os

def nothing(*arg):
    pass

def img2ret(img, minThreshold):
    img = img.copy()
    _, binaryzation = cv2.threshold(img, minThreshold, 255, cv2.THRESH_BINARY)
    mask = np.zeros(binaryzation.shape).astype(binaryzation.dtype)
    # mask = cv2.bitwise_not(mask, mask)
    # 找到所有的轮廓
    contours, _ = cv2.findContours(binaryzation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    

    # areas = [cv2.contourArea(c) for c in contours]
    # max_idx = np.argmax(areas)
    # print(max_idx)
    sorted_cnts = sorted(contours, key=lambda x: cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(x))), reverse=True)
    try:
        # print(1)
        cv2.fillPoly(mask,[sorted_cnts[1]],[255, 255, 255])
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, 2)
        cv2.drawContours(binaryzation, [sorted_cnts[1]], -1, (0, 0, 0), 2)
        result = cv2.bitwise_and(binaryzation,mask)


        # result = cv2.drawContours(binaryzation.copy(), sorted_cnts, 1, 0, cv2.FILLED) # cv2.FILLED
        # mask = cv2.bitwise_not(result, result)
        # ret = cv2.bitwise_and(binaryzation, result)
        # cv2.drawContours(ret, [sorted_cnts[1]], -1, (0, 0, 0), 1)
    except:
        # print(0)
        cv2.fillPoly(mask,[sorted_cnts[0]],[255, 255, 255])
        cv2.drawContours(binaryzation, [sorted_cnts[0]], -1, (0, 0, 0), 2)
        result = cv2.bitwise_and(binaryzation,mask)

        # result = cv2.drawContours(binaryzation.copy(), sorted_cnts, 0, 0, cv2.FILLED) # cv2.FILLED
        # mask = cv2.bitwise_not(result, result)
        # ret = cv2.bitwise_and(binaryzation, result)
        # cv2.drawContours(ret, [sorted_cnts[0]], -1, (0, 0, 0), 1)

    return img, result, mask

def saveResult():
    imgpath = r"C:\Users\Administrator\Desktop\issues\croped\kehen_unnamed_4_0.jpg"

def test(bb, gg, rr, GG):
    # imgpath = r"C:\Users\Administrator\Desktop\issues\croped\kehen_unnamed_4_0.jpg"
    img = cv2.imread(imgpath)
    Gimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = img.copy()
    b, g, r = cv2.split(img)
    # 二值化，100为阈值，小于100的变为255，大于100的变为0
    # cv2.THRESH_BINARY
    # cv2.THRESH_BINARY_INV
    # cv2.THRESH_TRUNC
    # cv2.THRESH_TOZERO_INV
    # cv2.THRESH_TOZERO
    # print(bb, gg, rr, GG)
    
    bimg, bresult, bstencil = img2ret(b, bb)
    gimg, gresult, gstencil = img2ret(g, gg)
    rimg, rresult, rstencil = img2ret(r, rr)
    Gimg, Gesult, Gstencil = img2ret(Gimg, GG)

    _, binaryzation = cv2.threshold(g, minThreshold, 255, cv2.THRESH_BINARY)
    stencil = np.zeros(binaryzation.shape).astype(binaryzation.dtype)
    # 找到所有的轮廓
    contours, _ = cv2.findContours(binaryzation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    

    # # areas = [cv2.contourArea(c) for c in contours]
    # # max_idx = np.argmax(areas)
    # # print(max_idx)
    # sorted_cnts = sorted(contours, key=lambda x: cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(x))), reverse=True)

    # result = cv2.drawContours(binaryzation.copy(), sorted_cnts, 1, 0, cv2.FILLED) # cv2.FILLED
    # result = cv2.bitwise_not(result)
    # result = cv2.bitwise_and(binaryzation,result)
    # cv2.drawContours(result, [sorted_cnts[1]], -1, (0, 0, 0), 1)
    # cv2.imshow("tmp", result)
    # cv2.waitKey()
    # exit(-1)
    # cv2.fillPoly(stencil,[sorted_cnts[1]],[255, 255, 255])
    # cv2.drawContours(binaryzation, [sorted_cnts[1]], -1, (0, 0, 0), 2)
    # result = cv2.bitwise_and(binaryzation,stencil)

    bresult = cv2.cvtColor(bresult, cv2.COLOR_GRAY2RGB)
    bstencil = cv2.cvtColor(bstencil, cv2.COLOR_GRAY2RGB)

    gresult = cv2.cvtColor(gresult, cv2.COLOR_GRAY2RGB)
    gstencil = cv2.cvtColor(gstencil, cv2.COLOR_GRAY2RGB)

    rresult = cv2.cvtColor(rresult, cv2.COLOR_GRAY2RGB)
    rstencil = cv2.cvtColor(rstencil, cv2.COLOR_GRAY2RGB)
    
    Gesult = cv2.cvtColor(Gesult, cv2.COLOR_GRAY2RGB)
    Gstencil = cv2.cvtColor(Gstencil, cv2.COLOR_GRAY2RGB)

    bhist = np.hstack([img, bresult, bstencil])
    ghist = np.hstack([img, gresult, gstencil])
    rhist = np.hstack([img, rresult, rstencil])
    Ghist = np.hstack([img, Gesult, Gstencil])

    Hist = np.vstack([bhist, ghist, rhist, Ghist])
    cv2.imshow('window', Hist)
    return Hist
    # cv2.imshow('masked', result)
    # cv2.imshow('binaryzation', stencil)
    # cv2.waitKey(0)

def t2():
    import cv2
    import numpy
    img = cv2.imread(r"C:\Users\Administrator\Desktop\issues\mask\huahen_unnamed_6.jpg")
    stencil = numpy.zeros(img.shape).astype(img.dtype)
    contours = [numpy.array([[100, 180], [200, 280], [200, 180]]), numpy.array([[280, 70], [12, 20], [80, 150]])]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result = cv2.bitwise_and(img, stencil)
    cv2.imshow('binaryzation', stencil)
    cv2.imshow('masked', result)
    cv2.waitKey(0)

# t2()
     
def main():
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('window', 800, 780)
    cv2.createTrackbar('b', 'window', minThreshold, maxThreshold, nothing)
    cv2.createTrackbar('g', 'window', minThreshold, maxThreshold, nothing)
    cv2.createTrackbar('r', 'window', minThreshold, maxThreshold, nothing)
    cv2.createTrackbar('gray', 'window', minThreshold, maxThreshold, nothing)


    # cv2.createTrackbar('Show', 'window',0,1,nothing)

    while True:
        bb = cv2.getTrackbarPos('b', 'window')
        gg = cv2.getTrackbarPos('g', 'window')
        rr = cv2.getTrackbarPos('r', 'window')
        GG = cv2.getTrackbarPos('gray', 'window')

        Hist = test(bb, gg, rr, GG)

        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            break
    dst = r"C:\Users\Administrator\Desktop\issues\croped\mask"
    # cv2.imwrite(os.path.join(dst, "{}={}-{}-{}-{}.jpg").format(ii, bb, gg, rr, GG), Hist)
    print("{}: {} {} {} {}".format(imgpath, bb, gg, rr ,GG))    
    cv2.destroyAllWindows()   

if __name__ == '__main__':
    # writen = ['10_201908302_2_1.jpg', '10_201908302_2_1.jpg', '10_201908302_5_0.jpg', '10_201908302_5_1.jpg', \
    #     '1_201908302_6_0.jpg', '1_201908302_8_0.jpg', '3_201908302_4_1.jpg', '3_201908302_5_0.jpg', \
    #         '3_201908302_7_1.jpg', '4_201908302_1_0.jpg', '4_201908302_1_1.jpg']
    minThreshold = 50
    maxThreshold = 254
    # imgpath = r"C:\Users\Administrator\Desktop\issues\croped\5_201908302_5_1.jpg"
    root = r"C:\Users\Administrator\Desktop\issues\croped"
    #for ii in os.listdir(root):
        #if ii not in writen:
            # imgpath = os.path.join(root, ii)
    imgpath = r'D:\code\image-processing-training\Transform_demo\test001.jpg'
        # print(imgpath)
    main()
    # writen = []
    # for ii in os.listdir(r"C:\Users\Administrator\Desktop\issues\croped\mask"):
    #     writen.append(ii.partition('=')[0])
    # print(writen)