import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Retinex.code.retinex import retinex_SSR, retinex_MSR, retinex_gimp, retinex_MSRCR, retinex_MSRCP, retinex_AMSR, \
    retinex_FM
from Retinex.code.tools import cv2_heq


def contrast_plot(src_img, retinex_imgs, labels=None):
    """draw the original image and a series of contrast imgs after using enhanced
       algorithm, you can set label for each of them"""
    n = len(retinex_imgs)
    h = int(np.sqrt(n))
    w = int(np.ceil(n / h)) + 1
    for i, img_temp in enumerate([src_img] + retinex_imgs, 0):
        if i == 0:
            plt.subplot(h, w, 1)
        else:
            plt.subplot(h, w, int(i + np.ceil(i / (w - 1))))
        if len(img_temp.shape) == 2:
            plt.imshow(img_temp, cmap='gray')
        else:
            plt.imshow(img_temp[:, :, ::-1])
        if labels is not None:
            plt.xlabel(labels[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    data = []
    columns = ['niqe_mean', 'niqe_std']
    rows = []
    label = ['src', 'SSR(15)', 'SSR(80)', 'SSR(250)', 'MSR(15,80,250,0.333)',
             'Gimp', 'MSRCR', 'MSRCP', 'FM', 'cv2 heq', 'Auto MSR', 'CLAHE',
             'ICM', 'RGHS', 'RayleighDistribution', 'GC']
    img = cv2.imread('D:/code/NRVQA/imgs/Origin/0003.jpg')
    clahe_img = cv2.imread('D:/code/NRVQA/imgs/CLAHE/0003.jpg')
    icm_img = cv2.imread('D:/code/NRVQA/imgs/ICM/0003.jpg')
    rghs_img = cv2.imread('D:/code/NRVQA/imgs/RGHS/0003.jpg')
    rayleigh_img = cv2.imread('D:/code/NRVQA/imgs/RayleighDistribution/0003.jpg')
    gc_img = cv2.imread('D:/code/NRVQA/imgs/GC/0003.jpg')
    contrast_plot(img, [retinex_SSR(img, 15), retinex_SSR(img, 80), retinex_SSR(img, 250), retinex_MSR(img),
                        retinex_gimp(img), retinex_MSRCR(img), retinex_MSRCP(img),
                        retinex_FM(img), cv2_heq(img), retinex_AMSR(img, )],label)
    with open("result.csv") as f:
        f_csv = csv.DictReader(f)
        for x, row1 in enumerate(f_csv):
            rows.append(row1['name'])
            data.append(row1)
    n_rows = len(data)
    cell_text = []
    for row in range(n_rows):
        y_offset = []
        y_offset.clear()
        for col in columns:
            y_offset.append(f'{float(data[row][col]):.3f}')
        cell_text.append(y_offset)

    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center',
                          cellLoc='center',
                          rowLoc='center')
    the_table.scale(0.5, 1)
    # Adjust layout to make room for the table:
    #plt.subplots_adjust(bottom=0.3)
    plt.axis('off')
    plt.show()
