import csv
import matplotlib.pyplot as plt
import numpy as np


def contrast_plot(src_img, retinex_imgs, labels=None):
    """draw the original image and a series of contrast imgs after using enhanced
       algorithm, you can set label for each of them"""
    n = len(retinex_imgs)
    h = int(np.sqrt(n))
    w = int(np.ceil(n / h)) + 1
    for i, img in enumerate([src_img] + retinex_imgs, 0):
        if i == 0:
            plt.subplot(h, w, 1)
        else:
            plt.subplot(h, w, int(i + np.ceil(i / (w - 1))))
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img[:, :, ::-1])
        if labels is not None:
            plt.xlabel(labels[i])
        plt.xticks([])
        plt.yticks([])


if __name__ == '__main__':
    data = []
    columns = ['niqe_mean', 'niqe_std']
    rows = []
    with open("result.csv") as f:
        f_csv = csv.DictReader(f)
        for i, row1 in enumerate(f_csv):
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
                          loc='left',
                          cellLoc='center',
                          rowLoc='center')
    # the_table.scale(0.3, 1)
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.6, bottom=0.2)
    plt.axis('off')
    plt.show()
