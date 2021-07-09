import cv2
import os
from PIL import Image
import natsort
import numpy as np

def Pic2Video(img_path):
    #imgPath = "D:/code/NRVQA/imgs/RayleighDistribution/"  # 读取图片路径
    imgPath = img_path
    videoPath = "D:/code/video/enhance/"  # 保存视频路径
    if not os.path.exists(videoPath):
        os.makedirs(videoPath)
        print("folder creation completed")
    video_name = "output002.mp4"
    images = os.listdir(imgPath)
    fps = 25  # 每秒25帧数

    # VideoWriter_fourcc为视频编解码器 ('I', '4', '2', '0') —>(.avi) 、('P', 'I', 'M', 'I')—>(.avi)、('X', 'V', 'I', 'D')—>(.avi)、('T', 'H', 'E', 'O')—>.ogv、('F', 'L', 'V', '1')—>.flv、('m', 'p', '4', 'v')—>.mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath + video_name, fourcc, fps, image.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])  # 这里的路径只能是英文路径
        # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
        print(im_name)
        videoWriter.write(frame)
    print("图片转视频结束！")
    videoWriter.release()
    cv2.destroyAllWindows()


def Video2Pic():
    videoPath = "D:/code/video/1.mp4"  # 读取视频路径
    imgPath = "D:/code/video/picOutput3/"  # 保存图片路径

    if not os.path.exists(imgPath):
        os.makedirs(imgPath)
        print("folder creation completed")

    cap = cv2.VideoCapture(videoPath)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
        suc, frame = cap.read()
        if frame is not None:
            frame_resize = cv2.resize(frame, None, fx=0.5, fy=0.5)
            cv2.imwrite(imgPath + str(frame_count).zfill(8) + ".png", frame_resize)
            cv2.waitKey(1)
            frame_count += 1
    cap.release()
    print("视频转图片结束！")

def picCrop():
    img1_path = "D:/code/video/picOutput2/"
    img2_path = "D:/code/NRVQA/imgs/RayleighDistribution/"
    output_path = 'output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("folder creation completed")
    files1 = os.listdir(img1_path)
    files1 = natsort.natsorted(files1)
    files2 = os.listdir(img2_path)
    files2 = natsort.natsorted(files2)

    if len(files1) == len(files2):
        print("视频合并开始")
        for i,file1 in enumerate(files1):
            file2 = files2[i]
            file1path = img1_path + "/" + file1
            file2path = img2_path + '/' + file2
            prefix = file1.split('.')[0]
            print('********    file   ********', file1, file2)
            img1 = cv2.imread(file1path)
            img2 = cv2.imread(file2path)

            # python里数组传递传递的是数组的引用
            # 对图像进行缩放操作，方便观察
            img1_resize = cv2.resize(img1, None, fx=0.5, fy=0.5)
            img2_resize = cv2.resize(img2, None, fx=0.5, fy=0.5)
            # 合并两张图像为一张
            half = int(np.shape(img1_resize)[1] / 2)
            diff = np.hstack((img1_resize[:, :half, :], img2_resize[:, half:, :]))
            print(f'开始合并图像  {file1}与{file2}')
            cv2.imwrite(output_path + prefix + '.jpg', diff)
        Pic2Video(output_path)
    else:
        raise ValueError('图片序列长度不一致，无法合并')


if __name__ == '__main__':
    Video2Pic()
    #Pic2Video()
    #picCrop()