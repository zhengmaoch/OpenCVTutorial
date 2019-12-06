import cv2 as cv
import numpy as np


def video_demo():
    # 捕获摄像头，用数字控制不同摄像头设备，0表示第一个摄像头
    vc = cv.VideoCapture(0)
    # 也可以打开视频文件，参数为视频文件地址
    # vc = cv.VideoCapture("images/video.avi")
    # 如果摄像头开启就读取视频数据
    while vc.isOpened():
        # 读取视频数据，返回读取结果和对应的数据帧，数据帧就是图像
        ret, frame = vc.read()
        # 如果数据帧为空（视频结束）则停止读取
        if frame is None:
            break
        # 如果读取数据正确，则显示数据
        if ret:
            # 将视频帧左右调换
            frame = cv.flip(frame, 1)
            # 将图像转换为灰度图像
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # 显示每一帧图像
            cv.imshow("video", frame)
            # 10毫秒关闭当前捕获的图像，显示下一个图像，如果用户按下“Esc”键，则停止读取，27是ESC键的ASCII码值
            if cv.waitKey(10) & 0xFF == 27:
                break
    # 释放摄像头资源
    vc.release()


def get_image_info(image):
    print(type(image))  #<class 'numpy.ndarray'>    numpy类型数组
    print(image.shape)  #打印图像的高度，宽度，通道数(608, 343, 3)3个方向
    print(image.size)   #打印图像的大小625632==>608*343*3
    print(image.dtype)  #dtype:每个像素点有3个通道，每个通道所占的位数：无符号的int8位uint8
    pixel_data = np.array(image)
    print(pixel_data)


# src = cv.imread("images/demo.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
#
# get_image_info(src)
video_demo()

# # 将图像转换为灰度图像
# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# cv.imshow("gray", gray)
# # 保持图像至指定的地址
# cv.imwrite("images/gray_image.jpg", gray)

cv.waitKey(0)
cv.destroyAllWindows()
