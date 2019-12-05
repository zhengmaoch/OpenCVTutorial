import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
分水岭算法主要用于图像分段，通常是把一副彩色图像灰度化，然后再求梯度图，最后在梯度图的基础上进行分水岭算法，求得分段图像的边缘线
1、获取灰度图像，二值化图像，进行形态学操作（开操作、闭操作、腐蚀、膨胀），消除噪点
1、距离变换

"""

def watershed_demo(image):
    """分水岭"""
    # 滤波降噪（关键）或者用高斯模糊进行降噪
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    # 转换为灰度图
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # 二值化将图像转化为黑色和白色部分
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    # 形态学操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # 2次开操作，消除图像中的噪点
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    # 3次膨胀，将目标对象进行放大
    sure_bg = cv.dilate(mb, kernel, iterations=3)
    cv.imshow("sure_bg", sure_bg)
    cv.imwrite("images/sure_bg_image.jpg", sure_bg)
    # 使用距离变换
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    # 对返回结果进行归一化处理，得到骨骼图像
    distout = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX) * 50
    cv.imshow("distout", distout)
    cv.imwrite("images/distout_image.jpg", distout)
    # 使用阈值进行二值化处理，获取前景色
    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow("sure_fg", surface)
    cv.imwrite("images/sure_fg_image.jpg", surface)
    # 将前景转化为整形
    surface_fg = np.uint8(surface)
    # 将背景减去前景得到差值
    unknown = cv.subtract(sure_bg, surface_fg)
    cv.imshow("unknown", unknown)
    cv.imwrite("images/unknown_image.jpg", unknown)
    # 获取markers，在markers中含有种子区域
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)
    print(markers[2])

    # watershed transform
    markers += 1
    # 像素操作
    markers[unknown == 255] = 0
    # 获取栅栏，栅栏区域设置为-1
    markers = cv.watershed(src, markers)
    # 栅栏区域设置为红色
    src[markers == -1] = [0, 0, 255]
    cv.imshow("watershed", src)
    cv.imwrite("images/watershed_image.jpg", distout)


src = cv.imread("images/coins.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

watershed_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
