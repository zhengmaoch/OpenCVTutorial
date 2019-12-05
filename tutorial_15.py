import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def pyramid_demo(image):
    """高斯金字塔"""
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_" + str(i), dst)
        temp = dst.copy()
    return pyramid_images


def lapalian_demo(image):
    """拉普拉斯金字塔"""
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("lapalian_down_" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            cv.imshow("lapalian_down_" + str(i), lpls)

# 图片大小必须是2的n次方大小，否则报错
src = cv.imread("c:/Users/fenjin/PycharmProjects/images/lena.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

lapalian_demo(src)
# cv.imwrite("c:/Users/fenjin/PycharmProjects/images/result_image.jpg", src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
