import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def canny_edge_demo(image):
    """Canny边缘提取"""
    # 高斯模糊进行降噪
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # X Grodient
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # Y Grodient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # edge
    # canny_edge = cv.Canny(xgrad, ygrad, 50, 150)
    canny_edge = cv.Canny(gray, 50, 150)
    cv.imshow("canny_edge", canny_edge)
    cv.imwrite("images/canny_edge_image.jpg", canny_edge)

    dst = cv.bitwise_and(image, image, mask=canny_edge)
    cv.imshow("color_edge", dst)
    cv.imwrite("images/color_edge_image.jpg", dst)


src = cv.imread("images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

t1 = cv.getTickCount()

canny_edge_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
