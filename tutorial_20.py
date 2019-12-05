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
    canny_edge = cv.Canny(gray, 20, 100)
    return canny_edge


def contours_demo(image):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    binary = canny_edge_demo(image)

    cv.imshow("binary", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)
        # 填充
        # cv.drawContours(image, contours, i, (0, 0, 255), -1)
        print(i)

    cv.imshow("contours", image)
    cv.imwrite("images/contours_image.jpg", image)


src = cv.imread("images/detect_blob.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

contours_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
