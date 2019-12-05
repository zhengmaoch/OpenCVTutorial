import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def open_demo(image):
    """开操作"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    open = cv.morphologyEx(binary, cv.MORPH_RECT, kernel)
    cv.imshow("open", open)
    cv.imwrite("images/open_image.jpg", open)


def close_demo(image):
    """开操作"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    close = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("close", close)
    cv.imwrite("images/close_image.jpg", close)


src = cv.imread("images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

open_demo(src)
close_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
