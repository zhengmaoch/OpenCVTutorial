import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def threshold_demo(image):
    """全局二制化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value %s" % ret)
    cv.imshow("threshold", binary)
    cv.imwrite("images/threshold_image.jpg", binary)


def local_threshold_demo(image):
    """局部二制化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("local_threshold", binary)
    cv.imwrite("images/local_threshold_image.jpg", binary)


def custom_threshold_demo(image):
    """自定义二制化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w * h])
    mean = m.sum() / (w * h)
    print("mean : %s" % mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("custom_threshold", binary)
    cv.imwrite("images/custom_threshold_image.jpg", binary)


src = cv.imread("images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

t1 = cv.getTickCount()

threshold_demo(src)
local_threshold_demo(src)
custom_threshold_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
