import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def erode_demo(image):
    """腐蚀"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # binary = canny_edge_demo(image)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    erode = cv.erode(binary, kernel)

    cv.imshow("erode", erode)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/erode_image.jpg", erode)

def dilate_demo(image):
    """膨胀"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # binary = canny_edge_demo(image)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilate = cv.dilate(binary, kernel)

    cv.imshow("dilate", dilate)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/dilate_image.jpg", dilate)


src = cv.imread("c:/Users/fenjin/PycharmProjects/images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

# erode_demo(src)
# dilate_demo(src)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilate = cv.dilate(src, kernel)
cv.imshow("dilate", dilate)
cv.imwrite("c:/Users/fenjin/PycharmProjects/images/dilate.jpg", dilate)

erode = cv.erode(src, kernel)
cv.imshow("erode", erode)
cv.imwrite("c:/Users/fenjin/PycharmProjects/images/erode.jpg", erode)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
