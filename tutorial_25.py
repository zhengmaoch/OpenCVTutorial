import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def watershed_demo(image):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=3)


    cv.imshow("sure_bg", sure_bg)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/sure_bg_image.jpg", sure_bg)

    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    distout = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX) * 50
    cv.imshow("distout", distout)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/distout_image.jpg", distout)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow("surface", surface)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/surface_image.jpg", surface)

src = cv.imread("c:/Users/fenjin/PycharmProjects/images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

watershed_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
