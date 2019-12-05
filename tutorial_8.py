import cv2 as cv
import numpy as np
"""保留边缘滤波EPF"""


def bilateralFilter_demo(image):
    """高斯双边模糊"""
    dst = cv.bilateralFilter(image, 0, 100, 15)
    cv.imshow("bilateralFilter", dst)
    cv.imwrite("images/bilateralFilter_image.jpg", dst)


def pyrMeanShiftFiltering_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("pyrMeanShiftFiltering", dst)
    cv.imwrite("images/pyrMeanShiftFiltering_image.jpg", dst)


src = cv.imread("images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

t1 = cv.getTickCount()

bilateralFilter_demo(src)
pyrMeanShiftFiltering_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
