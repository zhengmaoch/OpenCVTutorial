import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def detect_circle_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    # blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)
    cv.imshow("detect_circle", image)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/detect_circle_image.jpg", image)


src = cv.imread("c:/Users/fenjin/PycharmProjects/images/detect_blob.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

detect_circle_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
