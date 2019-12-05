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


def measure_object(image):
    """轮廓分析"""
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    binary = canny_edge_demo(image)

    # cv.imshow("binary", binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        mm = cv.moments(contour)
        cx = mm['m10'] / mm['m00']
        cy = mm['m01'] / mm['m00']
        cv.circle(dst, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)
        # cv.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print("contour area %s" % area)
        approxCurve = cv.approxPolyDP(contour, 4, True)
        print(approxCurve.shape)
        if approxCurve.shape[0] > 6:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        if approxCurve.shape[0] == 4:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)

    cv.imshow("approxCurve", dst)
    cv.imwrite("images/approxCurve_image.jpg", dst)


src = cv.imread("images/detect_blob.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

measure_object(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
