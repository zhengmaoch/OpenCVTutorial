import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def top_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    top_hat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 120
    top_hat = cv.add(top_hat, cimage)
    cv.imshow("top_hat", top_hat)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/top_hat_image.jpg", top_hat)


def black_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    black_hat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 120
    black_hat = cv.add(black_hat, cimage)
    cv.imshow("black_hat", black_hat)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/black_hat_image.jpg", black_hat)


def internal_external_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)

    dst1 = cv.subtract(image, em)
    dst2 = cv.subtract(dm, image)

    cv.imshow("internal", dst1)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/internal_image.jpg", dst1)

    cv.imshow("external", dst2)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/external_image.jpg", dst2)


src = cv.imread("c:/Users/fenjin/PycharmProjects/images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

# top_hat_demo(src)
# black_hat_demo(src)
internal_external_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
