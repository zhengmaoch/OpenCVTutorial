import cv2 as cv
import numpy as np


def blur_demo(image):
    """均值模糊"""
    dst = cv.blur(image, (5, 5))
    cv.imshow("blur", dst)
    cv.imwrite("images/blur_image.jpg", dst)


def median_blur_demo(image):
    """中值模糊"""
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur", dst)
    cv.imwrite("images/median_blur_image.jpg", dst)


def custom_blur_demo(image):
    """自定义模糊"""
    # kernel = np.ones([5, 5], np.float32) / 25
    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("custom_blur", dst)
    cv.imwrite("images/custom_blur_image.jpg", dst)


src = cv.imread("images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

blur_demo(src)
median_blur_demo(src)
custom_blur_demo(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
