import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    # grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    # grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradx", gradx)
    cv.imwrite("images/gradx_image.jpg", gradx)
    cv.imshow("grady", grady)
    cv.imwrite("images/grady_image.jpg", grady)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradxy", gradxy)
    cv.imwrite("images/gradxy_image.jpg", gradxy)


def lapalian_demo(image):
    # dst = cv.Laplacian(image, cv.CV_32F)
    # lpls = cv.convertScaleAbs(dst)
    # 拉普拉斯算子
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    dst = cv.filter2D(image,cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("lapalian", lpls)
    cv.imwrite("images/lapalian_image.jpg", lpls)


src = cv.imread("images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

t1 = cv.getTickCount()
# sobel_demo(src)
lapalian_demo(src)
# cv.imwrite("images/result_image.jpg", src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
