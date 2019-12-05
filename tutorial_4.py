import cv2 as cv
import numpy as np


def add_image(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("add_image", dst)
    # rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
    cv.imwrite("images/addimage.jpg", dst)


def subtract_image(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_image", dst)
    # rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
    cv.imwrite("images/subtractimage.jpg", dst)


def divide_image(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("divide_image", dst)
    # rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
    cv.imwrite("images/divideimage.jpg", dst)


def multiply_image(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("multiply_image", dst)
    # rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
    cv.imwrite("images/multiplyimage.jpg", dst)


def mean_image(m1, m2):
    M1 = cv.mean(m1)
    M2 = cv.mean(m2)
    m11, dev1 = cv.meanStdDev(m1)
    m12, dev2 = cv.meanStdDev(m2)

    print(M1)
    print(M2)
    print(m11)
    print(m12)
    print(dev1)
    print(dev2)

    h, w = m1.shape[:2]
    img = np.zeros([h, w], np.uint8)
    m, dev = cv.meanStdDev(img)
    print(m)
    print(dev)


def logic_image(m1, m2):
    # dst = cv.bitwise_and(m1, m2)
    # cv.imshow("and", dst)
    # cv.imwrite("images/andimage.jpg", dst)
    # dst = cv.bitwise_or(m1, m2)
    # cv.imshow("or", dst)
    # cv.imwrite("images/orimage.jpg", dst)
    # dst = cv.bitwise_not(m2)
    # cv.imshow("not", dst)
    # cv.imwrite("images/notimage.jpg", dst)
    dst = cv.bitwise_xor(m1, m2)
    cv.imshow("xor", dst)
    cv.imwrite("images/xorimage.jpg", dst)


def contrast_brightness(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow("con_bri", dst)
    cv.imwrite("images/con_bri_image.jpg", dst)


src = cv.imread("images/demo.jpg")
src1 = cv.imread("images/LinuxLogo.jpg")
src2 = cv.imread("images/WindowsLogo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
print(src1.shape)
print(src2.shape)

cv.imshow("input image", src)
# cv.imshow("image1", src1)
# cv.imshow("image2", src2)

t1 = cv.getTickCount()

# add_image(src1, src2)
# subtract_image(src1, src2)
# multiply_image(src1, src2)
# divide_image(src1, src2)
# mean_image(src1, src2)
# logic_image(src1, src2)
contrast_brightness(src, 1.2, 10)
t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
