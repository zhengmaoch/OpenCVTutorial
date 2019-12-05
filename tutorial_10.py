import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def equalHist_demo(image):
    """全局直方图均衡化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    dst = cv.equalizeHist(gray)
    cv.imshow("equalizeHist", dst)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/equalizeHist_image.jpg", dst)


def createCLAHEt_demo(image):
    """局部直方图均衡化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow("createCLAHE", dst)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/createCLAHE_image.jpg", dst)


def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbHist[np.int(index), 0] += 1
    return rgbHist


def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离： %s, 相关性： %s, 卡方： %s" % (match1, match2, match3))


src = cv.imread("c:/Users/fenjin/PycharmProjects/images/demo.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
t1 = cv.getTickCount()

# equalHist_demo(src)
# createCLAHEt_demo(src)

image1 = cv.imread("c:/Users/fenjin/PycharmProjects/images/addimage.jpg")
image2 = cv.imread("c:/Users/fenjin/PycharmProjects/images/xorimage.jpg")
cv.imshow("image1", image1)
cv.imshow("image2", image2)
hist_compare(image1, image2)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
