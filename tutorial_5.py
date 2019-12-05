import cv2 as cv
import numpy as np


def fill_color(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(copyImg, mask, (30, 30), (0, 255, 255), (100, 100, 100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color", copyImg)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/fill_color_image.jpg", copyImg)


def fill_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300, : ] = 255
    cv.imshow("fill_binary", image)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/fill_binary_image.jpg", image)

    # mask初始化为1
    mask = np.ones([402, 402, 1], np.uint8)
    # 填充区域mask初始化为0
    mask[101:301, 101:303] = 0
    # 只有mask不为1的区域才会被填充
    cv.floodFill(image, mask, (200, 200), (100, 2, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fill_mask", image)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/fill_mask_image.jpg", image)


src = cv.imread("c:/Users/fenjin/PycharmProjects/images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

# face = src[20:400, 20:400]
# gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
# rbg = cv.cvtColor(gray, cv.COLOR_BGR2RGB)
# src[20:400, 20:400] = rbg
# cv.imshow("face", src)

fill_color(src)
fill_binary()

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
