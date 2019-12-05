import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def big_image_binary(image):
    print(image.shape)
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            print(np.std(roi), np.mean(roi))
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 20)
            gray[row:row + ch, col:col + cw] = dst

            # dev = np.std(roi)
            # if dev < 15:
            #     gray[row:row + ch, col:col + cw] = 255
            # else:
            #     ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            #     gray[row:row+ch, col:col+cw] = dst
    cv.imwrite("images/big_binary_image.jpg", gray)


src = cv.imread("images/big.jpeg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)

t1 = cv.getTickCount()

big_image_binary(src)
# cv.imwrite("images/result_image.jpg", src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
