import cv2 as cv
import numpy as np


def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def gaussian_noise(image):
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv.imshow("gaussian_noise", image)
    cv.imwrite("c:/Users/fenjin/PycharmProjects/images/gaussian_noise_image.jpg", image)


src = cv.imread("c:/Users/fenjin/PycharmProjects/images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)


t1 = cv.getTickCount()

gaussian_noise(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

dst = cv.GaussianBlur(src, (5, 5), 0)
cv.imshow("gaussianBlur", dst)

cv.waitKey(0)
cv.destroyAllWindows()
