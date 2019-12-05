import cv2 as cv
import numpy as np


def access_pixels(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width : %s, height : %s channels : %s" % (width, height, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("pixels_demo", image)


def create_image():
    # img = np.zeros([400, 400, 3], np.uint8)
    # img[:, :, 0] = np.ones([400, 400]) * 255
    # cv.imshow("new image", img)

    # img = np.ones([400, 400, 1], np.uint8)
    # img = img * 127
    # cv.imshow("new image", img)

    m1 = np.ones([3, 3], np.uint8)
    m1.fill(1222.388)
    print(m1)

    m2 = m1.reshape([1, 9])
    print(m2)

    m3 = np.array([[2,3,4],[4,5,6],[7,8,9]], np.int32)
    # m3.fill(9)
    print(m3)

def inverse(image):
    dst = cv.bitwise_not(image)
    cv.imshow("imverse demo", dst)


src = cv.imread("images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

t1 = cv.getTickCount()
# access_pixels(src)
create_image()
# inverse(src)
t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
