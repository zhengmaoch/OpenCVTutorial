import cv2 as cv
import numpy as np


def access_pixels(image):
    print(image.shape)
    # height = image.shape[0]
    # width = image.shape[1]
    # channels = image.shape[2]
    height, width, channels = image.shape  #每个像素3个通道，通道顺序b,g,r
    print("width : %s, height : %s channels : %s" % (width, height, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                # 像素取反
                image[row, col, c] = 255 - pv
    cv.imshow("pixels_demo", image)


def create_image():
    # 创建一个三维数组高400，宽400，信号通道3个，初始都为0，每通道占8位个
    # img = np.zeros([400, 400, 3], np.uint8)
    # 将0号通道下[400,400]面积使用ones设置为1，之后乘以255，将其设置为255，注意：3个信道分别是b,g,r所以这里显示为蓝色
    # img[:, :, 0] = np.ones([400, 400]) * 255
    # cv.imshow("new image", img)

    # img = np.ones([400, 400, 1], np.uint8)
    # img = img * 127
    # cv.imshow("new image", img)

    m1 = np.ones([3, 3], np.uint8)
    m1.fill(1222.388)
    print(m1)

    # 数组维度转换
    m2 = m1.reshape([1, 9])
    print(m2)
    # [[122 122 122 122 122 122 122 122 122]]

    # 自定义数组
    m3 = np.array([[2,3,4],[4,5,6],[7,8,9]], np.int32)
    # m3.fill(9)
    print(m3)


def inverse(image):
    # OpenCV内置方法取反（直接使用c代码实现，效率更高）
    dst = cv.bitwise_not(image)
    cv.imshow("inverse_demo", dst)


src = cv.imread("images/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

# 获取时间，用于精度计时，操作系统启动所经过（elapsed）的毫秒数
t1 = cv.getTickCount()
# access_pixels(src)
create_image()
# inverse(src)
t2 = cv.getTickCount()
# getTickFrequency()是获取一秒钟结果的点数，获取秒数
print("time : %s s" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
