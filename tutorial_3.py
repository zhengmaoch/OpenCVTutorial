import cv2 as cv
import numpy as np


def save_video():
    cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('images/video.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv.flip(frame, 1)
            out.write(frame)
            cv.imshow("frame", frame)
            c = cv.waitKey(50)
            if c == 27:
                break
        else:
            break


def extrace_object_demo():
    capture = cv.VideoCapture("images/video.avi")
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([15, 43, 46])
        upper_hsv = np.array([20, 255, 255])
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        dst = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("video", frame)
        cv.imshow("mask", mask)
        cv.imshow("dst", dst)
        c = cv.waitKey(40)
        if c == 27:
            break


def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)
    Ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow("Ycrcb", Ycrcb)
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.imshow("rgb", rgb)


src = cv.imread("images/demo.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

t1 = cv.getTickCount()
# save_video()
# color_space_demo(src)
# b, g, r = cv.split(src)
# cv.imshow("blue", b)
# cv.imshow("green", g)
# cv.imshow("red", r)
#
# src = cv.merge([b, g, r])
# src[:, :, 0] = 0
# cv.imshow("changed image", src)
extrace_object_demo()
t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1) / cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
