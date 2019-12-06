import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def face_detect_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("data/lbpcascades/lbpcascade_frontalface.xml")
    faces = face_detector.detectMultiScale(gray, 1.1, 5)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.imshow("faces_detect", image)
    cv.imwrite("images/faces_detect_image.jpg", image)
    # cv.waitKey(10)


def face_detect_from_video():
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        face_detect_demo(frame)
        c = cv.waitKey(10)
        if c == 27:
            break


# src = cv.imread("images/demo.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# t1 = cv.getTickCount()

# face_detect_demo(src)
face_detect_from_video()


# t2 = cv.getTickCount()
# print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
