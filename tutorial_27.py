import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract as tess


def recognize_text(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    recognize_text = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("recognize_text", recognize_text)
    cv.imwrite("images/recognize_text_image.jpg", recognize_text)

    # cv.bitwise_not(recognize_text,recognize_text)
    text_image = Image.fromarray(recognize_text)

    # text = tess.image_to_string(text_image)
    # print("recognize result : %s" % text)


src = cv.imread("images/yzm1.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1 = cv.getTickCount()

recognize_text(src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))
cv.waitKey(0)
cv.destroyAllWindows()
