import cv2 as cv

src = cv.imread("c:/Users/fenjin/PycharmProjects/demo.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
cv.waitKey(0)
cv.destroyAllWindows()

print("Hello Python!")
