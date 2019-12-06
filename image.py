import cv2 as cv


# 读取图片，OpenCV读取图像格式默认是BGR
# 可以设置第二个参数[cv.IMREAD_COLOR | cv.IMREAD_GRAYSCALE]分别表示彩色图像或者为灰度图像
src = cv.imread("images/demo.jpg")
# 创建GUI窗口，形式为自适应大小
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 通过名字将图像显示到窗口
cv.imshow("input image", src)
# 等待用户操作，等待时间参数的单位是毫秒，0表示任意键终止
cv.waitKey(0)
# 从内存中销毁，释放资源
cv.destroyAllWindows()

print("Hello Python!")

# https://www.cnblogs.com/ssyfj/category/1245091.html