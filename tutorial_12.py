import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def template_demo():
    """模板匹配"""
    template = cv.imread("c:/Users/fenjin/PycharmProjects/images/template.jpg")
    target = cv.imread("c:/Users/fenjin/PycharmProjects/images/demo.jpg")
    cv.imshow("sample", template)
    cv.imshow("target", target)

    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = template.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, template, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        # cv.imshow("match_" + np.str(md), target)
        cv.imshow("match_" + np.str(md), result)
        cv.imwrite("c:/Users/fenjin/PycharmProjects/images/match_" + np.str(md) + "_image.jpg", target)


src = cv.imread("c:/Users/fenjin/PycharmProjects/images/demo.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)

t1 = cv.getTickCount()

template_demo()
# cv.imwrite("c:/Users/fenjin/PycharmProjects/images/result_image.jpg", src)

t2 = cv.getTickCount()
print("time : %s ms" % ((t2 - t1)/cv.getTickFrequency() * 1000))

cv.waitKey(0)
cv.destroyAllWindows()
