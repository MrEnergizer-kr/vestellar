import cv2 as cv
import numpy as np
import math

def nothing(x):
    pass

####cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)####
#cv.namedWindow('Canny')

#cv.createTrackbar('low threshold', 'Canny', 0, 1000, nothing)
#cv.createTrackbar('high threshold', 'Canny', 0, 1000, nothing)

#cv.setTrackbarPos('low threshold', 'Canny', 50)
#cv.setTrackbarPos('high threshold', 'Canny', 150)

cv.namedWindow('HoughLinesP')
cv.createTrackbar('threshold', 'HoughLinesP', 0, 1000, nothing)
cv.createTrackbar('minLength', 'HoughLinesP', 0, 200, nothing)
cv.createTrackbar('maxLineGap', 'HoughLinesP', 0, 100, nothing)

cv.setTrackbarPos('threshold', 'HoughLinesP', 150)
cv.setTrackbarPos('minLength', 'HoughLinesP', 50)
cv.setTrackbarPos('maxLineGap', 'HoughLinesP', 10)

src = cv.imread('../8. ROI & hough/images/perspective.jpg', cv.IMREAD_GRAYSCALE)
dst = cv.Canny(src, 209, 617, None, 3)
cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)



while (1):
    thresh = cv.getTrackbarPos('threshold', 'HoughLinesP')
    minLength = cv.getTrackbarPos('minLength', 'HoughLinesP')
    maxLineGap = cv.getTrackbarPos('maxLineGap', 'HoughLinesP')
####         cv.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)####
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, thresh, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
            cv.imshow('HoughLinesP', cdstP)

    if cv.waitKey(1) & 0xFF == 27: ## esc
        break


cv.destroyAllWindows()