import cv2 as cv


def nothing(x):
    pass


cv.namedWindow('Canny')

cv.createTrackbar('low threshold', 'Canny', 0, 1000, nothing)
cv.createTrackbar('high threshold', 'Canny', 0, 1000, nothing)

cv.setTrackbarPos('low threshold', 'Canny', 50)
cv.setTrackbarPos('high threshold', 'Canny', 150)

#img_gray = cv.imread('../9. perspective/perspective_dark.jpg', cv.IMREAD_GRAYSCALE)
img_gray = cv.imread('../8. ROI & hough/images/perspective.jpg', cv.IMREAD_GRAYSCALE)


while (1):
    low = cv.getTrackbarPos('low threshold', 'Canny')
    high = cv.getTrackbarPos('high threshold', 'Canny')


    img_canny = cv.Canny(img_gray, low, high)

    cv.imshow('Canny', img_canny)

    if cv.waitKey(1) & 0xFF == 27:
        break


cv.destroyAllWindows()