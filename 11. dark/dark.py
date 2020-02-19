import cv2

img_color = cv2.imread('../9. perspective/images/original.jpg')
height,width = img_color.shape[:2]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

lower_white = (0, 0, 171)
upper_white = (158, 54, 255)

#lower_white = np.array([0,0,0], dtype=np.uint8)
#upper_white = np.array([0,0,255], dtype=np.uint8)
#img_mask = cv2.inRange(inverse_img, lower_white, upper_white)
img_mask = cv2.inRange(img_hsv, lower_white, upper_white)

img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)


#cv2.imshow('img_color', img_color)
#cv2.imshow('img_mask', img_mask)
cv2.imshow('img_result', img_result)


cv2.waitKey(0)
cv2.destroyAllWindows()