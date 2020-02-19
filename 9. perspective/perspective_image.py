#-*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        points.append([x, y])

#img = cv2.imread('images/test.mp4')
img = cv2.imread('images/original.jpg')
cv2.imshow('img', img)
# [x,y] 좌표점을 4x2의 행렬로 작성
# 좌표점은 좌상->좌하->우상->우하
# pts1 = np.float32([[676, 361], [419, 605], [1236, 357], [1436, 597]])

#
points = []
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imshow('img', img)
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', on_mouse)  ## window name이 'img'인 윈도우에서 마우스 이벤트가 발생하면 on_mouse 함수가 호출 됨

    if len(points) == 4:
        pts1 = np.float32(points)

        # 좌표의 이동점
        pts2 = np.float32([[10,10],[10,1000],[1000,10],[1000,1000]])

        # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
        cv2.circle(img, (points[0][0], points[0][1]), 20, (255, 0, 0), -1)
        ##  cv2.circle(img, center, radian, color, thickness=-1 이면 원 안쪽을 채움)
        cv2.circle(img, (points[1][0], points[1][1]), 20, (0, 255, 0), -1)
        cv2.circle(img, (points[2][0], points[2][1]), 20, (0, 0, 255), -1)
        cv2.circle(img, (points[3][0], points[3][1]), 20, (0, 0, 0), -1)

        M = cv2.getPerspectiveTransform(pts1, pts2)

        dst = cv2.warpPerspective(img, M, (1100,1100))

        img_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        lower_white = (0, 0, 171)
        upper_white = (158, 54, 255)
        img_mask = cv2.inRange(img_hsv, lower_white, upper_white)
        dark_image = cv2.bitwise_and(dst, dst, mask=img_mask)

        plt.subplot(121),plt.imshow(img),plt.title('image')
        plt.subplot(122),plt.imshow(dst),plt.title('Perspective')
        #cv2.imwrite('perspective_dark.jpg', dark_image)
        #cv2.imwrite('perspective.jpg', dst)
        plt.show()

    if key == ord("q"):
        points = []
        img.release()
        cv2.destroyAllWindows()
        break