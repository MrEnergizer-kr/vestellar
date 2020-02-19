#-*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        points.append([x, y])


cap = cv2.VideoCapture('videos/test.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('result/output.avi', fourcc, 30.0, (width, height))
#img = cv2.imread('images/original.jpg')
#cv2.imshow('img', img)
# [x,y] 좌표점을 4x2의 행렬로 작성
# 좌표점은 좌상->좌하->우상->우하
# pts1 = np.float32([[676, 361], [419, 605], [1236, 357], [1436, 597]])

#
points = [] ## 여기에 각 cctv 마다의 좌표 정보가 저장됨
while cap.isOpened():
    ret, frame = cap.read()
    original_frame = np.copy(frame)  ## frame과 original_frame으로 나눔
    cv2.imshow('origianl frame', original_frame) ## BACKGROUND 에서 도는 영상은 original_frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imshow('img', frame)
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', on_mouse)  ## window name이 'img'인 윈도우에서 마우스 이벤트가 발생하면 on_mouse 함수가 호출 됨

    if len(points) == 4:
        cv2.destroyWindow('img')
        pts1 = np.float32(points)

        # 좌표의 이동점
        pts2 = np.float32([[10,10],[10,1000],[1000,10],[1000,1000]]) ## 이건 고정

        # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
        #cv2.circle(original_frame, (points[0][0],points[0][1]), 20, (255,0,0),-1)
        ##  cv2.circle(img, center, radian, color, thickness=-1 이면 원 안쪽을 채움)
        #cv2.circle(original_frame, (points[1][0],points[1][1]), 20, (0,255,0),-1)
        #cv2.circle(original_frame, (points[2][0],points[2][1]), 20, (0,0,255),-1)
        #cv2.circle(original_frame, (points[3][0],points[3][1]), 20, (0,0,0),-1)



        M = cv2.getPerspectiveTransform(pts1, pts2) # Matrix

        dst = cv2.warpPerspective(frame, M, (1100,1100)) ## 여기 중요
        ## 영상이면 영상으로 받음

        img_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        lower_white = (0, 0, 171)
        upper_white = (158, 54, 255)
        img_mask = cv2.inRange(img_hsv, lower_white, upper_white)
        dark_image = cv2.bitwise_and(dst, dst, mask=img_mask)

        #cv2.circle(dst, (10, 10), 20, (0, 255, 0), -1)
        #cv2.circle(dst, (10, 1000), 20, (0, 255, 0), -1)
        #cv2.circle(dst, (1000, 10), 20, (0, 0, 255), -1)
        #cv2.circle(dst, (1000, 1000), 20, (0, 0, 0), -1)
        #cv2.imshow('Perspective', dst) ## 원근법이 제거된 영상은 frame영상에 적용


        #cv2.imwrite('perspective_dark.jpg', dark_image)




    if key == ord("q"):
        writer.write(dst)  ## 저장
        points = []
        break


cap.release()
writer.release()
cv2.destroyAllWindows()