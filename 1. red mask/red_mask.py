import cv2
import numpy as np

Input_Video = "videos/Tracking_2.mp4"

cap = cv2.VideoCapture(Input_Video)

BasePoint = [415, 716]
ROI_BasePoint = [120, 252]

def Red_componet_extract():
    while (1):
        _, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #BGR을 HSV로 바꿈

    # 앞서 설명한 빨간색 계열의 범위
        lower_red = np.array([-10, 100, 100])
        upper_red = np.array([10, 255, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
    ##cv2.imshow('mask', mask) 흰색으로
        cv2.imshow('res', res)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()