import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture("Videos/tiny_1.mp4")
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)

sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor
# information to start saving a video file
ret, frame = cap.read()  # import image
ratio = 0.5  # resize ratio
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
width2, height2, channels = image.shape
# video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)

while True:
    ret, frame = cap.read()  # import image
    if not ret:  # if vid finish repeat ## 받아올 ret이 없으면 즉, 비디오가 끝났으면
        frame = cv2.VideoCapture("videos/Tracking_2.mp4") ## 다음걸 진행하라 여기에는 다음 영상이 들어올 자리
        continue
    if ret:  # if there is a frame continue with code
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
        cv2.imshow("image", image)  # @
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
        #cv2.imshow("gray", gray)  # @
        fgmask = sub.apply(gray)  # uses the background subtraction
        cv2.imshow("fgmask", fgmask)  # @
        # applies different thresholds to fgmask to try and isolate cars
        # just have to keep playing around with settings until cars are easily identifiable
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow("closing", closing)  # @
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("opening", opening)  # @
        dilation = cv2.dilate(opening, kernel)
        #cv2.imshow("dilation", dilation)  # @
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
        #cv2.imshow("retvalbin", retvalbin)  # @
        # creates contours
        # cv2.imshow('bins',bins)
        contours, hierarchy = cv2.findContours(bins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minarea = 1500
        # max area for contours, can be quite large for buses
        maxarea = 30000
        # vectors for the x and y locations of contour centroids in current frame

        for i in range(len(contours)):  # cycles through all contours in current frame
            if hierarchy[0, i, 3] == -1:  # heirarchy 리스트의 0번째의 i번째의 3번째가 -1 이면 --> [[3, -1, 1, *-1], [1, 3, 2, -1], ...], [next, prev, child, parent] 즉 parent만 본다는 의미
                area = cv2.contourArea(contours[i])  # contour면적
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    ## 모멘트는 무게중심점을 찾는 함수
                    ## https://076923.github.io/posts/Python-opencv-25/
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # gets bounding points of contour to create rectangle
                    # x,y is top left corner and w,h is width and height
                    x, y, w, h = cv2.boundingRect(cnt)
                    # creates a rectangle around contour
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Prints centroid text in order to double check later on
                    cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3,
                                (0, 0, 255), 1)
                    cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,
                                   line_type=cv2.LINE_8)
    cv2.imshow("countours", image)
    key = cv2.waitKey(20)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

### 결과 영상은 각 contour의 중심점을 모아서 사각형 박스를 그렸음
### 이러한 결과를 기반으로 contour의 중심점들의 평균 구간을 찾아내는 방식으로 진행해야 함
### 이 코드의 사용 목적은 영상에서 움직임이 없을때 yolo를 실행하지 않음(load를 줄이기 위해서)
### 만약 어떤 움직임을 감지하고자 할때 이 코드를 사용
### 이렇게 박스의 갯수가 4개(숫자는 나중에 정해줘도 됨)이상 되면 그때 yolo를 시작함과 동시에 tracker를 yolo의 반환값 좌표를 통해 tracking 시작한다.
### 이 contour내부의 점들 cx, cy는 yolo --> tracking시 검증할 비교 자료로 활용할 수 있음
### 이 프로그램은 큰 load가 없음