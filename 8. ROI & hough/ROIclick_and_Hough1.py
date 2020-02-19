import datetime
import time
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import math
from operator import itemgetter
from glob import glob
from typing import Dict
import imutils

def increase_contrastness(img):
    #-----Converting image to LAB Color model
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(15,15))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    ##cv2.imshow('final', final)
    ###
    return final

def white_color_mask(img):
    # white color mask
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(img, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190, 0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(img, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(img, img, mask)
    return masked

def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

def makebin(gray):
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    return cv2.bitwise_not(bin)

def find_squares(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)
    squares = []
    for gray in cv2.split(img):
        bin = makebin(gray)
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        corners = cv2.goodFeaturesToTrack(gray, len(contours) * 4, 0.2,15)
        ## (그레이이미지, 검출할 코너 갯수 = contour *4, 코너 검출 품질, 검출할 코너사이 최대거리)
        ##cv2.cornerSubPix(gray, corners, (6, 6), (-1, -1),(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 10, 0.1))
        ### 코너 값 조정

        ##print(contours)
        ##print(len(contours))
        for i, cnt in enumerate(contours):
            cnt_len = cv2.arcLength(cnt, True)
            if hierarchy[0, i, 3] == -1  and cv2.contourArea(cnt) > 20000 and cv2.contourArea(cnt) < 34000:
                ##print(cv2.contourArea(cnt))
                rect = cv2.boundingRect(cnt)
                if rect not in squares:
                    squares.append(rect)

    ##print(squares)
    ##print(corners)
    ##print(len(corners))
    return squares, corners, contours

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        points.append([x, y])

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def region_mask(frame,vertices):
    f_mask = np.zeros_like(frame)
    if len(f_mask.shape) == 2:  ### 2차원 배열이면
        cv2.fillPoly(f_mask, vertices, 255)  ### 다각형 생성 --> 사각형이 아닌 다각형
    else:
        cv2.fillPoly(f_mask, vertices, (255,) * f_mask.shape[2])  # in case, the input image has a channel dimension

    # images showing the region of interest only
    roi_images = cv2.bitwise_and(frame, f_mask)
    return roi_images

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

Input_Video = 'test.mp4'
#BasePath = "yolo-coco"
BasePath = 'yolo-coco'
BaseConfidence = 0.6 # 0.5
Base_threshold = 0.5 # 0.3

labelsPath = os.path.sep.join([BasePath, "coco.names"])
weightsPath = os.path.sep.join([BasePath, "yolov3.weights"])
configPath = os.path.sep.join([BasePath, "yolov3.cfg"])
##labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
#=============================================

np.random.seed(40)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

f_num = 0
points = []
cap = cv2.VideoCapture(Input_Video)
start_time = float(time.time())

list_of_determined_x = []
list_of_determined_y = []
list_of_determined_cnt = []

list_of_Defined_x = []
list_of_Defined_y = []
list_of_Defined_coordinates = []
list_of_Defined_ID = []
list_of_Defined_status = []
ID_num = 0

LAPLACIAN =2.0

while cap.isOpened():
    grabbed, frame = cap.read()
    f_num = f_num + 1
    print("Frame : ",f_num)



    Original_frame = np.copy(frame)
    empty = np.empty(frame.shape)
    #cv2.imshow('original', Original_frame)

    time1 = float(time.time() - start_time)
    print("진행시간 : %s" % (time1))

    key = cv2.waitKey(1) & 0xFF

    if True:
        bright = img = increase_contrastness(frame)
        masked = white_color_mask(bright)
        gray_images = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

        low_threshold = 932  # 좀 바꾸면 변동이 있긴 함
        high_threshold = 160
        canny = cv2.Canny(gray_images, low_threshold, high_threshold)
    

        edge_images = cv2.dilate(canny, (5, 5), 7)
        rows, cols = edge_images.shape[:2]  ## 1280 x 720

        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = (0, 0, 171)
        upper_white = (158, 54, 255)
        img_mask = cv2.inRange(img_hsv, lower_white, upper_white)
        dark_image = cv2.bitwise_and(frame, frame, mask=img_mask)


        if key == ord("s"):
            cv2.imshow('img', frame)
            cv2.namedWindow('img')
            cv2.setMouseCallback('img', on_mouse)  ## window name이 'img'인 윈도우에서 마우스 이벤트가 발생하면 on_mouse 함수가 호출 됨

        if len(points) == 4: ## points라는 리스트에 4개 좌표가 들어오면
            x = points[0]
            y = points[1]
            z = points[2] ## 좌표 4개로 사각형 만듦
            w = points[3]
            cv2.destroyWindow('img')

            vertices = np.array([[x, y, z, w]], dtype=np.int32)
            roi_images = region_mask(edge_images, vertices)
            cv2.imshow('roi', roi_images)
            ####
            list_of_lines = []
            list_of_lines = cv2.HoughLinesP(roi_images, rho=1, theta=np.pi / 360, threshold=100, minLineLength=210, maxLineGap=30)
            print(list_of_lines)  # 100

            blank_img = np.zeros((rows, cols, 3), np.uint8)

            if list_of_lines is None:
                pass
            else:
                for num in list_of_lines:
                    for x1, y1, x2, y2 in num:
                        # if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                        cv2.line(blank_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            hough_gray = cv2.cvtColor(blank_img, cv2.COLOR_RGB2GRAY)

            ##8. Contour
            squares, corners, contours = find_squares(hough_gray)
            squares = sorted(squares, key=itemgetter(1, 0, 2, 3))
            areas = []

            list_of_midpoint_x = []
            list_of_midpoint_y = []

            distance_gap = 200
            font = cv2.FONT_HERSHEY_SIMPLEX

            for s in squares:
                areas.append(s[2] * s[3])
                ##cv2.rectangle(Original_frame, (s[0],s[1]),(s[0]+s[2],s[1]+s[3]),(0,255,0),1)
                _cx = s[0] + int(s[2] / 2)
                _cy = s[1] + int(s[3] / 2)

                min_d = 1468  ## (0,0)으로부터 1280,720까지의 거리
                temp_num = 0;
                while temp_num < len(list_of_midpoint_x):
                    # print(list_of_midpoint_x[temp_num], list_of_midpoint_y[temp_num])
                    distance = math.sqrt(
                        pow(abs(_cx - list_of_midpoint_x[temp_num]), 2) + pow(abs(_cy - list_of_midpoint_y[temp_num]),
                                                                              2))
                    ##print("d:" ,distance)
                    if distance < min_d:
                        min_d = distance
                    temp_num = temp_num + 1

                # print("min:",min_d)

                if min_d > distance_gap:
                    list_of_midpoint_x.append(_cx)
                    list_of_midpoint_y.append(_cy)
                    ##print("cx,cy :",_cx,_cy)

                    ##Determined_point
                    if (len(list_of_determined_x) == 0):
                        list_of_determined_x.append(_cx)
                        list_of_determined_y.append(_cy)
                        list_of_determined_cnt.append(1)
                    else:
                        temp2_num = 0
                        Overlap_check = 0
                        while temp2_num < len(list_of_determined_x):
                            ##print("dx, dy :",temp2_num,list_of_determined_x[temp2_num],list_of_determined_y[temp2_num])
                            distance2 = math.sqrt(pow(abs(_cx - list_of_determined_x[temp2_num]), 2) + pow(
                                abs(_cy - list_of_determined_y[temp2_num]), 2))
                            ##print(distance2)

                            ##중복 체크
                            if distance2 < 100:
                                Overlap_check = 1
                                list_of_determined_cnt[temp2_num] = list_of_determined_cnt[temp2_num] + 1
                                ##print("cnt:", list_of_determined_cnt[temp2_num])
                            else:
                                pass

                            temp2_num = temp2_num + 1

                        if Overlap_check == 0:
                            list_of_determined_x.append(_cx)
                            list_of_determined_y.append(_cy)
                            list_of_determined_cnt.append(1)
                            ##print(list_of_determined_x)
                            ##print(list_of_determined_y)
                        else:
                            pass

                    cv2.circle(Original_frame, (int(_cx), int(_cy)), 3, (0, 0, 255), 10)
                    ##cv2.rectangle(Original_frame, (int(_cx) - 45, int(_cy) - 45), (int(_cx) + 45, int(_cy) + 45), (0, 0, 255),3)  ###임의의 거리 대입해놓음
                    ##cv2.putText(Original_frame, '50%', (int(_cx) + 45, int(_cy) - 45), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # cv2.circle(Original_img, (_cx, _cy), 3, (0, 0, 255), -1)
                ##cv2.rectangle(Original_frame, (int(_cx) - 45, int(_cy) - 45), (int(_cx) + 45, int(_cy) + 45), (0, 0, 255), 3)
                ##cv2.putText(Original_frame, '50%', (int(_cx) + 45, int(_cy) - 45), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                ##print(list_of_determined_x)
                ##print(list_of_determined_y)
                ##print(list_of_determined_cnt)

                temp3_num = 0
                while temp3_num < len(list_of_determined_x):
                    Probabilty = list_of_determined_cnt[temp3_num] / f_num * 100

                    ## 9. Define_Parkinglot
                    ## 150 프레임 이상동안 확률이 70프로 이상이면 Parking_lot으로 정의
                    if f_num > 150 and int(Probabilty) > 70:
                        Def_x = list_of_determined_x[temp3_num]
                        Def_y = list_of_determined_y[temp3_num]
                        if len(list_of_Defined_x) == 0:
                            list_of_Defined_x.append(list_of_determined_x[temp3_num])
                            list_of_Defined_y.append(list_of_determined_y[temp3_num])
                            list_of_Defined_coordinates.append(
                                [[Def_x - 45, Def_y - 45], [Def_x - 45, Def_y + 45], [Def_x + 45, Def_y + 45],
                                 [Def_x + 45, Def_y - 45]])
                            list_of_Defined_status.append(False)
                            list_of_Defined_ID.append(ID_num)
                            ID_num = ID_num + 1
                        else:
                            temp4_num = 0
                            checking_flag = 0
                            while temp4_num < len(list_of_Defined_x):
                                if list_of_Defined_x[temp4_num] == list_of_determined_x[temp3_num] and \
                                        list_of_Defined_y[temp4_num] == list_of_determined_y[temp3_num]:
                                    checking_flag = 1;
                                else:
                                    pass
                                temp4_num = temp4_num + 1
                            if checking_flag == 0:
                                list_of_Defined_x.append(list_of_determined_x[temp3_num])
                                list_of_Defined_y.append(list_of_determined_y[temp3_num])
                                list_of_Defined_coordinates.append(
                                    [[Def_x - 45, Def_y - 45], [Def_x - 45, Def_y + 45], [Def_x + 45, Def_y + 45],
                                     [Def_x + 45, Def_y - 45]])
                                list_of_Defined_ID.append(ID_num)
                                list_of_Defined_status.append(False)
                                ID_num = ID_num + 1

                    ##determined_x_y_출력
                    if int(Probabilty) > 10:
                        cv2.rectangle(Original_frame, (
                        int(list_of_determined_x[temp3_num]) - 60, int(list_of_determined_y[temp3_num]) - 60), (
                                      int(list_of_determined_x[temp3_num]) + 60,
                                      int(list_of_determined_y[temp3_num]) + 60), (0, 255, 0), 3)
                        cv2.putText(Original_frame, str(int(Probabilty)) + "%", (
                        int(list_of_determined_x[temp3_num]) + 60, int(list_of_determined_y[temp3_num]) - 40), font, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(Original_frame, "Parkinglot",
                                    (int(list_of_determined_x[temp3_num]) - 70,
                                     int(list_of_determined_y[temp3_num]) - 70),
                                    font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    temp3_num = temp3_num + 1

        if key == ord("q"):
            break

    cv2.imshow('frame', Original_frame)

    if key == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()