## 주차면 검출하여 주차영역의 Rectangler 표시

# 1. input_image
# 2. white, yellow masking out
# 3. RGB to GRAY
# 4. Canny Edge detect
# 5. region masking
    # 5.5 Hough Transform
    # 5.5.5 Hogh and ROI
# 6. Contour _ Rectangler Detect
# 7. Drawing Rectangler and Circle

##============================================

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
import time
import os

##======================================================================================================================
##======================================================================================================================
##======================================================================================================================
#----------------------------      function_list_for_parkinglot_detecting     ------------------------------------------

##대비 변경 함수
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


def region_mask(edge_images,vertices):
    f_mask = np.zeros_like(edge_images)
    if len(f_mask.shape) == 2:  ### 2차원 배열이면
        cv2.fillPoly(f_mask, vertices, 255)  ### 다각형 생성 --> 사각형이 아닌 다각형
    else:
        cv2.fillPoly(f_mask, vertices, (255,) * f_mask.shape[2])  # in case, the input image has a channel dimension

    # images showing the region of interest only
    roi_images = cv2.bitwise_and(edge_images, f_mask)
    return roi_images


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



#exc_func
##밝기_변경_함수
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

##======================================================================================================================
##======================================================================================================================
##======================================================================================================================
## -----------------      Set up for YOLO_ Car_ Detection_      --------------------------------------------------------

# construct the argument parse and parse the arguments

Input_Video = 'test.mp4'
#BasePath = "yolo-coco"
BasePath = 'yolo-coco'
BaseConfidence = 0.6 # 0.5
Base_threshold = 0.5 # 0.3

# load the COCO class labels our YOLO model was trained on

labelsPath = os.path.sep.join([BasePath, "coco.names"])
##labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])

LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(40)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([BasePath, "yolov3.weights"])
configPath = os.path.sep.join([BasePath, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# determine only the *output* layer names that we need from YOLO========================================================
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(Input_Video)
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1



##======================================================================================================================
##======================================================================================================================
##======================================================================================================================
#1. VIDEO_INPUT
cap = cv2.VideoCapture('test.mp4')

f_num = 0

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

while(cap.isOpened()and f_num<411):
    f_num = f_num+1
    print("Frame : ",f_num)
    grabbed, frame = cap.read()

    Original_frame = np.copy(frame)
    empty = np.empty(frame.shape)


    if f_num<155: ## 155프레임 (5초) 동안 주차면 인식 및 정의 수행
        ##==========================================================================================================##
        ##                                                                                                          ##
        ## 1. 대비 증가
        bright = img =increase_contrastness(frame)

        ## 2. 흰색, 노랑색 마스킹
        masked = white_color_mask(bright)

        ## 3. RGB2GRAY
        gray_images = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

        ## 4. canny
        ##low_threshold = 75
        low_threshold = 932  # 좀 바꾸면 변동이 있긴 함
        ##high_threshold = 255
        high_threshold = 160
        canny = cv2.Canny(gray_images, low_threshold, high_threshold)
        cv2.imshow('hi', canny)

        ## 5. dilate
        edge_images = cv2.dilate(canny, (5, 5), 7)

        ## 6. region_mask
        rows, cols = edge_images.shape[:2]  ## 1280 x 720
        #pt_1 = [5, 290]
        #pt_2 = [1271, 298]
        #pt_3 = [1275, 544]
        #pt_4 = [7, 528]
        ##추가
        ##k = cv2.waitKey(1) & 0xFF
        #==========여기서 부터 추가
        #if k == ord('s'):
        ##initBB = cv2.selectROI("Frame", frame, fromCenter=False,
		#	showCrosshair=True)
        #
        # 초기에만 selectROI 해주고 그 값을 각 cctv id마다 mapping 해준다.
        ## mysql에 데이터 저장하는 것까지 함수로 만든다 --> 추후 더 나은 주차면 알고리즘 적용되면 적용시키면 되기 때문
        ##끝
        pt_1 = [0, 0]     ##1920 x 1080
        pt_2 = [1920, 0]
        pt_3 = [1920, 1080]
        pt_4 = [0, 1080] ## 전체 영역 roi
        ## roi를 여기서 직접 좌표로 설정해주고 있음 --> 요거 수정 필요 마우스로 4개좌표 찍으면 사각형 vertice 나오게끔
        ## 추가적으로 이러한 roi를 자동으로 찾아나게끔 하는 알고리즘 추가해야함 --> 기둥을 찾고 그 기둥의 좌표 선으로 연결하여 그
        ## 안과 일부 밖의 부분까지 roi로 설정해 주어야 함
        vertices = np.array([[pt_1, pt_2, pt_3, pt_4]], dtype=np.int32)

        roi_images = region_mask(edge_images, vertices)
        cv2.imshow('roi', roi_images)


        ##7. Hough_transform
        list_of_lines =[]
        list_of_lines = cv2.HoughLinesP(roi_images, rho=1, theta=np.pi / 360, threshold=100, minLineLength=210,maxLineGap=30)
        print(list_of_lines)                                                                       #100

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
                #print(list_of_midpoint_x[temp_num], list_of_midpoint_y[temp_num])
                distance = math.sqrt(pow(abs(_cx - list_of_midpoint_x[temp_num]), 2) + pow(abs(_cy - list_of_midpoint_y[temp_num]), 2))
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
                if(len(list_of_determined_x)==0):
                    list_of_determined_x.append(_cx)
                    list_of_determined_y.append(_cy)
                    list_of_determined_cnt.append(1)
                else:
                    temp2_num = 0
                    Overlap_check = 0
                    while  temp2_num < len(list_of_determined_x):
                        ##print("dx, dy :",temp2_num,list_of_determined_x[temp2_num],list_of_determined_y[temp2_num])
                        distance2 = math.sqrt(pow(abs(_cx - list_of_determined_x[temp2_num]), 2) + pow(abs(_cy - list_of_determined_y[temp2_num]), 2))
                        ##print(distance2)

                        ##중복 체크
                        if distance2 <100:
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



                cv2.circle(Original_frame, (int(_cx), int(_cy)), 3, (0,0,255), 10)
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
                        list_of_Defined_coordinates.append([[Def_x-45,Def_y-45],[Def_x-45,Def_y+45],[Def_x+45,Def_y+45],[Def_x+45,Def_y-45]])
                        list_of_Defined_status.append(False)
                        list_of_Defined_ID.append(ID_num)
                        ID_num= ID_num + 1
                    else:
                        temp4_num=0
                        checking_flag=0
                        while temp4_num < len(list_of_Defined_x):
                            if list_of_Defined_x[temp4_num] == list_of_determined_x[temp3_num] and list_of_Defined_y[temp4_num] == list_of_determined_y[temp3_num]:
                                checking_flag =1;
                            else:
                                pass
                            temp4_num = temp4_num + 1
                        if checking_flag ==0:
                            list_of_Defined_x.append(list_of_determined_x[temp3_num])
                            list_of_Defined_y.append(list_of_determined_y[temp3_num])
                            list_of_Defined_coordinates.append([[Def_x-45,Def_y-45],[Def_x-45,Def_y+45],[Def_x+45,Def_y+45],[Def_x+45,Def_y-45]])
                            list_of_Defined_ID.append(ID_num)
                            list_of_Defined_status.append(False)
                            ID_num = ID_num + 1

                ##determined_x_y_출력
                if int(Probabilty) > 10:
                    cv2.rectangle(Original_frame, (int(list_of_determined_x[temp3_num]) - 60, int(list_of_determined_y[temp3_num]) - 60), (int(list_of_determined_x[temp3_num]) + 60, int(list_of_determined_y[temp3_num]) + 60), (0, 255,0),3)
                    cv2.putText(Original_frame, str(int(Probabilty))+"%", (int(list_of_determined_x[temp3_num]) + 60, int(list_of_determined_y[temp3_num]) - 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(Original_frame, "Parkinglot",
                                (int(list_of_determined_x[temp3_num]) - 70, int(list_of_determined_y[temp3_num]) - 70),
                                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                temp3_num =temp3_num + 1

    else: ##155프레임 이후 모션 디텍 팅 수행
        #print(list_of_midpoint_x, list_of_midpoint_y)

        #print("num of points", len(list_of_midpoint_x))


        ##print(list_of_Defined_coordinates)
        ##Point_format = {'id': 0, 'coordinates': [[381, 405], [380, 475], [455, 474], [452, 408]]},

        for index, p in enumerate(list_of_Defined_coordinates):
            Coordinates = np.array(p)
            # [[371, 401], [461, 401], [371, 491], [461, 491]]
            # [[689, 405], [779, 405], [689, 495], [779, 495]]
            rect = cv2.boundingRect(Coordinates)
            ##print(rect)
            ##(371, 401, 91, 91)
            ##(689, 405, 91, 91)

            new_Coordinates = Coordinates.copy()  ### 사각형 평행 이동 To (0,0)과 가깝게
            new_Coordinates[:, 0] = Coordinates[:, 0] - rect[0]
            new_Coordinates[:, 1] = Coordinates[:, 1] - rect[1]

            ##print(new_Coordinates)

            mask = cv2.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),  # 직사각형 크기만한 blank 이미지 생성
                [new_Coordinates],  ##새로운 좌표
                contourIdx=-1,  ##모든 Contour를 그림
                color=255,
                thickness=-1,
                lineType=cv2.LINE_8)

            ##cv2.imshow("mask",mask)
            mask = mask ==255

            blurred = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]  ## 원하는 사각형 이미지 가져옴
            ##cv2.imshow("roi",roi_gray)

            laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)  ##라플라시안 적용
            ## laplacina * mask 한 값의 절대값의 평균이 < 라플라시안 값보다 작으면
            list_of_Defined_status[index] = np.mean(np.abs(laplacian * mask)) < LAPLACIAN
            ##print("ma,la : ",len(mask),len(laplacian))
            if index == 0:
                cv2.imshow("lap", laplacian)
                print(np.mean(np.abs(laplacian * mask)))

        temp5_num = 0;
        while temp5_num < len(list_of_Defined_x):
            color=(255,255,255)
            if list_of_Defined_status[temp5_num] == True:
                color = (0,255,0)
            else:
                color = (0,0,255)

            cv2.rectangle(Original_frame,
                          (int(list_of_Defined_x[temp5_num]) - 70, int(list_of_Defined_y[temp5_num]) - 70),
                          (int(list_of_Defined_x[temp5_num]) + 70, int(list_of_Defined_y[temp5_num]) + 70), color,
                          3)
            cv2.putText(Original_frame, "Defined ID:" + str(list_of_Defined_ID[temp5_num]),
                        (int(list_of_Defined_x[temp5_num]) - 80, int(list_of_Defined_y[temp5_num]) - 80), font, 1,
                        color, 2, cv2.LINE_AA)
            temp5_num = temp5_num + 1
    ##print(list_of_Defined_status)



    ##                                                                                                          ##
    ##==========================================================================================================##

    ##==========================================================================================================##
    ##                                                                                                          ##
    ##                                          YOLO  CAR Detecting                                             ##

    '''
    if f_num % 10 == 1:
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        print("[INFO] {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > BaseConfidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, BaseConfidence, Base_threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.circle(Original_frame, (x+int(w/2), y+int(h/2)), 3, (255, 0, 0), 10)
                cv2.rectangle(Original_frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {}".format(LABELS[classIDs[i]], str(int(confidences[i] * 100)) + "%")
                cv2.putText(Original_frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    '''
    cv2.imshow('frame', Original_frame)



    ##                                                                                                          ##
    ##==========================================================================================================##
    ##cv2.imshow('frame', Original_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



##======================================================================================================================
##======================================================================================================================
cap.release()
cv2.destroyAllWindows()