## detect
## yolo

import yaml
##From coordinates_generator import CoordinatesGenerator
import cv2
import numpy as np
##from colors import *

#from TakePoints import TakePoints

import imutils
from imutils.video import FPS
import time
import os
import math

BasePath = "yolo-coco"
BaseConfidence = 0.3  #0.3
Base_threshold = 0.2  #0.3

def main():
	fps = FPS().start()
	writer = None
    ##일단 영상을 받아와서 영상을 움직이는 차량에 대해서만 보이게 한다.
	cap = cv2.VideoCapture("rtsp://210.89.190.107:10570/PSIA/Streaming/channels/0")
	#_, frame = cap.read()

	initBB = None

	YOLOINIT()

	## 처음 대조할 이미지를 저장해서 그 저장한 값과 비교하는 역할
	## 웹에서 영상을 불러올때 초기 이미지를 저장해서 비교군으로 저장하는 코드 필요
	first_frame = cv2.imread("empty.png")

	##link = "http://210.89.190.107:52273"
	##proxy_handler = urllib2.ProxyHandler({})
	##opener = urllib2.build_opener(proxy_handler)
	##req = urllib2.Request(link)
	##추가해야함
	first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
	first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
	while True:
		_, frame = cap.read()
		#'cap.read()의 return값은 2개인데 앞의 _,는 앞의 return값을 무시한다는 의미'

		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

		difference = cv2.absdiff(first_gray, gray_frame)
		_, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)


		first_frame = cv2.resize(first_frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
		frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
		difference = cv2.resize(difference, (1280, 720), interpolation=cv2.INTER_CUBIC)
		#cv2.imshow("First frame", first_frame)
		#cv2.imshow("Frame", frame)

		difference = cv2.dilate(difference, (25, 25), 50)
		mask3 = cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)  # 3 channel mask
		Substracted = cv2.bitwise_and(frame, mask3) ## Substracted가 원래는 im_thresh_color

		layerOutputs, start, end = YOLO_Detect(frame)

		# 3.YOLO_BOX_INFO(layerOutputs,BaseConfidence,Base_threshold))
		idxs, boxes, classIDs, confidences = YOLO_BOX_INFO(frame, layerOutputs, BaseConfidence, Base_threshold)

		# 4.검출된 화면의 X,Y 좌표 가져온다.
        # 검출됨 차량 수 만큼 좌표 가져옴
		Vehicle_x = []
		Vehicle_y = []
		Vehicle_w = []
		Vehicle_h = []

		#차량 포인트 가져옴
		Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h = Position(idxs, classIDs, boxes, Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h)

		#차량 포인트 그리기
		Draw_Points(frame, Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h)

		#blank_image = np.zeros((64, 1912, 3), np.uint8)
		#frame[0:64, 0:1912] = blank_image
		#frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
		# Substracted = cv2.resize(Substracted , (1280, 720), interpolation=cv2.INTER_CUBIC)

		fps.update()
		fps.stop()
		cv2.putText(frame, "FPS : " + "{:.2f}".format(fps.fps()), (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

		cv2.imshow("frame", frame)

		##cv2.imshow("difference", difference)
		##cv2.imshow("im_thresh_color", Substracted) ## --> 처리된 영상
		##cv2.imshow("image", frame) ## 원본 영상

		# yolo detect할 이미지는 im_thresh_color(Substracted로 이름 바꿈)를 계속해서 detect 시킬 것


		key = cv2.waitKey(30)
		if key == ord("q"):
			break

		##if key == ord("s"):
		##    cv2.imwrite("output.png",frame)  ##  저장인데 저장할 필욘 없다고 판단됨
		##    break

	##cap.release()
	##cv2.destroyAllWindows()

def YOLOINIT():
	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([BasePath, "coco.names"])

	global LABELS
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([BasePath, "yolov3-tiny.weights"])
	configPath = os.path.sep.join([BasePath, "yolov3-tiny.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	global net
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# determine only the *output* layer names that we need from YOLO====================================================
	global ln
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream, pointer to output video file, and
	# frame dimensions

	(W, H) = (None, None)

	# try to determine the total number of frames in the video file
	try:
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
			else cv2.CAP_PROP_FRAME_COUNT
		total = int(cap.get(prop))
		print("[INFO] {} total frames in video".format(total))

	# an error occurred while trying to determine the total
	# number of frames in the video file
	except:
		print("[INFO] could not determine # of frames in video")
		print("[INFO] no approx. completion time can be provided")
		total = -1
	return total
#end YOLOINIT()

def YOLO_Detect(frame): ## 이건 cpu로 돌리는 yolo
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	# blob : 영상 데이터를 포함할 수 있는 다차원 데이터 표현 방식
	# blob은 Mat 타입의 4차원 행렬로 반환됨(N:영상갯수,C:채널갯수,H:영상의 세로,W:영상의 가로)
	# 인자는 blobFromImage(image, scalefactor : 입력 영상 픽셀 값에 곱할 값, size() : 출력영상의 크기,
	# swapRB : (B,G,R)에서 (R,G,B)로 바꿀지 결정 true 면 바꿈, crop : 입력영상의 크기를 변경한 후 크롭을 수행할 것인지
	# scalefactor에 1/255.f의 의미는 입력영상 픽셀 값 범위를 0~1로 정규화한 딥러닝 모델 사용시 쓰는 값
	net.setInput(blob) ## blob을 network 입력으로 설정한다.
	start = time.time()
	layerOutputs = net.forward(ln) ## forward로 실행
	end = time.time()

	print("[INFO] {:.6f} seconds".format(end - start))
	return layerOutputs,start,end
#end YOLO_Detect()

def YOLO_BOX_INFO(frame,layerOutputs,BaseConfidence,Base_threshold):

	H, W = frame.shape[:2]  ## 1280 x 720
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
	return idxs, boxes , classIDs, confidences

def Position(idxs,classIDs,boxes,Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h):



    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            ##검출된 이미지가 차량(2) 또는 트럭(7) 인경우에
			## yolo.coco 리스트에 있는 종류들
			## flatten은 리스트들을 하나로 합쳐주는 함수
#			n = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]

#			def flatten(n):
#				org = []
#				for i in n:
#					if (isinstance(i, list)):
#						org += flatten(i)
#					else:
#						org.append(i)
#				return org
#
#			print(flatten(n))
            if classIDs[i] == 2 or classIDs[i] == 7:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # 검출된 번호에 맞게 차량의 위치, 크기 정보 대입


                Vehicle_x.append(x)
                Vehicle_y.append(y)
                Vehicle_w.append(w)
                Vehicle_h.append(h)


    return Vehicle_x, Vehicle_y, Vehicle_w, Vehicle_h

def Draw_Points(frame,Vehicle_x,Vehicle_y,Vehicle_w,Vehicle_h):
    if len(Vehicle_x) > 0:
        for i in range(0, len(Vehicle_x), 1):

            # 보여주기위한 칼라화면
            cv2.circle(frame, (Vehicle_x[i] + int(Vehicle_w[i] / 2), Vehicle_y[i] + Vehicle_h[i]), 5, (0, 255, 0),
                       -1)
#end func

def roi(frame,vertices):
    f_mask = np.zeros_like(frame)
    if len(f_mask.shape) == 2:               ### 2차원 배열이면
        cv2.fillPoly(f_mask, vertices, 255)  ### 다각형 생성
    else:
        cv2.fillPoly(f_mask, vertices, (255,) * f_mask.shape[2])  # in case, the input image has a channel dimension

    # images showing the region of interest only
    roi_images = cv2.bitwise_and(frame, f_mask)
    return roi_images
# end function

## tracking
## 3초 트래킹 1초 detecting 구조
## swap 영역 만들어서 3초 트래킹한 데이터 swap 영역에 저장
## 1초 detecting한

if __name__ == '__main__':
    main()