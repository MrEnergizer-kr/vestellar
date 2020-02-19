import cv2
import numpy as np

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


points = []

#grabbed, frame = cap.read()
cap = cv2.VideoCapture('test.mp4')




while cap.isOpened():
    grabbed, frame = cap.read()

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = (0, 0, 171)
    upper_white = (158, 54, 255)
    img_mask = cv2.inRange(img_hsv, lower_white, upper_white)
    dark_image = cv2.bitwise_and(frame, frame, mask=img_mask)

    if frame is None:
        break

    Original_frame = np.copy(frame)
    cv2.imshow('original', Original_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imshow('img', frame)
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', on_mouse)  ## window name이 'img'인 윈도우에서 마우스 이벤트가 발생하면 on_mouse 함수가 호출 됨

    if len(points) == 4: ## points라는 리스트에 2개 좌표가 들어오면
        x = points[0]
        y = points[1]
        z = points[2]  ## 좌표 4개로 사각형 만듦
        w = points[3]
        cv2.destroyWindow('img')

        vertices = np.array([[x, y, z]], dtype=np.int32)
        roi_images = region_mask(dark_image, vertices)
        cv2.imshow('roi', roi_images)
        ## 삼각형 여러개를 ROI로 받을 수 있게끔 작업 필요

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()