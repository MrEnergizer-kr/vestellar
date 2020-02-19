import numpy as np
import cv2

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        points.append([x, y])

def region_mask(frame,vertices):
    f_mask = np.zeros_like(frame)
    if len(f_mask.shape) == 2:  ### 2차원 배열이면
        cv2.fillPoly(f_mask, vertices, 255)  ### 다각형 생성 --> 사각형이 아닌 다각형
    else:
        cv2.fillPoly(f_mask, vertices, (255,) * f_mask.shape[2])  # in case, the input image has a channel dimension

    # images showing the region of interest only
    roi_images = cv2.bitwise_and(frame, f_mask)
    return roi_images

##이미지 불러오기
image = cv2.imread("../9. perspective/result/perspective.jpg")
img_h = image.shape[0]
#img_w = image.shape[1]

Original_frame = np.copy(image)
cv2.imshow("Original", Original_frame)

# 어떤 범위를 잘라낼지 마스크 만들기(그냥 사각형 하나)

points = []

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imshow('img', image)
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', on_mouse)  ## window name이 'img'인 윈도우에서 마우스 이벤트가 발생하면 on_mouse 함수가 호출 됨

    if len(points) == 2:  ## points라는 리스트에 2개 좌표가 들어오면
        x = [(points[0][0], 0)]
        y = [(points[0][0], img_h)]
        z = [(points[1][0], 0)]
        w = [(points[1][0], img_h)]
        print(points)
        cv2.destroyWindow('img')
        vertices = np.array([x, y, z, w], dtype=np.int32)
        print(vertices)
        roi_images = region_mask(image, vertices)
        cv2.imshow('roi', roi_images)
    if key == ord("q"):
        break

'''mask = np.zeros(image.shape[:2], dtype = "uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75, cY + 75), 255, -1)
cv2.imshow("Mask", mask)

# 위에서 만든 사각형과 내가 불러온 그림중에 AND 연산으로 겹치는 부분 보여주기
masked = cv2.bitwise_not(image, image, mask = mask)
#masked = cv2.bitwise_or(image, image, mask = mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)'''