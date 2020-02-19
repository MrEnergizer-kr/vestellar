import cv2
import numpy as np

red = np.uint8([[[0, 0, 255]]]) #BGR
hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
print(hsv_red)
## 결과 값의 특정색과 비슷한 계열의 색상은 [Hue-10,100,100]에서 [Hue+10,255,255]까지의 범위로 나타낼 수 있다
## hue는 BGR * 0.5
## 즉, 빨간색 계열은 [-10,100,100]에서 [10,255,255]까지