import numpy as np

np.random.seed(0)
## seed : 어떤 특정한 시작 숫자를 정해주면 컴퓨터가 정해진 알고리즘에 의해 마치 난수처럼 보이는 수열을 생성한다.
## 이러한 시작 숫자를 seed 라고 한다.
## 일단 생성된 난수는 다음번 난수 생성을 위한 시드값이 된다. 즉, seed값은 한번만 정하면 된다.
## seed(seed0) --> 난수1 , 난수1(seed1) --> 난수2
## seed값은 보통 현재시간을 이용하여 정해지지만 사람이 수동으로 설정할 수 있다.
## 특정한 시트값이 사용되면 그 다음에 만들어지는 난수들은 모두 예측 할 수 있다.

## seed를 설정하는 명령은 seed()이다.
## 고정 seed값을 설정해서 암호같은 형식으로 사용하는건가?

np.random.randint(5, size=(2, 4))
## >>> array([[4, 0, 2, 1],
##            [3, 2, 2, 0]])
## 0~4사이의 정수로 2x4 배열을 생성

np.random.randint(1, 40, size=(2, 4))
## 1~39사이의 정수로 2x4 배열을 생성


#==========================================#
import cv2

cap = cv2.VideoCapture(fn)

if not cap.isOpened(): 
    print "could not open :",fn
    return

length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
## cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT) --> 프레임 정보 중 프레임 수 확인
## cap.get(cv2.cv.CV_CAP_PROP_FRAME_WITDH) --> 프레임 정보 중 프레임 너비 확인
## cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) --> 프레임 정보 중 프레임 높이 확인
## cap.get(cv2.cv.CV_CAP_PROP_FPS) --> 프레임 정보 중 FPS 확인
#========================================

# 이미지에서 blue영역
# blue 영역의 from ~ to
ret, frame = cap.read()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_blue = np.array([110, 50, 50]) # from
upper_blue = np.array([130, 255, 255]) # to
mask = cv2.inRange(hsv, lower_blue, upper_blue)
