import cv2

cap = cv2.VideoCapture("test.mp4")

# FPS = cap.get(cv2.CAP_PROP_FPS) 어차피 rtsp로 받아오는데 그땐 정상적으로 출력됨 빠르거나 느리지 않음
#print(FPS)
## 이미지 속성 변경 3 = width, 4 = height
##cap.set(3, 1920);
##cap.set(4, 1080);

while (cap.isOpened()):
    grabbed, frame = cap.read()

    if frame is None:
        break

    cv2.imshow('img', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()