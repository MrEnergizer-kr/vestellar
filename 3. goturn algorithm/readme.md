goturnTracker.py는 시험 코드이며 초기에 bbox를 코드속에 설정해 주어야 함
new.py는 간단하게 물체를 인식하게끔하는 코드이며 실제 진행해보면 간단하게 물체를 detect함

goturn의 특징은 기본적으로 deep learning 기반입니다.
caffeemodel을 사용함으로써 GPU를 사용할 수 있게 되는데 caffe의 GPU에서는 100FPS, OPENCV CPU에서는 20FPS로 실행된다.
즉, caffe GPU를 사용하도록 설정을 해줘야 한다.

caffemodel에 훈련된 세트가 많을 수록 물체를 detecting하는데 있어서 어려움이 있었다. 가령
손바닥을 추적할때 얼굴위로 손바닥을 움직였을때 추적기가 얼굴에 걸리고 빠져나오지 않는 현상이 발생한다.
