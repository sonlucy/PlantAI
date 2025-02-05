## 🌿식물 병해 식별 시스템

## 개요
인공지능 모델과 웹 소켓 통신을 활용해 농가에서 보다 쉽게 병해에 대한 정보를 얻을 수 있도록 함으로써 해당 병해의 적법한 대응을 용이하게 함을 목표로 한다.


## 기능 블록도
![Image](https://github.com/user-attachments/assets/3cc4315f-4847-4465-8184-161761dec174)
- client : 웹 소켓을 통해 이미지를 바이너리 타입으로 전송하는 장치. 스마트폰 카메라, 웹캠 등으로 촬영한 이미지를 서버로 전송하고 서버에서 수신한 결과를 출력
- server : client에서 웹 소켓을 통해 수신한 바이너리 타입 이미지를 (size,channel)의 3차원 이미지로 변경한 뒤 병해 식별 모델에 입력해 예측값을 전달받아 client로 전송
- AI service : 전이학습(Transfer learning)과  미세조정(Fine tuning)을 거쳐 훈련된 모델이며 EfficientnetB7을 기반 모델로 하여  훈련 데이터셋의 클래스 당 이미지 수는 1000, 검증 데이터셋의 클래스 당 이미지 수는 100으로 하여 훈련
- 모델의 데이터셋은 Aihub의 노지 작물 질병 데이터셋을 활용하여 구축하였으며 모델이 식별하도록 훈련한 식물 종과 병해는 고추, 고추 탄저병, 배추, 배추 검은썩음병, 애호박, 애호박 흰가루병이다.

## 기능 순서도
![Image](https://github.com/user-attachments/assets/ea5d1217-0ba1-450d-8525-49ad3b4266a7)
- 웹 소켓을 활용해 이미지를 바이너리로 변환해 클라이언트에서 서버로 전송하고, 이를 다시 이미지로 변환해 AI service로 전송하여 상호 연동

## 상태 천이도
![Image](https://github.com/user-attachments/assets/dcb0d69b-bef8-42fa-9cf4-463318d4d34f)


## 주요 기능
- 다양한 작물(고추, 토마토, 오이 등)의 질병을 이미지 기반으로 분석
- 데이터 전처리 및 정규화 기능 지원
- 학습 및 평가 데이터셋을 위한 커스텀 데이터로더
- EfficientNet 및 ResNet50을 활용한 모델 학습 및 평가
- 학습 히스토리 저장 및 시각화


## 기술 스택
### 언어
- Python 3.8+
- html5
- css3
- javascript (ES6+)

### 라이브러리 및 프레임워크
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pickle
- Scikit-learn (옵션)

## 실행 결과
![Image](https://github.com/user-attachments/assets/e1301fab-b286-4239-bcd9-b1682ec19455)
