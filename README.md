# Hand Gesture PC Control

웹캠을 통해 사용자의 손 제스처를 인식하고, 인식된 제스처를 기반으로 컴퓨터의 미디어 기능을 제어하는 프로젝트입니다.

MediaPipe Hand Landmarker를 이용해 손의 21개 랜드마크 좌표를 추출하고, XGBoost 모델을 이용해 사용자의 제스처를 분류합니다.  
분류된 제스처는 재생, 정지, 볼륨 조절, 다음 곡 등의 PC 명령으로 연결됩니다.

---

## 프로젝트 개요

기존의 키보드나 마우스 입력 대신, 웹캠을 통해 사용자의 손 모양을 인식하여 컴퓨터를 제어하는 비접촉식 제스처 인터페이스를 구현하는 것이 목적입니다.

웹캠 영상에서 손을 탐지한 뒤, 손가락 관절의 위치를 기반으로 제스처를 분류합니다.

전체 처리 과정은 다음과 같습니다.

```text
Webcam
  ↓
MediaPipe Hand Landmarker
  ↓
21 Hand Landmarks
  ↓
Landmark Normalization
  ↓
XGBoost Classifier
  ↓
Gesture Classification
  ↓
PC Media Control
인식하는 제스처

현재 총 5개의 손 제스처를 인식합니다.

제스처	분류 이름	기능
주먹 쥐기	stop	재생 정지
손바닥 펴기	play	재생
검지만 위로 펴기	volume_up	볼륨 증가
엄지만 아래로 향하기	volume_down	볼륨 감소
검지와 중지를 V자로 펴기	next	다음 곡
사용 기술
Python
OpenCV
MediaPipe Tasks
MediaPipe Hand Landmarker
XGBoost
NumPy
Pandas
Scikit-learn
PyAutoGUI
손 랜드마크

MediaPipe Hand Landmarker를 사용하여 한 손에서 총 21개의 랜드마크를 추출합니다.

각 랜드마크는 다음과 같은 좌표를 가집니다.

x, y, z

따라서 한 손에서 사용하는 기본 특징 수는 다음과 같습니다.

21 Landmarks × 3 Coordinates
= 63 Features

이 63개의 특징을 이용해 XGBoost 모델이 손 제스처를 분류합니다.

데이터 처리

카메라에서 얻은 랜드마크 좌표를 그대로 사용할 경우 손의 위치나 카메라와의 거리에 따라 값이 크게 달라질 수 있습니다.

이를 줄이기 위해 다음과 같은 정규화를 적용합니다.

손목 랜드마크를 기준점으로 사용
모든 랜드마크에서 손목 좌표를 빼서 위치 차이 제거
손목과 중지 MCP 사이의 거리를 기준으로 크기 정규화

이를 통해 손이 화면의 왼쪽이나 오른쪽에 있거나 카메라에서 멀고 가까운 경우에도 비슷한 특징값을 얻을 수 있도록 구성했습니다.

프로젝트 구성

프로젝트는 데이터 수집, 라벨링, 학습, 실시간 인식 단계로 분리되어 있습니다.

01_hand_capture.py
02_make_dataset.py
03_train_model.py
04_hand_detect.py
1. 데이터 수집

01_hand_capture.py

웹캠을 통해 손 이미지를 캡처하고 동시에 MediaPipe Hand Landmarker를 이용해 21개의 손 랜드마크를 추출합니다.

저장되는 데이터:

hand_0.jpg
hand_1.jpg
hand_2.jpg
...

그리고 각 이미지에 대응하는 랜드마크가 다음과 같이 저장됩니다.

keypoints.csv
2. 데이터 라벨링 및 Dataset 생성

02_make_dataset.py

촬영한 이미지를 사람이 직접 각 제스처 폴더로 분류합니다.

예:

hand_img/person/

stop/
play/
volume_up/
volume_down/
next/

각 폴더 이름을 정답 라벨로 사용하여 기존 keypoints.csv와 결합합니다.

최종적으로 다음 파일을 생성합니다.

dataset.csv
3. XGBoost 모델 학습

03_train_model.py

생성된 dataset.csv의 손 랜드마크 좌표를 이용해 XGBoost 분류 모델을 학습합니다.

학습 데이터와 테스트 데이터는 분리하여 성능을 평가하며 다음 지표를 확인합니다.

Accuracy
Precision
Recall
F1-score
Confusion Matrix

학습 완료 후 모델은 다음 파일로 저장됩니다.

hand_model.xgb
4. 실시간 손 제스처 인식

04_hand_detect.py

웹캠 영상에서 실시간으로 손을 인식한 후 학습된 XGBoost 모델을 이용해 제스처를 분류합니다.

각 손마다 다음 정보를 화면에 표시합니다.

손 랜드마크
Bounding Box
예측한 제스처
예측 Confidence
Left / Right Hand

예측 Confidence가 일정 기준 이하일 경우 오인식을 줄이기 위해 UNKNOWN으로 처리합니다.

제스처 안정화

실시간 영상에서는 한 프레임의 오인식만으로 PC 명령이 실행될 경우 잘못된 동작이 발생할 수 있습니다.

이를 방지하기 위해 같은 제스처가 여러 프레임 연속으로 인식될 경우에만 명령을 실행합니다.

예:

VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP

→ Volume Up 실행

이를 통해 일시적인 오검출로 인한 잘못된 명령 실행을 줄였습니다.

PC 제어

인식된 제스처는 PyAutoGUI를 이용해 Windows 미디어 키 입력으로 변환됩니다.

예:

volume_up
→ Volume Up

volume_down
→ Volume Down

next
→ Next Track

볼륨 조절은 제스처를 유지하고 있을 경우 일정 간격으로 반복 실행되도록 구성되어 있습니다.

반면 play, stop, next와 같은 명령은 같은 자세를 유지하고 있는 동안 반복 실행되지 않도록 처리했습니다.

폴더 구조 예시
hand/
│
├─ 01_hand_capture.py
├─ 02_make_dataset.py
├─ 03_train_model.py
├─ 04_hand_detect.py
│
├─ models/
│   └─ hand_landmarker.task
│
└─ hand_img/
    └─ person/
        ├─ classes.json
        ├─ keypoints.csv
        ├─ dataset.csv
        ├─ hand_model.xgb
        │
        ├─ stop/
        ├─ play/
        ├─ volume_up/
        ├─ volume_down/
        └─ next/
classes.json

학습에 사용할 제스처 목록은 classes.json에서 관리합니다.

[
    "stop",
    "play",
    "volume_up",
    "volume_down",
    "next"
]

이 파일의 순서가 XGBoost 모델에서 사용하는 클래스 번호와 연결됩니다.

특징
웹캠 기반 실시간 손 제스처 인식
MediaPipe 기반 21개 손 관절 추출
XGBoost 기반 사용자 정의 제스처 분류
손 위치 및 크기에 대한 랜드마크 정규화
Confidence Threshold를 이용한 오인식 감소
연속 프레임 기반 제스처 안정화
여러 손 동시 탐지 가능
Windows 미디어 제어 기능
향후 개선 방향

현재 프로젝트는 정적인 손 모양을 중심으로 분류합니다.

추후 다음과 같은 기능을 추가할 수 있습니다.

여러 사용자의 데이터 추가를 통한 일반화 성능 향상
데이터 증강 적용
왼손 / 오른손별 특징 보정
LSTM, GRU 등을 이용한 연속 동작 인식
사용자별 제스처 커스터마이징
Spotify, YouTube Music 등 특정 프로그램 직접 제어
손 추적 ID를 이용한 다중 사용자 제어

이 버전이면 단순 코드 설명이 아니라 **“이 프로젝트가 뭔지 → 어떤 기술을 썼는지 → 데이터는 어떻게 만들었는지 → 모델은 어떻게 학습하는지 → 실제로 어떻게 PC를 제어하는지”**까지 README 하나로 설명할 수 있어.
