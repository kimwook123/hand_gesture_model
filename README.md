# ✋ Hand Gesture PC Control

<p align="center">
  <b>웹캠으로 손 제스처를 인식해 PC 미디어 기능을 제어하는 프로젝트</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-00BFA5">
  <img src="https://img.shields.io/badge/XGBoost-Classifier-EB5B29">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/PyAutoGUI-PC%20Control-lightgrey">
</p>

---

## 📌 프로젝트 소개

**Hand Gesture PC Control**은 웹캠을 통해 사용자의 손 모양을 실시간으로 인식하고,  
인식된 제스처를 이용해 컴퓨터의 미디어 기능을 제어하는 프로젝트입니다.

MediaPipe Hand Landmarker를 사용해 손의 **21개 랜드마크**를 추출하고,  
이를 기반으로 XGBoost 모델이 사용자의 손 제스처를 분류합니다.

분류된 결과는 다음과 같은 PC 제어 명령으로 연결됩니다.

```text
웹캠
  ↓
MediaPipe Hand Landmarker
  ↓
21개 손 랜드마크 추출
  ↓
랜드마크 정규화
  ↓
XGBoost 분류
  ↓
제스처 인식
  ↓
PC 미디어 제어
```
🎯 인식하는 제스처   
손 제스처	클래스	실행 기능   
✊ 주먹 쥐기	stop	재생 정지   
🖐 손바닥 펴기	play	재생   
☝ 검지만 위로 펴기	volume_up	볼륨 증가   
👎 엄지만 아래로 향하기	volume_down	볼륨 감소   
✌ V자 만들기	next	다음 곡   
🛠 사용 기술
Computer Vision
OpenCV
MediaPipe Tasks
MediaPipe Hand Landmarker
Machine Learning
XGBoost
Scikit-learn
NumPy
Pandas
PC Control
PyAutoGUI
Language
Python
🖐 Hand Landmark

MediaPipe Hand Landmarker는 한 손에서 총 21개의 랜드마크를 추출합니다.

각 랜드마크는 다음 좌표를 가집니다.

x, y, z

따라서 한 손에서 사용하는 특징 수는:

21 Landmarks × 3 Coordinates
= 63 Features

입니다.

이 63개의 특징을 XGBoost 모델의 입력값으로 사용합니다.

🔄 데이터 처리 과정

손 랜드마크의 원본 좌표를 그대로 사용할 경우 다음 요소의 영향을 크게 받을 수 있습니다.

손의 화면 위치
카메라와의 거리
손의 크기
사용자별 손 크기 차이

이를 줄이기 위해 랜드마크 정규화를 적용했습니다.

정규화 과정
1. 손목 좌표를 기준점으로 설정
2. 모든 랜드마크에서 손목 좌표를 차감
3. 손목과 중지 MCP 간 거리를 기준으로 크기 정규화

이를 통해 손이 화면 어디에 위치하더라도 비슷한 특징값을 얻을 수 있도록 구성했습니다.

📂 프로젝트 구조
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
📸 01. 학습 데이터 수집
01_hand_capture.py

웹캠을 이용해 손 이미지를 촬영합니다.

동시에 MediaPipe Hand Landmarker가 손의 21개 랜드마크를 추출합니다.

생성되는 이미지:

hand_0.jpg
hand_1.jpg
hand_2.jpg
...

랜드마크 데이터:

keypoints.csv

CSV에는 이미지 이름과 함께 총 63개의 랜드마크 특징이 저장됩니다.

image_name
x0, y0, z0
x1, y1, z1
...
x20, y20, z20
🏷 02. 데이터 라벨링
02_make_dataset.py

촬영된 이미지를 사람이 직접 제스처별 폴더로 이동합니다.

예:

hand_img/person/

├─ stop/
├─ play/
├─ volume_up/
├─ volume_down/
└─ next/

파일 이름은 변경하지 않고 폴더만 이동합니다.

예:

hand_10.jpg
    ↓
volume_up/hand_10.jpg

각 폴더 이름을 정답 라벨로 사용하여 keypoints.csv와 결합합니다.

최종적으로:

dataset.csv

가 생성됩니다.

🧠 03. XGBoost 모델 학습
03_train_model.py

생성한 dataset.csv를 사용하여 XGBoost 분류 모델을 학습합니다.

학습 과정:

dataset.csv
   ↓
랜드마크 정규화
   ↓
Train / Test 분리
   ↓
XGBoost 학습
   ↓
모델 평가

평가 지표:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

학습 완료 후 모델은 다음 파일로 저장됩니다.

hand_model.xgb
🎥 04. 실시간 제스처 인식
04_hand_detect.py

웹캠 영상에서 실시간으로 손을 탐지합니다.

웹캠
 ↓
MediaPipe Hand Landmarker
 ↓
63개 특징 추출
 ↓
랜드마크 정규화
 ↓
XGBoost 예측
 ↓
제스처 분류

화면에는 다음 정보가 표시됩니다.

손 랜드마크
Bounding Box
제스처 이름
예측 Confidence
Left / Right Hand
⚠️ UNKNOWN 처리

XGBoost는 입력이 들어오면 학습된 클래스 중 하나를 반드시 선택하려는 특성이 있습니다.

따라서 학습하지 않은 손 모양도 잘못 분류될 수 있습니다.

이를 줄이기 위해 Confidence Threshold를 적용했습니다.

PRED_THRESHOLD = 0.75

예측 확률이 기준보다 낮으면:

UNKNOWN

으로 처리합니다.

🛡 제스처 안정화

실시간 영상에서는 손을 움직이는 과정에서 순간적으로 잘못된 제스처가 인식될 수 있습니다.

예:   
PLAY
 ↓
VOLUME_UP
 ↓
NEXT
 ↓
PLAY
   
이런 오작동을 방지하기 위해 일정 프레임 이상 동일한 제스처가 유지될 때만 명령을 실행합니다.

VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP
VOLUME_UP

        ↓

Volume Up 실행

설정값:

STABLE_FRAME_COUNT = 8
💻 PC 제어

PyAutoGUI를 사용해 인식된 제스처를 Windows 미디어 키 입력으로 변환합니다.

volume_up
→ 시스템 볼륨 증가

volume_down
→ 시스템 볼륨 감소

next
→ 다음 곡
볼륨 제어

볼륨 제스처를 유지하고 있으면 일정 간격으로 반복 실행됩니다.

VOLUME_REPEAT_DELAY = 0.30
단발성 명령

다음 제스처는 같은 자세를 유지하고 있어도 반복 실행되지 않습니다.

play
stop
next

한 번 손 모양을 풀고 다시 제스처를 취해야 재실행됩니다.

📄 classes.json

학습에 사용할 제스처 목록은 classes.json에서 관리합니다.

[
    "stop",
    "play",
    "volume_up",
    "volume_down",
    "next"
]

이 순서는 XGBoost 모델의 클래스 번호와 연결됩니다.

예:

0 → stop
1 → play
2 → volume_up
3 → volume_down
4 → next
🚀 실행 순서
1. 01_hand_capture.py
       ↓
   이미지 및 랜드마크 수집

2. 손 이미지를 클래스별 폴더로 직접 분류
       ↓

3. 02_make_dataset.py
       ↓
   dataset.csv 생성

4. 03_train_model.py
       ↓
   XGBoost 모델 학습

5. 04_hand_detect.py
       ↓
   실시간 손 제스처 인식 및 PC 제어
✨ 주요 기능
✅ 웹캠 기반 실시간 손 탐지   
✅ 손가락 21개 랜드마크 추출   
✅ 사용자 정의 손 제스처 데이터셋 생성   
✅ XGBoost 기반 다중 클래스 분류   
✅ 랜드마크 위치/크기 정규화   
✅ Confidence Threshold 적용   
✅ 여러 프레임 기반 제스처 안정화   
✅ 여러 손 동시 탐지   
✅ Windows 미디어 제어   
📈 향후 개선 방향
여러 사용자 데이터 추가
좌우 손 차이에 대한 보정
데이터 증강
손 회전 및 각도 변화 대응
사용자별 제스처 커스터마이징
LSTM / GRU 기반 동적 제스처 인식
손 추적 ID 적용
Spotify / YouTube Music / VLC 직접 제어
GUI 기반 제스처 등록 기능
📌 Summary

MediaPipe Hand Landmarker로 손의 21개 랜드마크를 추출하고,
XGBoost로 사용자의 손 제스처를 분류하여
손 동작만으로 PC 미디어 기능을 제어하는 프로젝트입니다.


이 버전이 GitHub에서는 훨씬 보기 좋을 거야. 특히 상단 배지 + 이모지 + 단계별 설명이 있어서 처음 보는 사람도 프로젝트 흐름을 빠르게 이해할 수 있어.
