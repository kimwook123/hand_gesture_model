# ✋ Hand Gesture PC Control

웹캠으로 사용자의 손 제스처를 실시간 인식하고, 인식된 제스처를 이용해 PC의 미디어 기능을 제어하는 프로젝트입니다.

MediaPipe Hand Landmarker로 손의 21개 랜드마크를 추출하고, XGBoost 모델을 이용해 손 제스처를 분류합니다.

분류 결과에 따라 재생, 정지, 볼륨 조절, 다음 곡 등의 명령을 실행합니다.

---

## 🎯 프로젝트 목표

키보드나 마우스를 직접 사용하지 않고, 웹캠 앞에서 손 제스처만으로 PC의 미디어 기능을 제어하는 비접촉식 인터페이스를 구현하는 것이 목표입니다.

전체 처리 흐름은 다음과 같습니다.

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
```

---

## ✋ 인식하는 제스처

| 손 제스처 | 클래스명 | 실행 기능 |
| --- | --- | --- |
| ✊ 주먹 쥐기 | `stop` | 재생 정지 |
| 🖐 손바닥 펴기 | `play` | 재생 |
| ☝ 검지만 위로 펴기 | `volume_up` | 볼륨 증가 |
| 👎 엄지만 아래로 향하기 | `volume_down` | 볼륨 감소 |
| ✌ V자 만들기 | `next` | 다음 곡 |

---

## 🛠 사용 기술

### Computer Vision

- OpenCV
- MediaPipe Tasks
- MediaPipe Hand Landmarker

### Machine Learning

- XGBoost
- Scikit-learn
- NumPy
- Pandas

### PC Control

- PyAutoGUI

### Language

- Python

---

## 🖐 Hand Landmark

MediaPipe Hand Landmarker는 한 손에서 총 **21개의 랜드마크**를 추출합니다.

각 랜드마크는 `x`, `y`, `z` 좌표를 가지며, 한 손에서 사용하는 특징 수는 다음과 같습니다.

```text
21 Landmarks × 3 Coordinates = 63 Features
```

이 63개의 특징을 XGBoost 모델의 입력값으로 사용합니다.

---

## 🔄 데이터 처리 방식

손 랜드마크의 원본 좌표를 그대로 사용할 경우 다음 요소의 영향을 받을 수 있습니다.

- 손의 화면 위치
- 카메라와의 거리
- 손 크기
- 사용자별 손 크기 차이

이를 줄이기 위해 랜드마크 정규화를 적용합니다.

### 정규화 과정

1. 손목 랜드마크를 기준점으로 설정
2. 모든 랜드마크에서 손목 좌표를 차감
3. 손목과 중지 MCP 사이의 거리를 기준으로 크기 정규화

이를 통해 손이 화면의 왼쪽이나 오른쪽에 위치하거나 카메라와의 거리가 달라져도 비슷한 특징값을 얻도록 구성했습니다.

---

## 📂 프로젝트 구조

```text
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
        ├─ stop/
        ├─ play/
        ├─ volume_up/
        ├─ volume_down/
        └─ next/
```

---

## 📸 1. 학습 데이터 수집

### `01_hand_capture.py`

웹캠을 통해 손 이미지를 촬영하고, 동시에 MediaPipe Hand Landmarker로 손의 21개 랜드마크를 추출합니다.

### 저장되는 이미지 예시

```text
hand_0.jpg
hand_1.jpg
hand_2.jpg
...
```

### 저장되는 키포인트 데이터

```text
keypoints.csv
```

CSV에는 이미지 이름과 21개 랜드마크의 `x`, `y`, `z` 좌표가 저장됩니다.

---

## 🏷 2. 이미지 분류 및 데이터셋 생성

### `02_make_dataset.py`

촬영한 손 이미지를 직접 확인한 뒤 각 제스처 폴더로 이동합니다.

```text
hand_img/person/
├─ stop/
├─ play/
├─ volume_up/
├─ volume_down/
└─ next/
```

파일 이름은 변경하지 않고 폴더만 이동합니다.

예:

```text
hand_10.jpg
→ volume_up/hand_10.jpg
```

각 폴더 이름을 정답 라벨로 사용하여 `keypoints.csv`와 결합하고, 최종적으로 `dataset.csv`를 생성합니다.

---

## 🧠 3. XGBoost 모델 학습

### `03_train_model.py`

생성된 `dataset.csv`를 이용해 XGBoost 분류 모델을 학습합니다.

```text
dataset.csv
  ↓
Landmark Normalization
  ↓
Train / Test Split
  ↓
XGBoost Training
  ↓
Model Evaluation
```

### 평가 지표

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

학습이 완료되면 모델을 `hand_model.xgb`로 저장합니다.

---

## 🎥 4. 실시간 손 제스처 인식

### `04_hand_detect.py`

웹캠 영상에서 실시간으로 손을 탐지하고, 학습된 XGBoost 모델로 제스처를 분류합니다.

```text
Webcam
  ↓
MediaPipe Hand Landmarker
  ↓
63 Features
  ↓
Normalization
  ↓
XGBoost Prediction
  ↓
Gesture Recognition
```

화면에는 다음 정보가 표시됩니다.

- 손 랜드마크
- Bounding Box
- 예측 제스처
- 예측 Confidence
- Left / Right Hand

---

## ⚠️ UNKNOWN 처리

XGBoost는 입력값이 들어오면 학습된 클래스 중 하나를 선택하려는 특성이 있습니다.

따라서 학습하지 않은 손 모양도 잘못 분류될 수 있어 Confidence Threshold를 적용합니다.

```python
PRED_THRESHOLD = 0.75
```

예측 확률이 기준보다 낮으면 `UNKNOWN`으로 처리합니다.

---

## 🛡 제스처 안정화

실시간 영상에서는 손을 움직이는 도중 순간적으로 다른 제스처가 인식될 수 있습니다.

이를 줄이기 위해 같은 제스처가 일정 프레임 이상 연속으로 인식될 경우에만 명령을 실행합니다.

```python
STABLE_FRAME_COUNT = 8
```

예를 들어 `VOLUME_UP`이 8프레임 연속 인식되면 볼륨 증가 명령을 실행합니다.

---

## 💻 PC 제어

PyAutoGUI를 이용해 인식된 제스처를 Windows 미디어 키 입력으로 변환합니다.

- `stop` → 재생 정지
- `play` → 재생
- `volume_up` → 시스템 볼륨 증가
- `volume_down` → 시스템 볼륨 감소
- `next` → 다음 곡

볼륨 관련 제스처는 손 모양을 유지하고 있으면 일정 간격으로 반복 실행됩니다.

```python
VOLUME_REPEAT_DELAY = 0.30
```

반면 `play`, `stop`, `next`는 같은 자세를 유지하고 있어도 반복 실행되지 않도록 처리합니다.

---

## 📄 classes.json

학습에 사용할 제스처 목록은 `classes.json`에서 관리합니다.

```json
[
  "stop",
  "play",
  "volume_up",
  "volume_down",
  "next"
]
```

이 순서는 XGBoost 클래스 번호와 연결됩니다.

```text
0 → stop
1 → play
2 → volume_up
3 → volume_down
4 → next
```

---

## 🚀 실행 순서

### 1. 손 이미지 및 랜드마크 수집

```bash
python 01_hand_capture.py
```

### 2. 수집한 이미지를 제스처별 폴더로 직접 분류

```text
stop/
play/
volume_up/
volume_down/
next/
```

### 3. 데이터셋 생성

```bash
python 02_make_dataset.py
```

### 4. XGBoost 모델 학습

```bash
python 03_train_model.py
```

### 5. 실시간 손 제스처 인식 및 PC 제어

```bash
python 04_hand_detect.py
```

---

## ✨ 주요 기능

- 웹캠 기반 실시간 손 탐지
- 손의 21개 랜드마크 추출
- 사용자 정의 손 제스처 데이터셋 생성
- XGBoost 기반 다중 클래스 분류
- 손 위치 및 크기 정규화
- Confidence Threshold 적용
- 연속 프레임 기반 제스처 안정화
- 여러 손 동시 탐지
- Windows 미디어 제어

---

## 📈 향후 개선 방향

- 여러 사용자 데이터 추가
- 왼손과 오른손 차이에 대한 보정
- 데이터 증강 적용
- 손 회전 및 각도 변화 대응
- 사용자별 제스처 커스터마이징
- LSTM / GRU 기반 동적 제스처 인식
- 손 추적 ID 적용
- Spotify / YouTube Music / VLC 직접 제어
- GUI 기반 제스처 등록 기능

---

## 📌 Summary

MediaPipe Hand Landmarker를 이용해 손의 21개 랜드마크를 추출하고, XGBoost를 통해 손 제스처를 분류하여 손 동작만으로 PC 미디어 기능을 제어하는 프로젝트입니다.
