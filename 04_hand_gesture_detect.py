import numpy as np
import cv2
import mediapipe as mp
import xgboost as xgb
import pyautogui
import os
import json
import time


# ============================================================
# 경로
# ============================================================

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

os.chdir(script_dir)


save_dir = "./hand_img/person/"

model_path = "./models/hand_landmarker.task"

weight_file = os.path.join(
    save_dir,
    "hand_model.xgb"
)

classes_file = os.path.join(
    save_dir,
    "classes.json"
)


# ============================================================
# 설정
# ============================================================

# XGBoost 예측 신뢰도
PRED_THRESHOLD = 0.75

# 몇 프레임 연속 같은 제스처가 나와야 실행할지
STABLE_FRAME_COUNT = 8

# 볼륨 명령 반복 간격
VOLUME_REPEAT_DELAY = 0.30

# NEXT / PLAY / STOP 등을 다시 실행하기 위한 최소 시간
ACTION_COOLDOWN = 1.0

# 실제 컴퓨터 제어
# 처음 테스트할 때 False
# 정상 동작 확인 후 True
ENABLE_PC_CONTROL = True


# ============================================================
# 파일 검사
# ============================================================

if not os.path.exists(model_path):

    print(
        f"Error: {model_path} 없음"
    )

    exit()


if not os.path.exists(weight_file):

    print(
        f"Error: {weight_file} 없음"
    )

    exit()


if not os.path.exists(classes_file):

    print(
        f"Error: {classes_file} 없음"
    )

    exit()


# ============================================================
# 클래스
# ============================================================

with open(
    classes_file,
    encoding="utf-8"
) as fp:

    gesture_classes = json.load(fp)


print(
    "Gesture classes:",
    gesture_classes
)


# ============================================================
# XGBoost
# ============================================================

gesture_model = xgb.XGBClassifier()

gesture_model.load_model(
    weight_file
)


# ============================================================
# MediaPipe Tasks
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=model_path
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.6,

    min_hand_presence_confidence=0.6,

    min_tracking_confidence=0.6
)


landmarker = (
    HandLandmarker
    .create_from_options(options)
)


# ============================================================
# 손 연결선
# ============================================================

HAND_CONNECTIONS = [

    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


# ============================================================
# 색상
# ============================================================

GESTURE_COLORS = [

    (0, 0, 255),      # stop

    (0, 255, 0),      # play

    (255, 0, 0),      # volume_up

    (0, 255, 255),    # volume_down

    (255, 0, 255)     # next
]


UNKNOWN_COLOR = (
    128,
    128,
    128
)


# ============================================================
# 랜드마크 정규화
# ============================================================

def normalize_landmarks(landmarks):

    points = np.array([

        [
            p.x,
            p.y,
            p.z
        ]

        for p in landmarks

    ], dtype=np.float32)


    # --------------------------------------------------------
    # 손목을 원점으로 이동
    # --------------------------------------------------------

    wrist = points[0].copy()

    points = points - wrist


    # --------------------------------------------------------
    # 손목 -> 중지 MCP 거리로 크기 정규화
    # --------------------------------------------------------

    scale = np.linalg.norm(
        points[9]
    )


    if scale < 1e-6:

        scale = 1.0


    points = points / scale


    return points.flatten()


# ============================================================
# 손 그리기
# ============================================================

def draw_hand(
    frame,
    landmarks,
    color
):

    h, w = frame.shape[:2]


    points = [

        (
            int(p.x * w),
            int(p.y * h)
        )

        for p in landmarks

    ]


    # 연결선
    for start, end in HAND_CONNECTIONS:

        cv2.line(

            frame,

            points[start],

            points[end],

            color,

            2
        )


    # 관절
    for point in points:

        cv2.circle(

            frame,

            point,

            4,

            color,

            -1
        )


# ============================================================
# PC 명령 실행
# ============================================================

def execute_action(gesture):

    print(
        f"[ACTION] {gesture}"
    )


    # 테스트 모드라면 실제 명령 실행 안 함
    if not ENABLE_PC_CONTROL:
        return


    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if gesture == "stop":

        # Windows Media Play/Pause
        pyautogui.press(
            "playpause"
        )


    # --------------------------------------------------------
    # PLAY
    # --------------------------------------------------------

    elif gesture == "play":

        pyautogui.press(
            "playpause"
        )


    # --------------------------------------------------------
    # VOLUME UP
    # --------------------------------------------------------

    elif gesture == "volume_up":

        pyautogui.press(
            "volumeup"
        )


    # --------------------------------------------------------
    # VOLUME DOWN
    # --------------------------------------------------------

    elif gesture == "volume_down":

        pyautogui.press(
            "volumedown"
        )


    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    elif gesture == "next":

        pyautogui.press(
            "nexttrack"
        )


# ============================================================
# 카메라
# ============================================================

cap = cv2.VideoCapture(0)


cap.set(

    cv2.CAP_PROP_FRAME_WIDTH,

    1280
)


cap.set(

    cv2.CAP_PROP_FRAME_HEIGHT,

    720
)


if not cap.isOpened():

    print(
        "카메라를 열 수 없습니다."
    )

    landmarker.close()

    exit()


# ============================================================
# 제스처 안정화 변수
# ============================================================

previous_gesture = "UNKNOWN"

stable_count = 0

last_action = None

last_action_time = 0


# ============================================================
# Main
# ============================================================

while cap.isOpened():

    success, frame = cap.read()


    if not success:
        break


    # ========================================================
    # 거울 모드
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # RGB 변환
    # ========================================================

    rgb_frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    timestamp_ms = int(
        time.monotonic() * 1000
    )


    # ========================================================
    # MediaPipe 손 검출
    # ========================================================

    result = landmarker.detect_for_video(

        mp_image,

        timestamp_ms
    )


    # 이 프레임에서 가장 확실하게 인식된 제스처
    frame_best_gesture = "UNKNOWN"

    frame_best_confidence = 0.0


    # ========================================================
    # 각 손 처리
    # ========================================================

    if result.hand_landmarks:

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            # =================================================
            # Feature 생성
            # =================================================

            features = normalize_landmarks(
                landmarks
            )


            features = features.reshape(
                1,
                -1
            )


            # =================================================
            # XGBoost 확률 예측
            # =================================================

            probabilities = (
                gesture_model
                .predict_proba(features)[0]
            )


            pred = int(
                np.argmax(
                    probabilities
                )
            )


            confidence = float(
                probabilities[pred]
            )


            # =================================================
            # Threshold
            # =================================================

            if confidence >= PRED_THRESHOLD:

                gesture_name = (
                    gesture_classes[pred]
                )


                color = (

                    GESTURE_COLORS[

                        pred
                        % len(GESTURE_COLORS)

                    ]
                )


                # ---------------------------------------------
                # 여러 손 중 가장 confidence 높은 손만
                # 실제 명령 후보로 사용
                # ---------------------------------------------

                if (
                    confidence
                    > frame_best_confidence
                ):

                    frame_best_confidence = (
                        confidence
                    )

                    frame_best_gesture = (
                        gesture_name
                    )


            else:

                gesture_name = "UNKNOWN"

                color = UNKNOWN_COLOR


            # =================================================
            # Bounding Box
            # =================================================

            h, w = frame.shape[:2]


            xs = [

                int(p.x * w)

                for p in landmarks
            ]


            ys = [

                int(p.y * h)

                for p in landmarks
            ]


            padding = 25


            x1 = max(
                min(xs) - padding,
                0
            )


            y1 = max(
                min(ys) - padding,
                0
            )


            x2 = min(
                max(xs) + padding,
                w
            )


            y2 = min(
                max(ys) + padding,
                h
            )


            # =================================================
            # 랜드마크
            # =================================================

            draw_hand(

                frame,

                landmarks,

                color
            )


            # =================================================
            # Bounding Box
            # =================================================

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                color,

                2
            )


            # =================================================
            # Label
            # =================================================

            text = (

                f"{gesture_name.upper()} "

                f"{confidence:.2f}"
            )


            cv2.putText(

                frame,

                text,

                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),

                cv2.FONT_HERSHEY_DUPLEX,

                0.75,

                color,

                2
            )


            # =================================================
            # Left / Right
            # =================================================

            if (
                result.handedness

                and hand_index
                < len(result.handedness)
            ):

                handedness = (

                    result
                    .handedness[hand_index][0]
                    .category_name
                )


                cv2.putText(

                    frame,

                    handedness,

                    (
                        x1,
                        min(
                            y2 + 25,
                            h - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    color,

                    2
                )


    # ========================================================
    # 제스처 안정화
    # ========================================================

    if (
        frame_best_gesture
        == previous_gesture
    ):

        stable_count += 1


    else:

        previous_gesture = (
            frame_best_gesture
        )

        stable_count = 1


    # ========================================================
    # UNKNOWN이면 재실행 가능하도록 초기화
    # ========================================================

    if frame_best_gesture == "UNKNOWN":

        last_action = None


    # ========================================================
    # 명령 실행
    # ========================================================

    if (

        frame_best_gesture
        != "UNKNOWN"

        and stable_count
        >= STABLE_FRAME_COUNT

    ):

        current_time = time.time()


        # ----------------------------------------------------
        # 볼륨 명령
        #
        # 손을 계속 들고 있으면 일정 간격으로 계속 조절
        # ----------------------------------------------------

        if frame_best_gesture in [

            "volume_up",

            "volume_down"

        ]:

            if (
                current_time
                - last_action_time
                >= VOLUME_REPEAT_DELAY
            ):

                execute_action(
                    frame_best_gesture
                )

                last_action_time = (
                    current_time
                )

                last_action = (
                    frame_best_gesture
                )


        # ----------------------------------------------------
        # STOP / PLAY / NEXT
        #
        # 같은 자세를 계속 유지한다고 반복 실행하지 않음
        # 손 모양을 풀었다가 다시 해야 실행
        # ----------------------------------------------------

        else:

            if (
                frame_best_gesture
                != last_action

                and current_time
                - last_action_time
                >= ACTION_COOLDOWN
            ):

                execute_action(
                    frame_best_gesture
                )


                last_action = (
                    frame_best_gesture
                )


                last_action_time = (
                    current_time
                )


    # ========================================================
    # 화면 왼쪽 상단 상태 표시
    # ========================================================

    cv2.putText(

        frame,

        f"Gesture: {frame_best_gesture.upper()}",

        (20, 40),

        cv2.FONT_HERSHEY_DUPLEX,

        0.9,

        (0, 0, 255),

        2
    )


    cv2.putText(

        frame,

        (
            f"Stable: "
            f"{stable_count}/"
            f"{STABLE_FRAME_COUNT}"
        ),

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 0, 255),

        2
    )


    cv2.putText(

        frame,

        (
            f"Control: "
            f"{'ON' if ENABLE_PC_CONTROL else 'OFF'}"
        ),

        (20, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 0, 255),

        2
    )


    # ========================================================
    # 화면
    # ========================================================

    cv2.imshow(

        "Hand Gesture PC Control",

        frame
    )


    # Q 종료
    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):

        break


# ============================================================
# 종료
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()