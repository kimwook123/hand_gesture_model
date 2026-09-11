import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import os
import time
import json


# ============================================================
# 경로 설정
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

save_dir = "./hand_img/person/"
model_path = "./models/hand_landmarker.task"

keypoint_path = os.path.join(save_dir, "keypoints.csv")
classes_file = os.path.join(save_dir, "classes.json")

os.makedirs(save_dir, exist_ok=True)


# ============================================================
# 파일 확인
# ============================================================

if not os.path.exists(model_path):
    print(f"Error: {model_path} 파일이 없습니다.")
    print("MediaPipe Hand Landmarker 모델을 넣어주세요.")
    exit()

if not os.path.exists(classes_file):
    print(f"Error: {classes_file} 파일이 없습니다.")
    exit()


# ============================================================
# 클래스 로드
# ============================================================

with open(classes_file, encoding="utf-8") as fp:
    gesture_classes = json.load(fp)

for gesture_name in gesture_classes:
    os.makedirs(
        os.path.join(save_dir, gesture_name),
        exist_ok=True
    )

print(
    f"제스처 {len(gesture_classes)}종:",
    gesture_classes
)


# ============================================================
# MediaPipe Tasks
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


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


landmarker = HandLandmarker.create_from_options(
    options
)


# ============================================================
# 기존 CSV 로드
# ============================================================

if os.path.exists(keypoint_path) and os.path.getsize(keypoint_path) > 0:

    old_df = pd.read_csv(keypoint_path)

    all_data = old_df.to_dict("records")

    print(f"기존 키포인트 {len(all_data)}개 로드")

else:

    all_data = []

    print("기존 keypoints.csv가 없거나 비어 있습니다. 새로 시작합니다.")


# ============================================================
# 이미지 번호 결정
# ============================================================

existing_indices = []

for root, dirs, files in os.walk(save_dir):

    for file in files:

        if (
            file.startswith("hand_")
            and file.endswith(".jpg")
        ):

            try:

                index = int(
                    file
                    .replace("hand_", "")
                    .replace(".jpg", "")
                )

                existing_indices.append(index)

            except ValueError:
                pass


if existing_indices:

    image_index = max(existing_indices) + 1

else:

    image_index = 0


print(
    f"이미지 번호 {image_index}부터 시작"
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
# 랜드마크 그리기
# ============================================================

def draw_landmarks(frame, landmarks):

    h, w = frame.shape[:2]

    points = []

    for landmark in landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )


    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2
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

    print("Error: 카메라를 열 수 없습니다.")

    landmarker.close()

    exit()


# ============================================================
# 데이터 수집 설정
# ============================================================

# 0.15초마다 저장
interval_sec = 0.15

# 이번 실행에서 저장할 개수
frame_total = 500

saved_count = 0

last_save_time = 0


# ============================================================
# Main
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break


    # 사용자가 보기 편하게 거울 화면
    frame = cv2.flip(
        frame,
        1
    )


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe Image
    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    timestamp_ms = int(
        time.monotonic() * 1000
    )


    # ========================================================
    # Hand Landmark 검출
    # ========================================================

    result = landmarker.detect_for_video(

        mp_image,

        timestamp_ms
    )


    view = frame.copy()


    # ========================================================
    # 손 발견
    # ========================================================

    if result.hand_landmarks:

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            draw_landmarks(
                view,
                landmarks
            )


            h, w = frame.shape[:2]


            xs = [
                int(p.x * w)
                for p in landmarks
            ]

            ys = [
                int(p.y * h)
                for p in landmarks
            ]


            padding = 40


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


            cv2.rectangle(

                view,

                (x1, y1),

                (x2, y2),

                (255, 0, 0),

                2
            )


        # ====================================================
        # 저장
        # ====================================================

        now = time.time()


        if (
            now - last_save_time
            >= interval_sec

            and saved_count
            < frame_total
        ):

            last_save_time = now


            for hand_index, landmarks in enumerate(
                result.hand_landmarks
            ):

                h, w = frame.shape[:2]


                xs = [
                    int(p.x * w)
                    for p in landmarks
                ]

                ys = [
                    int(p.y * h)
                    for p in landmarks
                ]


                padding = 40


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


                hand_crop = frame[
                    y1:y2,
                    x1:x2
                ]


                if hand_crop.size == 0:
                    continue


                # ============================================
                # 파일명
                # ============================================

                image_name = (
                    f"hand_{image_index}.jpg"
                )


                output_path = os.path.join(
                    save_dir,
                    image_name
                )


                cv2.imwrite(
                    output_path,
                    hand_crop
                )


                # ============================================
                # CSV 데이터
                # ============================================

                data = {

                    "image_name":
                    image_name
                }


                # 21개 landmark
                for i, landmark in enumerate(
                    landmarks
                ):

                    data[f"x{i}"] = landmark.x
                    data[f"y{i}"] = landmark.y
                    data[f"z{i}"] = landmark.z


                # handedness 저장
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

                else:

                    handedness = "Unknown"


                data["handedness"] = handedness


                all_data.append(
                    data
                )


                image_index += 1

                saved_count += 1


                if saved_count >= frame_total:
                    break


    # ========================================================
    # 화면 출력
    # ========================================================

    cv2.putText(

        view,

        f"Saved: {saved_count}/{frame_total}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 0, 255),

        2
    )


    cv2.imshow(

        "Hand Data Capture",

        view
    )


    if saved_count >= frame_total:
        break


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 종료
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


# ============================================================
# CSV
# ============================================================

df = pd.DataFrame(
    all_data
)


df.to_csv(

    keypoint_path,

    index=False
)


print()
print("==============================")
print("수집 완료")
print("==============================")

print(
    f"이번 실행 저장: {saved_count}"
)

print(
    f"전체 CSV: {len(df)}"
)

print(
    f"Saved: {keypoint_path}"
)