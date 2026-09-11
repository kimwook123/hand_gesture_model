import pandas as pd
import os
import json


# ============================================================
# 경로
# ============================================================

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

os.chdir(script_dir)


save_dir = "./hand_img/person/"

keypoint_file = os.path.join(
    save_dir,
    "keypoints.csv"
)

dataset_file = os.path.join(
    save_dir,
    "dataset.csv"
)

classes_file = os.path.join(
    save_dir,
    "classes.json"
)


# ============================================================
# classes.json
# ============================================================

if not os.path.exists(classes_file):

    print(
        f"Error: {classes_file} 없음"
    )

    exit()


with open(
    classes_file,
    encoding="utf-8"
) as fp:

    gesture_classes = json.load(fp)


# ============================================================
# 이미지 → label
# ============================================================

image_to_label = {}


for gesture_name in gesture_classes:

    folder = os.path.join(
        save_dir,
        gesture_name
    )


    if not os.path.isdir(folder):

        print(
            f"{gesture_name}: 폴더 없음"
        )

        continue


    image_files = [

        file

        for file in os.listdir(folder)

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


    print(
        f"{gesture_name}: "
        f"{len(image_files)}장"
    )


    for image_name in image_files:

        image_to_label[
            image_name
        ] = gesture_name


# ============================================================
# CSV
# ============================================================

df = pd.read_csv(
    keypoint_file
)


df["label"] = (
    df["image_name"]
    .map(image_to_label)
)


# 폴더에 들어가지 않은 이미지는 제외
df = df.dropna(
    subset=["label"]
)


# ============================================================
# 저장
# ============================================================

df.to_csv(

    dataset_file,

    index=False
)


print()
print("==============================")
print("데이터셋 생성 완료")
print("==============================")


print(
    df["label"]
    .value_counts()
    .to_string()
)


print()
print(
    f"전체: {len(df)}개"
)

print(
    f"Saved: {dataset_file}"
)