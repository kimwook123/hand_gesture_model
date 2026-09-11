import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

dataset_file = os.path.join(
    save_dir,
    "dataset.csv"
)

classes_file = os.path.join(
    save_dir,
    "classes.json"
)

weight_file = os.path.join(
    save_dir,
    "hand_model.xgb"
)


# ============================================================
# 클래스
# ============================================================

with open(
    classes_file,
    encoding="utf-8"
) as fp:

    gesture_classes = json.load(fp)


label_to_id = {

    name: i

    for i, name
    in enumerate(gesture_classes)
}


# ============================================================
# Dataset
# ============================================================

df = pd.read_csv(
    dataset_file
)


print()
print("===== 데이터 =====")

print(
    df["label"]
    .value_counts()
    .to_string()
)


# ============================================================
# 데이터 검증
# ============================================================

empty_classes = [

    name

    for name in gesture_classes

    if name not in set(df["label"])
]


if empty_classes:

    print(
        "Error: 학습 데이터가 없는 클래스:",
        empty_classes
    )

    exit()


# ============================================================
# 랜드마크 정규화
# ============================================================

def normalize_landmarks(row):

    # 21 × XYZ
    points = np.array([

        [
            row[f"x{i}"],
            row[f"y{i}"],
            row[f"z{i}"]
        ]

        for i in range(21)

    ], dtype=np.float32)


    # --------------------------------------------------------
    # 손목을 원점으로
    # --------------------------------------------------------

    wrist = points[0].copy()

    points = (
        points - wrist
    )


    # --------------------------------------------------------
    # 손목 -> 중지 MCP(9번) 거리로 크기 정규화
    # --------------------------------------------------------

    scale = np.linalg.norm(
        points[9]
    )


    if scale < 1e-6:

        scale = 1.0


    points = (
        points / scale
    )


    # 21 × 3 → 63
    return points.flatten()


# ============================================================
# X, y 생성
# ============================================================

X = np.array([

    normalize_landmarks(row)

    for _, row
    in df.iterrows()

])


y = (
    df["label"]
    .map(label_to_id)
    .values
)


print()
print(
    "Feature shape:",
    X.shape
)


# ============================================================
# Train / Test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print(
    "Train:",
    len(X_train)
)

print(
    "Test:",
    len(X_test)
)


# ============================================================
# XGBoost
# ============================================================

model = xgb.XGBClassifier(

    objective="multi:softprob",

    eval_metric="mlogloss",

    n_estimators=400,

    max_depth=5,

    learning_rate=0.04,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=2,

    reg_alpha=0.1,

    reg_lambda=1.0,

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 학습
# ============================================================

print()
print("Training started...")


start_time = time.time()


model.fit(

    X_train,

    y_train,

    eval_set=[

        (X_train, y_train),

        (X_test, y_test)

    ],

    verbose=False
)


elapsed = (
    time.time()
    - start_time
)


print(
    f"Training Time: "
    f"{elapsed:.2f} sec"
)


# ============================================================
# 평가
# ============================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(

    y_test,

    y_pred
)


print()
print("==============================")
print("Accuracy")
print("==============================")

print(
    f"{accuracy:.4f}"
)


print()
print("==============================")
print("Classification Report")
print("==============================")


print(
    classification_report(

        y_test,

        y_pred,

        target_names=gesture_classes
    )
)


print()
print("==============================")
print("Confusion Matrix")
print("==============================")


print(
    confusion_matrix(

        y_test,

        y_pred
    )
)


# ============================================================
# 저장
# ============================================================

model.save_model(
    weight_file
)


print()
print(
    f"Saved: {weight_file}"
)