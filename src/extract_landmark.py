import cv2
import mediapipe as mp
import os
import csv
from utils import process_landmarks

DATASET_DIR = "data/dataset"
OUTPUT_FILE = "data/processed/dataset.csv"

os.makedirs("data/processed", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

data_rows = []

for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing: {label}")

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        data = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

        data = process_landmarks(data)

        if data is not None:
            data.append(label)
            data_rows.append(data)

print("Saving CSV...")

with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_rows)

print("✅ Dataset siap")