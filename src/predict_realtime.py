import cv2
import mediapipe as mp
import pickle
import numpy as np
from utils import process_landmarks

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pred_buffer = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    data = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        data = process_landmarks(data)

        if data is not None:
            data = np.array(data).reshape(1, -1)

            proba = model.predict_proba(data)
            confidence = np.max(proba)

            if confidence > 0.8:
                pred = model.predict(data)[0]
                pred_buffer.append(pred)

                if len(pred_buffer) > 10:
                    pred_buffer.pop(0)

    # stabilisasi output
    label_pred = ""

    if pred_buffer:
        label_pred = max(set(pred_buffer), key=pred_buffer.count)

    cv2.putText(frame, f"Hasil: {label_pred}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("BISINDO Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()