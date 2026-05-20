import cv2
import mediapipe as mp
import os

# ===== SETUP AWAL =====
label = input("Masukkan label awal: ")
jumlah = int(input("Jumlah gambar tiap capture: "))

def get_folder(label):
    return f"data/dataset/{label}"

folder_path = get_folder(label)
os.makedirs(folder_path, exist_ok=True)

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

img_count = len(os.listdir(folder_path))

print("\n=== CONTROL ===")
print("SPACE = capture")
print("L = ganti label")
print("Q = keluar\n")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    tangan_terdeteksi = False

    if result.multi_hand_landmarks:
        tangan_terdeteksi = True
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ===== TAMPILAN =====
    status = "Terdeteksi" if tangan_terdeteksi else "Tidak"
    warna = (0, 255, 0) if tangan_terdeteksi else (0, 0, 255)

    cv2.putText(frame, f"Label: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"Status: {status}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, warna, 2)

    cv2.imshow("Collect Image", frame)

    key = cv2.waitKey(1)

    # ===== GANTI LABEL =====
    if key == ord('l'):
        label = input("\nMasukkan label baru: ")

        # format otomatis
        label = label.upper() if len(label) == 1 else label.lower()

        folder_path = get_folder(label)
        os.makedirs(folder_path, exist_ok=True)

        img_count = len(os.listdir(folder_path))

        print(f"Label diganti ke: {label}")

    # ===== CAPTURE =====
    if key == 32:  # SPACE
        for i in range(jumlah):

            # countdown 3 detik
            for detik in range(2, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                cv2.putText(frame, f"{label} dalam: {detik}", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                cv2.imshow("Collect Image", frame)
                cv2.waitKey(1000)

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                img_count += 1
                img_name = f"{label}_{img_count}.jpg"
                img_path = os.path.join(folder_path, img_name)

                cv2.imwrite(img_path, frame)
                print(f"✅ {img_name} disimpan di {folder_path}")
            else:
                print("❌ Tangan tidak terdeteksi")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()