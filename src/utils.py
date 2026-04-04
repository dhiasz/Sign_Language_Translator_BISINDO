import math

def normalize_landmarks(landmarks):
    if len(landmarks) == 0:
        return landmarks

    base_x = landmarks[0]
    base_y = landmarks[1]
    base_z = landmarks[2]

    normalized = []

    # translasi (relatif ke titik awal)
    for i in range(0, len(landmarks), 3):
        normalized.append(landmarks[i] - base_x)
        normalized.append(landmarks[i+1] - base_y)
        normalized.append(landmarks[i+2] - base_z)

    # scaling (biar tidak tergantung jarak kamera)
    max_value = max([abs(v) for v in normalized]) if normalized else 1

    if max_value != 0:
        normalized = [v / max_value for v in normalized]

    return normalized


def add_distance_features(data):
    features = data.copy()

    def dist(i, j):
        x1, y1 = data[i*3], data[i*3+1]
        x2, y2 = data[j*3], data[j*3+1]
        return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

    # pasangan jari penting
    pairs = [
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20)
    ]

    for p in pairs:
        features.append(dist(p[0], p[1]))

    return features


def process_landmarks(data):
    # support 1 tangan & 2 tangan
    if len(data) == 63:
        data = data + [0] * 63
    elif len(data) != 126:
        return None

    data = normalize_landmarks(data)
    data = add_distance_features(data)

    return data