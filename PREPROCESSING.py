import os
import pickle
import mediapipe as mp
import cv2
import sys

sys.stdout.reconfigure(encoding='utf-8')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

dataset_path = "sign_language_dataset"

data = []
labels = []

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)

    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_vals, y_vals, z_vals = [], [], []

                print(f"\nProcessing Image: {img_path}")
                print(f"Hand landmarks for label: {label}")

                for i, lm in enumerate(hand_landmarks.landmark):
                    x_vals.append(lm.x)
                    y_vals.append(lm.y)
                    z_vals.append(lm.z)
                    print(f"  Landmark {i}: (x: {lm.x:.6f}, y: {lm.y:.6f}, z: {lm.z:.6f})")

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_vals))
                    data_aux.append(lm.y - min(y_vals))
                    data_aux.append(lm.z - min(z_vals))

                data.append(data_aux)
                labels.append(label)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\nData saved successfully!")