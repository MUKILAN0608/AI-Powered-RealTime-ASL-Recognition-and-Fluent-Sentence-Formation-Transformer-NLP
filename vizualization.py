import os
import pickle
import numpy as np
import pandas as pd
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mediapipe as mp
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import sys

sys.stdout.reconfigure(encoding='utf-8')

dataset_path = "sign_language_dataset"


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


data = []
labels = []

for label in tqdm(os.listdir(dataset_path), desc="Processing Dataset"):
    label_path = os.path.join(dataset_path, label)
    
    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_vals, y_vals, z_vals = [], [], []

                for lm in hand_landmarks.landmark:
                    x_vals.append(lm.x)
                    y_vals.append(lm.y)
                    z_vals.append(lm.z)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_vals))
                    data_aux.append(lm.y - min(y_vals))
                    data_aux.append(lm.z - min(z_vals))

                data.append(data_aux)
                labels.append(label)

# Convert to Numpy Arrays
data = np.array(data)
labels = np.array(labels)

# Save Preprocessed Data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\n✅ Dataset Preprocessing Complete! Data saved in data.pickle.")

df = pd.DataFrame(data)
df['label'] = labels  

report = sv.analyze(df)
report.show_html('sweetviz_report.html')  
print("✅ Sweetviz report saved as sweetviz_report.html")


plt.figure(figsize=(12, 6))
sns.countplot(x=labels, palette='coolwarm', order=pd.Series(labels).value_counts().index)
plt.title("Dataset Label Distribution")
plt.xlabel("Signs")
plt.ylabel("Image Count")
plt.xticks(rotation=90)
plt.show()


pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette="tab10", alpha=0.8)
plt.title("PCA: Principal Component Analysis (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc='best')
plt.show()


tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_data = tsne.fit_transform(data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=labels, palette="tab10", alpha=0.8)
plt.title("t-SNE: Sign Language Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(loc='best')
plt.show()

# 5️⃣ Pairwise Landmark Distance Heatmap
distances = pdist(data[:, :21], metric='euclidean')
distance_matrix = squareform(distances)

plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, cmap='coolwarm', square=True)
plt.title("Heatmap of Landmark Distances")
plt.show()


def create_montage(image_folder, montage_size=(4, 4)):
    images = []
    for label in os.listdir(image_folder):
        label_path = os.path.join(image_folder, label)
        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path)[:montage_size[0] * montage_size[1]]:
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                images.append(img)

    if len(images) == 0:
        print("⚠️ No images found for montage!")
        return

    rows = []
    for i in range(0, len(images), montage_size[1]):
        rows.append(np.hstack(images[i:i+montage_size[1]]))

    montage = np.vstack(rows)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
    plt.title("Sign Language Image Montage")
    plt.axis('off')
    plt.show()

# Call Montage Function
create_montage("sign_language_dataset")
