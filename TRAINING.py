import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys

sys.stdout.reconfigure(encoding='utf-8')
# Load preprocessed data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as sign_language_model.pkl")
