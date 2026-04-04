import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

df = pd.read_csv("data/processed/dataset.csv", header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Akurasi: {accuracy * 100:.2f}%")

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model disimpan")