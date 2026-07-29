import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
# Read dataset
df = pd.read_csv("accident_data.csv")

# Label Encoding
encoders = {}

for col in ["Place", "Weather", "Traffic", "Risk"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features
X = df[["Place", "Weather", "Traffic", "Speed"]]

# Target
y = df["Risk"]

# Train Model
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X, y)

# Save Model
joblib.dump((model, encoders), "accident_model.pkl")

print("Model Trained Successfully")