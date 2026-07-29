import pandas as pd
import matplotlib.pyplot as plt
import os

# Create graphs folder if it doesn't exist
os.makedirs("static/graphs", exist_ok=True)

# Read dataset
df = pd.read_csv("accident_data.csv")

# --------------------------
# Graph 1 - Place Wise
# --------------------------
plt.figure(figsize=(8,5))
df["Place"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Accidents by Place")
plt.xlabel("Place")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/graphs/place.png")
plt.close()

# --------------------------
# Graph 2 - Weather Wise
# --------------------------
plt.figure(figsize=(6,6))
df["Weather"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.ylabel("")
plt.title("Weather Distribution")
plt.tight_layout()
plt.savefig("static/graphs/weather.png")
plt.close()

# --------------------------
# Graph 3 - Risk Wise
# --------------------------
plt.figure(figsize=(7,5))
df["Risk"].value_counts().plot(kind="bar", color="orange")
plt.title("Risk Level")
plt.xlabel("Risk")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/graphs/risk.png")
plt.close()

print("Graphs Created Successfully")