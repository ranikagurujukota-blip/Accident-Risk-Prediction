from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import joblib

app = Flask(__name__)

model, encoders = joblib.load("accident_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction", methods=["GET","POST"])
def prediction():

    result = ""

    if request.method=="POST":

        place = request.form["place"]
        weather = request.form["weather"]
        traffic = request.form["traffic"]
        speed = int(request.form["speed"])

        p = encoders["Place"].transform([place])[0]
        w = encoders["Weather"].transform([weather])[0]
        t = encoders["Traffic"].transform([traffic])[0]

        pred = model.predict([[p,w,t,speed]])

        result = encoders["Risk"].inverse_transform(pred)[0]

    return render_template("prediction.html",result=result)


@app.route("/charts")
def charts():
    return render_template("charts.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/performance")
def performance():

    df = pd.read_csv("accident_data.csv")

    from sklearn.preprocessing import LabelEncoder

    for col in ["Place","Weather","Traffic","Risk"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df[["Place","Weather","Traffic","Speed"]]
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 2)
    recall = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 2)
    f1 = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 2)

    return render_template(
        "performance.html",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )

if __name__=="__main__":
    app.run(debug=True)