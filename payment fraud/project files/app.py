from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("fraud_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST"])
def submit():

    data = [
        float(request.form['step']),
        int(request.form['type']),
        float(request.form['amount']),
        float(request.form['oldbalanceOrg']),
        float(request.form['newbalanceOrig']),
        float(request.form['oldbalanceDest']),
        float(request.form['newbalanceDest']),
        int(request.form['isFlaggedFraud'])
    ]

    # Debug print (keep this)
    print("INPUT:", data)

    prob = model.predict_proba([data])[0][1]
    print("FRAUD PROB:", prob)

    is_fraud = prob > 0.35

    percent = round(prob * 100, 1)

    return render_template(
    "submit.html",
    is_fraud=is_fraud,
    prob=round(prob, 3),
    percent=percent
    )



if __name__ == "__main__":
    app.run(debug=True)
