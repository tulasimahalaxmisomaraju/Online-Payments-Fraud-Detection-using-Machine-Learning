import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ======================
# LOAD DATA
# ======================

df = pd.read_csv("fraud.csv")

print(df.head())
print(df.info())


# ======================
# DROP UNUSED COLUMNS
# ======================

for col in ["nameOrig", "nameDest"]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


# ======================
# ENCODE TYPE COLUMN
# ======================

if df["type"].dtype == "object":
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"])


# ======================
# SPLIT DATA
# ======================

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ======================
# TRAIN MODELS
# ======================

models = {
    "RandomForest": RandomForestClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "SVC": SVC(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

best_score = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)

    print("\n", name)
    print("Accuracy:", score)
    print(classification_report(y_test, pred))

    if score > best_score:
        best_score = score
        best_model = model


# ======================
# SAVE BEST MODEL
# ======================

pickle.dump(best_model, open("fraud_model.pkl", "wb"))
print("\nBest model saved as fraud_model.pkl")
