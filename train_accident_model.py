import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("accident_risk_data.csv")

X = df.drop("risk", axis=1)
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

with open("accident_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Accident risk ML model trained & saved")
