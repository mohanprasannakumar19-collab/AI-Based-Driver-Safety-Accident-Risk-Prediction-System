import pickle
import numpy as np

with open("accident_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_accident_risk(ear, closed_frames, drowsy, time_of_day, speed):
    features = np.array([[ear, closed_frames, drowsy, time_of_day, speed]])
    return model.predict(features)[0]
