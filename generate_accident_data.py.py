import pandas as pd
import numpy as np

np.random.seed(42)
data = []

for _ in range(1500):
    ear = np.random.uniform(0.15, 0.35)
    closed_frames = np.random.randint(0, 40)
    drowsy = 1 if ear < 0.23 and closed_frames > 15 else 0
    time_of_day = np.random.choice([0, 1])  # 0=day, 1=night
    speed = np.random.choice([0, 1, 2])     # low, medium, high

    if drowsy == 1 and speed == 2 and time_of_day == 1:
        risk = 2   # HIGH
    elif drowsy == 1 or speed == 2:
        risk = 1   # MEDIUM
    else:
        risk = 0   # LOW

    data.append([ear, closed_frames, drowsy, time_of_day, speed, risk])

df = pd.DataFrame(data, columns=[
    "ear", "closed_frames", "drowsy", "time_of_day", "speed", "risk"
])

df.to_csv("accident_risk_data.csv", index=False)
print("accident_risk_data.csv created successfully")
