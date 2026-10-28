
from pathlib import Path
import joblib
import pandas as pd

# Exact model path
MODEL_PATH = Path(r"D:\CA2\model.joblib")

print("Looking for model at:", MODEL_PATH)

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}])

prediction = model.predict(sample)

print("Model loaded successfully!")
print("Prediction:", int(prediction[0]))
print(
    "Survival:",
    "Survived" if prediction[0] == 1 else "Did not survive"
)