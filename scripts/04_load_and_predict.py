"""ปรับจาก Material 3.4: โหลด @staging และทำนายแถวแรกของแต่ละคลาส."""

import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")
    model_uri = "models:/cancer-classifier-prod@staging"
    print(f"Loading model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    # index 0 เป็น malignant และ index 19 เป็น benign ในข้อมูลชุดนี้
    sample_indices = [y.index[y.eq(label)][0] for label in (0, 1)]
    sample_data = X.loc[sample_indices]
    predictions = model.predict(sample_data)

    print("Sample index | Actual    | Predicted | Correct")
    for sample_index, prediction in zip(sample_indices, predictions, strict=True):
        actual_label = int(y.loc[sample_index])
        predicted_label = int(prediction)
        actual_name = data.target_names[actual_label]
        predicted_name = data.target_names[predicted_label]
        correct = actual_label == predicted_label
        print(f"{sample_index:12} | {actual_name:9} | {predicted_name:9} | {correct}")


if __name__ == "__main__":
    load_and_predict()
