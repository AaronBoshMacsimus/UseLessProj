# classify_file.py
from lightweight_classifier_module import LightweightShapeClassifier  # adjust if your class is in georeal.py
import os

img = "test_picamera2.jpg"
model_path = "lightweight_shape_model.pkl"

clf = LightweightShapeClassifier()
clf.load_model_lightweight(model_path)

pred, conf = clf.predict(img)
shape_type = "3D" if pred in clf.shape_classes_3d else "2D"
print(f"Prediction: {pred} ({shape_type}), confidence={conf:.3f}")
