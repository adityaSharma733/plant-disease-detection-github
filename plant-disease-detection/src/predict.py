import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = Path("models/plant_disease_model.keras")
CLASS_PATH = Path("models/class_names.txt")

if len(sys.argv) < 2:
    print("Usage: python src/predict.py path/to/leaf.jpg")
    raise SystemExit(1)

image_path = Path(sys.argv[1])

model = tf.keras.models.load_model(MODEL_PATH)
class_names = CLASS_PATH.read_text(encoding="utf-8").splitlines()

image = Image.open(image_path).convert("RGB").resize((224, 224))
array = np.array(image) / 255.0
array = np.expand_dims(array, axis=0)

prediction = model.predict(array, verbose=0)[0]
index = int(np.argmax(prediction))

print("Prediction:", class_names[index])
print("Confidence:", f"{prediction[index] * 100:.2f}%")
