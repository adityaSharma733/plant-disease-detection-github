from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("data/plantvillage")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Dataset not found. Put the class folders inside data/plantvillage/"
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Classes:", class_names)

normalization = tf.keras.layers.Rescaling(1.0 / 255)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    normalization,
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save(MODEL_DIR / "plant_disease_model.keras")

with open(MODEL_DIR / "class_names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

print("Model saved successfully.")
