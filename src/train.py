from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


PIC_SIZE = 48
NUM_CLASSES = 7


def build_generators(data_dir: Path, batch_size: int):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at {train_dir} and {test_dir}. "
            "Please place your dataset inside data/train and data/test."
        )

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(PIC_SIZE, PIC_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    validation_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=(PIC_SIZE, PIC_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator


def build_model() -> Sequential:
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(NUM_CLASSES, activation="softmax"))

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history, output_path: Path) -> None:
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title("Training vs Validation Loss")
    plt.ylabel("Loss")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.title("Training vs Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_and_plot_confusion_matrix(model, validation_generator, output_path: Path) -> None:
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes
    class_names = list(validation_generator.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_model_json(model, output_path: Path) -> None:
    output_path.write_text(model.to_json())


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN for facial emotion recognition.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset root.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument(
        "--output_dir", type=str, default="models", help="Directory to save model artifacts."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_generator, validation_generator = build_generators(data_dir, args.batch_size)
    model = build_model()

    checkpoint = ModelCheckpoint(
        filepath=str(output_dir / "best_model.weights.h5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.n // train_generator.batch_size),
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.n // validation_generator.batch_size),
        callbacks=[checkpoint],
    )

    save_model_json(model, output_dir / "emotion_model.json")
    plot_training_history(history, output_dir / "training_history.png")
    evaluate_and_plot_confusion_matrix(model, validation_generator, output_dir / "confusion_matrix.png")

    print("\nTraining complete.")
    print(f"Artifacts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
