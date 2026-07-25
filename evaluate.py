from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data import read_sign_mnist_csv
from src.utils import ensure_directory, save_json


CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ASL model.")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = ensure_directory(args.output_dir or model_path.parent)

    images, labels = read_sign_mnist_csv(args.test_csv)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(model_path)
    evaluation = model.evaluate(dataset, return_dict=True, verbose=1)

    probabilities = model.predict(dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    labels_order = sorted(np.unique(labels).tolist())
    cm = confusion_matrix(labels, predictions, labels=labels_order)
    report = classification_report(
        labels,
        predictions,
        labels=labels_order,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    results = {
        **{key: float(value) for key, value in evaluation.items()},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    save_json(results, output_dir / "evaluation.json")

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\nEvaluation results:")
    for key, value in evaluation.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
