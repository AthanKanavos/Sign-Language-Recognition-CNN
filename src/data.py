from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


AUTOTUNE = tf.data.AUTOTUNE


def read_sign_mnist_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "label" not in frame.columns:
        raise ValueError(f"Expected a 'label' column in {csv_path}")

    labels = frame["label"].to_numpy(dtype=np.int64)
    images = frame.drop(columns=["label"]).to_numpy(dtype=np.float32)

    if images.shape[1] != 784:
        raise ValueError(
            f"Expected 784 pixel columns, but found {images.shape[1]} in {csv_path}"
        )

    images = images.reshape(-1, 28, 28, 1) / 255.0
    return images, labels


def _augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )


def _make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
    use_augmentation: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(images),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size)

    if training and use_augmentation:
        augmentation = _augmenter()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    return dataset.prefetch(AUTOTUNE)


def load_datasets(
    train_csv: str | Path,
    test_csv: str | Path,
    batch_size: int = 32,
    validation_size: float = 0.15,
    seed: int = 42,
    use_augmentation: bool = True,
):
    train_images, train_labels = read_sign_mnist_csv(train_csv)
    test_images, test_labels = read_sign_mnist_csv(test_csv)

    x_train, x_val, y_train, y_val = train_test_split(
        train_images,
        train_labels,
        test_size=validation_size,
        random_state=seed,
        stratify=train_labels,
    )

    train_ds = _make_dataset(
        x_train, y_train, batch_size, True, seed, use_augmentation
    )
    val_ds = _make_dataset(
        x_val, y_val, batch_size, False, seed, False
    )
    test_ds = _make_dataset(
        test_images, test_labels, batch_size, False, seed, False
    )

    return train_ds, val_ds, test_ds
