from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from src.data import load_datasets
from src.models import build_model
from src.utils import ensure_directory, save_history, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reconstructed CNN for Sign Language MNIST."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--architecture", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_directory(
        Path(args.output_dir) / f"architecture_{args.architecture}"
    )
    save_json(vars(args), output_dir / "config.json")

    train_ds, val_ds, test_ds = load_datasets(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
        seed=args.seed,
        use_augmentation=not args.no_augmentation,
    )

    model = build_model(
        architecture=args.architecture,
        learning_rate=args.learning_rate,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(output_dir / "final_model.keras")
    save_history(history, output_dir)

    results = model.evaluate(test_ds, return_dict=True, verbose=1)
    save_json(
        {key: float(value) for key, value in results.items()},
        output_dir / "test_metrics.json",
    )

    print("\nTest metrics:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
