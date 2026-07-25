# Enhancing Sign Language Recognition using Deep Convolutional Neural Networks

TensorFlow/Keras implementation of the convolutional neural network architectures presented in the paper:

**Enhancing Sign Language Recognition using Deep Convolutional Neural Networks**

## Task

Multi-class classification of American Sign Language alphabet images using the **Sign Language MNIST** dataset.

The dataset contains:

- 27,455 training examples
- 7,172 test examples
- 28 × 28 grayscale images
- 24 classes
- Labels corresponding to A–Z, excluding J and Z because those signs involve motion

## Dataset

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/datamunge/sign-language-mnist
```

Expected files:

```text
sign_mnist_train.csv
sign_mnist_test.csv
```

## CNN Architectures

### Architecture 1

```text
(Conv2D ×2 → BatchNormalization → MaxPooling2D → Dropout) ×3
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

### Architecture 2

```text
((Conv2D → BatchNormalization) ×2 → MaxPooling2D → Dropout) ×3
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

### Architecture 3

```text
(Conv2D ×3 → BatchNormalization → MaxPooling2D → Dropout) ×3
→ Conv2D ×2
→ BatchNormalization
→ MaxPooling2D
→ Dropout
→ GlobalAveragePooling2D
→ Flatten
→ Dense(256)
→ Dropout
→ Softmax Output
```

## Implementation Details

The implementation follows the methodology presented in the paper and uses the following configuration:

- Input shape: `28 × 28 × 1`
- Number of classes: `24`
- Kernel size: `3 × 3`
- Filter progression: `32 → 64 → 128 → 256`
- Activation: ReLU
- Optimizer: Adam
- Learning rate: `1e-3`
- Loss: Sparse categorical cross-entropy
- Block dropout: `0.25`
- Dense dropout: `0.50`
- Default epochs: `100`
- Optional training augmentation
- Random seed: `42`

## Project Structure

```text
Sign-Language-Recognition-CNN/
├── README.md
├── LICENSE
├── requirements.txt
├── train.py
├── evaluate.py
├── .gitignore
├── outputs/
└── src/
    ├── __init__.py
    ├── data.py
    ├── models.py
    └── utils.py
```

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Training

Architecture 1:

```bash
python train.py \
  --train-csv "path/to/sign_mnist_train.csv" \
  --test-csv "path/to/sign_mnist_test.csv" \
  --architecture 1 \
  --batch-size 32 \
  --epochs 100
```

Architecture 2:

```bash
python train.py \
  --train-csv "path/to/sign_mnist_train.csv" \
  --test-csv "path/to/sign_mnist_test.csv" \
  --architecture 2 \
  --batch-size 32 \
  --epochs 100
```

Architecture 3:

```bash
python train.py \
  --train-csv "path/to/sign_mnist_train.csv" \
  --test-csv "path/to/sign_mnist_test.csv" \
  --architecture 3 \
  --batch-size 64 \
  --epochs 100
```

Batch sizes evaluated in the paper:

```text
32, 64, 128, 256
```

## Evaluation

```bash
python evaluate.py \
  --test-csv "path/to/sign_mnist_test.csv" \
  --model-path "outputs/architecture_3/best_model.keras"
```

The evaluation script produces:

- Loss
- Accuracy
- Confusion matrix
- Classification report
- Per-class precision, recall, and F1-score

## Published Results

The paper reports a maximum accuracy of approximately **98.73%** and a minimum loss of **0.0539**.

The third architecture generally achieved the best performance, especially with smaller batch sizes.

Results may vary depending on the software environment, preprocessing pipeline, random initialization, hyperparameter configuration, and hardware platform.

## Citation

```bibtex
@inproceedings{DBLP:conf/iisa/KanavosPMM23,
  author       = {Athanasios Kanavos and
                  Orestis Papadimitriou and
                  Phivos Mylonas and
                  Manolis Maragoudakis},
  editor       = {Nikolaos G. Bourbakis and
                  George A. Tsihrintzis and
                  Maria Virvou},
  title        = {Enhancing Sign Language Recognition Using Deep Convolutional Neural
                  Networks},
  booktitle    = {14th International Conference on Information, Intelligence, Systems
                  {\&} Applications, {IISA} 2023, Volos, Greece, July 10-12, 2023},
  pages        = {1--4},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/IISA59645.2023.10345865},
  doi          = {10.1109/IISA59645.2023.10345865},
  timestamp    = {Mon, 03 Mar 2025 21:13:05 +0100},
  biburl       = {https://dblp.org/rec/conf/iisa/KanavosPMM23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## License

This project is released under the MIT License.
