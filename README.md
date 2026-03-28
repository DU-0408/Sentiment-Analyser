# 🧠 Sentiment Analysis with Linear Classifiers

> MITx 6.86x Machine Learning — Project 1

A from-scratch implementation of linear classifiers for **sentiment analysis** on Amazon product reviews, classifying them as positive (+1) or negative (−1).

## 📌 Overview

This project builds a complete text classification pipeline:

1. **Implement core ML algorithms** — Perceptron, Average Perceptron, and Pegasos (SVM)
2. **Build an NLP feature pipeline** — bag-of-words with stopword removal
3. **Train & evaluate** classifiers on real-world review data
4. **Tune hyperparameters** and identify the most explanatory words for sentiment

## 🏗️ Project Structure

```
.
├── project1.py          # Core ML algorithm implementations
├── main.py              # Training, evaluation & hyperparameter tuning
├── utils.py             # Data loading, plotting & tuning utilities
├── test.py              # Unit tests for all implementations
├── reviews_train.tsv    # Training dataset (~2 MB)
├── reviews_val.tsv      # Validation dataset
├── reviews_test.tsv     # Test dataset
├── reviews_submit.tsv   # Submission dataset
├── toy_data.tsv         # 2D toy dataset for visualization
└── stopwords.txt        # Stopwords list for bag-of-words filtering
```

## ⚙️ Algorithms Implemented

### Part I — Linear Classifiers

| Algorithm | Description |
|---|---|
| **Hinge Loss** | Single-point and averaged hinge loss computation |
| **Perceptron** | Classic online learning algorithm with single-step updates |
| **Average Perceptron** | Averages parameters over all updates for better generalization |
| **Pegasos (SVM)** | Stochastic sub-gradient descent for SVM with L2 regularization |

### Part II — Text Classification

| Component | Description |
|---|---|
| **Classify** | Predicts ±1 labels using learned parameters θ and θ₀ |
| **Bag of Words** | Converts review text to a word-index dictionary with stopword removal |
| **Feature Extraction** | Transforms reviews into numerical feature vectors (supports binary & count modes) |
| **Classifier Accuracy** | End-to-end training + evaluation on train/validation splits |

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib (for plotting)

### Installation

```bash
git clone https://github.com/DU-0408/Sentiment-Analyser.git
cd sentiment-analysis-linear-classifiers
pip install numpy matplotlib
```

### Run Tests

```bash
python test.py
```

### Run the Full Pipeline

```bash
python main.py
```

This trains the **Pegasos** classifier (T=25, λ=0.01) — the best-performing model — on the full training set, evaluates on the test set, and prints the top-10 most explanatory words for sentiment.

## 📊 Results

The best classifier (Pegasos with T=25, λ=0.01) achieves strong accuracy on the test set. The model also identifies the most explanatory words — words whose learned weights most strongly indicate positive or negative sentiment.

## 📝 Key Concepts

- **Hinge Loss**: The loss function underlying SVMs; penalizes predictions that are correct but not confident enough.
- **Perceptron**: A simple online algorithm that updates weights whenever a misclassification occurs.
- **Pegasos**: An efficient SVM solver using stochastic gradient descent with a decaying learning rate (η = 1/√t).
- **Bag of Words**: Represents text as a vector of word counts (or binary indicators), ignoring word order.

## 📄 License

This project is part of the MITx 6.86x coursework. Please follow academic integrity guidelines if you are currently enrolled.
