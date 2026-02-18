# Semi-Supervised Learning for Medical Survival Prediction

## Executive Summary

This project explores supervised and semi-supervised learning approaches for predicting short- and medium-term patient survival outcomes using clinical and hospitalization data.

We compare:

- K-Nearest Neighbors (KNN)
- Self-Training KNN
- Transductive Support Vector Machines (TSVM)

The goal is to evaluate whether leveraging unlabeled data improves classification performance under limited labeled samples.

---

## Problem Statement

Given structured clinical, demographic, and physiological data, we aim to classify patients into survival categories based on hospitalization and mortality signals.

The task is multi-class classification with partially labeled data.

We investigate whether semi-supervised methods (Self-Training & TSVM) outperform standard supervised learning.

---

## Data Processing Pipeline

Implemented in `EDA.py`.

Key steps:

- Target construction from survival indicators
- Categorical encoding (demographics, diagnosis groups, DNR status)
- Missing value handling
- Feature binning for physiological variables
- Outlier removal using IQR filtering
- Correlation-based feature filtering
- Optional PCA dimensionality reduction

The dataset is normalized before modeling.

---

## Experimental Setup

Defined in `main.py`.

Data is split into:

- Labeled training set
- Unlabeled training set
- Test set

Stratified custom splitting preserves class distribution across subsets.

Multiple random seeds are evaluated to test robustness.

---

## Models

### 1️ Supervised KNN

Implemented in `KNN.py`.

Hyperparameters explored:
- Distance metric (Euclidean / Manhattan)
- Weighting (uniform / distance)
- Number of neighbors (5, 10, 15)

Performance evaluated across multiple seeds.

---

### 2️ Self-Training KNN

A semi-supervised wrapper using `SelfTrainingClassifier`.

Unlabeled samples are iteratively pseudo-labeled and incorporated into training.

Comparison performed between:
- Regular KNN
- Self-training KNN

---

### 3️ Transductive SVM (TSVM)

Implemented in `newTSvm.py`.

Approaches:
- One-vs-All
- One-vs-One

Hyperparameters:
- Kernel: RBF
- Gamma
- Cu (unlabeled penalty strength)

Voting mechanism applied for multi-class prediction.

---

## Results & Observations

- KNN achieved moderate classification performance.
- Self-training provided marginal improvement depending on seed.
- TSVM performance was sensitive to hyperparameters.
- Semi-supervised gains were not consistent across seeds.

Key insight:

Unlabeled data does not automatically improve performance.  
Semi-supervised learning is highly dependent on data distribution assumptions (cluster assumption / manifold assumption).

---

## Technical Stack

- Python
- Pandas / NumPy
- Scikit-learn
- LAMDA_SSL (TSVM implementation)
- Matplotlib / Seaborn
- OpenPyXL (Excel result logging)

---

## Key Learning Outcomes

- Semi-supervised learning implementation
- Custom stratified splitting
- Multi-class TSVM via One-vs-One voting
- Experimental evaluation across random seeds
- Model sensitivity analysis
- Handling partially labeled datasets

---


