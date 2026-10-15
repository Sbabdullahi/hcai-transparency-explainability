# -*- coding: utf-8 -*-
#HCAI
# ============================================================
# WEEK 3-4: HUMAN-CENTERED AI (HCAI)
# Deep Learning on UCI MNIST with HITL Override Mechanism.
#Author: S.B. Abdullahi
# ============================================================

# ------------------------------------------------------------
# CELL 1: INSTALL LIBRARIES
# ------------------------------------------------------------
"""
Library definitions:
- ucimlrepo: Official UCI ML Repository Python client (fetches datasets)
- numpy: Numerical computing (ndarray = n-dimensional array)
- pandas: Tabular data (DataFrame = 2D labeled table)
- matplotlib/seaborn: Visualization
- scikit-learn: Classical ML utilities (splits, metrics)
- tensorflow/keras: Deep learning framework (neural networks)
- shap: SHapley Additive exPlanations (interpretability)
"""

!pip install -q ucimlrepo pandas numpy matplotlib seaborn scikit-learn
!pip install -q tensorflow shap

# ------------------------------------------------------------
# CELL 2: IMPORTS
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)

import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# Reproducibility (deterministic behavior)
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")

# ------------------------------------------------------------
# CELL 3: LOAD UCI MNIST DATASET (ALL 10 CLASSES)
# ------------------------------------------------------------
"""
We fetch MNIST from UCI (ID=683). To keep the notebook fast for
low-level students, we SUBSAMPLE: e.g., 500 samples per class.
10 classes × 500 = 5,000 total samples (down from 70,000).
"""
# ------------------------------------------------------------
# CELL 3: LOAD MNIST VIA KERAS
# ------------------------------------------------------------
"""
Keras provides a built-in loader for the MNIST dataset.
This fetches the standard 60,000 train / 10,000 test split.
"""
from tensorflow import keras

print("Loading MNIST from Keras...")
(x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.mnist.load_data()

print(f"Original dataset: X_train={x_train_full.shape}, y_train={y_train_full.shape}")
print(f"                  X_test={x_test_full.shape}, y_test={y_test_full.shape}")
print(f"Classes: {sorted(np.unique(y_train_full))}")

# ------------------------------------------------------------
# CELL 4: SUBSAMPLE — ALL 10 CLASSES, FEWER SAMPLES PER CLASS
# ------------------------------------------------------------
# ------------------------------------------------------------
# CELL 4 (ADAPTED): SUBSAMPLE FROM THE KERAS LOADED DATA
# ------------------------------------------------------------
SAMPLES_PER_CLASS = 500  # Keep this for fast training

# Combine the training data for stratified sampling
import pandas as pd

# Flatten images for the DataFrame
X_flat = x_train_full.reshape(x_train_full.shape[0], -1)

df_full = pd.DataFrame(X_flat)
df_full['label'] = y_train_full

# Stratified subsample
df_sub = (df_full
          .groupby('label', group_keys=False)
          .apply(lambda g: g.sample(n=min(SAMPLES_PER_CLASS, len(g)), random_state=42))
          .reset_index(drop=True))

print(f"\nSubsampled dataset: {df_sub.shape}")
print(f"Class distribution:\n{df_sub['label'].value_counts().sort_index()}")

# Extract back to numpy arrays for the model
X_sub = df_sub.drop(columns=['label']).values.astype('float32')
y_sub = df_sub['label'].values.astype('int64')

# Reshape and normalize (rest of your preprocessing)
X_images = X_sub.reshape(-1, 28, 28, 1) / 255.0
y_onehot = keras.utils.to_categorical(y_sub, num_classes=10)


# ------------------------------------------------------------
# CELL 5: PREPROCESS: RESHAPE, NORMALIZE, ONE-HOT
# ------------------------------------------------------------
"""
Concepts:
- Reshape: 784-vector → 28×28 image (CNN needs spatial structure)
- Normalize: pixel values [0, 255] → [0, 1] (faster convergence)
- One-hot encoding: label 3 → [0,0,0,1,0,0,0,0,0,0]
"""

X = df_sub.drop(columns=['label']).values.astype('float32')
y = df_sub['label'].values.astype('int64')

# Normalize: divide by 255 (max pixel intensity)
X = X / 255.0

# Reshape to images: (N, 28, 28, 1)
X_images = X.reshape(-1, 28, 28, 1)

# One-hot encode labels (for categorical crossentropy)
y_onehot = utils.to_categorical(y, num_classes=10)

print(f"X_images shape: {X_images.shape}")
print(f"y_onehot shape: {y_onehot.shape}")
print(f"Pixel range: [{X.min():.2f}, {X.max():.2f}]")

# ------------------------------------------------------------
# CELL 6: TRAIN / TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_images, y_onehot, test_size=0.2, random_state=42, stratify=y
)

# Also keep integer labels for evaluation
_, _, y_train_int, y_test_int = train_test_split(
    X_images, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------------------------------------
# CELL 7: VISUALIZE SAMPLE DIGITS (HCAI: human understanding)
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
for digit in range(10):
    idx = np.where(y_train_int == digit)[0][0]
    axes[0, digit].imshow(X_train[idx].reshape(28, 28), cmap='gray')
    axes[0, digit].set_title(f"Digit {digit}", fontsize=9)
    axes[0, digit].axis('off')

    idx2 = np.where(y_train_int == digit)[0][1]
    axes[1, digit].imshow(X_train[idx2].reshape(28, 28), cmap='gray')
    axes[1, digit].axis('off')

plt.suptitle("Sample MNIST Digits (Two Examples per Class)", fontsize=12)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# CELL 8: ARCHITECTURE 1: MLP (Multilayer Perceptron)
# ------------------------------------------------------------
"""
MLP: Fully connected feedforward network.
Definition:
- Input layer: 784 neurons (flattened image)
- Hidden layers: Dense (fully connected) with ReLU activation
- Output layer: 10 neurons with Softmax (probability distribution)

ReLU(z) = max(0, z)  — introduces non-linearity
Softmax(z_i) = exp(z_i) / Σ exp(z_j)  — converts to probabilities
"""

def build_mlp():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),                          # 28×28 → 784
        layers.Dense(128, activation='relu'),      # Hidden layer 1
        layers.Dropout(0.3),                       # Regularization
        layers.Dense(64, activation='relu'),       # Hidden layer 2
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')     # Output: 10 classes
    ], name="MLP")
    return model

mlp = build_mlp()
mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
mlp.summary()

# ------------------------------------------------------------
# CELL 9: TRAIN MLP
# ------------------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_mlp = mlp.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
mlp_loss, mlp_acc = mlp.evaluate(X_test, y_test, verbose=0)
print(f"\n[MLP] Test Accuracy: {mlp_acc:.4f}")

# ------------------------------------------------------------
# CELL 10: ARCHITECTURE 2: CNN (Convolutional Neural Network)
# ------------------------------------------------------------
"""
CNN: Specialized for images using convolution operations.
Key layers:
- Conv2D: Slides a filter (kernel) over the image to detect features
  Formula: (f * I)(i,j) = Σ_m Σ_n f(m,n) · I(i+m, j+n)
- MaxPooling2D: Downsamples by taking max in each window (translation invariance)
- Flatten + Dense: Classifier head
"""

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Conv block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Conv block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ], name="CNN")
    return model

cnn = build_cnn()
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
cnn.summary()

# ------------------------------------------------------------
# CELL 11: TRAIN CNN
# ------------------------------------------------------------
history_cnn = cnn.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

cnn_loss, cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)
print(f"\n[CNN] Test Accuracy: {cnn_acc:.4f}")

# ------------------------------------------------------------
# CELL 12: TRAINING CURVES (HCAI: transparency in learning)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# MLP
axes[0].plot(history_mlp.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history_mlp.history['val_accuracy'], label='Val', linewidth=2)
axes[0].set_title(f'MLP — Accuracy (Test: {mlp_acc:.3f})', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

# CNN
axes[1].plot(history_cnn.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history_cnn.history['val_accuracy'], label='Val', linewidth=2)
axes[1].set_title(f'CNN — Accuracy (Test: {cnn_acc:.3f})', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# CELL 13: CONFUSION MATRIX (CNN)
# ------------------------------------------------------------
y_pred_cnn = np.argmax(cnn.predict(X_test, verbose=0), axis=1)

cm = confusion_matrix(y_test_int, y_pred_cnn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('CNN Confusion Matrix on MNIST Test Set', fontweight='bold')
plt.tight_layout()
plt.show()

print("\nClassification Report (CNN):")
print(classification_report(y_test_int, y_pred_cnn, digits=3))

# ------------------------------------------------------------
# CELL 14: HUMAN-IN-THE-LOOP (HITL) OVERRIDE SIMULATION
# ------------------------------------------------------------
"""
HCAI Core Concept:
A deployed system sends low-confidence predictions to a human reviewer.
The human either ACCEPTS or OVERRIDES the model.
Overrides are logged and fed back into training.
"""

CONFIDENCE_THRESHOLD = 0.90  # If model confidence < 90%, send to human

y_proba_cnn = cnn.predict(X_test, verbose=0)
y_pred_cnn = np.argmax(y_proba_cnn, axis=1)
confidence = np.max(y_proba_cnn, axis=1)

# Identify low-confidence samples (candidates for HITL review)
low_conf_mask = confidence < CONFIDENCE_THRESHOLD
high_conf_mask = ~low_conf_mask

print(f"Total test samples: {len(y_test_int)}")
print(f"High-confidence (auto): {high_conf_mask.sum()} "
      f"({high_conf_mask.mean():.1%})")
print(f"Low-confidence (HITL):  {low_conf_mask.sum()} "
      f"({low_conf_mask.mean():.1%})")

# Simulate human reviewer: corrects when model is wrong
def human_review(x, model_pred, true_label, review_cost=1.0):
    """
    Simulated human reviewer.
    Returns: final_label, was_override
    Real world: human sees image and decides independently.
    Here: we assume the human is perfect (ground truth = true_label).
    """
    if model_pred == true_label:
        return model_pred, False   # Human agrees
    else:
        return true_label, True    # Human overrides

# Build hybrid decision: model for high-conf, human for low-conf
final_pred = y_pred_cnn.copy()
override_log = []

for i in range(len(y_test_int)):
    if low_conf_mask[i]:
        final_label, was_override = human_review(
            X_test[i], y_pred_cnn[i], y_test_int[i]
        )
        final_pred[i] = final_label
        override_log.append({
            'index': i,
            'model_pred': y_pred_cnn[i],
            'true_label': y_test_int[i],
            'final_label': final_label,
            'confidence': confidence[i],
            'override': was_override
        })

override_df = pd.DataFrame(override_log)

# ------------------------------------------------------------
# CELL 15: HITL PERFORMANCE COMPARISON
# ------------------------------------------------------------
auto_only_acc = accuracy_score(y_test_int[high_conf_mask],
                                y_pred_cnn[high_conf_mask])
model_all_acc = accuracy_score(y_test_int, y_pred_cnn)
hitl_acc = accuracy_score(y_test_int, final_pred)

print("=" * 60)
print("PERFORMANCE: MODEL-ONLY vs HUMAN-IN-THE-LOOP")
print("=" * 60)
print(f"Model-only accuracy (all):        {model_all_acc:.4f}")
print(f"Model-only accuracy (high-conf):  {auto_only_acc:.4f}")
print(f"HITL hybrid accuracy (all):       {hitl_acc:.4f}")
print(f"Improvement from HITL:            "
      f"+{(hitl_acc - model_all_acc)*100:.2f} percentage points")

if len(override_df) > 0:
    print(f"\nOverrides made by human: {override_df['override'].sum()}")
    print(f"Override rate (of low-conf): "
          f"{override_df['override'].mean():.1%}")

# ------------------------------------------------------------
# CELL 16: VISUALIZE HITL; CONFIDENCE DISTRIBUTION
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Confidence histogram
axes[0].hist(confidence, bins=30, color='steelblue', edgecolor='black')
axes[0].axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--',
                linewidth=2, label=f'Threshold = {CONFIDENCE_THRESHOLD}')
axes[0].set_xlabel('Model Confidence'); axes[0].set_ylabel('Count')
axes[0].set_title('Prediction Confidence Distribution', fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Accuracy vs threshold
thresholds = np.linspace(0.5, 0.99, 20)
auto_fracs, hitl_accs = [], []
for t in thresholds:
    mask = confidence >= t
    if mask.sum() > 0:
        auto_fracs.append(mask.mean())
        # Hybrid: model for high-conf, perfect human for rest
        hybrid = y_pred_cnn.copy()
        hybrid[~mask] = y_test_int[~mask]
        hitl_accs.append(accuracy_score(y_test_int, hybrid))
    else:
        auto_fracs.append(0); hitl_accs.append(1.0)

ax2 = axes[1]
ax2.plot(thresholds, hitl_accs, 'o-', color='green', linewidth=2,
         label='HITL accuracy')
ax2.plot(thresholds, auto_fracs, 's--', color='orange', linewidth=2,
         label='Fraction auto-decided')
ax2.set_xlabel('Confidence Threshold')
ax2.set_ylabel('Value')
ax2.set_title('Trade-off: Accuracy vs Automation Rate', fontweight='bold')
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# CELL 17: OVERRIDE FEEDBACK INTO TRAINING DATA POOL
# ------------------------------------------------------------
"""
HCAI Feedback Loop:
Overrides become NEW labeled training samples (with higher trust weight).
This simulates how a deployed HITL system improves over time.
"""

print("=" * 60)
print("OVERRIDE FEEDBACK LOOP")
print("=" * 60)

if len(override_df) > 0:
    # Collect overridden samples
    override_indices = override_df[override_df['override']]['index'].values
    X_override = X_test[override_indices]
    y_override = override_df[override_df['override']]['final_label'].values
    y_override_onehot = utils.to_categorical(y_override, 10)

    print(f"Overridden samples collected: {len(X_override)}")
    print(f"Class distribution of overrides:")
    print(pd.Series(y_override).value_counts().sort_index())

    # Augment training set with overrides
    X_train_aug = np.concatenate([X_train, X_override], axis=0)
    y_train_aug = np.concatenate([y_train, y_override_onehot], axis=0)

    print(f"\nOriginal training size: {len(X_train)}")
    print(f"Augmented training size: {len(X_train_aug)}")

    # Retrain CNN on augmented data
    print("\nRetraining CNN with human-verified overrides...")
    cnn_retrained = build_cnn()
    cnn_retrained.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    cnn_retrained.fit(
        X_train_aug, y_train_aug,
        validation_split=0.1,
        epochs=10, batch_size=64,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3,
                                  restore_best_weights=True)],
        verbose=0
    )

    _, retrained_acc = cnn_retrained.evaluate(X_test, y_test, verbose=0)
    print(f"\nOriginal CNN accuracy:  {cnn_acc:.4f}")
    print(f"Retrained CNN accuracy: {retrained_acc:.4f}")
    print(f"Delta: {(retrained_acc - cnn_acc)*100:+.2f} pp")
else:
    print("No overrides occurred — model was perfect on low-conf samples.")

# ------------------------------------------------------------
# CELL 18: HCAI ETHICS; BIAS AUDIT OF OVERRIDES
# ------------------------------------------------------------
"""
Critical HCAI question: Are overrides uniformly distributed across classes?
If the human reviewer overrides digit '1' frequently but never '8',
the retrained model will be skewed. This is a fairness check.
"""

if len(override_df) > 0:
    overrides_by_class = (override_df[override_df['override']]
                          .groupby('true_label').size()
                          .reindex(range(10), fill_value=0))

    plt.figure(figsize=(10, 4))
    plt.bar(overrides_by_class.index, overrides_by_class.values,
            color='coral', edgecolor='black')
    plt.xlabel('True Digit Class')
    plt.ylabel('Number of Overrides')
    plt.title('Override Distribution by Class — Bias Audit',
              fontweight='bold')
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nBias Audit: Overrides per class")
    print(overrides_by_class)
    print(f"\nMax/Min ratio: "
          f"{overrides_by_class.max() / max(overrides_by_class.min(), 1):.2f}")
    print("If ratio >> 1, the reviewer may be biased toward certain digits.")

# ------------------------------------------------------------
# CELL 19: SAVE MODELS & OVERRIDE LOG
# ------------------------------------------------------------
mlp.save('mnist_mlp.h5')
cnn.save('mnist_cnn.h5')
if len(override_df) > 0:
    override_df.to_csv('override_log.csv', index=False)
print("✓ Models and override log saved.")

# ------------------------------------------------------------
# CELL 20: FINAL SUMMARY
# ------------------------------------------------------------
print("=" * 60)
print("FINAL SUMMARY — WEEK 2 HCAI")
print("=" * 60)
print(f"Dataset: UCI MNIST (subsampled to {SAMPLES_PER_CLASS}/class)")
print(f"Total samples used: {len(df_sub)}")
print(f"\nMLP test accuracy:                {mlp_acc:.4f}")
print(f"CNN test accuracy:                {cnn_acc:.4f}")
print(f"HITL hybrid accuracy:             {hitl_acc:.4f}")
print(f"\nConfidence threshold:             {CONFIDENCE_THRESHOLD}")
print(f"HITL review rate:                 {low_conf_mask.mean():.1%}")
if len(override_df) > 0:
    print(f"Overrides:                        {override_df['override'].sum()}")
    print(f"Feedback loop:                    Retrained CNN")
print("=" * 60)
print("Key HCAI Lesson: Human oversight is not a cost — it's an")
print("investment in accuracy, trust, and long-term model improvement.")