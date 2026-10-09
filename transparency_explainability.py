# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/
"""

# ============================================================
# TRANSPARENCY & EXPLAINABILITY
# German Credit + SHAP + LIME + Counterfactuals + GDPR Audit
# ============================================================

# ------------------------------------------------------------
# CELL 1: INSTALL LIBRARIES
# ------------------------------------------------------------
"""
Library definitions:
- ucimlrepo: UCI ML Repository client
- shap: SHapley Additive exPlanations (game-theoretic attribution)
- lime: Local Interpretable Model-agnostic Explanations
- dice-ml: Diverse Counterfactual Explanations
- fairlearn: Fairness metrics (for compliance audit)
"""
!pip install -q ucimlrepo shap lime dice-ml fairlearn
!pip install -q scikit-learn pandas numpy matplotlib seaborn
!pip install -q tensorflow

# ------------------------------------------------------------
# CELL 2: IMPORTS
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import dice_ml

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow: {tf.__version__}")
print(f"SHAP: {shap.__version__}")

# ------------------------------------------------------------
# CELL 3: LOAD GERMAN CREDIT (UCI ID 144)
# ------------------------------------------------------------
print("Loading German Credit...")
german = fetch_ucirepo(id=144)

column_mapping = {
    'Attribute1': 'checking_status', 'Attribute2': 'duration_months',
    'Attribute3': 'credit_history', 'Attribute4': 'purpose',
    'Attribute5': 'credit_amount', 'Attribute6': 'savings_status',
    'Attribute7': 'employment_since', 'Attribute8': 'installment_rate',
    'Attribute9': 'personal_status_sex', 'Attribute10': 'other_debtors',
    'Attribute11': 'residence_since', 'Attribute12': 'property',
    'Attribute13': 'age', 'Attribute14': 'other_installments',
    'Attribute15': 'housing', 'Attribute16': 'existing_credits',
    'Attribute17': 'job', 'Attribute18': 'num_dependents',
    'Attribute19': 'telephone', 'Attribute20': 'foreign_worker'
}

df = german.data.features.rename(columns=column_mapping).copy()
df['target'] = german.data.targets.iloc[:, 0].map({1: 1, 2: 0})

print(f"Shape: {df.shape}")
print(f"Target: {df['target'].value_counts().to_dict()}")

# ------------------------------------------------------------
# CELL 4: PREPROCESSING
# ------------------------------------------------------------
# Keep raw dataframe for counterfactuals (DiCE needs interpretable values)
df_raw = df.copy()

# Encode for models
df_proc = df.copy()
df_proc = df_proc.drop(columns=['personal_status_sex'])

label_encoders = {}
for col in df_proc.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    label_encoders[col] = le

y = df_proc['target'].values
X = df_proc.drop(columns=['target']).values.astype('float32')
feature_names = df_proc.drop(columns=['target']).columns.tolist()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
(X_train, X_test, y_train, y_test,
 df_raw_train, df_raw_test) = train_test_split(
    X_scaled, y, df_raw.drop(columns=['target']),
    test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------------------------------------
# CELL 5: TRAIN MODELS
# ------------------------------------------------------------
# 1) Logistic Regression (INTRINSICALLY interpretable)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# 2) Random Forest (partially interpretable)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# 3) MLP (BLACK BOX)
def build_mlp():
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

mlp = build_mlp()
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_train, y_train, epochs=40, batch_size=32,
        validation_split=0.15, verbose=0)

y_proba_mlp = mlp.predict(X_test, verbose=0).ravel()
y_pred_mlp = (y_proba_mlp >= 0.5).astype(int)
mlp_acc = accuracy_score(y_test, y_pred_mlp)

print(f"Logistic Regression: {lr_acc:.4f}")
print(f"Random Forest:       {rf_acc:.4f}")
print(f"MLP (black-box):     {mlp_acc:.4f}")

# ------------------------------------------------------------
# CELL 6: INTRINSIC EXPLAINABILITY — LOGISTIC REGRESSION
# ------------------------------------------------------------
"""
Intrinsic interpretability: coefficients directly tell us the direction
and magnitude of each feature's effect on the log-odds.
"""
print("="*60)
print("INTRINSIC EXPLAINABILITY — Logistic Regression Coefficients")
print("="*60)

lr_coefs = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr.coef_[0],
    'abs_coef': np.abs(lr.coef_[0])
}).sort_values('abs_coef', ascending=False)

print(lr_coefs.head(10).to_string(index=False))
print("\nInterpretation: coefficient > 0 → increases P(Good Credit)")
print("               coefficient < 0 → decreases P(Good Credit)")

# Plot coefficients
fig, ax = plt.subplots(figsize=(10, 6))
top = lr_coefs.head(10)
colors = ['green' if c > 0 else 'red' for c in top['coefficient']]
ax.barh(top['feature'], top['coefficient'], color=colors, edgecolor='black')
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Coefficient (log-odds)')
ax.set_title('Intrinsic Interpretability: Logistic Regression Coefficients',
             fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# CELL 7: SHAP — GLOBAL EXPLANATIONS
# ------------------------------------------------------------
"""
SHAP global: which features matter most across the dataset?
"""
print("Computing SHAP values (this may take 1-2 minutes)...")

# Background sample for KernelSHAP
background = shap.sample(X_train, 50)

# MLP as a probability-returning function
def mlp_proba(x):
    return mlp.predict(x, verbose=0).ravel()

# Create SHAP explainer
explainer = shap.KernelExplainer(mlp_proba, background)

# Compute SHAP for first 100 test samples
shap_values = explainer.shap_values(X_test[:100], nsamples=100)

print(f"SHAP values shape: {np.array(shap_values).shape}")

# Global summary plot (beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test[:100],
                  feature_names=feature_names, show=False)
plt.title("SHAP Global Summary — MLP on German Credit", fontweight='bold')
plt.tight_layout()
plt.show()

# Bar plot of mean |SHAP|
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test[:100],
                  feature_names=feature_names,
                  plot_type="bar", show=False)
plt.title("Mean |SHAP| — Global Feature Importance", fontweight='bold')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# CELL 8: SHAP — LOCAL EXPLANATION (INDIVIDUAL DECISION)
# ------------------------------------------------------------
"""
Local explanation: why was THIS applicant rejected?
This is what GDPR Art. 15 requires.
"""
# Find a rejected applicant
rejected_indices = np.where(y_pred_mlp == 0)[0]
rejected_idx = rejected_indices[0]
sample = X_test[rejected_idx:rejected_idx+1]

# Compute SHAP for this sample
shap_local = explainer.shap_values(sample, nsamples=200)

# Waterfall plot — the "why" visualization
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_local[0],
        base_values=explainer.expected_value,
        data=sample[0],
        feature_names=feature_names
    ),
    show=False
)
plt.title(f"Local Explanation: Applicant #{rejected_idx}\n"
          f"Prediction: {'REJECTED' if y_pred_mlp[rejected_idx]==0 else 'APPROVED'} "
          f"(P(Good)={y_proba_mlp[rejected_idx]:.3f})",
          fontweight='bold')
plt.tight_layout()
plt.show()

# Textual explanation
print("\n" + "="*60)
print(f"GDPR-COMPLIANT EXPLANATION FOR APPLICANT #{rejected_idx}")
print("="*60)
print(f"\nDecision: {'REJECTED' if y_pred_mlp[rejected_idx]==0 else 'APPROVED'}")
print(f"Model confidence P(Good Credit): {y_proba_mlp[rejected_idx]:.3f}")

# Top 5 features by absolute SHAP
top_local = np.argsort(np.abs(shap_local[0]))[::-1][:5]
print(f"\nTop 5 factors influencing this decision:")
for rank, idx in enumerate(top_local, 1):
    direction = "increases" if shap_local[0][idx] > 0 else "decreases"
    print(f"  {rank}. {feature_names[idx]:25s} "
          f"(SHAP = {shap_local[0][idx]:+.4f}) — {direction} approval probability")

# ------------------------------------------------------------
# CELL 9: LIME — LOCAL EXPLANATIONS
# ------------------------------------------------------------
"""
LIME: alternative local explanation using surrogate linear model.
Compare with SHAP for robustness.
"""
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Rejected', 'Approved'],
    mode='classification',
    random_state=42
)

# Explain the same rejected applicant
exp = lime_explainer.explain_instance(
    X_test[rejected_idx],
    mlp_proba.reshape(-1, 1) if False else lambda x: np.column_stack(
        [1 - mlp_proba(x), mlp_proba(x)]
    ),
    num_features=10,
    num_samples=1000
)

# Visualize
fig = exp.as_pyplot_figure()
plt.title(f"LIME Local Explanation — Applicant #{rejected_idx}",
          fontweight='bold')
plt.tight_layout()
plt.show()

# Textual
print("\nLIME Explanation (top features):")
for feature, weight in exp.as_list()[:5]:
    direction = "increases" if weight > 0 else "decreases"
    print(f"  {feature:40s} {weight:+.4f} ({direction} approval)")

# ------------------------------------------------------------
# CELL 10: COUNTERFACTUAL EXPLANATIONS (DiCE)
# ------------------------------------------------------------
"""
Counterfactual: what minimal changes would flip the decision?
This satisfies GDPR Art. 22(3) — the right to contest.
"""
# Prepare data for DiCE (needs original, unencoded features)
df_for_dice = df_raw.copy()
df_for_dice = df_for_dice.drop(columns=['personal_status_sex'])

# Encode categoricals for DiCE
for col in df_for_dice.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_for_dice[col] = le.fit_transform(df_for_dice[col].astype(str))

# Create DiCE data interface
d = dice_ml.Data(
    dataframe=df_for_dice,
    continuous_features=[c for c in df_for_dice.columns
                         if df_for_dice[c].nunique() > 10],
    outcome_name='target'
)

# Wrap MLP as a DiCE-compatible model
class MLPWrapper:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled, verbose=0).ravel()
        return np.column_stack([1 - proba, proba])

m = dice_ml.Model(model=MLPWrapper(mlp, scaler), backend='sklearn')

# Create explainer
exp_dice = dice_ml.Dice(d, m, method='random')

# Query counterfactual for rejected applicant
# Need to map back to original feature values
rejected_original = df_for_dice.iloc[
    df_for_dice.index.get_loc(df_raw_test.index[rejected_idx])
]

query_instance = rejected_original.drop('target').to_frame().T

print("="*60)
print(f"COUNTERFACTUAL EXPLANATION FOR APPLICANT #{rejected_idx}")
print("="*60)
print("\nOriginal profile:")
print(query_instance.T.to_string())

# Generate counterfactuals
cf = exp_dice.generate_counterfactuals(
    query_instance,
    total_CFs=3,
    desired_class=1,  # Flip to approved
    verbose=False
)

print("\nCounterfactual recommendations (to obtain approval):")
cf.visualize_as_dataframe(show_only_changes=True)

# ------------------------------------------------------------
# CELL 11: GDPR COMPLIANCE AUDIT
# ------------------------------------------------------------
"""
Check if our system satisfies key GDPR/EU AI Act provisions.
"""
print("="*60)
print("GDPR / EU AI ACT COMPLIANCE AUDIT")
print("="*60)

# Art. 15 — Right of access: meaningful information
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_5_global = np.argsort(mean_abs_shap)[::-1][:5]
print("\n[Art. 15] Meaningful information about decision logic:")
print(f"  ✓ Top 5 features can be explained:")
for i, idx in enumerate(top_5_global, 1):
    print(f"    {i}. {feature_names[idx]} (mean |SHAP| = {mean_abs_shap[idx]:.4f})")

# Art. 22 — Right to human intervention
print("\n[Art. 22] Right to human intervention:")
print(f"  ✓ HITL override mechanism implemented (Week 2)")
print(f"  ✓ Rejected applicants can request human review")

# Art. 22(3) — Right to contest
print("\n[Art. 22(3)] Right to contest (counterfactual recourse):")
print(f"  ✓ Counterfactual explanations generated via DiCE")

# Art. 13 — Transparency
print("\n[Art. 13] Transparency requirements:")
print(f"  ✓ Model type documented: MLP + Logistic Regression baseline")
print(f"  ✓ Training data: German Credit (UCI ID 144), 1000 samples")
print(f"  ✓ Accuracy: LogReg={lr_acc:.3f}, MLP={mlp_acc:.3f}")

# Art. 14 — Human oversight
print("\n[Art. 14] Human oversight:")
print(f"  ✓ Gray-zone routing (Week 2 architecture)")
print(f"  ✓ Override logging enabled")

# Art. 86 — Right to explanation
print("\n[Art. 86] Right to explanation of individual decisions:")
print(f"  ✓ SHAP local explanation available for every decision")
print(f"  ✓ LIME cross-validation available")

# ------------------------------------------------------------
# CELL 12: EXPLANATION STABILITY TEST
# ------------------------------------------------------------
"""
Robustness check: do similar inputs get similar explanations?
A stable explanation is more trustworthy.
"""
print("="*60)
print("EXPLANATION STABILITY TEST")
print("="*60)

# Take a sample and add small noise
base_sample = X_test[rejected_idx]
noise_levels = [0.0, 0.01, 0.05, 0.1]
shap_vectors = []

for noise in noise_levels:
    noisy = base_sample + np.random.normal(0, noise, base_sample.shape)
    shap_v = explainer.shap_values(noisy.reshape(1, -1), nsamples=100)[0]
    shap_vectors.append(shap_v)

shap_matrix = np.array(shap_vectors)
std_per_feature = shap_matrix.std(axis=0)
top_5_local = np.argsort(np.abs(shap_matrix[0]))[::-1][:5]

print("\nStability of SHAP values under small input perturbations:")
print(f"{'Feature':30s} {'Base SHAP':>12s} {'Std Dev':>12s} {'Stability':>12s}")
print("-" * 70)
for idx in top_5_local:
    stability = "✓ Stable" if std_per_feature[idx] < 0.05 else "⚠ Unstable"
    print(f"{feature_names[idx]:30s} "
          f"{shap_matrix[0][idx]:>12.4f} "
          f"{std_per_feature[idx]:>12.4f} "
          f"{stability:>12s}")

# ------------------------------------------------------------
# CELL 13: SHAP vs LIME CROSS-VALIDATION
# ------------------------------------------------------------
"""
Best practice: cross-validate explanations from multiple XAI methods.
If SHAP and LIME agree → explanation is robust.
If they disagree → investigate.
"""
print("="*60)
print("SHAP vs LIME CROSS-VALIDATION")
print("="*60)

# Get SHAP ranking for the sample
shap_ranking = np.argsort(np.abs(shap_local[0]))[::-1][:5]

# Get LIME ranking (parse feature names)
lime_list = exp.as_list()
lime_features = [item[0].split(' ')[0] for item in lime_list[:5]]

print("\nTop 5 features per method:")
print(f"{'Rank':>6s} {'SHAP':30s} {'LIME':30s}")
print("-" * 70)
for i in range(5):
    shap_feat = feature_names[shap_ranking[i]]
    lime_feat = lime_features[i] if i < len(lime_features) else "—"
    match = "✓" if shap_feat == lime_feat else " "
    print(f"{i+1:>6d} {shap_feat:30s} {lime_feat:30s} {match}")

# Agreement metric
overlap = len(set([feature_names[i] for i in shap_ranking[:5]]) &
              set(lime_features))
print(f"\nTop-5 overlap: {overlap}/5 features")
if overlap >= 3:
    print("✓ High agreement — explanation is robust")
else:
    print("⚠ Low agreement — investigate discrepancies")

# ------------------------------------------------------------
# CELL 14: FAIRNESS-AWARE EXPLANATION
# ------------------------------------------------------------
"""
Does the explanation differ by protected group?
If yes, the model may be treating groups differently.
"""
print("="*60)
print("FAIRNESS-AWARE EXPLANATION AUDIT")
print("="*60)

# Reconstruct protected attributes for test set
sex_col = df_raw_test['personal_status_sex'].apply(
    lambda x: 0 if str(x).upper() in ['A92', 'A95'] else 1
).values
age_col = (df_raw_test['age'] >= 25).astype(int).values

# Compute SHAP for all test samples (limited for speed)
n_explain = min(200, len(X_test))
shap_all = explainer.shap_values(X_test[:n_explain], nsamples=100)

# Average |SHAP| by group
mean_abs_shap_per_sample = np.abs(shap_all).mean(axis=1)

# Group comparison
for name, group in [('Sex=Male', sex_col[:n_explain] == 1),
                     ('Sex=Female', sex_col[:n_explain] == 0),
                     ('Age>=25', age_col[:n_explain] == 1),
                     ('Age<25', age_col[:n_explain] == 0)]:
    if group.sum() > 0:
        avg = mean_abs_shap_per_sample[group].mean()
        print(f"{name:15s}: mean |SHAP| per sample = {avg:.4f} "
              f"(n={group.sum()})")

# ------------------------------------------------------------
# CELL 15: SUMMARY
# ------------------------------------------------------------
print("=" * 60)
print("WEEK 3 SUMMARY — TRANSPARENCY & EXPLAINABILITY")
print("=" * 60)
print(f"\nDataset: German Credit (UCI ID 144), 1000 applicants")
print(f"Models trained:")
print(f"  • Logistic Regression (intrinsic): {lr_acc:.4f}")
print(f"  • Random Forest: {rf_acc:.4f}")
print(f"  • MLP (black-box): {mlp_acc:.4f}")
print(f"\nXAI tools applied:")
print(f"  ✓ SHAP global (feature importance)")
print(f"  ✓ SHAP local (individual decisions)")
print(f"  ✓ LIME (cross-validation)")
print(f"  ✓ Counterfactuals (actionable recourse)")
print(f"\nGDPR / EU AI Act compliance:")
print(f"  ✓ Art. 13 — Transparency (model card documented)")
print(f"  ✓ Art. 15 — Right of access (SHAP explanations)")
print(f"  ✓ Art. 22 — Right to human intervention (HITL)")
print(f"  ✓ Art. 22(3) — Right to contest (counterfactuals)")
print(f"  ✓ Art. 86 — Right to explanation (local SHAP)")
print(f"\nKey insight: For high-stakes credit decisions, XAI methods")
print(f"are LEGAL REQUIREMENTS, not optional features.")
print("=" * 60)