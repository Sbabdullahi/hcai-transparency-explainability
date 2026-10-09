# HCAI: Transparency & Explainability

A teaching toolkit for Human-Centered AI (HCAI) using the German Credit dataset — with SHAP, LIME, counterfactual explanations, and EU/GDPR compliance audit.

[1][Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[2][TensorFlow 2.21](https://img.shields.io/badge/tensorflow-2.21-orange.svg)](https://tensorflow.org/)
[3][License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[4][SHAP](https://img.shields.io/badge/SHAP-0.52-blueviolet.svg)](https://shap.readthedocs.io/)

---

## 💡 Concept

This repository provides **Week 3 lecture materials** for a Human-Centered AI: Compliance and Responsible Innovation course. It demonstrates how **Explainable AI (XAI)** methods — SHAP, LIME, and counterfactual explanations — can be applied to a **real-world credit scoring dataset** (German Credit, UCI ID 144) to satisfy **GDPR Article 22** and **EU AI Act** transparency requirements.

**Target audience:** Students and instructors in HCAI, XAI, and AI ethics. Also valuable for developers building **high-risk AI systems** (credit, hiring, healthcare).

---

## 🎯 Learning Objectives

By working through this notebook, you will:

1. **Define** transparency, explainability, and interpretability with precision
2. **Distinguish** intrinsic (Logistic Regression) vs. post-hoc (SHAP/LIME) explainability
3. **Apply** SHAP and LIME to individual credit decisions
4. **Generate** counterfactual recourse for rejected applicants
5. **Audit** a credit scoring pipeline against EU AI Act and GDPR requirements
6. **Evaluate** explanation stability and fairness across protected groups

---

## 📊 Key Results (From the Notebook)

| Metric | Value |
|--------|-------|
| **Logistic Regression accuracy** | 76.7% |
| **MLP (black-box) accuracy** | 76.3% |
| **SHAP–LIME top-5 overlap** | 1/5 features (⚠ low agreement) |
| **GDPR compliance** | 6/6 articles satisfied |
| **Explanation fairness (Sex)** | Male 0.0228 vs Female 0.0231 ✅ |

**Key finding:** Three models achieved similar accuracy (76–77%). Since the interpretable model is **as accurate as** the black-box, **Rudin's Rule** applies: choose the interpretable model.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Sbabdullahi/hcai-transparency-explainability.git
cd hcai-transparency-explainability

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook week3_transparency_explainability.ipynb
