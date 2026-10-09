# Model Card: German Credit Scoring with XAI

**Version:** 1.0  
**Date:** 2026-09-08  
**Author:** S.B. Abdullahi  
**Contact:** sba_elt.bch@outlook.com 
**License:** MIT  
**Compliance:** EU AI Act Annex III(5)(b) — High-Risk Credit Scoring

---

## 1. Model Details

### 1.1 Basic Information

| Field | Value |
|-------|-------|
| **Model type** | Logistic Regression (primary), MLP (comparison) |
| **Framework** | scikit-learn, TensorFlow/Keras |
| **Task** | Binary classification — Good/Bad credit |
| **Input** | 19 features (German Credit dataset) |
| **Output** | P(Good Credit) ∈ [0, 1] |
| **Training date** | 2026-09-09 |

### 1.2 Intended Use

**Primary use:** Teaching HCAI principles (Transparency & Explainability).

**Secondary use:** Reference implementation for XAI on tabular data.

**❌ Out-of-scope uses:**
- Real-world credit scoring decisions
- Any high-stakes financial application
- Discriminatory profiling

### 1.3 Model Architecture

**Logistic Regression (primary):**
- Linear model: `P(Y=1|X) = σ(β₀ + Σ βᵢXᵢ)`
- Interpretable coefficients
- Input: 19 standardized features

**MLP (comparison):**
- 3 layers: 19 → 64 → 32 → 1
- Activation: ReLU, Sigmoid (output)
- Dropout: 0.3
- Parameters: ~3,400

---

## 2. Intended Purpose

### 2.1 Primary Function

Classify loan applicants as "Good" or "Bad" credit based on 19 features from the German Credit dataset.

### 2.2 Human Oversight

- **HITL integration:** Gray-zone routing for low-confidence predictions
- **Override mechanism:** Human loan officer has final decision
- **Explanation layer:** SHAP/LIME for every decision

---

## 3. Factors

### 3.1 Relevant Factors

| Factor | Type | Source |
|--------|------|--------|
| **Age** | Protected | Demographics |
| **Sex** | Protected (via `personal_status_sex`) | Demographics |
| **Foreign worker status** | Protected | Demographics |
| **Marital status** | Potential proxy | Demographics |

### 3.2 Evaluation Factors

Model evaluated on:
- Overall accuracy
- Fairness metrics (DP, EO, DI)
- Explanation stability
- GDPR/AI Act compliance

---

## 4. Metrics

### 4.1 Model Performance

| Metric | Logistic Regression | MLP |
|--------|---------------------|-----|
| **Accuracy** | 76.7% | 76.3% |
| **Training samples** | 700 | 700 |
| **Test samples** | 300 | 300 |

### 4.2 Fairness Metrics

| Attribute | DP Diff | EO Diff | DI Ratio |
|-----------|---------|---------|----------|
| **Sex** | 0.126 | 0.185 | 0.845 |
| **Age** | 0.160 | 0.133 | 0.799 ⚠ |

**Interpretation:** Age DI ratio = 0.799 is slightly below the EEOC 0.8 threshold.

### 4.3 Explanation Metrics

| Metric | Value |
|--------|-------|
| **SHAP stability** | ✓ All top features stable |
| **SHAP–LIME agreement** | 1/5 (⚠ low) |
| **Explanation fairness** | Male 0.0228 vs Female 0.0231 ✅ |

---

## 5. Evaluation Data

### 5.1 Dataset

- **Name:** German Credit (Statlog)
- **Source:** UCI ML Repository (ID 144)
- **DOI:** https://doi.org/10.24432/C5NC77
- **Size:** 1,000 samples, 20 features
- **Class balance:** 700 Good / 300 Bad
- **Collection period:** 1970s–1990s Germany

### 5.2 Train/Test Split

- **Training:** 700 (70%)
- **Test:** 300 (30%)
- **Stratification:** By target class

### 5.3 Known Limitations

- Historical dataset (reflects 1990s lending practices)
- Small sample size (limited generalization)
- Contains historical bias (see Fairness section)

---

## 6. Quantitative Analyses

### 6.1 Performance by Group

| Group | Sample size | Approval rate |
|-------|-------------|---------------|
| **Male** | 690 | 72.3% |
| **Female** | 310 | 64.8% |
| **Age ≥ 25** | 851 | 71.9% |
| **Age < 25** | 149 | 59.1% |

### 6.2 Explanation Coverage

- ✅ SHAP explanations for every decision
- ✅ LIME cross-validation available
- ✅ Counterfactual recourse via DiCE
- ✅ Stability tested under input perturbation

---

## 7. Ethical Considerations

### 7.1 Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Historical bias** | High | Documented; fairness audit provided |
| **SHAP–LIME disagreement** | Medium | Both methods provided; users should cross-validate |
| **Small dataset** | Medium | Documented; not for production |
| **Proxy discrimination** | Medium | Protected attributes audited |

### 7.2 Legal Compliance

| Regulation | Article | Status |
|-----------|---------|--------|
| **GDPR** | Art. 13 (Transparency) | ✅ Model card |
| **GDPR** | Art. 15 (Right of access) | ✅ SHAP explanations |
| **GDPR** | Art. 22 (Automated decisions) | ✅ HITL override |
| **GDPR** | Art. 22(3) (Right to contest) | ✅ Counterfactuals |
| **EU AI Act** | Art. 13 (Transparency) | ✅ This document |
| **EU AI Act** | Art. 14 (Human oversight) | ✅ HITL architecture |
| **EU AI Act** | Art. 15 (Accuracy/robustness) | ✅ Metrics reported |
| **EU AI Act** | Art. 86 (Right to explanation) | ✅ Local SHAP |

### 7.3 Environmental Impact

- **Training time:** < 2 minutes on CPU
- **Carbon footprint:** Negligible (< 0.01 kg CO₂)
- **Model size:** < 1 MB

---

## 8. Caveats and Recommendations

### 8.1 Known Caveats

1. **Not production-ready:** This is a teaching demonstration.
2. **Historical bias:** Dataset reflects past discriminatory lending practices.
3. **Explanation fragility:** SHAP and LIME disagree on 80% of top features.
4. **No external validation:** Model tested on a single split.

### 8.2 Recommendations for Use

**For teaching:**
- ✅ Use as a reference implementation
- ✅ Adapt for your own courses
- ✅ Cite appropriately

**For real-world deployment:**
- ❌ Do NOT deploy as-is
- ⚠️ Requires: retraining on modern data, extensive fairness audit, regulatory review
- ⚠️ Must include: continuous monitoring, override logging, bias detection

---

## 9. AI Use Disclosure

This model card was drafted with assistance from **Claude 4 Sonnet** (Anthropic, 2026) and **DeepSeek V-4.1-Flash** (DeepSeek, 2026). All content was reviewed and validated by the human author.

**Citation:**
> Anthropic. (2026). *Claude 4 Sonnet* [Large language model]. https://claude.ai
> DeepSeek. (2026). *DeepSeek V-4.1-Flash* [Large language model]. https://www.deepseek.com
> Abdullahi, S.B. (2026). Age-stratified machine learning using de-identified clinical and transcriptomic data for pediatric appendicitis,
  International Journal of Medical Informatics, Volume 220, 1 November 2026, 106632.
> Abdullahi, S.B. & Mahbub, T. (2026). A blockchain-verified explainable AI pipeline for multi-scale agri-food carbon accounting,
  Computers & Industrial Engineering, Volume 221, 2026.


---

## 10. Citation

If you use this model or its documentation, please cite:

```bibtex
@misc{hcai_credit_model_2026,
  title={hcai-transparency-explainability},
  author={S.B. Abdullahi},
  year={2026},
  howpublished={\url{https://github.com/Sbabdullahi/hcai-transparency-explainability}},
  note={HCAI: compliance and responsible innovation teaching materials}
}