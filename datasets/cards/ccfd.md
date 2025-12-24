# CCFD Dataset Card (Credit Card Fraud Detection)

## Overview
- **Name:** ccfd
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Domain:** Financial fraud detection

## Dataset Statistics
- **Samples:** 284,807 transactions
- **Features:** 30 (excluding target)
- **Classes:** 2 (binary classification)
- **Fraud rate:** 0.17% (492 frauds out of 284,807)

## Label Meaning
- **0:** Legitimate transaction
- **1:** Fraudulent transaction

## Feature Description
| Feature | Type | Description |
|---------|------|-------------|
| Time | Numeric | Seconds elapsed between this transaction and the first transaction |
| V1-V28 | Numeric | PCA-transformed features (anonymized for confidentiality) |
| Amount | Numeric | Transaction amount |

## Known Issues / Leakage Risks
- **Time feature:** Could introduce temporal leakage if not handled properly
- **PCA features:** Already anonymized, no feature engineering possible
- **Extreme imbalance:** 0.17% positive class requires careful evaluation metrics

## Preprocessing Notes
- V1-V28 are already scaled (PCA output)
- `Amount` should be scaled separately
- `Time` can be normalized or dropped
- No missing values

## Split Strategy
- **Recommended:** Stratified random split (maintains class ratio)
- **Alternative:** Time-based split (use Time feature for temporal ordering)
- **Train/Val/Test:** 60% / 20% / 20%

## Evaluation Considerations
- Use PR-AUC as primary metric (handles imbalance better than ROC-AUC)
- Report fraud recall/precision specifically
- Accuracy is misleading due to extreme imbalance (99.83% baseline)

## Citation
```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification.
IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.
```
