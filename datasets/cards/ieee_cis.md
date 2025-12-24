# IEEE-CIS Dataset Card (Fraud Detection)

## Overview
- **Name:** ieee_cis
- **Source:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **Domain:** E-commerce fraud detection

## Dataset Statistics
- **Samples:** 590,540 transactions (train set)
- **Features:** 392 (transaction only) or 434 (with identity)
- **Classes:** 2 (binary classification)
- **Fraud rate:** 3.50% (20,663 frauds)

## Label Meaning
- **0:** Legitimate transaction
- **1:** Fraudulent transaction

## Feature Groups
| Group | Count | Description |
|-------|-------|-------------|
| TransactionDT | 1 | Timedelta from reference datetime |
| TransactionAmt | 1 | Transaction amount (USD) |
| ProductCD | 1 | Product code (categorical) |
| card1-card6 | 6 | Card information |
| addr1, addr2 | 2 | Address information |
| dist1, dist2 | 2 | Distance features |
| P_emaildomain, R_emaildomain | 2 | Email domains |
| C1-C14 | 14 | Counting features |
| D1-D15 | 15 | Timedelta features |
| M1-M9 | 9 | Match features (categorical) |
| V1-V339 | 339 | Vesta engineered features |

## Known Issues / Leakage Risks
- **High missing rates:** Many V-features have 50%+ missing values
- **TransactionDT:** Could leak temporal information
- **Card features:** Some may identify users across transactions
- **Memory intensive:** Full dataset requires significant RAM

## Preprocessing Notes
- Handle missing values (median/mode imputation recommended)
- Encode categorical features (ProductCD, card4, card6, M1-M9, email domains)
- Scale numeric features
- Consider dropping high-missing columns (>90% missing)

## Split Strategy
- **Recommended:** Stratified random split
- **Alternative:** Time-based split using TransactionDT
- **Train/Val/Test:** 60% / 20% / 20%

## Evaluation Considerations
- Use PR-AUC as primary metric
- Competition used ROC-AUC but PR-AUC is more informative for imbalanced data
- Consider cost-sensitive evaluation (fraud has higher cost than false alarm)

## Memory/Performance Notes
- Full dataset: ~2GB RAM for transaction data alone
- Use `sample_frac` parameter for faster iteration
- Consider feature selection to reduce dimensionality

## Citation
```
IEEE Computational Intelligence Society (IEEE-CIS) and Vesta Corporation.
IEEE-CIS Fraud Detection Competition, Kaggle, 2019.
```
