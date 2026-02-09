# LCLD Dataset Card (Lending Club Loan Default)

## Overview
- **Name:** lcld
- **Source:** [Lending Club open data](https://www.lendingclub.com/info/download-data.action)
- **Domain:** Credit risk / loan default prediction

## Dataset Statistics
- **Raw samples:** ~2,260,668 loans
- **After filtering:** ~1,340,973 (Current loans dropped — outcome unknown)
- **Features:** 63 (after column removal)
- **Classes:** 2 (binary classification)
- **Default rate:** ~19.6% (Charged Off + Default)

## Label Meaning
- **0:** Good loan (Fully Paid, Late, In Grace Period)
- **1:** Default (Charged Off, Default, Does not meet credit policy: Charged Off)

Loans with status `Current` are excluded because the outcome is not yet known.

## Feature Description
| Group | Examples | Type | Count |
|-------|----------|------|-------|
| Loan terms | loan_amnt, funded_amnt, int_rate, installment, term, grade | Numeric/Categorical | ~8 |
| Borrower profile | annual_inc, emp_length, home_ownership, verification_status | Numeric/Categorical | ~5 |
| Credit history | delinq_2yrs, inq_last_6mths, open_acc, revol_bal, total_acc | Numeric | ~40+ |
| Application | purpose, addr_state, application_type | Categorical | ~3 |

## Known Issues / Leakage Risks
- **Post-origination columns removed:** total_pymnt, recoveries, out_prncp, last_pymnt_*, settlement_*, hardship_* — these are known only after loan outcome
- **High missing columns dropped:** 43 columns with >50% missing removed
- **`Current` loans excluded:** These have unknown outcomes and would introduce label noise
- **Grade/sub_grade correlation:** These are Lending Club's own risk assessments and strongly predict default; consider excluding for harder benchmarks

## Preprocessing Notes
- 11 categorical features require encoding (OneHotEncoder)
- ~52 numeric features — StandardScaler recommended
- Some numeric features have low missing rates (<15%) — imputation handled by preprocessor
- `term` stored as string (" 36 months", " 60 months") — treated as categorical

## Split Strategy
- **Recommended:** Stratified random split (maintains class ratio)
- **Train/Val/Test:** 60% / 20% / 20%

## Citation
```
LendingClub. Lending Club Loan Data. https://www.lendingclub.com/
```
