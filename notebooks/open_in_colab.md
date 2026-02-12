# Opening Notebooks in Google Colab

## Quick Links (update after pushing to GitHub)

- **Experiment Runner**:
  `https://colab.research.google.com/github/YOUR_USERNAME/Capstone_FraudBench/blob/master/notebooks/colab_runner.ipynb`

- **Debug Notebook**:
  `https://colab.research.google.com/github/YOUR_USERNAME/Capstone_FraudBench/blob/master/notebooks/colab_debug.ipynb`

## Manual Steps

1. Go to [Google Colab](https://colab.research.google.com)
2. File > Open notebook > GitHub tab
3. Enter your repo URL
4. Select the notebook to open

## Google Drive Setup (One-Time)

Upload your datasets to Google Drive with this structure:

```
MyDrive/
└── FraudBench/
    ├── data/
    │   ├── CCFD/
    │   │   └── creditcard.csv
    │   ├── ieee-fraud-detection/
    │   │   ├── train_transaction.csv
    │   │   └── train_identity.csv
    │   ├── LCLD/
    │   │   └── loan.csv
    │   └── Sparkov/
    │       ├── fraudTrain.csv
    │       └── fraudTest.csv
    ├── results/       (auto-created by notebook)
    ├── models/        (auto-created by notebook)
    └── logs/          (auto-created by notebook)
```

Use `scripts/prepare_drive_data.py` to create this structure locally, then upload:

```bash
python scripts/prepare_drive_data.py --data-dir ./datasets --output-dir ./drive_upload
```
