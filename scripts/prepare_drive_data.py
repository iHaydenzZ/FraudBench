"""
Prepare datasets for Google Drive upload.

Usage:
    python scripts/prepare_drive_data.py --data-dir ./datasets --output-dir ./drive_upload

Creates a clean directory structure that can be uploaded to Google Drive
at MyDrive/FraudBench/data/.
"""
import argparse
import os
import shutil

# Expected dataset structure (must match datasets/loader.py)
DATASET_STRUCTURE = {
    "CCFD": ["creditcard.csv"],
    "ieee-fraud-detection": ["train_transaction.csv", "train_identity.csv"],
    "LCLD": ["loan.csv"],
    "Sparkov": ["fraudTrain.csv", "fraudTest.csv"],
}


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Google Drive")
    parser.add_argument(
        "--data-dir",
        default="./datasets",
        help="Local directory containing raw datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--output-dir",
        default="./drive_upload",
        help="Output directory to upload to Drive (default: ./drive_upload)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name, expected_files in DATASET_STRUCTURE.items():
        src_dir = os.path.join(args.data_dir, dataset_name)
        dst_dir = os.path.join(args.output_dir, dataset_name)

        if not os.path.exists(src_dir):
            print(f"  SKIP  {dataset_name}/ (not found at {src_dir})")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        # Copy all CSV files from the source
        copied = 0
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if f.endswith(".csv"):
                    src_file = os.path.join(root, f)
                    dst_file = os.path.join(dst_dir, f)
                    shutil.copy2(src_file, dst_file)
                    size_mb = os.path.getsize(dst_file) / 1e6
                    print(f"  COPY  {dataset_name}/{f} ({size_mb:.1f} MB)")
                    copied += 1

        if copied == 0:
            print(f"  WARN  {dataset_name}/ — no CSV files found")

        # Check for expected files
        for ef in expected_files:
            if not os.path.exists(os.path.join(dst_dir, ef)):
                print(f"  MISS  {dataset_name}/{ef} — expected but not found")

    print(
        f"\nDone. Upload '{args.output_dir}/' to Google Drive > MyDrive > FraudBench > data"
    )


if __name__ == "__main__":
    main()
