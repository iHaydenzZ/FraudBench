from datasets.loader import load_dataset
from datasets.splitter import split_dataset

def test_dataset_flow():
    print("Testing load_dataset...")
    dataset = load_dataset("dummy_dataset")
    print(f"Loaded dataset: {dataset.meta['name']}")
    print(f"Features: {len(dataset.feature_names)}")
    print(f"X shape: {dataset.X.shape}")
    print(f"y shape: {dataset.y.shape}")
    
    print("\nTesting split_dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(dataset)
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    assert X_train.shape[0] == 600
    assert X_val.shape[0] == 200
    assert X_test.shape[0] == 200
    print("Split sizes correct!")

if __name__ == "__main__":
    test_dataset_flow()
