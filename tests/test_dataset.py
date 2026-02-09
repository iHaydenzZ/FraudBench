"""Tests for dataset loading and splitting."""
import pytest
import pandas as pd
import os
from datasets.loader import load_dataset, DatasetObj, DEFAULT_DATA_ROOT
from datasets.splitter import split_dataset


class TestDatasetLoader:
    """Tests for dataset loader functionality."""

    def test_invalid_dataset_raises_error(self):
        """Test that loading an invalid dataset raises ValueError."""
        with pytest.raises(ValueError, match="not implemented"):
            load_dataset("nonexistent_dataset")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DEFAULT_DATA_ROOT, "CCFD", "creditcard.csv")),
    reason="CCFD dataset not available"
)
class TestCCFDLoader:
    """Tests for CCFD dataset loader."""

    def test_load_ccfd(self):
        """Test loading CCFD dataset."""
        dataset = load_dataset("ccfd", config={"sample_frac": 0.01})

        assert isinstance(dataset, DatasetObj)
        assert dataset.meta["name"] == "ccfd"
        assert len(dataset.feature_names) == 30
        assert "Time" in dataset.feature_names
        assert "Amount" in dataset.feature_names

    def test_ccfd_feature_types(self):
        """Test that all CCFD features are numeric."""
        dataset = load_dataset("ccfd", config={"sample_frac": 0.01})

        for ftype in dataset.feature_types.values():
            assert ftype == "numeric"

    def test_ccfd_fraud_rate(self):
        """Test that fraud rate is in expected range."""
        dataset = load_dataset("ccfd", config={"sample_frac": 0.1})

        fraud_rate = dataset.meta["fraud_rate"]
        # CCFD has ~0.17% fraud rate, allow some variance due to sampling
        assert 0.0001 < fraud_rate < 0.01


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DEFAULT_DATA_ROOT, "ieee-fraud-detection", "train_transaction.csv")),
    reason="IEEE-CIS dataset not available"
)
class TestIEEECISLoader:
    """Tests for IEEE-CIS dataset loader."""

    def test_load_ieee_cis(self):
        """Test loading IEEE-CIS dataset."""
        dataset = load_dataset("ieee_cis", config={"sample_frac": 0.01})

        assert isinstance(dataset, DatasetObj)
        assert dataset.meta["name"] == "ieee_cis"
        assert len(dataset.feature_names) > 300  # Should have 392 features

    def test_ieee_cis_has_categorical(self):
        """Test that IEEE-CIS has categorical features."""
        dataset = load_dataset("ieee_cis", config={"sample_frac": 0.01})

        cat_count = sum(1 for t in dataset.feature_types.values() if t == "categorical")
        assert cat_count > 0  # Should have some categorical features

    def test_ieee_cis_fraud_rate(self):
        """Test that fraud rate is in expected range."""
        dataset = load_dataset("ieee_cis", config={"sample_frac": 0.1})

        fraud_rate = dataset.meta["fraud_rate"]
        # IEEE-CIS has ~3.5% fraud rate
        assert 0.01 < fraud_rate < 0.1


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DEFAULT_DATA_ROOT, "LCLD", "loan.csv")),
    reason="LCLD dataset not available"
)
class TestLCLDLoader:
    """Tests for LCLD dataset loader."""

    def test_load_lcld(self):
        """Test loading LCLD dataset."""
        dataset = load_dataset("lcld", config={"sample_frac": 0.01})

        assert isinstance(dataset, DatasetObj)
        assert dataset.meta["name"] == "lcld"
        assert len(dataset.feature_names) > 50  # Should have ~63 features
        assert "loan_amnt" in dataset.feature_names
        assert "int_rate" in dataset.feature_names

    def test_lcld_feature_types(self):
        """Test that LCLD has both numeric and categorical features."""
        dataset = load_dataset("lcld", config={"sample_frac": 0.01})

        cat_count = sum(1 for t in dataset.feature_types.values() if t == "categorical")
        num_count = sum(1 for t in dataset.feature_types.values() if t == "numeric")
        assert cat_count > 0
        assert num_count > 0

    def test_lcld_fraud_rate(self):
        """Test that default rate is in expected range."""
        dataset = load_dataset("lcld", config={"sample_frac": 0.1})

        fraud_rate = dataset.meta["fraud_rate"]
        # LCLD has ~19.6% default rate
        assert 0.10 < fraud_rate < 0.30

    def test_lcld_no_leakage_columns(self):
        """Test that post-origination leakage columns are removed."""
        dataset = load_dataset("lcld", config={"sample_frac": 0.01})

        leakage = ["total_pymnt", "recoveries", "out_prncp", "last_pymnt_amnt"]
        for col in leakage:
            assert col not in dataset.feature_names


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DEFAULT_DATA_ROOT, "Sparkov", "fraudTrain.csv")),
    reason="Sparkov dataset not available"
)
class TestSparkovLoader:
    """Tests for Sparkov dataset loader."""

    def test_load_sparkov(self):
        """Test loading Sparkov dataset."""
        dataset = load_dataset("sparkov", config={"sample_frac": 0.01})

        assert isinstance(dataset, DatasetObj)
        assert dataset.meta["name"] == "sparkov"
        assert "amt" in dataset.feature_names
        assert "category" in dataset.feature_names

    def test_sparkov_feature_types(self):
        """Test that Sparkov has both numeric and categorical features."""
        dataset = load_dataset("sparkov", config={"sample_frac": 0.01})

        cat_count = sum(1 for t in dataset.feature_types.values() if t == "categorical")
        num_count = sum(1 for t in dataset.feature_types.values() if t == "numeric")
        assert cat_count > 0
        assert num_count > 0

    def test_sparkov_fraud_rate(self):
        """Test that fraud rate is in expected range."""
        dataset = load_dataset("sparkov", config={"sample_frac": 0.1})

        fraud_rate = dataset.meta["fraud_rate"]
        # Sparkov has ~0.52% fraud rate
        assert 0.001 < fraud_rate < 0.02

    def test_sparkov_no_pii_columns(self):
        """Test that PII columns are removed."""
        dataset = load_dataset("sparkov", config={"sample_frac": 0.01})

        pii = ["first", "last", "cc_num", "street", "dob", "trans_num"]
        for col in pii:
            assert col not in dataset.feature_names


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DEFAULT_DATA_ROOT, "CCFD", "creditcard.csv")),
    reason="CCFD dataset not available"
)
class TestDatasetSplitter:
    """Tests for dataset splitting functionality."""

    @pytest.fixture
    def dataset(self):
        """Load CCFD dataset with small sample for testing."""
        return load_dataset("ccfd", config={"sample_frac": 0.01})

    def test_split_sizes(self, dataset):
        """Test that splits have correct sizes."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            dataset,
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            save_indices=False
        )

        total = len(dataset.X)
        assert len(X_train) == pytest.approx(total * 0.6, rel=0.05)
        assert len(X_val) == pytest.approx(total * 0.2, rel=0.05)
        assert len(X_test) == pytest.approx(total * 0.2, rel=0.05)

    def test_split_stratification(self, dataset):
        """Test that splits maintain class balance."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            dataset,
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            save_indices=False
        )

        # Check class ratios are similar
        orig_ratio = dataset.y.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        assert abs(train_ratio - orig_ratio) < 0.01
        assert abs(test_ratio - orig_ratio) < 0.01

    def test_split_reproducibility(self, dataset):
        """Test that same seed produces same splits."""
        split1 = split_dataset(dataset, random_state=42, save_indices=False)
        split2 = split_dataset(dataset, random_state=42, save_indices=False)

        pd.testing.assert_frame_equal(split1[0], split2[0])  # X_train
        pd.testing.assert_frame_equal(split1[2], split2[2])  # X_test

    def test_split_save_load_isolation(self, dataset, tmp_path):
        """Test that split indices are saved and loaded correctly using isolated tmp_path."""
        # First split - should save
        split1 = split_dataset(
            dataset,
            random_state=123,
            save_indices=True,
            output_dir=str(tmp_path)
        )

        # Second split - should load from cache
        split2 = split_dataset(
            dataset,
            random_state=123,
            save_indices=True,
            output_dir=str(tmp_path)
        )

        pd.testing.assert_frame_equal(split1[0], split2[0])  # X_train
        pd.testing.assert_frame_equal(split1[2], split2[2])  # X_test

        # Verify file was created with correct naming
        n_samples = len(dataset.X)
        expected_file = tmp_path / f"split_indices_ccfd_n{n_samples}_seed123.json"
        assert expected_file.exists()

    def test_split_cache_invalidation_on_size_change(self, tmp_path):
        """Test that cache is invalidated when dataset size changes."""
        # Load two different sized datasets
        dataset_small = load_dataset("ccfd", config={"sample_frac": 0.005})
        dataset_larger = load_dataset("ccfd", config={"sample_frac": 0.01})

        # Split small dataset
        split_dataset(
            dataset_small,
            random_state=42,
            save_indices=True,
            output_dir=str(tmp_path)
        )

        # Split larger dataset with same seed - should create new file, not reuse
        split_dataset(
            dataset_larger,
            random_state=42,
            save_indices=True,
            output_dir=str(tmp_path)
        )

        # Both files should exist with different sizes in filename
        files = list(tmp_path.glob("split_indices_*.json"))
        assert len(files) == 2
