"""Tests for the C2 attacker-capability masks (spec §1.6)."""

from constraints.masks import (
    LCLD_IMMUTABLE_RAW,
    SPARKOV_IMMUTABLE_RAW,
    build_processed_mutable_mask,
    get_mutable_mask,
)


class TestBuildProcessedMutableMask:
    def test_numeric_immutable_frozen_mutable_free(self):
        names = ["loan_amnt", "int_rate", "dti", "annual_inc"]
        mask = build_processed_mutable_mask(names, LCLD_IMMUTABLE_RAW)
        # int_rate is immutable; loan_amnt/dti/annual_inc are mutable.
        assert list(mask) == [True, False, True, True]

    def test_ohe_columns_frozen_by_prefix(self):
        names = ["verification_status_Verified", "verification_status_Not", "purpose_car"]
        mask = build_processed_mutable_mask(names, LCLD_IMMUTABLE_RAW)
        # verification_status is immutable -> both OHE columns frozen; purpose free.
        assert list(mask) == [False, False, True]


class TestGetMutableMask:
    def test_lcld_dispatch(self):
        names = ["loan_amnt", "int_rate"]
        assert list(get_mutable_mask("LCLD", names)) == [True, False]

    def test_ieee_freezes_d_fields_only(self):
        names = ["TransactionAmt", "D1", "D15", "C1", "card4_visa"]
        mask = get_mutable_mask("IEEE-CIS", names)
        # D1/D15 frozen; everything else mutable.
        assert list(mask) == [True, False, False, True, True]

    def test_sparkov_freezes_geography_and_identity(self):
        names = ["amt", "state_CA", "merch_lat", "gender_M", "category_food"]
        mask = get_mutable_mask("Sparkov", names)
        # amt + category mutable; state/merch_lat/gender frozen.
        assert list(mask) == [True, False, False, False, True]
        assert "state" in SPARKOV_IMMUTABLE_RAW

    def test_ccfd_all_mutable(self):
        names = ["V1", "V2", "Amount"]
        assert get_mutable_mask("CCFD", names).all()
