"""Tests for the C2 attacker-capability masks (spec §1.6)."""

from constraints.masks import (
    IEEE_CIS_MUTABLE_RAW,
    LCLD_IMMUTABLE_RAW,
    SPARKOV_IMMUTABLE_RAW,
    build_processed_mutable_mask,
    build_processed_mutable_mask_inverted,
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


class TestBuildProcessedMutableMaskInverted:
    def test_only_allowlisted_features_are_mutable(self):
        names = ["TransactionAmt", "addr1", "D1", "C1", "V14"]
        mask = build_processed_mutable_mask_inverted(names, IEEE_CIS_MUTABLE_RAW)
        # Allow-list members mutable; everything else (D/C/V blocks) frozen.
        assert list(mask) == [True, True, False, False, False]

    def test_ohe_prefix_in_allowlist_is_mutable(self):
        names = ["ProductCD_W", "ProductCD_C", "card4_visa"]
        mask = build_processed_mutable_mask_inverted(names, IEEE_CIS_MUTABLE_RAW)
        # ProductCD is allow-listed -> its OHE columns mutable; card4 is not.
        assert list(mask) == [True, True, False]


class TestGetMutableMask:
    def test_lcld_dispatch(self):
        names = ["loan_amnt", "int_rate"]
        assert list(get_mutable_mask("LCLD", names)) == [True, False]

    def test_ieee_freezes_all_but_transaction_facing_fields(self):
        # The wide immutable set is the spec §4.4 / thesis Table 12 mask: only
        # TransactionAmt/ProductCD/addr/dist are mutable; V*/C*/D*/card* frozen.
        names = ["TransactionAmt", "ProductCD_W", "addr1", "dist1", "D1", "C1", "V14", "card4_visa"]
        mask = get_mutable_mask("IEEE-CIS", names)
        assert list(mask) == [True, True, True, True, False, False, False, False]

    def test_sparkov_freezes_geography_and_identity(self):
        names = ["amt", "state_CA", "merch_lat", "gender_M", "category_food"]
        mask = get_mutable_mask("Sparkov", names)
        # amt + category mutable; state/merch_lat/gender frozen.
        assert list(mask) == [True, False, False, False, True]
        assert "state" in SPARKOV_IMMUTABLE_RAW

    def test_ccfd_all_mutable(self):
        names = ["V1", "V2", "Amount"]
        assert get_mutable_mask("CCFD", names).all()
