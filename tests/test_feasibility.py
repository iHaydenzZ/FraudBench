"""Tests for the dataset-aware feasibility evaluator (spec §1.6).

Fixtures are synthetic processed/raw frames built by hand (the real datasets are
not present locally), exercising the OHE-block folding, binding-constraint
selection, and per-row Protocol-B mask.
"""

import pandas as pd

from constraints.feasibility import (
    evaluate_feasibility,
    get_constraint_names,
    normalize_dataset,
)


def _g1_installment(loan, rate, term):
    r = rate / 12.0 / 100.0
    factor = (1.0 + r) ** term
    return loan * r * factor / (factor - 1.0)


class TestNormalisation:
    def test_accepts_registry_and_loader_casing(self):
        assert normalize_dataset("IEEE-CIS") == "ieee_cis"
        assert normalize_dataset("ieee_cis") == "ieee_cis"
        assert normalize_dataset("LCLD") == "lcld"

    def test_constraint_names_per_dataset(self):
        assert get_constraint_names("CCFD") == []
        assert get_constraint_names("LCLD")[0] == "g1"
        assert "s_state_ohe" in get_constraint_names("Sparkov")


class TestCCFDNegativeControl:
    def test_no_constraints_means_everything_feasible(self):
        X_proc = pd.DataFrame({"V1": [0.1, -0.2, 0.3], "Amount": [1.0, 2.0, 3.0]})
        res = evaluate_feasibility("CCFD", X_proc)
        assert res.aggregate_feasibility == 1.0
        assert res.main_failed_constraint == "none"
        assert res.feasible_row_mask.all()
        assert res.per_constraint == {}


class TestLCLD:
    def _frames(self):
        # 3 rows: row0 fully feasible, row1 breaks g1 (wrong instalment),
        # row2 breaks g3 (bankruptcies > pub_rec).
        loan, rate, term = 10000.0, 12.0, 36.0
        good_inst = _g1_installment(loan, rate, term)
        X_raw = pd.DataFrame(
            {
                "loan_amnt": [loan, loan, loan],
                "int_rate": [rate, rate, rate],
                "term": [term, term, term],
                "installment": [good_inst, good_inst + 50.0, good_inst],
                "open_acc": [3.0, 3.0, 3.0],
                "total_acc": [5.0, 5.0, 5.0],
                "pub_rec": [1.0, 1.0, 0.0],
                "pub_rec_bankruptcies": [0.0, 0.0, 2.0],
            }
        )
        # term OHE valid one-hot for all rows (36 months selected).
        X_proc = pd.DataFrame({"term_36": [1.0, 1.0, 1.0], "term_60": [0.0, 0.0, 0.0]})
        return X_proc, X_raw

    def test_per_constraint_and_aggregate(self):
        X_proc, X_raw = self._frames()
        res = evaluate_feasibility("LCLD", X_proc, X_raw=X_raw)
        assert res.per_constraint["g1"] == 2 / 3  # row1 fails
        assert res.per_constraint["g2"] == 1.0
        assert res.per_constraint["g3"] == 2 / 3  # row2 fails
        assert res.per_constraint["g4_term_ohe"] == 1.0
        # Only row0 passes the full conjunction.
        assert res.aggregate_feasibility == 1 / 3
        assert list(res.feasible_row_mask) == [True, False, False]

    def test_binding_is_lowest_pass_rate(self):
        X_proc, X_raw = self._frames()
        res = evaluate_feasibility("LCLD", X_proc, X_raw=X_raw)
        # g1 and g3 tie at 2/3; registry order resolves the tie to g1.
        assert res.main_failed_constraint == "g1"


class TestSparkovOHEFolding:
    def test_state_ohe_is_binding_and_folded(self):
        # state block invalid for 2/3 rows (sums != 1) => s_state_ohe = 1/3,
        # the lowest pass rate => binding. All other constraints pass.
        X_proc = pd.DataFrame(
            {
                "state_CA": [1.0, 0.4, 0.6],
                "state_NY": [0.0, 0.4, 0.6],
                "category_a": [1.0, 1.0, 1.0],
                "category_b": [0.0, 0.0, 0.0],
                "gender_M": [1.0, 0.0, 1.0],
                "gender_F": [0.0, 1.0, 0.0],
            }
        )
        X_raw = pd.DataFrame(
            {
                "amt": [10.0, 20.0, 30.0],
                "city_pop": [1000.0, 2000.0, 3000.0],
                "merch_lat": [40.0, 41.0, 42.0],
                "merch_long": [-100.0, -101.0, -102.0],
            }
        )
        res = evaluate_feasibility("sparkov", X_proc, X_raw=X_raw)
        assert res.per_constraint["s_state_ohe"] == 1 / 3
        assert res.per_constraint["s_category_ohe"] == 1.0
        assert res.per_constraint["s_amt_positive"] == 1.0
        assert res.main_failed_constraint == "s_state_ohe"
        # Aggregate conjunction folds the OHE block: only row0 has a valid state.
        assert res.aggregate_feasibility == 1 / 3

    def test_merch_bbox_and_positivity_break_aggregate(self):
        X_proc = pd.DataFrame({"state_CA": [1.0], "state_NY": [0.0]})
        X_raw = pd.DataFrame({"amt": [-5.0], "city_pop": [1000.0], "merch_lat": [40.0], "merch_long": [-100.0]})
        res = evaluate_feasibility("Sparkov", X_proc, X_raw=X_raw)
        assert res.per_constraint["s_amt_positive"] == 0.0
        assert res.aggregate_feasibility == 0.0


class TestIEEE:
    def test_ohe_and_sign_constraints(self):
        X_proc = pd.DataFrame(
            {
                "ProductCD_W": [1.0, 0.5],  # row1 invalid one-hot
                "ProductCD_C": [0.0, 0.5],
                "card4_visa": [1.0, 1.0],
                "card4_mc": [0.0, 0.0],
                "card6_debit": [1.0, 1.0],
                "card6_credit": [0.0, 0.0],
            }
        )
        X_raw = pd.DataFrame({"TransactionAmt": [50.0, 50.0], "C1": [0.0, 0.0], "D1": [1.0, 1.0]})
        res = evaluate_feasibility("IEEE-CIS", X_proc, X_raw=X_raw)
        assert res.per_constraint["i_product_ohe"] == 0.5
        assert res.per_constraint["i_card4_ohe"] == 1.0
        assert res.per_constraint["i_amt_positive"] == 1.0
        assert res.main_failed_constraint == "i_product_ohe"
        assert list(res.feasible_row_mask) == [True, False]


class TestMissingColumnsArePermissive:
    def test_absent_constraint_columns_do_not_lower_feasibility(self):
        # No state/category/gender OHE and no raw numeric cols present: every
        # constraint is vacuously satisfied.
        X_proc = pd.DataFrame({"unrelated": [0.0, 1.0]})
        res = evaluate_feasibility("sparkov", X_proc, X_raw=pd.DataFrame(index=X_proc.index))
        assert res.aggregate_feasibility == 1.0
        assert res.main_failed_constraint == "none"
