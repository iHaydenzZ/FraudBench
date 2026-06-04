Decisions: HopSkipJump dropped from ICDM scope (future work); all results regenerated fresh.

Completed:
- NB1 CAPGD protocol grid: MLP x 4 datasets x {none,AT,input_validation} x 3 seeds,
  protocols A/B/C1/C2 (CCFD A/B only), eps sweep on no-defence; folded-OHE aggregate.
  (294 rows, single weight hash per same_model_group.)
- NB2 Square model-family: 4 datasets x 3 models x {A,B}, adversarial examples saved.
  (96 rows incl. 24 protocol=not_applicable Protocol-C placeholders.)
- NB3 consolidation: master registry (390 rows), coverage + summary,
  Kendall-Tau (strong + free), PR-AUC vs ROC-AUC, thesis-consistency cross-check,
  golden anchors (0 WARN), figures.

Findings to confirm in writing:
- Same-model C2 vs between-model thesis Tables 11-12: IEEE-CIS robust acc
  0.886 (anchor 0.883 - matches); LCLD robust acc 0.171
  (anchor 0.153 - small same-model shift, recorded not force-fitted).
- Sparkov folded aggregate: binding constraint s_state_ohe (expected s_state_ohe);
  aggregate 0.000009 (~0.0001 as expected).
- CCFD robust PR-AUC reconciled to: 0.598
  +/- 0.262 (vs thesis 0.581+/-0.102;
  the ICDM draft's +/-0.225 matches neither registry and is superseded).
  NB2 note: CCFD MLP clean PR-AUC 0.683 vs anchor 0.633 - within
  CCFD's known seed variance.
- Strong B-vs-C Kendall-Tau distances per dataset:
  IEEE-CIS: B-vs-C1 weighted-KT distance = 0.0  (adversarial_training > none > input_validation  ->  adversarial_training > none > input_validation)
  IEEE-CIS: B-vs-C2 weighted-KT distance = 0.0  (adversarial_training > none > input_validation  ->  adversarial_training > none > input_validation)
  LCLD: B-vs-C1 weighted-KT distance = 0.0  (adversarial_training > none > input_validation  ->  adversarial_training > none > input_validation)
  LCLD: B-vs-C2 weighted-KT distance = 0.0  (adversarial_training > none > input_validation  ->  adversarial_training > none > input_validation)
  Sparkov: B-vs-C1 weighted-KT distance = 0.0  (adversarial_training > none > input_validation  ->  adversarial_training > none > input_validation)
  Sparkov: B-vs-C2 weighted-KT distance = 0.0  (adversarial_training > none > input_validation  ->  adversarial_training > none > input_validation)

Deferred (future work):
- HopSkipJump (6 cached rows exist, unused); black-box in-attack Protocol C for
  tree/ensemble (Cartella-style); CAA/MOEVA head-to-head; FA-AT; CTGAN;
  CCFD extra seeds; M2 ordinal.
