# FraudBench: Thesis + ICAIF 投稿双阶段计划

> **起草日期**: 2026-05-06
> **项目**: FraudBench — Domain-specific adversarial robustness benchmark for financial fraud detection
> **目标会议**: ICAIF 2026 (Milan, Nov 14–17, 2026)
> **学校课程**: ELEC5021 Capstone Project B, University of Sydney

---

## 1. 关键时间线 (Key Timeline)

| Milestone | 日期 | 距 2026-05-06 | 性质 |
|---|---|---|---|
| 学校 thesis draft | 2026-05-15 | 9 天 | Internal — supervisor review |
| 学校 thesis final | 2026-05-29 | 23 天 | Internal — submission |
| ICAIF 2026 投稿 | 2026-08-02 | 88 天 | **External — primary deadline** |

### 1.1 Submission constraints

**学校 thesis (ELEC5021 Capstone Project B):**
- Draft (2026-05-15): supervisor review on *structure, technical content, presentation* — **不要求实验结果完整**
- Final (2026-05-29): ≥ 40 normally-spaced pages excluding appendices
- 接受 negative results, 接受 long limitations sections

**ICAIF 2026 paper:**
- 8 pages, ACM 2-column format
- **No supplementary materials / appendices** accepted
- Submission via Microsoft CMT system

### 1.2 双阶段策略

Thesis (5.29) 与 ICAIF paper (8.02) 是 **两份不同文档**,不是 "压缩-扩展" 关系:

| 维度 | Thesis | ICAIF paper |
|---|---|---|
| 体量 | ≥ 40 页 | 8 页双栏 |
| 角色 | 实验仓库 (experimental repository) | Focused single-claim narrative |
| 内容 | 所有 dataset / ablation / negative results | 1 main table + 1 ablation + 1 transfer figure |
| 写作方式 | 厚, 详细 | 薄, punchy |

**Phase 2 不从 thesis 删减, 而是从空白文档重写.**

---

## 2. FA-AT 三层目标 (Tiered Specification)

### Tier 0 — Minimum Viable (by 5.15 draft)

**目的**: 让 supervisor review *方法成立性*, 不是 *结果显著性*.

**Required:**
- [ ] FA-AT method 章节: per-feature ε mapping 公式 + cost-sensitive loss 形式 + algorithm pseudocode
- [ ] Mutability 分类表 (mutability classification table): fully mutable / semi-mutable / immutable, 每个 dataset 一份
- [ ] Ablation 设计 (ablation plan): 列出要做的 ablations + motivation, **不要求结果**
- [ ] LCLD 单 seed sanity check: 代码不崩 + loss 下降

**Not required:**
- 不需要 effect size, multi-seed, 完整 ablation 结果

**论文写法**: "Preliminary results pending; full multi-seed evaluation in Section X."

### Tier 1 — Thesis-grade (by 5.29 final)

- [ ] 3 datasets: LCLD + Sparkov + IEEE-CIS (CCFD 因 PCA 用作 negative case)
- [ ] 3 seeds + Wilcoxon signed-rank test + Cohen's d
- [ ] vs. standard AT 在 robust PR-AUC 上的 head-to-head
- [ ] 2 ablations:
  - (i) 去掉 cost weighting (保留 per-feature ε)
  - (ii) 去掉 per-feature ε (保留 cost weighting)
- [ ] Constraint feasibility 报告: FA-AT 生成对抗样本的 g1 / g4 satisfaction rate
- [ ] Cross-attack transfer 章节 (thesis 里是 main result, 不是 appendix)
- [ ] §D Degenerate model audit 章节

### Tier 2 — ICAIF-grade (by 8.02)

- [ ] 4 datasets, 5 seeds
- [ ] **Compact** 2×2 ablation table: per-feature ε on/off × cost weighting on/off
- [ ] Cross-attack robustness figure (1 张)
- [ ] vs. Foe for Fraud (Aug 2025) differentiation paragraph
- [ ] Robust PR-AUC + constraint-aware feasibility 双指标
- [ ] **砍掉**: 长 ablation, supplementary, 详细 dataset cards, §D degenerate audit (thesis 已包含)

---

## 3. 88 天分阶段计划

### Phase 1A: Draft Sprint (5.06 → 5.15, 9 天)

**核心原则**: 写为主, 不是跑为主. Structural completeness > experimental completeness.

#### 5.06 – 5.09 (4 天)

**写作:**
- [ ] Paper outline + contribution list (半天, **最先做**)
- [ ] Sec 1 Introduction
- [ ] Sec 3 Benchmark Design
- [ ] Sec 6.1 FA-AT method spec (公式 + algorithm + mutability table)
- [ ] OHE-validity formal definition + Sec 5 structural floor 概念框架

**实验:**
- [ ] §A' IEEE-CIS sensitivity sweep (~20 min compute)
- [ ] HopSkipJump 剩余 runs 收尾

#### 5.10 – 5.14 (5 天)

**写作:**
- [ ] Sec 2 Related Work (含 Foe for Fraud differentiation 计划)
- [ ] Sec 4 Evaluation Protocol
- [ ] Sec 5 (含已有 TabularBench comparison + feasibility audit 结果)
- [ ] Sec 7 Limitations + Future Work (初稿)

**实验:**
- [ ] OHE projection repair 实现 (FA-AT baseline 之一)
- [ ] FA-AT LCLD 单 seed 跑通 (Tier 0 sanity check)

#### 5.15

- [ ] **提交 draft thesis**

---

### Phase 1B: Thesis 填充 (5.16 → 5.29, 14 天)

**核心原则**: 跑为主, 写填充.

#### 5.16 – 5.22 (一周, 实验为主)

- [ ] FA-AT LCLD multi-seed (3 seeds)
- [ ] FA-AT Sparkov dataset adaptation + multi-seed
- [ ] FA-AT IEEE-CIS multi-seed
- [ ] Ablation runs:
  - Cost weighting on/off
  - Per-feature ε on/off
- [ ] Cross-attack transfer (CAPGD-trained 模型 vs. Square / HSJ attacks)
- [ ] Cross-dataset OHE failure rate 表
- [ ] §D Degenerate model audit
- [ ] Dataset cards (4 份, formal style)

> **5.22 mid-point checkpoint** — 见 §5 Plan B trigger.

#### 5.23 – 5.29 (一周, 写作为主)

- [ ] Sec 6 完整结果章节 (FA-AT 三 dataset + ablation + cross-attack)
- [ ] Sec 5 完整 (structural floor + cross-dataset OHE 表)
- [ ] Sec 7 Limitations 修订 (基于实际结果)
- [ ] Abstract + executive summary
- [ ] 通读 + 内部 review

#### 5.29

- [ ] **提交 thesis final**

---

### Phase 2: ICAIF Compression + Extension (5.30 → 8.02, 65 天)

**核心原则**: 重写, 不是删减.

#### Month 1 (June)

- [ ] FA-AT 扩展到 IEEE-CIS + CCFD (Tier 2 multi-dataset)
- [ ] 5 seeds 完整重跑
- [ ] Cross-attack transfer 5-seed 版本
- [ ] vs. Foe for Fraud 详细 differentiation 实验

#### Month 2 (July)

- [ ] 重写 ICAIF 8 页双栏论文 (**从空白文档开始, 不要 copy-paste from thesis**)
- [ ] Reproducibility package + GitHub repo 公开
- [ ] Quickstart notebook
- [ ] Pip-installable package
- [ ] Static leaderboard scaffold

#### 7.27 – 8.02 (最后一周)

- [ ] 冻结结果
- [ ] Co-author review (Dr. Chen + Yitian)
- [ ] Camera-ready 检查
- [ ] **提交**

---

## 4. 实验任务排序

### 4.1 必做 (Critical Path)

| ID | Task | 阶段 | Effort |
|---|---|---|---|
| E1 | §A' IEEE-CIS sensitivity sweep | Phase 1A | ~20 min |
| E2 | HopSkipJump 收尾 | Phase 1A | small |
| E3 | OHE projection repair 实现 | Phase 1A | medium |
| E4 | FA-AT 实现 + LCLD sanity check | Phase 1A | medium |
| E5 | OHE-only marginal FSR quantification | Phase 1A/1B | small |
| E6 | Cross-dataset OHE failure rate 表 | Phase 1B | small |
| E7 | FA-AT multi-dataset multi-seed (Tier 1) | Phase 1B | **large** |
| E8 | FA-AT ablation (2 ablations) | Phase 1B | medium |
| E9 | Cross-attack transfer | Phase 1B | medium |
| E10 | §D Degenerate model audit | Phase 1B | small |
| E11 | FA-AT Tier 2 expansion (4 dataset, 5 seeds) | Phase 2 | **large** |

### 4.2 砍掉 (Cut)

| Task | Reason |
|---|---|
| §C Sparkov OHE projection (作为独立实验) | Subsumed by cross-dataset OHE 表 |
| BAF integration | 移交 Yitian, 不在 critical path |
| §B CCFD variance 重跑 | 用 caveat 段落处理即可 |

### 4.3 写作任务

| ID | Task | 阶段 |
|---|---|---|
| W1 | Paper outline + contribution list | Phase 1A day 1 |
| W2 | OHE-validity formal definition | Phase 1A |
| W3 | Sec 1 Introduction | Phase 1A |
| W4 | Sec 2 Related Work | Phase 1A |
| W5 | Sec 3 Benchmark Design | Phase 1A |
| W6 | Sec 4 Protocol | Phase 1A |
| W7 | Sec 5 Existing results (TabularBench / feasibility) | Phase 1A |
| W8 | Sec 6.1 FA-AT method spec | Phase 1A |
| W9 | Sec 6.2 FA-AT results | Phase 1B |
| W10 | Sec 7 Limitations | Phase 1A 初稿 + 1B 修订 |
| W11 | Dataset cards (4 份) | Phase 1B |
| W12 | Abstract | Phase 1B 末 |
| W13 | ICAIF 8 页重写 | Phase 2 Month 2 |

### 4.4 基础设施 (Infrastructure)

| ID | Task | 阶段 |
|---|---|---|
| I1 | Reproducibility package (uv lock + configs) | Phase 1B + Phase 2 |
| I2 | Automated figure generation | Phase 1B |
| I3 | Quickstart notebook | Phase 2 Month 2 |
| I4 | Pip-installable package | Phase 2 Month 2 |
| I5 | Static leaderboard scaffold | Phase 2 Month 2 |
| I6 | Extensibility guide | Phase 2 Month 2 |

---

## 5. Plan B Trigger (5.22 Checkpoint)

5.22 是 Phase 1B mid-point. 基于此时的 FA-AT vs. standard AT 对比结果决定 Phase 2 framing:

| 观测 | 行动 | ICAIF framing |
|---|---|---|
| Cohen's d ≥ 0.5 在 ≥ 2 datasets | 继续 Tier 1 + Tier 2 路径 | **A** — FA-AT primary |
| Cohen's d 在 0.2 – 0.5 之间 | Tier 1 写完 thesis, Phase 2 加大 dataset 体量 | A 但弱化 |
| Cohen's d < 0.2 在所有 datasets | Thesis 诚实报告 negative finding | **B** — OHE structural floor primary, FA-AT secondary |

**重点**: Thesis 接受 negative result, 5.22 即使 FA-AT 不显著也不影响 5.29 提交. Plan B 只影响 ICAIF framing.

---

## 6. 风险点 (Risks)

1. **FA-AT 不显著** — 5.22 checkpoint trigger Plan B (见 §5)
2. **8 页 + no appendix 约束** — Phase 2 不能直接压缩 thesis, 必须重写
3. **Foe for Fraud (Aug 2025) differentiation** — Per-feature ε 是 *defense-time* 创新, transferability 是 *attack-time* 分析, 两者方向不同, 但 reviewer 不会自动看出, 需明示
4. **Cross-attack transfer 阴性结果** — 如果 FA-AT 对 CAPGD 鲁棒但对 HSJ 不鲁棒, 工业适用性受质疑, 需在 Limitations 诚实讨论
5. **OHE projection repair 在 LCLD 上 g4 verifiability 修复有限** — g1 nonlinear 仍是 dominant killer (~98% failure rate), OHE-only repair 只能在 g4 上贡献

---

## 7. Panel 视角

### Researcher (Dr. Chen, supervisor)
5.15 draft 看的是 *方法描述* 和 *ablation 计划*. 公式严谨性 + mutability 分类 motivation + cost weighting motivation 比 results 重要十倍.

### Practitioner (银行风控工程师)
Cross-attack robustness figure 是上线决策依据. FA-AT 训练成本 + inference latency + production data drift 鲁棒性写进 ICAIF 8 页对工业界引用关键.

### Skeptical Reviewer (ICAIF PC)
8 页论文不能像 thesis 压缩. Main table 必须显示 FA-AT 在 ≥ 3/4 datasets 上 *统计显著* 优于 standard AT, 否则评 "marginal and inconsistent gains".

### End-user (FraudBench 使用者)
Thesis 给方法 + dataset 细节; Paper 给 "为什么用 FraudBench 而非 TabularBench". 两份文档目标人群不同, 必须分开写.

---

## 8. 下一步行动 (Next Actions)

按推荐顺序:

1. **今天 / 明天**: Paper outline + contribution list (thesis + ICAIF 共用骨架, 半天)
2. **2–3 天内**: FA-AT method spec (公式 + algorithm pseudocode + mutability 分类表)
3. **5.10 前**: OHE projection repair 实现 + LCLD sanity check
4. **5.15**: 提交 draft thesis

### 待确认事项

- [ ] Thesis 是否包含独立 "Background / Literature Review" 章节 (capstone 通常要求, 影响 outline 前几章结构)
- [ ] Thesis 是否需要包含 "Project Management / Timeline" 章节 (部分 capstone 要求)
- [ ] 与 Yitian 的 BAF 协同 timeline (是否在 ICAIF 投稿前完成 integration)

---

*文档版本*: v1.0 (2026-05-06)
*下次更新*: 5.15 draft 提交后
