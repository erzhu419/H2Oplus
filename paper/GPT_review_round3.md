# H2Oplus Round 3 审稿意见

## 总体建议

**Major Revision for transportation journals / Minor-to-Major Revision for an empirical RL benchmark venue.**

这一版相比 Round 2 又有实质进步。标题已经降为 **Real-Topology SUMO Benchmark**，不再暗示 calibrated digital twin；Introduction 和 Conclusion 也明确承认：H2O+ 在 default scenario 中略高于 ep39，但这个排序在 3×3 stress grid 中不成立。Table V 现在加入了 ep39、多种 H2O+、1000-epoch pure-online SAC、zero-hold、BC、retuned Daganzo，并补了 paired statistics、Pareto plot 和 per-scenario heatmap。这些修改让论文从“强推 H2O+ 优于 operator”转向“系统诊断 H2O+ 在 transit holding 中的适用边界”，可信度明显提高。

但当前版本仍不适合直接投交通类顶刊。核心原因是：**论文现在诚实地展示了 H2O+ 并没有稳定超过 ep39，也没有在乘客侧指标上占优；同时 Table V 的方法数量、evaluation count、paired statistics 和 operational-metric 叙述存在若干内部不一致。** 如果这些不修，审稿人会认为论文主张已经被实验结果削弱，且结果表的 accounting 不够可靠。

## 已经修好的关键问题

1. **标题和定位更稳。**  
   标题从 real corridor 降成 real-topology SUMO benchmark，Discussion 也明确说不是 AVL/APC-calibrated digital twin。这比上一版更适合当前证据。

2. **H2O+ vs ep39 的 claim 降调了。**  
   新版明确写：default scenario 中 `-658K` vs `-666K` 的轻微优势不外推到 3×3 stress grid；multi-scenario 下 ep39、H2O+ 和 1000-epoch SAC 是 near-tie。

3. **Table V 补了上一轮要求的关键行。**  
   现在有 ep39、SIM-online `Contrastive only`、SUMO-online oracle、1000-epoch pure-online SAC、BC、zero-hold、Daganzo α=0.3/0.4/0.6。

4. **pure-online SAC 叙事更诚实。**  
   文中承认 catastrophic failure 只适用于 200-epoch parity budget；1000-epoch SAC 是 credible RL baseline。

5. **Algorithm 1 已改成 generic loop。**  
   不再把 Contrastive + Q-floor 固定成唯一算法，这解决了上一轮“Algorithm 与推荐 recipe 不一致”的问题。

6. **Data-efficiency caveat 加上了。**  
   Table IV 现在说明该 sweep 用 `Contrastive + Q-floor` 是 historical reason，而 deployment-best 是 `Contrastive only`。

7. **运营目标冲突被显式呈现。**  
   Pareto plot 和 reward-vs-passenger discussion 是正确方向，说明作者没有继续把 shaped reward 等同于运营价值。

## 主要问题

### 1. 当前主结论已经变弱，需要重新包装论文贡献

新版结果显示：

- Table V 中 ep39 reward 最好：`-1634 ± 253K`；
- SUMO-online H2O+ oracle 是 `-1682 ± 269K`；
- deployment-relevant SIM-online H2O+ 是 `-1724 ± 159K`；
- 1000-epoch pure-online SAC 是 `-1739 ± 152K`；
- H2O+ 不稳定超过 ep39；
- pure-online SAC 在某些 passenger-side 指标上很有竞争力。

这意味着论文不能再以 “H2O+ preferable to competitors under operator constraints” 作为强主张。更准确的主线应该是：

> H2O+ can recover most of a strong operator/reference policy under a multi-fidelity simulator, reliably beats unlearned baselines, and reveals fidelity-dependent component choices; however, it does not consistently outperform ep39 or dominate passenger-side operational metrics.

建议把 Abstract 的 “whether H2O+ is preferable to its competitors” 改成更弱的 “when H2O+ is competitive, where it fails to dominate, and which components matter”。这会让论文更像一篇诚实的 empirical evaluation，而不是被结果反噬的 method paper。

### 2. Table V 的 accounting 有明显错误

当前文字写：

- “Ten methods are compared”
- “totalling 90 evaluations”

但 Table V 实际有 **11 行**：

1. ep39
2. H2O+ TransDisc only
3. H2O+ TransDisc + Q-floor
4. Pure-online SAC 1000 ep
5. H2O+ Contrastive only SIM-online
6. Pure-online SAC 200 ep
7. Zero-hold
8. BC
9. Daganzo α=0.3
10. Daganzo α=0.4
11. Daganzo α=0.6

而按表中 `n` 计算，evaluation count 也不是 90。若 9 个普通单-checkpoint方法各 9 scenarios，再加 1000-epoch SAC 的 27 和 SIM-online Contrastive 的 24，总数至少是 132，具体还取决于哪些行共享 training seeds / checkpoints。

这是当前最需要马上修的复现问题。建议新增一个小 inventory table 或一句明确 accounting：

- 每一行有几个 policy checkpoint；
- 每个 checkpoint eval 几个 scenarios；
- 是否对 training seeds 先取均值再和 scenario 配对；
- 为什么 SIM-online Contrastive 是 `n=24` 而不是 `3 × 9 = 27`；
- Table V 的 SE 是按 scenario、checkpoint-scenario、还是先 seed-aggregate 后 scenario 计算。

### 3. Paired statistics 仍有不一致和解释问题

Table VI 有两个明显问题：

1. H2O+ vs ep39：`95% CI [-77, -12]` 不包含 0，但 Wilcoxon `p=0.055`。文本称 “borderline-significant in the wrong direction”。如果 bootstrap CI 是主统计，它已经支持 ep39 优于 H2O+；如果 Wilcoxon 是主统计，则 CI 的解释需要降级。现在两者并列但结论不清。

2. H2O+ TransDisc + Q-floor：文本说 “indistinguishable from H2O+ TransDisc-only”，但 Table VI 给 `Δ=+16`, CI `[+5,+27]`, `p=0.027`。这不是 indistinguishable，而是小但统计上显著的差异。

此外，Table VI 的 reference 是 **H2O+ TransDisc-only SUMO-online oracle**，不是 deployment-relevant SIM-online policy。交通/部署语境下更应该报告：

- SIM-online Contrastive vs ep39；
- SIM-online Contrastive vs 1000-epoch pure-online SAC；
- SIM-online Contrastive vs zero-hold / BC / Daganzo；
- SUMO-online oracle vs SIM-online deployment gap。

否则 Table VI 的显著性主要证明 oracle H2O+ 的优势，而不是 operators 实际可用的 low-fidelity H2O+。

### 4. Table V 混合不同 `n` 和不同训练种子，统计层级不清

Table V 中：

- ep39、zero-hold、BC、Daganzo 多数是 `n=9`；
- pure-online 1000 ep 是 `n=27`；
- SIM-online Contrastive 是 `n=24`；
- BC 是 1 training seed；
- pure-online 200 ep 是 legacy single training seed；
- SUMO-online H2O+ oracle 似乎是单 checkpoint 或 4-seed selection 后的某个 checkpoint。

这些行直接用 mean ± SE 放在一起，会让读者误以为不确定性口径一致。实际应该区分：

- training seed uncertainty；
- scenario uncertainty；
- checkpoint-selection uncertainty；
- deterministic policy under scenario variation。

建议用 hierarchical bootstrap 或至少先按 training seed 聚合，再做 scenario-paired difference。若计算量有限，表注必须说明当前 SE 是 descriptive，不是统一 inferential uncertainty。

### 5. Passenger-side metric 叙述存在内部矛盾

Table V 中：

- lowest passenger wait 是 **Pure-online SAC 200 ep legacy**：`447.1s`；
- pure-online SAC 1000 ep 是 `505.0s`，并不是最低；
- best HW CV 是 ep39：`0.627`，不是 pure-online SAC 200 ep 的 `0.628`，虽然差距极小。

但 Fig. 5 caption 写：

> pure-online SAC (1,000 epoch retrain) has the lowest per-leg waiting

正文也有 “pure-online SAC has ... lowest headway CV” 的表述。这和 Table V 不一致。建议统一为：

- 200-epoch legacy SAC has the lowest waiting but poor reward / single seed；
- ep39 has the best HW CV by a tiny margin;
- 1000-epoch SAC is reward-competitive but no longer best on passenger wait；
- passenger-wait finding should be treated as a warning signal, not a robust superiority claim.

### 6. 交通运营结论仍然没有闭合

作者现在正确承认 reward 与 passenger utility 不完全一致，但这带来一个更深问题：**如果 passenger waiting / headway CV 是更运营相关的指标，H2O+ 的优势到底是什么？**

目前最稳的说法是：

- H2O+ 优化 shaped reward 能接近 ep39；
- H2O+ 显著好于 unlearned baselines；
- H2O+ 不是 passenger-wait 最优；
- ep39 仍是最强综合 reference；
- pure-online SAC 在 passenger-wait 目标下不能忽略。

若目标是交通顶刊，需要进一步补：

- total passenger waiting time；
- in-vehicle delay；
- total passenger travel time；
- number of completed trips / boarded passengers；
- holding time by line；
- line-level passenger wait distribution；
- reward-vs-passenger metric correlation；
- passenger-weighted reward 或 reward re-weighting sensitivity。

否则论文只能作为 RL benchmark / component-diagnosis paper，而不是“公交运营改善”论文。

### 7. deployment-relevant H2O+ 的位置仍不够突出

Round 2 的一个问题是 Table V 没有 SIM-online best；这一版补了，但它被放在 Table V 第五行，且主 paired stats 仍围绕 SUMO-online oracle。由于论文的 operator constraint 是只能在 cheap simulator 中做 Stage 2，真正主方法应是：

> H2O+ Contrastive only (SIM-online deployment)

建议把 Table V 排序或分组改成：

- Operator / references；
- Deployment-relevant learned policies；
- Same-fidelity oracle diagnostics；
- Unlearned baselines。

这样能避免读者把 SUMO-online oracle 误认为主方法。当前 “H2O+ TransDisc only” 这一行在 Table V 中太显眼，而它并不符合 deployment setting。

### 8. ep39 的身份仍需要更清楚

文中有几种说法：

- operator reference policy；
- ep39 deterministic SAC checkpoint included in `Doff`；
- operator-tuned heuristic reference；
- reference policy from prior deployment/experiment。

如果 ep39 是一个 deterministic SAC checkpoint，而不是真实 operator rule 或真实上线策略，就不应称为 operator reference policy。建议统一为以下之一：

- “best prior SUMO-trained reference checkpoint ep39”；
- “operator-inspired reference controller”；
- “reference policy included in the offline buffer”。

交通审稿人会非常在意 “operator policy” 是否真实来自公交公司运行规则。

### 9. “Data-efficiency 3.4×” 的主张仍偏强

Table IV 的 H2O+ column 用的是 `Contrastive + Q-floor`，而当前 deployment-best 是 `Contrastive only`。表注说 ratio robust to recipe choice，但正文没有给证据。另一个问题是 data-efficiency 的评估仍在 default/SUMO eval setting，而 multi-scenario stress grid 已经证明 default ranking 不稳定。

建议：

- 把 data-efficiency claim 降为 default-scenario finding；
- 或在 3×3 stress grid 上验证 200K H2O+ 是否仍接近 full-data pure offline；
- 至少提供 recipe-choice sensitivity 的数字。

### 10. 仍缺少标准 RL baselines

当前论文已经对交通 baselines 补了不少，但如果论文定位是 H2O+ 与 offline-to-online RL 的经验评估，仍缺：

- RLPD；
- WSRL；
- AWAC；
- IQL；
- TD3+BC；
- BC-on-ep39-only。

如果不补，建议删弱 “competitors” 这类宽泛说法，只说 “the baselines tested here”。否则审稿人会认为 H2O+ 没有和最近 offline-to-online 方法充分比较。

## 次要问题

1. Abstract 仍写 “whether H2O+ is preferable to its competitors”，与现在 near-tie / not consistently surpass ep39 的结果不完全匹配。

2. Table VI 中 “H2O+ DARC + Q-floor” 与 Table V 行名 “H2O+ TransDisc + Q-floor” 命名不一致。

3. Figure 4 仍写 “Best H2O+ beats ep39 by 3%”，但 Conclusion 说这个 default single-scenario ordering 不外推。图内标注应加 “single default scenario only”。

4. Figure 6 caption 说 “relative ranking is preserved cell by cell”，但正文又说 top ranking shifts with OD。应改成 “bottom ranking is preserved; top ranking shifts”。

5. Method/Problem 部分仍有一些 “real corridor / calibrated boarding/alighting” 表述，和 Discussion 的 “not calibration claim” 稍有张力。

6. `\newcommand{\tbd}` 仍在 `main.tex`，虽然不影响 PDF，但提交前最好清理。

7. LaTeX log 没有 undefined refs/citations，但有 hyperref PDF string warnings；小问题，提交前可以清掉。

## 建议的最低修复清单

1. **修正 Table V accounting。**  
   统一 method count、evaluation count、`n=24` 原因、SE 口径，并加一个 evaluation inventory。

2. **重做 Table VI 或补一张 deployment-focused statistics table。**  
   以 SIM-online Contrastive 为主方法，报告 vs ep39、1000-epoch SAC、zero-hold、BC、Daganzo 的 paired differences。

3. **修正统计解释冲突。**  
   CI 不含 0 但 p=0.055、p=0.027 却说 indistinguishable，这些必须统一。

4. **修正 passenger-metric 文本和图注。**  
   明确到底是 200-epoch SAC 还是 1000-epoch SAC 拿到最低 waiting；不要说 pure-online lowest HW CV，除非表格一致。

5. **进一步降调主张。**  
   从 “H2O+ preferable to competitors” 改成 “H2O+ competitive with ep39 and strong online SAC on shaped reward, while exposing reward-operational trade-offs”。

6. **把 deployment-relevant SIM-online 放到主叙事中心。**  
   SUMO-online oracle 应作为 diagnostic，不应成为 Table VI 主 reference。

7. **统一 ep39 身份。**  
   如果它是 deterministic SAC checkpoint，就不要叫 operator heuristic / operator reference policy，除非有真实运营依据。

8. **补 passenger-level evidence 或明确论文不是运营改进 claim。**  
   若不重跑 reward，需要把 passenger utility 的结论写成 limitation，而不是 deployment recommendation。

9. **收缩 data-efficiency claim。**  
   写成 default-scenario data-efficiency，或补多场景验证。

10. **清理图注和 residual wording。**  
    特别是 Fig. 4、Fig. 5、Fig. 6、Related Work 中 sim-to-real wording、`competitors` wording。

## 结论

Round 3 已经比 Round 2 更成熟，尤其是作者愿意把 “H2O+ does not consistently beat ep39” 放进摘要和结论，这一点很重要。当前稿件最有价值的贡献不是“提出一个比 operator 更好的控制器”，而是：

> 在公交 holding 的 multi-fidelity offline-to-online setting 中，H2O+ 可以接近强 reference policy，显著优于 unlearned baselines，并揭示 discriminator / Q-floor / reward-shape 的适用边界。

如果按这个定位，论文有潜力。但现在 Table V/VI 的 accounting 和统计解释必须先修；交通运营指标也必须进一步降调或补实验证据。我的建议仍是 **Major Revision**，但已经从 “结构性问题很多” 变成 “主线可保留，需修统计、命名和运营解释”。
