# H2Oplus Round 2 审稿意见

## 总体建议

**Major Revision / 当前版本仍不建议直接投交通类顶刊。**

这一版相比 Round 1 有明显进步：论文已经把贡献从“提出两个新算法组件”改成了更合理的 **empirical H2O+ recipe / component evaluation**；主表也把 **SIM-online** 明确作为 deployment-relevant setting，把 **SUMO-online** 标成 same-fidelity oracle；还补了 3×3 scenario grid、passenger waiting、headway CV、Jain fairness、Daganzo、BC，以及 action-invariance 的 data-level check。这些修改回应了上一轮最核心的几条批评。

但如果目标是 IEEE T-ITS / TR-C / TRE 这类交通类顶刊，当前版本仍有几个会卡住接收的问题。最关键的是：**多场景鲁棒性表和主表的推荐方法不一致，且乘客侧指标与 reward ranking 发生冲突**。Table V 中 pure-online SAC 的 passenger waiting time 和 headway CV 反而最好，而论文仍把 H2O+ winning 主要建立在 cumulative reward 上。对于交通领域审稿人，这会转化成一个直接问题：H2O+ 到底改善了什么运营目标？

## 已明显改善的地方

1. **主贡献定位更准确。**  
   摘要、Introduction 和 Conclusion 都明确说这不是新算法贡献，而是 H2O+ 在 transit holding 中的 empirical recipe 和 evidence。这比上一版把 Contrastive + Q-floor 包装成核心方法更稳。

2. **SIM-online 被放回主表。**  
   Table I 现在明确列出 deployment-relevant SIM-online：`Contrastive only = -658 ± 23K`，并把 `SUMO-online = -646 ± 16K` 标为 oracle / not deployable。上一轮最大的设定冲突基本解决。

3. **Q-floor claim 降低了。**  
   现在承认 Q-floor 是 mixed-effect stabilizer，不再强说它是核心贡献。这和 ablation 数字更一致。

4. **ep39 数据泄漏/公平性 caveat 写清楚了。**  
   文中明确说 ep39 transitions 是 `Doff` 的一部分，因此超过 ep39 不能理解成完全独立发现了新策略。

5. **补了多场景和运营指标。**  
   3 SUMO seeds × 3 OD scales、passenger waiting、HW CV、large-gap、Jain fairness 都是必要补充。

6. **Action-invariance 不再只靠 learned discriminator。**  
   新增了 PCA-grid / action-bin 的 data-level check，并报告 `r_hat = 0.156/0.169`。这比上一版只做 permutation diagnostic 更好。

7. **“real corridor” 的 calibration caveat 更诚实。**  
   Discussion 现在明确说这是 real topology synthetic-demand benchmark，不是 calibrated digital twin。这对避免过强 claim 很重要。

## 主要问题

### 1. Table V 的 H2O+ variant 与主表推荐方法不一致

Table I 中 deployment-relevant 最优方法是：

- `Contrastive only = -658 ± 23K`

但 Table V 的多场景鲁棒性主行变成：

- `H2O+ full (TransDisc + Q-floor) = -1681 ± 269K`
- `H2O+ DARC + Cal-QL = -1697 ± 272K`

这产生三个问题：

1. **“full” 的定义在全文不一致。** Table I 里 `full` 是 `Contrastive + Q-floor + KL`，而 Table V 里 `full` 写成 `TransDisc + Q-floor`。
2. **多场景表没有评估 Table I 的 deployment-best 方法 `Contrastive only`。** 因此 Table V 不能证明主推荐 recipe 在多场景下稳健。
3. **Table V 的 `TransDisc + Q-floor` 在 Table III/SIM-online ablation 中并不是强配置。** Table III 里类似配置 `sim_darc_calql = -694`，弱于 `sim_is = -658`。为什么它在 3×3 scenario grid 中成为主 H2O+ row，需要解释。

建议：统一命名并重跑/补报 Table V。至少应包含：

- `H2O+ SIM-online best: Contrastive only`
- `H2O+ full: Contrastive + Q-floor + KL`（如果仍称 full）
- `TransDisc + Q-floor`（作为 ablation，不要叫 full）
- zero-hold、ep39、BC、Daganzo

否则现在的 multi-scenario robustness 不能支撑主表中的 method recommendation。

### 2. 乘客侧运营指标与 reward ranking 冲突，削弱交通贡献

Table V 显示：

- H2O+ cumulative reward 最好：`-1681K`
- Pure-online SAC passenger waiting time 最好：`447.1s`
- Pure-online SAC HW CV 最好：`0.628`
- H2O+ passenger waiting time 约 `494.8s`

论文已经诚实指出 reward-shape vs operational-metric divergence，但这会让交通审稿人进一步追问：如果 passenger waiting time 和 headway CV 更接近运营价值，为什么 H2O+ 是更好的公交控制策略？

当前表述仍然把 H2O+ “wins” 建立在 cumulative reward 上，但 Discussion 又说 cumulative reward 不是 fully faithful proxy，passenger-side columns 更 operationally meaningful。两者放在一起会让主结论不稳。

建议：

1. 把 passenger waiting / total passenger delay / headway CV 作为主运营指标，而不是附属指标。
2. 给出 reward 与 passenger metrics 的相关性分析。
3. 做 Pareto comparison：reward、passenger wait、holding time、large-gap、fairness。
4. 若 H2O+ 只赢 reward 不赢 passenger metrics，应把结论改成 “H2O+ optimizes the paper's shaped reward, but pure-online SAC may be preferable under passenger-wait objective”。
5. 最好补一个 reward re-tuning 或 passenger-weighted reward 的 sensitivity，否则交通顶刊会认为 reward 设计没有对齐运营目标。

### 3. 统计推断仍不足，尤其是 Table V

Table V 报告 9 scenarios 的 mean ± standard error，但没有给 paired test、confidence interval for differences、或 hierarchical bootstrap。由于这些场景是 common random numbers 下 paired 的，最合适的证据应该是 per-scenario paired differences。

例如 H2O+ vs pure-online SAC 的差值是 `338K`，但两边 SE 分别是 `269` 和 `166`；如果不利用 paired structure，显著性并不显然。H2O+ 两个 variant 的差值只有 `16K`，论文说 statistically indistinguishable 是合理的，但也应正式报告 paired CI。

更严重的是，Table I 混合了：

- 训练 seed variance；
- single deterministic ep39；
- approximate zero-hold；
- 4-seed / 5-seed H2O+ rows；
- Table V 的 scenario-level variance。

建议统一主统计口径：

- default Table I：训练 seed-level mean ± std/SE；
- Table V：scenario-paired mean difference ± CI；
- 若有 training seed 和 scenario 两层，应做 hierarchical bootstrap；
- heuristic / zero-hold / Daganzo 也必须在同一 scenario grid 上报告；
- ep39 不能只给 single deterministic run 就和 `-658 ± 23K` 做 3%/1% 差异比较。

### 4. Pure-online SAC 的叙事变得矛盾

论文一方面说 pure-online SAC fails catastrophically：`-1654 ± 329K`，一个 seed diverges。另一方面 Table V 用 **1000 epochs** pure-online SAC，并显示它：

- passenger waiting time 最好；
- HW CV 最好；
- cumulative reward 虽差，但并非 no-operation collapse。

这会让审稿人觉得 pure-online 的失败是 reward-specific 或训练预算/seed-specific，而不是运营意义上的失败。尤其 sim-core 是 cheap simulator，1000 epochs 并不一定过分。

建议：

1. 明确 Table I 的 pure-online 是 200 epochs，Table V 是 1000 epochs，两者训练预算不同。
2. 不要再用 “fails catastrophically” 概括所有 pure-online 结果；应限定为 “fails on shaped cumulative reward at 200 epochs/default setting”。
3. 报告 1000-epoch pure-online 的 training seeds，而不是只做 scenario evaluation。
4. 如果 pure-online 在 passenger wait 上最好，应承认它是一个 strong operational baseline under passenger-wait objective。

### 5. 多场景 reward magnitudes 差异过大，需要更清楚解释

Table I 中 H2O+ 大约 `-650K`，Table V 中 H2O+ 大约 `-1680K`。论文解释说 Table V 包含 lower OD scales 和未见 seeds，且 lower OD scales 下 headway penalties dominate。

这个解释不够直观：OD scale 降低通常会减少 passenger/dwell variability，为什么 reward 反而显著变差，需要数据支撑。现在把 9 个场景直接平均，也可能掩盖某几个 seed 或 OD scale 的极端情况。

建议补：

- 3×3 per-scenario reward table 或 heatmap；
- 每个 OD scale 下的 decision count、completed trips、mean headway、per-decision reward；
- reward 是否随 OD scale 可比，是否需要按 event count 或 passenger count normalize；
- 哪些 scenario 导致 `-650K -> -1680K` 的量级变化。

否则 Table V 虽然看起来是 robustness test，但 reward scale 的可解释性不足。

### 6. Baseline 仍不足以支撑“preferable to competitors”

新版补了 BC 和 Daganzo，但标准 offline-to-online / offline RL baseline 仍然缺：

- IQL；
- AWAC；
- RLPD；
- WSRL；
- TD3+BC；
- BC on ep39-only；
- ep39 multi-scenario deterministic evaluation；
- Daganzo/Xuan-style controller with validation-tuned parameter。

文中说这些 remain outside current scope，但 Introduction/Conclusion 又说 H2O+ is preferable to competitors under deployment constraints。对于交通顶刊，这个 competitor set 仍偏窄。

尤其 Daganzo 只用 textbook `α = 0.6`，而单 seed retuned `α = 0.3/0.4` 已经接近 zero-hold。若要把 Daganzo 放进主表，应做 validation-tuned α，并在 9 scenarios 上评估 tuned controller。

### 7. Method/Algorithm 仍和推荐 recipe 不完全一致

论文现在说不提出新算法，推荐 fidelity-dependent recipe。但 Algorithm 1 仍叫：

`H2O+ with contrastive IS and bootstrap offline-Q floor`

而实验最优是：

- SIM-online：`Contrastive only`
- SUMO-online oracle：`TransDisc only`
- Q-floor：mixed / optional stabilizer

这会造成阅读混乱：Algorithm 1 看起来仍在定义一个 “full method”，但这个 full method 不是最优 recipe。

建议：

1. 把 Algorithm 1 改成 generic H2O+ training loop with selectable discriminator / optional Q-floor。
2. 不要在 algorithm title 中固定 Contrastive + Q-floor。
3. 把 “Proposed estimator” 改成 “Contrastive variant”。
4. 全文统一 `full`、`best`、`oracle`、`deployment-relevant` 的命名。

### 8. “real corridor / calibrated” 仍是交通顶刊的硬门槛

Discussion 现在承认不是 calibration claim，这是好的。但若目标是交通类顶刊，仅承认 limitation 可能还不够。当前稿件仍在标题/摘要/贡献中强调 “real corridor”，但没有提供：

- route-level travel-time MAPE；
- dwell-time calibration error；
- boarding/alighting validation；
- observed vs simulated headway distribution；
- observed vs simulated bunching / large-gap rate；
- calibration day vs test day split。

如果这些都没有，建议标题和摘要进一步降级为：

> real-topology SUMO benchmark

而不是容易被理解为 calibrated real-world digital twin 的 “real bus corridor”。

### 9. Action-invariance data-level check 有进步，但证据还偏弱

新增 `r_hat = 0.156` 是有用的，但当前验证仍有几个问题：

- PCA 5×5 / 6×6 grid 很粗，可能掩盖局部 action-dependent gaps；
- 只用 50K transitions，未报告每个 cell 的 sample count 和覆盖率；
- “conventional 0.30 supported threshold” 需要引用或解释，否则像经验阈值；
- Discussion 写 “on solid ground here” 语气偏强；
- paired sim-core rollouts under matching initial-state seeds 的构造需要更详细说明。

建议把它写成 supportive diagnostic，而不是 assumption 被充分验证。附录最好放 per-cell heatmap、coverage、bootstrap CI。

### 10. 还有一些协议和表述不一致

以下属于小但显眼的问题，建议提交前统一：

- Training protocol 写 every configuration 5 seeds，但 Table I/II 仍有 4-seed rows。
- Table V 文中说 “operator's reference policy expressed as deterministic controller (zero-hold)”；zero-hold 不是 ep39，也不是 operator reference policy。
- Baseline citation 在 PDF 中渲染成 `Nair et al. 7, Ball et al. 8, Zhou et al. 10`，格式不正常。
- Related Work 仍多次说 “sim-to-real IS estimator”，而全文主线说 deliberately avoid sim-to-real；建议统一成 multi-fidelity sim-to-sim / sim-to-sim IS correction。
- Data-efficiency table 使用 `H2O+ (Contrastive + Q-floor)`，但主表 deployment-best 是 `Contrastive only`；需要说明为什么 data-efficiency 不用 best recipe。
- Metrics 段说 operational indicators 包括 bunching/severe bunching，但 Table V 报的是 large-gap/Jain/passenger wait；两个 metrics set 需要统一。
- `Zero-hold ≈ -1600K` 仍是 approximate，应该在 default 和 3×3 scenario grid 都给正式数值。
- Main log 没看到 citation undefined，但有 hyperref PDF string warning；不是科学问题，但提交前应清掉。

## 次要建议

1. Table I 的 `-658 ± 23K` vs ep39 `-666K` 只有 8K 差距，不能说 strong improvement。建议写 “roughly matches/slightly exceeds in this single deterministic comparison”。

2. Table V 的 passenger waiting time 约 450–500s，很高。需要定义是 per-leg waiting、包含 transfer waiting 还是 station waiting，否则交通审稿人会质疑单位和合理性。

3. H2O+ 和 H2O+ DARC+CalQL 在 Table V reward 几乎一样，说明 Q-floor / DARC variant 的多场景差异很小。不要在多场景结论里过度强调 recipe。

4. 如果 pure-online SAC holds are short and rare，建议报告 holding time distribution，否则 reward-shape explanation 缺证据。

5. `Daganzo over-holds` 的结论需要 tuned α 的多场景证据；单 seed α=0.3/0.4 只能作为 hint。

6. Figure 3 diagnostics 来自 SUMO-online logs，但 deployment-relevant 结论来自 SIM-online。建议补 SIM-online diagnostics 或明确 Figure 3 只解释 oracle setting。

7. `Q-floor sufficient to prevent reward-scale drift` 这类旧语气在 Method 里仍偏强；既然实证 mixed，建议继续降调。

## 建议的最低修复清单

1. **统一 H2O+ variant 命名和推荐 recipe。**  
   明确 `Contrastive only` 是否是 deployment-relevant canonical recipe；不要把 `TransDisc + Q-floor` 也叫 full。

2. **重做/补充 Table V。**  
   在 3×3 scenario grid 中评估 Table I 的 deployment-best `Contrastive only`，并加入 ep39、tuned Daganzo、zero-hold、BC。

3. **把乘客侧指标提升为主结论。**  
   如果 H2O+ 不赢 passenger wait/HW CV，应把结论改成 reward-specific，并加入 Pareto / reward-alignment analysis。

4. **做 paired statistical tests。**  
   Table V 用 common random numbers，就报告 paired differences 和 CI；default ep39/zero-hold 也要多场景评估。

5. **澄清 pure-online SAC。**  
   区分 200-epoch failure 和 1000-epoch operational metrics；不要再笼统说 pure-online catastrophic failure。

6. **补 per-scenario breakdown。**  
   解释为什么 Table V reward 比 Table I 负很多，并报告 per-decision/passenger-normalized reward。

7. **补强交通 baseline。**  
   至少 tuned Daganzo、多场景 ep39、BC-on-ep39；若能加 RLPD/WSRL/IQL/AWAC 更好。

8. **降级 real-corridor claim 或补 calibration。**  
   没有 MAPE/RMSE/observed headway validation，就称 real-topology benchmark 更稳。

9. **把 Algorithm 1 改成 generic recipe。**  
   不要让 algorithm title 看起来仍在推一个实验上并不最优的 full method。

10. **清理表述和引用格式。**  
    特别是 `zero-hold` vs `ep39 operator reference`、4/5 seeds、`Nair et al. 7` 等。

## 结论

这一版已经从 “Weak Reject” 往前推进了一大步，论文主线更诚实、更像一篇 empirical evaluation paper。若投 applied ML / RL benchmark venue，当前版本经过一轮小到中等修改可能有机会。但若目标是交通类顶刊，仍建议 **Major Revision**。

现在最大的卡点不是 H2O+ 本身，而是交通解释：**H2O+ 赢的是 shaped cumulative reward，但 pure-online SAC 赢了 passenger waiting 和 HW CV。** 在交通审稿语境下，乘客等待和服务规律性比 RL reward 更重要。除非作者能证明 reward 与运营目标对齐，或把结论改成 reward-specific benchmark result，否则“preferable for transit operators”这条主张仍然站不稳。
