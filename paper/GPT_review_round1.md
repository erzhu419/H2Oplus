# H2Oplus 审稿意见

## 总体建议

**Major Revision / Weak Reject。当前版本不建议直接投交通类顶刊。**

这篇论文研究 hybrid offline-to-online RL 在公交 holding control 中的应用：用 SUMO 生成的离线数据预训练，再在低保真 sim-core 中在线交互，最后回到 SUMO 上评估。选题有应用价值，12 条线路、389 辆公交、675K transitions 的规模也比很多单线公交控制论文更接近真实系统。论文也主动承认这不是严格 sim-to-real，而是 multi-fidelity sim-to-sim，这是加分点。

但按 IEEE T-ITS、TR-C、TRE 这类交通类顶刊标准，当前版本还有几个会直接影响接收的硬问题：主实验结果和论文设定不完全对齐，方法主线与最优实验结论相互打架，统计与评估场景太单一，交通运营指标不足，若干数字和实验协议存在内部不一致。尤其是主表最好的 `-646K` 来自 **SUMO-online** 设定，而论文的核心动机是“不能在高保真目标环境中在线训练，只能用低保真 simulator”。这会让审稿人质疑主结论是否真的支持低保真 H2O+ 部署。

## 论文贡献概括

论文的核心主张包括：

1. 在公交 holding 任务中，纯 online RL 在低保真 simulator 中训练会失败，纯 offline RL 能达到较好水平但仍弱于 operator heuristic。
2. H2O+ 结合 offline buffer 和 simulator interaction 后，可以超过 pure online、pure offline 和 ep39 heuristic。
3. 针对公交 sparse action 和 reward-scale gap，论文讨论了两个组件：Contrastive InfoNCE discriminator 和 bootstrap offline-Q floor。
4. 170-configuration ablation 显示：SUMO-online 下最强的是简单 TransitionDiscriminator without Q-floor；SIM-online 下 Contrastive discriminator 更好；Q-floor 效果混合。

## 优点

1. **问题有实际意义。**  
   公交 holding 的在线探索确实难以在真实系统中执行，offline-to-online RL 是一个合理方向。

2. **场景规模不错。**  
   12 条线路、389 辆车、675K transitions，比很多 toy network 或单线 holding 任务更有说服力。

3. **论文没有过度声称真实 sim-to-real。**  
   文中多处说明评估环境仍是 SUMO，真正 sim-to-real 需要 historical replay 和 pilot deployment，这个表述比直接声称 real-world deployment 更稳妥。

4. **ablation 覆盖面较广。**  
   同时比较 offline-only、SUMO-online、SIM-online、data composition、data size，能暴露组件是否真的有用。

5. **一些负结果有价值。**  
   例如 Q-floor 并非稳定提升、DARC-style factored DynamicsDiscriminator 在公交场景表现差、不同 online simulator 下 discriminator preference 会翻转。这些发现比单纯包装一个“新方法”更可信。

## 主要问题

### 1. 主结果使用 SUMO-online，和论文核心设定冲突

论文问题设定是：offline data 来自高保真 SUMO，online interaction 只能来自低保真 sim-core，最终在 SUMO 上评估。这个设定对应的是 SIM-online H2O+。

但主表中最重要的结论 `H2O+ (TransDisc, no Q-floor) = -646 ± 16K` 写明来自 **SUMO-online seeds**。这意味着 Stage 2 online training 使用了和 evaluation 同源的高保真 SUMO，而不是低保真 sim-core。若如此，这个结果更像是“在目标仿真器中 online fine-tuning”的 oracle upper bound，不应作为低保真 H2O+ 的主结论。

相比之下，真正符合论文部署动机的 SIM-online 最好结果是 `sim_is = -658K`，而论文提出的 full variant `sim_full = -705K` 甚至明显弱于 TransDisc 和 Contrastive-only。这会影响三条核心 claim：

1. `-646K` 是否能代表低保真 simulator H2O+。
2. H2O+ 是否真的稳定超过 ep39 heuristic。
3. Contrastive + Q-floor 是否是应推荐给 transit operator 的方法。

建议把主结果表改为以 **SIM-online** 为主，SUMO-online 作为 oracle / same-fidelity ablation。若保留 `-646K` 为 headline result，必须明确它不是低保真部署设定下的结果。

### 2. 方法主线和实验结论不一致：提出的两个组件不是最优方法

摘要和 Method 部分将 Contrastive discriminator 和 offline-Q floor 作为两个 targeted modifications 来展开，但 ablation 的主结论是：

- SUMO-online 最好的是 `TransDisc only = -646K`；
- `Contrastive + Q-floor + KL = -663K`；
- SIM-online 最好的是 `Contrastive only = -658K`；
- `Contrastive + Q-floor + KL = -705K` 是 SIM-online 中很差的一档；
- Q-floor 的效果是 mixed，有些配置还会伤害性能。

这说明当前最强实证贡献不是“提出两个改进组件”，而是“系统验证 H2O+ 在公交 holding 中哪些组件有效，并发现原 H2O+ recipe 不一定迁移”。论文需要重写贡献定位。否则审稿人会问：如果 proposed full method 不是最优，为什么要把它作为算法主线？

建议：

1. 把贡献从 “we propose two targeted modifications” 改成 “we evaluate and adapt H2O+ components for transit, finding that simple TransDisc or Contrastive-only is preferable depending on fidelity gap”。
2. 明确推荐方法：SUMO-online 是 TransDisc-only，SIM-online 是 Contrastive-only，而不是 Contrastive + Q-floor + KL。
3. Q-floor 只能作为 diagnostic/optional stabilizer，不宜作为核心贡献。

### 3. 交通类顶刊会质疑“真实走廊”证据不足

论文说是 real 12-line corridor，但目前主要证据是 SUMO network 和线路规模。交通类审稿人通常会继续追问：

- SUMO 是否用真实 AVL/APC/IC-card/GTFS 数据校准？
- travel time、dwell time、passenger arrival、boarding/alighting demand 的 validation error 是多少？
- 校准日和测试日是否分离？
- 固定 OD profile 是否只是一个 synthetic demand profile？
- 12 条线路和 389 辆车的调度、站点客流、交通流是否来自真实运营记录？

如果没有这些说明，“real bus corridor” 容易被认为只是 real topology + simulated demand，而不是可信的 real-network benchmark。

建议补一个 calibration/validation subsection，至少报告：

- route-level travel-time MAPE/RMSE；
- stop-level arrival/departure headway distribution；
- dwell-time distribution；
- passenger boarding/alighting demand fit；
- simulated vs observed bunching/headway CV；
- train/calibration/test days 的划分。

### 4. 评估只用固定 SUMO seed 和固定 OD profile，泛化不足

论文写 “all reported numbers use the same SUMO seed for reproducibility” 和 fixed OD profile。这样虽然可复现，但顶刊审稿会认为评估场景太窄。当前结果只说明方法在一个固定仿真日或一个固定 demand scenario 上有效，不足以支持 robust transit control。

尤其 `ep39 = -666K` 似乎是单次 deterministic run，没有方差；而 RL 方法报告 4 或 5 个 training seeds。这不能严格证明 `-646K` 对 heuristic 的 3% 改进显著。

建议至少增加：

- 5-10 个 evaluation seeds；
- 不同 passenger demand days；
- traffic intensity shift；
- OD perturbation；
- route-level disturbance 或 incident；
- common random numbers 下的 paired evaluation；
- heuristic baseline 也在同样 seed set 上评估。

统计上应以 seed-level 或 scenario-level mean 做 paired test，而不是只报告 training-seed variance。

### 5. 交通运营指标不足，reward improvement 不能直接等价为服务改善

论文提到 per-decision cost、headway CV、bunching rate、severe bunching rate，但主结果表几乎只报告 cumulative reward 和 per-step reward。交通类顶刊不会只接受 shaped RL return，尤其当前 reward 是 headway deviation + holding penalty 的加权和。

需要补充更直接的运营指标：

- passenger waiting time；
- passenger in-vehicle delay；
- total passenger travel time；
- excess waiting time；
- headway CV by route and direction；
- large-gap rate；
- holding time distribution；
- number of held events；
- completed trips / skipped trips / decision count；
- line-level fairness，避免 aggregate reward 掩盖个别线路恶化。

特别是 episode return 是 asynchronous event reward 的总和。不同策略可能改变固定 18,000 秒内的 event/decision count，从而影响 cumulative reward。虽然论文报告 per-step reward，但仍应报告每个方法的 decision count、completed trips 和 passenger-weighted metrics。

### 6. baseline 公平性仍不够

目前 baseline 列表中有 Zero-hold、ep39、Daganzo、offline RL、CQL、H2O+ DARC 等，但主表只展示了很少一部分，Daganzo cooperative、H2O+ default/best-of-sweep、reward rescale 等没有完整进入主结果表。

此外，offline buffer 包含 ep39 reference policy 的 155K transitions。拿学习方法与 ep39 heuristic 比较时，必须承认方法已经看过 ep39 行为数据。建议增加：

- Behavior cloning on full data；
- Behavior cloning on ep39-only data；
- IQL / TD3+BC / AWAC / RLPD / WSRL 等更标准 offline-to-online baseline；
- Daganzo 或 Xuan-style analytical holding 在同一 evaluation seed set 下的完整结果；
- pure online SAC 更长训练预算，因为 sim-core 理论上 cheap，200 epochs 可能不足以证明 online RL 失败。

如果计算预算有限，至少要把当前 baseline 的缺口写进 limitation，避免过强 claim。

### 7. 理论部分的 InfoNCE density-ratio 论证还不够严谨

Lemma 写到 InfoNCE 最优 score 满足：

`f*(s,s') = log p_real(s'|s) / p_sim(s'|s) + c(s)`

但 Bellman residual 中使用的是 absolute IS weight `exp f(s,s')`。这里的 `c(s)` 并不会在训练中自动“cancel”，因为权重会跨不同 state 混合、裁剪，并直接乘到 TD error 上。若 `c(s)` 未校准，得到的是 state-dependent scaled ratio，不一定是合法 IS correction。

此外，从 action-marginalized ratio 推到 action-conditional ratio 需要的假设比当前写法更强。仅说明 `P(d | s,a,s') = P(d | s,s')` 可能不足以保证 `p_real(s'|s,a)/p_sim(s'|s,a)` 与 `p_real(s'|s)/p_sim(s'|s)` 等价，尤其 offline 和 online 数据的 behavior action distribution 也不同。

建议：

1. 降低理论语气，把 InfoNCE 明确写成 heuristic plug-in estimator。
2. 解释 `c(s)` 如何校准、归一化或为什么不影响优化。
3. 把 assumption 改成更完整的 density-ratio condition。
4. 把 action-invariance test 写成支持性 diagnostic，而不是证明。

### 8. Action-invariance 的“数据级验证”承诺没有真正兑现

Experiments 的 Q4 和 Method 部分都说会做 data-level check，定义了 histogram density ratio `rho_hat` 和 action-dependence ratio。但正文 Analysis 只报告了 learned discriminator output under action permutation：Contrastive 0.96，TransitionDisc 0.79。论文自己也承认这不严格验证底层数据分布。

Appendix 里也没有看到真正的 histogram/binning data-level result。这个会被审稿人抓住，因为它直接支撑 Contrastive discriminator 的核心假设。

建议把 data-level check 正式补上，报告：

- action bins 如何定义；
- state-transition bins 如何构造；
- 每个 bin 的 sample count；
- action-dependence ratio 的均值和置信区间；
- 哪些状态区域 assumption fails。

如果数据不支持 assumption，就把 Contrastive 定位为 empirical variant，而不是 theoretically justified IS estimator。

### 9. Q-floor 的动机、符号和实证作用都需要澄清

论文说低保真 simulator 的 reward magnitude 只有 SUMO 的 0.2，因此 LCB target 会 drift toward simulator reward scale，Q-floor 可以防止 sim-side target 被拉低。但 reward 是负数时，“smaller magnitude”通常意味着 reward 更接近 0，不一定是“拉低”。而 `max(target, q_floor)` 对负 Q 的实际作用也需要解释清楚：它到底是在防止过低估计，还是在阻止 reward-scale mismatch 导致的过乐观？

更重要的是，实证上 Q-floor 并不稳定提升：

- SUMO-online 中 `TransDisc + Q-floor` 弱于 `TransDisc only`；
- SIM-online 中 `Contrastive + Q-floor` 弱于 `Contrastive only`；
- full method 加 KL/warmup 后更差。

因此 Q-floor 不能作为强贡献。建议：

1. 明确 reward/Q 的符号约定和 floor 激活频率；
2. 报告 Q-scale drift 曲线、floor activation rate；
3. 把 Q-floor 改为 optional stabilizer；
4. 删除“become necessary”或“sufficient to prevent drift”这类过强表述。

### 10. 实验协议和数字存在多处内部不一致

以下问题虽然看似细节，但会影响顶刊审稿人对实验可信度的判断：

- 摘要和正文说 170-configuration ablation；Computational cost 又说 full 100-configuration sweep；Appendix 说 full 48-configuration sweep。
- Experiments 写 every configuration 5 seeds；主表 H2O+ 行是 4 seeds；SUMO-online ablation 也是 4 seeds。
- Method Stage 1 写 `beta_LCB = -2`，Experiments/Appendix 写 `beta_LCB = 1.0`。如果公式是 `mu - beta sigma`，负 beta 还会变成 optimism。
- Method 写 discriminator warmup 是 5000 gradient steps / first 50 epochs；Appendix 写 warmup epochs 20 when enabled。
- Experiments 写 pairwise Wilcoxon signed-rank tests，但表格没有 p-value，也没有说明如何处理 4 vs 5 seeds、single-run heuristic。
- Abstract 最后说 “transit holding sim-to-real”，但 Introduction/Conclusion 又说 deliberately avoid sim-to-real；keywords 也包含 sim-to-real transfer。
- Baselines 列出 Daganzo cooperative、DARC default/best-of-sweep、reward rescale，但主表没有完整呈现这些结果。

建议先做一次全文数字和协议审计，把所有配置数量、seed 数、训练环境、评估环境、hyperparameters、统计检验统一。

## 次要问题

1. `Zero-hold ≈ -1600K` 和 `ep39 = -666K` 没有 variance，建议同样在多 evaluation seeds 下报告。

2. 主表中 `Pure online RL = -1654 ± 329K` 方差很大，说明失败结论可能被一个 diverged seed 主导。建议报告 median/IQR 和每个 seed 的结果。

3. `Pure offline RL with full data` 和 data-efficiency table 中 full-data pure offline 分别写 `-749 ± 26K`、`-750 ± 28K`，虽接近但最好统一。

4. “pure offline plateaus at 200K transitions” 需要更谨慎。当前 200K、400K、675K 数字接近，但应说明数据抽样方式和 source composition 是否一致，否则 plateau 可能来自采样策略。

5. `same SUMO seed for reproducibility` 不应作为主评估 protocol，只能作为 deterministic debugging protocol。

6. `Mreal` 被称为 real/evaluation environment，但实际仍是 SUMO。建议所有地方统一称 high-fidelity SUMO target simulator，避免 reviewer 误解。

7. `operator's current best-performing rule-based controller` 这个说法需要证据。如果 ep39 只是 prior SUMO experiment 中最好的 rule checkpoint，不宜称 operator current policy。

8. Appendix 里的 pipeline fixes 很有价值，但目前像工程排错记录。建议保留，但不要让它们分散主论文贡献。

9. 计算成本部分需要与实验数量对齐，否则 reproducibility 会被质疑。

10. 如果论文投交通期刊，建议减少 RL 术语堆叠，增加公交控制语境下的解释，例如 holding policy 如何影响乘客等待、headway regularity 和 route-level equity。

## 建议的最低修复清单

1. **重写主结果逻辑。**  
   SIM-online 作为主表；SUMO-online 作为 oracle/same-fidelity ablation。不要用 SUMO-online 的 `-646K` 支撑低保真 simulator deployment claim。

2. **重定位贡献。**  
   从“提出 Contrastive + Q-floor 新方法”改为“系统评估 H2O+ 在公交 holding 中的组件适用性，并给出 fidelity-dependent recipe”。

3. **补全交通运营指标。**  
   报告 passenger waiting time、in-vehicle delay、holding time、headway CV、bunching/large-gap rate、decision count、completed trips，并做 route-level breakdown。

4. **做多场景评估。**  
   至少增加多个 evaluation seeds、demand perturbations、traffic perturbations。heuristic 和 RL 方法必须在同一 seed/scenario set 上比较。

5. **补 simulator calibration。**  
   说明真实数据来源，并报告 SUMO 与真实运营数据之间的 validation error。

6. **清理实验协议。**  
   统一 170/100/48 configs、4/5 seeds、warmup epochs、LCB beta、Q-floor definition、statistical tests。

7. **补 action-invariance 数据级验证。**  
   不要只报告 learned discriminator permutation sensitivity。

8. **降低 Q-floor claim。**  
   用 activation rate 和 Q-scale drift 曲线支撑它的作用；若仍 mixed，就把它写成 optional component。

9. **补 baseline。**  
   至少加入 BC、ep39-only BC、Daganzo/Xuan analytical holding、多 seed heuristic、一个标准 offline-to-online baseline。

10. **明确开源复现入口。**  
    给出每个主表 row 的 exact command/config/checkpoint selection/evaluation seed set。

## 结论

这篇稿子有一个值得做的方向：把 H2O+ 引入公交 holding，并通过大规模 ablation 展示哪些 RL 组件在 transit sparse-action/multi-fidelity setting 下真的有效。但当前版本的主结果、方法叙事和实验设定还没有完全对齐。对交通类顶刊而言，最关键的问题不是公式是否复杂，而是结果是否能证明真实公交运营意义、是否在多场景下稳健、是否和低保真 simulator training 的部署设定一致。

我的建议是先按上述清单做一轮大修，再投。若现在直接投，比较可能被评为 **Major Revision 或 Reject with encouragement to resubmit**；最容易被卡住的点是 SUMO-online headline result、交通指标不足、多场景泛化不足和实验协议不一致。
