# Project Roadmap: H2O+ Integration with Snapshot-based Buffer Reset
基于对 @[H2Oplus] 代码库、@[LSTM-RL/env/sim.py] 以及 @[SUMO_ruiguang] 的分析

**任务目标**: 实现一套基于 H2O+ 的 Sim-to-Real 强化学习框架，利用低保真仿真器 (`LSTM-RL/env/sim.py`) 和少量真实路网数据 (`SUMO_ruiguang`) 训练鲁棒策略。

**利用 `H2O+` 框架，结合：**
1.  **Offline Data**: 来自复杂的 `SUMO_ruiguang` (视为 "Real World")。
2.  **Online Sim**: 来自简化的 `LSTM-RL/env/sim.py` (视为 "Dynamics Gap Simulator")。
3.  **Algorithm**: 使用 H2O+ (或 CQL) 进行混合训练。

**核心机制**:
1.  **上帝模式重置 (God-Mode Reset)**: 利用真实数据的全局快照 (Snapshot) 强制重置仿真器状态，消除 `LSTM-RL/env`的时间漂移(随着仿真运行，`sim.py`产出的数据分布离`SUMO_ruiguang`产出的越来越远)。
2.  **上下文感知判别器 (Context-Aware Discriminator)**: 利用全局快照提取宏观交通特征 (Context)，帮助判别器精准识别 Sim 与 Real 的分布差异。判别器不仅看单步状态，而是看宏观路网从 $z_t$ 到 $z_{t+1}$ 的**时空转移**，精准识别违背物理惯性的“假”仿真数据。
3.  **判别器引导的动态截断 (Discriminator-Guided Early Truncation)**:
实时监控仿真信度 $w$，一旦发现仿真偏离真实逻辑，立即截断并重新重置，极大提升样本效率。

---

## Phase 0: 核心数据协议定义 (Core Data Protocols)

**执行优先**: 在编写任何环境代码前，必须先定义好共享的数据结构和工具函数。建议新建文件 `common/data_utils.py`。

### 0.1 快照数据结构 (Snapshot Schema)
`LSTM-RL/env/sim.py` 和 `SUMO_ruiguang` 必须生成**结构完全一致**的字典，用于描述某一时刻的全局系统状态,每一次 "公交到站决策事件"，产生一个snapshot，包括一个传统RL用的tuple，用于训练offline RL，以及包含其他所有车辆当前时刻信息的infors。
这个结构代表了我们对客观世界状态的最佳估计 (Best Estimate)。在 SUMO 实验中直接取真值；在实车部署中，这代表经过清洗和推断后的数据。

```python
# Type Definition in pseudo-code
Transition = {
    # --- Part A: 微观 RL 数据 (Standard Offline RL) ---
    # 描述 "Ego Bus" (当前触发决策的车辆) 的状态和行为
    "global_time":float,     # 以"事件"为核心的索引
    "obs": np.ndarray,       # Shape: (state_dim,)，若state已经为最后一站则删除该条数据
    "action": np.ndarray,    # Shape: (action_dim,), 执行的驻站时间，由算法返回得到，非控制状态为0
    "reward": float,         # 单步奖励，用LSTM-RL/env/bus.py中相似的计算方式计算得到
    "next_obs": np.ndarray,  # 需要暂时空置，待车辆到达下一站才获取到
    "terminal": bool,        # 是否结束

    # --- Part B: 宏观快照 (For Reset & Discriminator) ---
    # 描述决策时刻的全系统物理状态
    "infos": {
        "snapshot": {
            "global_time": float,   # 绝对仿真时间
            "ego_bus_id": str,      # 标记谁是当前决策的主角
            
            # 全网车辆列表 (包含 Ego Bus 和 背景车辆)
            "all_buses": [
                {
                    "id": str,
                    "pos": float,   
                    # 距线路起点的绝对距离 (m) -> 核心对齐字段。
                    #"pos 必须是归一化到线路起点的线性距离，且需处理环线或折返线的坐标跳变。参考LSTM-RL/env/bus.py中的absolute_distance的定义方式"
                    "speed": float, # 当前速度 (m/s)
                    'last_stop_index': int, # 刚经过的站点索引 (0~N)
                    'ratio_to_next': float, # 距离下一站的进度百分比 (0.0~1.0)
                    "load": int,    # 车内人数
                    "is_ego": bool  # 方便快速索引
                },
                # ... 必须包含路网上所有活跃车辆
            ],
            
            # 全网站点列表(在真实中用推断值)
            "all_stations": [
                {"id": str, 
                "index": int,
                "pos": float,         #必须包含站点的线性绝对距离 (m)，用于空间装箱
                "waiting_count": int, # 站台滞留人数 (推断值)
                "arrival_rate":float # (可选) 该时段的历史平均到达率，辅助Sim生成新客流
                },
                # ...
            ]
        }
    }
}
```

### 0.2 上下文提取函数 (Context Extractor)
为了保留空间信息，我们将路网离散化为K个空间段 (Segments)，生成一个矩阵作为 Discriminator 的输入，而不是几个标量。

```python
def extract_structured_context(snapshot: dict, num_segments=10) -> np.ndarray:
    """
    将线性路网切分为 num_segments 个段，计算每一段的交通指纹。
    Input: SnapshotDict
    Output: Flat Vector, shape = (num_segments * 3, )
            包含 [速度分布, 车辆密度分布, 拥堵分布]
    """
    total_length = ROUTE_LENGTH # 假设已知线路总长
    segment_len = total_length / num_segments
    
    # 初始化 buckets
    seg_speeds = [[] for _ in range(num_segments)]
    seg_counts = [0] * num_segments
    seg_loads  = [0] * num_segments 
    seg_waiting = [0] * num_segments 
    
    # 1. 将车辆映射到空间段中 (Spatial Binning)
    for bus in snapshot['all_buses']:
        # 计算该车在哪一段 (0 ~ K-1)
        idx = int(bus['pos'] / segment_len)
        idx = min(idx, num_segments - 1)
        
        seg_speeds[idx].append(bus['speed'])
        seg_counts[idx] += 1
        seg_loads[idx] += bus['load']
    for st in snapshot['all_stations']:
        idx = int(st['pos'] / segment_len)
        idx = min(idx, num_segments - 1)
        seg_waiting[idx] += st['waiting_count']
        
    # 2. 计算统计特征
    # 特征 1: 速度分布 (SimBus 通常均匀，Real 在红绿灯路段低)
    vec_speed = np.array([np.mean(s) if s else 30.0 for s in seg_speeds]) / 30.0
    
    # 特征 2: 密度分布 (哪里车多)
    vec_density = np.array(seg_counts) / 5.0 # 归一化
    
    # 特征 3: 站点拥挤度 (映射 waiting 到空间段)
    vec_waiting = np.array(seg_waiting) / 20.0
    
    # 3. 拼接并展平
    # 判别器将看到一个长度为 30 的向量，清晰地描述了整条路哪里快、哪里慢、哪里人多
    z = np.concatenate([vec_speed, vec_density, vec_waiting])
    return z.astype(np.float32)


# [新增] 在 extract_structured_context 之后添加
import torch

def aggregate_temporal_contexts(z_list: list) -> np.ndarray:
    """
    解决异步问题：在 Ego Bus 从上一站到本站的时间段内，可能产生了 n 个不同的 z。
    利用统计池化 (Statistical Pooling) 将这 n 个 (30,) 维向量，
    聚合成一个定长的、描述这段时间整体路况波动的特征向量。
    
    Input: z_list, 长度为 n 的 List[np.ndarray(30,)]
    Output: Z_temporal, shape=(120,), 包含[Mean, Max, Min, Var]
    """
    if len(z_list) == 0:
        return np.zeros(30 * 4, dtype=np.float32)
        
    z_tensor = torch.tensor(np.array(z_list)) # Shape: (n, 30)
    
    z_mean = torch.mean(z_tensor, dim=0)
    z_max, _ = torch.max(z_tensor, dim=0)
    z_min, _ = torch.min(z_tensor, dim=0)
    # 如果只有1个z，方差为0
    z_var = torch.var(z_tensor, dim=0, unbiased=False) if len(z_list) > 1 else torch.zeros(30)
    
    # 拼接：30*4 = 120 维
    z_temporal = torch.cat([z_mean, z_max, z_min, z_var], dim=-1).numpy()
    
    return z_temporal.astype(np.float32)
```

<!-- **新方案** : 输入 30 个数（空间图谱）。判别器能判断 “为什么第 3 段（十字路口）没有降速？” 或者 “为什么第 5 段（学校）没有很多人？”
结果: 这极大地增强了 H2O+ 区分 Sim/Real 的能力，迫使 Policy 在那些 SimBus 模拟不准的“特征路段”（红绿灯、拥堵点）更加保守，而在路况简单的路段更加自信。 -->

---

## Phase 1: 真实环境封装与数据采集 (Real World - /SUMO_ruiguang)

**目标**: 将 `SUMO_ruiguang` 封装为 Gym 环境，并采集包含 Snapshot 的全量数据。

### 1.1 `SumoGymWrapper` 实现

*   **文件**: `envs/sumo_wrapper.py`
*   **现状**: `rl_env.py` 是 Event-driven 的，返回字典结构状态。
*   **基类**: `gym.Env`
*   **策略**: 直接利用现有的高效 `rl_bridge.py` 逻辑。该 Bridge 已经实现了 `Libsumo` 加速和事件队列管理，性能优异。
*   **关键点**: Wrapper 内部维护一个 `while` 循环调用 `bridge.fetch_events()`，直到获得下一个 Action 请求，从而适配 Gym 的 `step()` 阻塞式接口。
*   **机制**: env.step(action) 不是前进固定的 1 秒，而是前进直到任意一辆公交车到站。
```python
def step(self, action):
    # 1. 对上一辆 Ego Bus 应用 Action (设定驻站)
    self._apply_hold(self.last_ego_id, action)
    
    # 2. 仿真循环，直到下一辆车需要决策
    while True:
        traci.simulationStep()
        arrived_bus = self._check_any_bus_arrival()
        if arrived_bus:
            self.last_ego_id = arrived_bus
            return self._get_state(arrived_bus), ...
```

### 1.2 `collect_data_sumo.py` 实现

### 1.2.1 异步数据采集与时序对齐 (Asynchronous Data Alignment)

**核心挑战**: 公交系统不同于标准 Gym 环境，动作 ($a_t$) 与下一状态 ($s_{t+1}$) (并不是在t+1时刻的状态，只是约定俗成，实际上下一站上下乘客后的当前状态)之间存在较长的物理时间延迟，且不同车辆是异步到达的。

**解决方案**: 采用 **"Pending Cache (挂起缓存)"** 机制，跨时间步拼接完整的 Transition 元组。可以参考`sac_v2_bus.py`或 `sac_v2_bus_SUMO.py`中使用state_dict和action_dict并在state_dict的长度凑足后才存入buffer的办法。

**注意**:无论在offline环境还是online环境，都要采样相同的数据格式，即(s, a, r, s', snapshot_T1, snapshot_T2)，并记录来源是sim还是real

#### A. 时序逻辑定义
一条用于 Offline RL 和 H2O+ 的标准数据 `(s, a, r, s', snapshot_T1, snapshot_T2)` 必须由两个时间点的事件共同构成：

*   **时刻 $T_1$ (上一站)**: 车辆到达站点 $k$。
    *   产生: `current_obs` ($s$), `action` ($a$), **`snapshot` ($Snapshot_{T1}$)**。
    *   动作: **暂存 (Cache)** 这些数据，因为此时不知道奖励和下一状态。
    *   *注意: Snapshot 必须捕获 $T_1$ 时刻的全网状态，用于后续 Reset 回到做出决策的那一刻。*
*   **时刻 $T_2$ (当前站)**: 车辆到达站点 $k+1$。
    *   产生: `next_obs` ($s'$), `reward` ($r$)。
    *   动作: **结算 (Settle)** 上一站的缓存，生成完整 Tuple 并存入 Buffer。
    *   *注意: 此时也要捕获 $T_2$ 时刻的全网状态，用于后面给discriminator判别sim还是real*
#### B. `collect_data_sumo.py` 核心实现逻辑

```python
# 缓存字典: Key=VehicleID, Value=Dict(上一次决策时的上下文)
global_z_buffer =[]     # 用于暂存仿真步推进过程中产生的所有 z

def on_simulation_step():
    # 获取当前仿真步内所有完成靠站、需要决策的车辆列表

    global global_z_buffer

    arrived_buses = sumo_env.get_arrived_buses()
    
    # 如果有车到站，说明路网状态发生了有意义的更新
    if len(arrived_buses) > 0:
        current_snapshot = capture_full_system_snapshot()
        current_z = extract_structured_context(current_snapshot)
        global_z_buffer.append(current_z)

    for bus_id in arrived_buses:
        # 1. 获取当前时刻(T2)的状态 -> 作为 s'
        current_obs = sumo_env.get_state(bus_id)
        
        current_time = sumo_env.get_time()
        current_snapshot = sumo_env.get_snapshot()
        z_t = extract_structured_context(current_snapshot) #压缩得到30维向量z_t
        # ---> 挂起 (obs_t, action_t, z_t)，Ego Bus 继续开往下一站 ...
        next_obs = current_obs
        next_snapshot = current_snapshot
        z_next = extract_structured_context(next_snapshot)


        # --- [结算逻辑] ---
        # 如果该车在上一站有未结单的决策
        if bus_id in pending_transitions:
            prev_data = pending_transitions.pop(bus_id)
            
            # 计算延迟奖励 r (根据 T1 的预测和 T2 的实际情况)
            # 例如: reward = -(current_headway_variance)
            reward = calculate_reward(prev_data['obs'], current_obs) #复用LSTM-RL/env/bus.py中的reward计算方式一样
            # 【关键变动】提取从上一站到这一站期间累积的所有 z
            # prev_data['z_list_start_idx'] 记录了当时全局 buffer 的长度
            z_list_for_this_transition = global_z_buffer[prev_data['z_list_start_idx']:]
            z_temporal_fixed = aggregate_temporal_contexts(z_list_for_this_transition)
            
            # 存入 Replay Buffer
            # 关键: info['snapshot'] 必须是 prev_data['snapshot'] (T1时刻的快照)
            replay_buffer.add(
                obs=prev_data['obs'],
                action=prev_data['action'],
                reward=reward,
                next_obs=current_obs,
                terminal=False,
                z_temporal=z_temporal_fixed,
                infos={'snapshot': prev_data['snapshot']} 
            )
            
        # --- [新开单逻辑] ---
        # 如果车辆未到达终点，需要进行下一次决策
        if not is_route_end(bus_id):
            # 1. 立即捕获当前时刻(T2)的全网快照 -> 作为下一次的 snapshot
            # 注意: 必须标记当前的 bus_id 为 ego_vehicle
            snapshot_now = capture_full_system_snapshot(ego_id=bus_id)
            
            # 2. 决策
            action = behavior_policy.get_action(current_obs)
            sumo_env.apply_action(bus_id, action)
            
            # 3. 存入缓存，等待下一次到达
            pending_transitions[bus_id] = {
                'obs': current_obs,
                'action': action,
                'snapshot': snapshot_now,
                'timestamp': current_time,
                'z_list_start_idx': len(global_z_buffer) # 记录当前累积起点
            }
```

#### C. 对 H2O+ 的影响
这种采集方式确保了：
1.  **Reset 有效性**: 当我们用 Buffer 中的 `snapshot` 重置 SimBus 时，SimBus 会回到 $T_1$ 时刻。此时 Agent 看到的观测值正是 $s$，它可以尝试输出一个新的动作 $a'$，从而产生新的反事实轨迹。
2.  **Critic 训练正确性**: $r$ 和 $s'$ 真实反映了在真实动力学下，执行 $a$ 后的结果。

*   **输出**: `datasets/sumo_offline_full.hdf5` (或 pickle)。

#### 验证：通过训练offline RL,用collect_data_sumo.py和SumoGymWrapper采集的数据训练offline RL，并通过收敛性确认以上工作的完成情况。
---

## Phase 2: 仿真环境改造 (Sim World - /LSTM-RL/env)

**目标**: 改造 `LSTM-RL/env`中的`sim.py`/`bus.py`等依赖，使其支持多线路仿真，和并使其支持“写入快照(Step)”和“读取快照(Reset)”。

**动机**: 为了减小 Dynamics Gap，必须确保 `LSTM-RL/env` (Sim) 的基础数据和运行特性与 `SUMO_ruiguang` (Real) 高度一致。

#### 2.1 静态数据对齐 (Static Data Alignment)
-   **现状差异**: `SUMO` 使用 XML 定义路网和时刻表，`LSTM-RL` 使用 Excel (`stop_news.xlsx`, `time_table.xlsx` 等)。
-   **校准策略**: 编写数据转换脚本 `xml_to_excel_converter.py`。
    -   解析 `SUMO_ruiguang` 的 `save_obj_*.xml` 文件。
    -   生成 `LSTM-RL/env` 所需的 Excel 格式。
    -   将 SUMO 历史红绿灯延误转化为 SimBus 路段的**等效平均限速**。
    -   **关键点**: 站点 ID、线路 ID、站点距离 (`distance`)、发车时间 (`launch_time`) 必须一一对应。确保 `LSTM-RL/env/sim.py` 读取的 `route_length` 和 `station_positions` 与 `SUMO_ruiguang`中的 XML 文件中的定义**误差 < 1%**。如果路网长度不对，位置映射就会失效。其余`LSTM-RL/env`中的客流/路况(速度)尽可能贴合`SUMO_ruiguang`中的.

#### 2.2 动力学参数 (Dynamics)
-   **现状差异**: `SUMO` 有完整的信号灯相位逻辑 (`Signal` 类)，`LSTM-RL` 仅有简单的 `Route` 和 `V_max`，无信号灯实体。
-   **校准策略 (简化版)**: 不在 `LSTM-RL` 中重写复杂的信号灯逻辑，而是通过**等效降速**来模拟。
    -   **统计**: 从 `SUMO` 历史运行数据中统计每条 Edge 的**平均通过时间** (包含红灯等待)。
    -   **应用**: 将该等效平均速度填入 `LSTM-RL` 的 `route_news.xlsx` 中的速度限制或参数中。
    -   **复用机制**: `LSTM-RL` 的 Bus 复用机制与 `SUMO` 的 Trip 机制不同，这影响不大，只要确保同一时刻在线车辆的行为符合物理规律即可。在生成 Gym Wrapper 时，根据 `trip_id` 唯一标识 Agent。


### 2.3 `BusSimEnv` 接口改造(上帝模式)
*   **文件**: `LSTM-RL/env`
*   **需求 A: 输出快照 (Symmetry for Discriminator)**
    *   在 `step()` 返回的 `info` 字典中，必须调用 `self._build_snapshot()`，返回符合 **Phase 0 定义** 的 `SnapshotDict`。
    
*   **需求 B: 快照重置 (Reset Mechanism)**
    *   实现 `reset(snapshot=None)` 接口。
    ```python
    def reset(self, snapshot: dict = None):
        self.cleanup() # 清空环境
        
        if snapshot is None:
            return self._reset_standard() # t=0 发车
        else:
            # 时光倒流
            self.current_time = snapshot['global_time']
            
            # 重建物理实体
            for b_data in snapshot['all_buses']:
                new_bus = self.spawn_bus(
                    id=b_data['id'],
                    pos=b_data['pos'], # 映射回 SimBus 坐标
                    speed=b_data['speed'],
                    load=b_data['load']
                )
            
            # 恢复站点
            for s_data in snapshot['all_stations']:
                self.stations[s_data['id']].queue = s_data['waiting_count']
            self.set_passenger_arrival_rate(station_id, station_data['arrival_rate'])
            # 关键：寻找 snapshot['ego_bus_id'] 指向的那辆车
            # 并返回它的 State 作为初始 Obs
            return self._get_bus_state(snapshot['ego_bus_id'])
    ```
*   **验证**: 通过训练online RL,用之前已收敛的`sac_v2_bus.py`或`sac_v2_bus_SUMO.py`在改造后的`LSTM-RL/env`上训练,以验证该部分改造工作的正确性。
---

## Phase 3: H2O+ 算法深度集成 (H2O+ Integration)

**目标**: 修改 H2O+ 训练循环，实现混合重置与加权更新。

### 3.1 缓冲区管理 (Buffer Management)
*   `OfflineBuffer`: 加载 `datasets/sumo_offline_full.hdf5`。数据不可变。
    *   **预处理**: 在加载时，对所有数据调用 `extract_context(info['snapshot'])`，生成 `info` 向量并缓存。
*   `OnlineBuffer`: 这里的 Sim 数据是动态生成的。
    *   **运行时**: 每次 `env.step()` 后，调用 `extract_context(info['snapshot'])` 得到 `info`，存入 Buffer

> ⚠️ **CRITICAL PROMPT FOR AI CODING ASSISTANTS (关于状态撕裂与轨迹截断):**
> 在编写本节的 Replay Buffer 存储逻辑时，必须严格遵守马尔可夫决策过程 (MDP) 的时序边界！
> 1. **严禁时空连结**：当触发 `p < P_RESET` 导致仿真器从 `t=100` 瞬间重置回真实快照的 `t=50` 时，这在物理上是一次“时空跳变”。
> 2. **代码强制要求**：在执行 `sim_env.reset(snapshot)` 之前，如果上一段仿真还有未完结的 Transition，必须将其强制打上 `truncated=True` 或 `done=True` 的标签存入 Buffer。
> 3. **防止 TD-Error 崩溃**：绝对不能让 Critic 网络的 Bellman Equation 出现类似 $Q(s_{99}, a) = r + \gamma \max Q(s_{50}, a')$ 的计算！重置前后的状态在时序上是完全断开的，必须通过 `done/truncated` 掩码 (Mask) 切断梯度传导。


### 3.2 混合重置与动态早停训练循环 (Training Loop)
这里引入了**判别器引导的动态截断**：如果不连续或偏离真实路况，提前结束本回合仿真。
*   **文件**: `algorithms/h2o_plus/train.py`
*   **伪代码逻辑**:
    ```python
    # 初始化

    WARMUP_EPISODES = 20 # 或者是前 10% 的训练步数

    offline_buffer.load("sumo_offline_full.pkl")
    online_buffer = ReplayBuffer()

    for episode in range(MAX_EPISODES):
        # --- A. 混合重置策略 ---
        if random.random() < P_RESET (e.g., 0.5):
            # Mode: Buffer Reset (解决 Drift)
            real_batch = offline_buffer.sample(1)
            snapshot = real_batch['infos']['snapshot'][0]
            obs = sim_env.reset(snapshot=snapshot)
            z_sim_curr = extract_structured_context(snapshot_curr[0])
            z_sim_next = extract_structured_context(snapshot_next[0])
            max_steps = H_ROLLOUT (e.g., 20) # 短程修补
        else:
            # Mode: Standard (全程规划)
            obs = sim_env.reset(snapshot=None)
            z_sim_curr = extract_structured_context(sim_env.get_current_snapshot())
            z_sim_next = extract_structured_context(sim_env.get_next_snapshot())
            max_steps = FULL_LENGTH
            
        # --- B. 数据交互与收集 ---
        for t in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, info = sim_env.step(action)
            
            # 实时提取 Sim 的宏观特征
            z_sim_curr = extract_structured_context(info['snapshot'][0])
            z_sim_next = extract_structured_context(info['snapshot'][1])
            # 实时计算当前转移的"真实度" w
            with torch.no_grad():
                # 判别器输出的是 logit，需要经过 sigmoid 映射到 0~1 表示概率
                logit = discriminator(obs, action, next_obs, z_sim_curr, z_sim_next)
                prob_real = torch.sigmoid(logit).item()
                
                # 在 H2O 中，w = P(real) / P(sim) = prob_real / (1 - prob_real + epsilon)
                # 为防止除零，加入微小常数；为防止截断过于敏感，可用 prob_real 直接替代评估信度
                w = prob_real / (1.0 - prob_real + 1e-8)
            
            # 核心：动态截断 (Early Exit)
            if w < W_THRESHOLD and episode > WARMUP_EPISODES: # e.g., 0.1
                # 仿真器已经严重偏离真实动力学，立即截断
                online_buffer.add(obs, action, reward, next_obs, done=False, truncated=True, z=z_sim)
                break # 提前跳出循环，重新从真实快照 Reset
            else:
                # 表现合理，继续 Rollout
                online_buffer.add(obs, action, reward, next_obs, done, truncated=False, z=z_sim)
                obs = next_obs
                z_sim_curr = z_sim_next
            
            if done: break
            
            # --- C. H2O+ 更新步骤 (每步或每N步) ---
            if ready_to_train:
                # 1. 采样
                batch_real = offline_buffer.sample(BATCH_SIZE) # 带 z_real
                batch_sim = online_buffer.sample(BATCH_SIZE)   # 带 z_sim
                
                # 2. 训练 Discriminator
                d_loss = update_discriminator(batch_real, batch_sim)
                
                # 3. 计算 Importance Weight
                # w = P_real / P_sim ≈ exp(logit_real - logit_sim) 
                # 具体公式参考 H2O 论文，通常使用 sigmoid 输出处理
                w = compute_weights(discriminator, batch_sim)
                
                # 4. 训练 Critic (带权重)
                # Loss = w * (Q - Target_Q)^2 + (1-w) * (Q - Target_Q_Conservative)^2 (可选保守项)
                # 简化版: Loss = w * (Q_sim - Target)^2 + (Q_real - Target)^2
                c_loss = update_critic(batch_real, batch_sim, w)
                
                # 5. 训练 Actor (标准 SAC)
                a_loss = update_actor(batch_sim) # 通常只在 Sim 数据上更新 Actor
    ```

    
> ⚠️ **CRITICAL PROMPT FOR AI CODING ASSISTANTS (关于因果律与防信息泄露):**
> 在编写 Actor 动作选择与 Discriminator 推断逻辑时，必须严格区分“决策时刻 ($T_1$)”与“结算时刻 ($T_2$)”，严禁发生未来信息泄露 (Information Leakage)！
> 1. **动作决策 (Action Selection)**：在 $T_1$ 时刻（公交刚到站），Actor 网络 `policy(obs)` 只能且必须只能接收当前的局部微观观测 `obs_t`。此时路网的宏观演变 `z_temporal` 尚未发生，绝不能将其作为 Actor 的输入。
> 2. **信度评估 (Weight Calculation)**：Discriminator 的计算 $w = D(obs_t, a_t, obs_{t+1}, z\_temporal)$ 必须是**事后诸葛亮 (Post-evaluation)**。只有当公交车在 $T_2$ 时刻到达下一站，凑齐了从 $T_1$ 到 $T_2$ 期间收集到的宏观快照序列并聚合出 `z_temporal` 后，才能进行前向推断。
> 3. **变量边界**：计算出的信度权重 $w$ 仅用于两处：(A) 存入 Buffer，在更新 Critic 损失函数时作为 Importance Weight；(B) 决定是否触发 Early Truncation 放弃该条轨迹的后续 rollout。**绝对不能将 $w$ 或 $z\_temporal$ 泄露给当前的 Actor。**



### 3.3 上下文转移判别器 (Contextual Transition Discriminator)
判别器必须评估**全局状态转移 $(z_t \to z_{t+1})$** 的合理性。
*   **输入**: `concat(state, action, next_state, z_t, z_t+1)`。
*   **逻辑**: 如果 SimBus 从一个“局部拥堵”的 $z_t$，仅凭单车动作，瞬间变成了“全网畅通”的 $z_{t+1}$，判别器将识别出违背物理定律，输出低 $w$。
*   **Critic 更新**: 使用 $w \cdot (Q - \mathcal{T}Q)^2$ 进行加权更新。

---

## Phase 4: 验证清单 (Definition of Done)

在提交论文或大规模跑实验前，请按顺序验证以下三点：

1.  **接口一致性验证**: 打印 `SUMO_wrapper.observation_space` 和 `BusSimEnv.observation_space`，必须完全相同。
2.  **重置有效性验证**:
    *   从 Offline Data 拿一个 Snapshot（例如 t=3600s, 3辆车）。
    *   调用 `sim_env.reset(snapshot)`。
    *   检查 `sim_env` 内部是否真的生成了3辆车，且位置、速度、时间完全一致。
3.  **判别器敏感度验证**:
    *   手动构造一个“极度顺畅”的 Snapshot (Sim特征) 和一个“极度拥堵”的 Snapshot (Real特征)。
    *   检查 Discriminator 是否能给出显著不同的分数。
#### Structured Contex & Buffer Reset消融实验
    分别用简单的mean/var only contex对比structured contex，以及buffer reset vs 直接用sim的策略在SUMO跑做评价。
---
## Appendix A: 上下文感知判别器 (Context-Aware Discriminator)

**定位**: 替代 Phase 3.3 中简单的 MLP 判别器。
**核心思想**: 放弃手动提取均值/方差 ($z$)，改用 **Ego-Centric Attention (以自我为中心的注意力机制)** 自动学习 Ego Bus 与背景交通流的时空依赖关系。

### A.1 网络架构设计
该架构利用 Transformer Block 处理变长的车辆集合，具有**置换不变性 (Permutation Invariance)** 和 **距离敏感性**。

*   **Input**:
    *   `ego_feat`: 当前决策车辆特征 `(batch, state_dim)`
    *   `context_feat`: 背景车辆特征序列 `(batch, max_buses, state_dim)` (需 Padding)
    *   `mask`: 掩码矩阵，标记 Padding 的位置
*   **Architecture**:
    1.  **Embedding**: 将物理特征映射为高维隐向量。
    2.  **Cross-Attention**:
        *   **Query**: Ego Embedding
        *   **Key/Value**: Context Embedding (包含 Ego 自身)
        *   *物理含义*: 自动关注前车、后车以及拥堵路段的车辆，忽略远端无关车辆。
    3.  **Classifier**: 输出 Logit。

### A.2 PyTorch 参考实现

```python
import torch
import torch.nn as nn

class SimpleTransitionDiscriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, context_dim=30, hidden_dim=256):
        """
        context_dim = 30 (假设你把路线切成10段，每段提取速度、密度、等待人数 3 个特征)
        """
        super().__init__()
        
        # 输入维度 = 当前观测 + 动作 + 下一观测 + 动作前全网路况(z_t) + 动作后全网路况(z_t+1)
        input_dim = obs_dim + act_dim + obs_dim + context_dim + context_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出 Logit
        )

    def forward(self, obs, act, next_obs, z_curr, z_next):
        """
        所有的输入都是 1D 的 vector (Batch, Dim)
        """
        x = torch.cat([obs, act, next_obs, z_curr, z_next], dim=-1)
        return self.net(x)

        
```

---

## Appendix B: 递归状态估计器 (Recursive State Estimator)

**定位**: 解决 Phase 0 中真实世界数据缺失（Unknown Waiting/Load）的问题。
**核心思想**: 利用 **卡尔曼滤波 (Kalman Filter)** 的“预测-修正”思想，维护全网站点的人数信念状态 (Belief State)，而无需每次重新计算积分。

### B.1 数学模型
*   **状态向量 ($X_t$)**: 全网站点当前滞留人数向量。
*   **预测步 (Predict)**: $X_{t+\Delta t} = X_t + \lambda \cdot \Delta t$
    *   $\lambda$: 历史平均到达率向量 (Arrival Rate)。
*   **修正步 (Correct)**: $X_{new} = \max(0, X_{pred} - u_{board})$
    *   $u_{board}$: 推测上车人数。

### B.2 状态估计器实现 (Python)

此模块应集成在 `BusSimEnv` 和 `collect_data_sumo.py` 中，作为数据预处理层。

```python
import numpy as np

class TrafficStateEstimator:
    def __init__(self, station_ids, arrival_rates):
        """
        Args:
            station_ids: 站点ID列表
            arrival_rates: Dict {station_id: rate_per_second} (来自历史统计)
        """
        self.station_map = {sid: i for i, sid in enumerate(station_ids)}
        self.rates = np.array([arrival_rates[sid] for sid in station_ids])
        
        # 信念状态 (Belief State)
        self.queues = np.zeros(len(station_ids))
        self.last_update_time = 0.0

    def predict_until(self, current_time):
        """
        [积分步]：根据流逝时间推演队列增长
        """
        dt = current_time - self.last_update_time
        if dt > 0:
            # 简单泊松过程期望：Rate * Time
            # 进阶：可在此处加入高斯噪声模拟不确定性
            self.queues += self.rates * dt
            self.last_update_time = current_time

    def correct_on_arrival(self, station_id, estimated_board):
        """
        [修正步]：车辆到站带走乘客
        Args:
            estimated_board: 
                - SUMO中: 直接读取真值
                - Real World: (Dwell_Time - Dead_Time) / Time_Per_Person
        """
        idx = self.station_map[station_id]
        
        # 1. 先同步到当前时刻
        # self.predict_until(now) # 需传入当前时间
        
        # 2. 修正状态 (人数不能为负)
        self.queues[idx] = max(0.0, self.queues[idx] - estimated_board)
        
        return self.queues[idx]

    def get_snapshot_state(self):
        """返回用于构建 Snapshot 的数据"""
        return self.queues.copy()

```
## Appendix C: 判别器训练与标签平滑 (Discriminator Training & Regularization)

**定位**: 解决少量 Offline 数据下，判别器极易“死记硬背”数值而丧失语义泛化能力的问题。
**核心策略**: 对真实数据的宏观特征 $z$ 注入随机噪声，并放弃绝对的 0/1 标签，改用软标签 (Label Smoothing)。

### C.1 训练循环参考实现

```python
# 假设已初始化: discriminator = SimpleTransitionDiscriminator(...)
# 使用 BCEWithLogitsLoss，因为它内置了 Sigmoid，数值更稳定
criterion = nn.BCEWithLogitsLoss() 

def train_discriminator_step(offline_buffer, online_buffer, optimizer_D, batch_size):
    # ---------------------------------------------------------
    # 1. 真实数据 (SUMO) 前向传播
    # ---------------------------------------------------------
    real_batch = offline_buffer.sample(batch_size)
    
    real_obs = real_batch['obs']
    real_act = real_batch['action']
    real_next_obs = real_batch['next_obs']
    
        # 【更新】现在的 z 已经是聚合了 t 到 t+1 期间波动的 120 维特征
    real_z_temporal = real_batch['z_temporal'] 
    
    # 注入噪声增强鲁棒性
    noise = torch.randn_like(real_z_temporal) * 0.05
    
    # 判别器输入：微观起点 + 微观动作 + 微观终点 + 期间的宏观波动
    real_logits = discriminator(real_obs, real_act, real_next_obs, real_z_temporal + noise)
    
    # [关键防御 1: 数据增强] 为真实的交通图谱注入 5% 的高斯噪声
    # 迫使判别器学习“拥堵的范围”而不是“拥堵的绝对数值”
    noise_curr = torch.randn_like(real_z_curr) * 0.05
    noise_next = torch.randn_like(real_z_next) * 0.05
    
    real_logits = discriminator(real_obs, real_act, real_next_obs, 
                                real_z_curr + noise_curr, 
                                real_z_next + noise_next)
    
    # [关键防御 2: 标签平滑] 真实数据标签设为 0.9，而非绝对的 1.0
    real_labels = torch.full_like(real_logits, 0.9)
    loss_real = criterion(real_logits, real_labels)

    # ---------------------------------------------------------
    # 2. 仿真数据 (SimBus) 前向传播
    # ---------------------------------------------------------
    sim_batch = online_buffer.sample(batch_size)
    
    sim_obs = sim_batch['obs']
    sim_act = sim_batch['action']
    sim_next_obs = sim_batch['next_obs']
    sim_z_curr = sim_batch['z_curr']
    sim_z_next = sim_batch['z_next']
    
    sim_logits = discriminator(sim_obs, sim_act, sim_next_obs, sim_z_curr, sim_z_next)
    
    # 仿真数据标签设为 0.1，而非绝对的 0.0
    sim_labels = torch.full_like(sim_logits, 0.1)
    loss_sim = criterion(sim_logits, sim_labels)

    # ---------------------------------------------------------
    # 3. 梯度回传
    # ---------------------------------------------------------
    total_d_loss = loss_real + loss_sim
    optimizer_D.zero_grad()
    total_d_loss.backward()
    optimizer_D.step()
    
    return total_d_loss.item()
```

### 鲁棒性增强策略 (Robustness Strategy)

为了防止推测误差导致 H2O+ 训练崩塌，在 **Phase 3 (训练循环)** 中建议加入以下策略：

1.  **噪声重置 (Noisy Reset)**:
    在执行 `sim_env.reset(snapshot)` 时，不要精准还原推测出的 `waiting` 或 `load`，而是注入噪声：
    ```python
    # 在 BusSimEnv._reset_from_snapshot 中
    noise = np.random.normal(0, 0.15) # 15% 的估计误差
    bus.load = int(data['load'] * (1 + noise))
    station.queue = int(data['waiting'] * (1 + noise))
    ```
    *目的*: 迫使 Policy 学会在状态估计不准的情况下依然表现良好 (Domain Randomization 思想)。

2.  **特征模糊化**:
    Discriminator 训练时，可以对 Context Feature 中的 `waiting_count` 维度进行 Dropout 或添加噪声，降低判别器对这一“不可靠特征”的依赖权重。