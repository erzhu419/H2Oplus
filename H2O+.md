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

## 项目目录结构 (Project Layout)

所有代码均位于 `H2Oplus/bus_h2o/` 下，必须在该目录下运行 (`cd H2Oplus/bus_h2o`).

```
H2Oplus/bus_h2o/
├── common/
│   └── data_utils.py          # build_edge_linear_map, extract_structured_context,
│                              # set_route_length, ZOnlyDiscriminator, compute_z_importance_weight
├── sim_core/
│   ├── sim.py                 # env_bus (单线基类), MultiLineEnv (多线聚合)
│   ├── bus.py                 # Bus, BusState 枚举
│   ├── route.py               # Route (路段降速模型)
│   └── station.py             # Station
├── envs/
│   └── bus_sim_env.py         # BusSimEnv(env_bus), MultiLineSimEnv(MultiLineEnv)
├── sumo_env/
│   ├── rl_bridge.py           # SumoRLBridge — SUMO侧的核心接口
│   ├── sumo_snapshot.py       # bridge_to_snapshot, bridge_to_full_snapshot
│   └── rl_env.py              # SumoBusHoldingEnv (勿在快照测试中导入, 会冲突 libsumo)
├── calibrated_env/            # MultiLineSimEnv 各线路标定数据 (Excel + config.json)
│   ├── 7X/data/*.xlsx
│   ├── 7S/data/*.xlsx
│   └── ...                    # 每条线路一个子目录
├── network_data/
│   └── a_sorted_busline_edge.xml  # build_edge_linear_map 的输入文件 (非 .net.xml!)
└── test_snapshot_reset.py      # 方案 2 验证脚本

“真实”侧 SUMO 文件位于上两层:
  SUMO_ruiguang/online_control/          # SumoRLBridge 的 root_dir
  SUMO_ruiguang/online_control/sim_obj/  # SUMO 对象模型 (需加入 sys.path)
```

> [!IMPORTANT]
> **环境变量**: 必须设置 `LIBSUMO_AS_TRACI=1`，否则 `import traci` 会走 TraCI 协议而非直接调用 libsumo。

---

## Phase 0: 核心数据协议定义 (Core Data Protocols)

**执行优先**: 在编写任何环境代码前，必须先定义好共享的数据结构和工具函数。建议新建文件 `common/data_utils.py`。

### 0.0 坐标对齐工具 (Edge → Linear Distance Mapper)

**必做前置任务**。`SUMO_ruiguang` 原生坐标系是 Edge ID + lane offset，而 Snapshot 协议要求 `pos` 是归一化到线路起点的**一维线性绝对距离 (m)**（与 `LSTM-RL/env/bus.py` 中 `absolute_distance` 一致）。这个映射在两处都是关键依赖：
- **Reset 时**：从 offline buffer 抽取 snapshot，需把 `pos` 还原为 SimBus 的坐标，否则车辆位置错乱。
- **特征提取时**：`extract_structured_context` 依赖 `pos` 做空间装箱 (Spatial Binning)，两边坐标系不一致则 `z_t` 和 `z_{t+1}` 的 Real/Sim 比较完全失效。

```python
# common/data_utils.py

def build_edge_to_linear_map(sumo_net_xml_path: str, route_edge_ids: list) -> dict:
    """
    解析 SUMO 路网 XML，沿路线计算每条 Edge 起点到线路起点的累计距离。
    返回字典: edge_id -> cumulative_start_distance (m)
    使用时: linear_pos = edge_map[edge_id] + lane_offset

    Args:
        sumo_net_xml_path: SUMO .net.xml 文件路径
        route_edge_ids:    路线上各 Edge 的有序列表（从起点到终点）
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(sumo_net_xml_path)
    root = tree.getroot()
    edge_lengths = {}
    for edge in root.findall('edge'):
        eid = edge.get('id')
        lane = edge.find('lane')
        if lane is not None:
            edge_lengths[eid] = float(lane.get('length', 0.0))

    edge_map = {}
    cumulative = 0.0
    for eid in route_edge_ids:
        edge_map[eid] = cumulative
        cumulative += edge_lengths.get(eid, 0.0)
    return edge_map   # {edge_id: 该 Edge 起点距线路起点的累计距离 (m)}


ROUTE_EDGE_MAP = {}   # 全局单例，由 main() 初始化后注入

def sumo_pos_to_linear(edge_id: str, lane_offset: float) -> float:
    """将 traci 返回的 (edge_id, lanePosition) 转为线性绝对距离。"""
    return ROUTE_EDGE_MAP.get(edge_id, 0.0) + lane_offset
```



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

> [!WARNING]
> **Schema 一致性说明**: 以上 0.1 是初始设计文档，包含 `last_stop_index`, `ratio_to_next`, `is_ego` 等字段。实际实现中，`restore_full_system_snapshot` 需要的字段已扩展为 14 个（参见 §2.4 的全保真 SnapshotDict Schema）。`bridge_to_full_snapshot` 和 `capture_full_system_snapshot` 输出的都是 §2.4 中的实际 Schema，而非上面的简化版。简化版仅用于理解设计意图。

#### env_bus 内部属性速查 (快照操作需读写的字段)

```python
# sim_core/sim.py  class env_bus
self.bus_all: list[Bus]          # 所有上路车辆列表
self.stations: list[Station]     # 站点列表(station.station_id 为整数)
self.timetables: list[Timetable] # 发车时刻表(timetable.launched 标记是否已发车)
self.routes: list[Route]         # 路段列表
self.current_time: int/float     # 当前仿真时间 (s)
self.bus_id: int                 # 下一个新车辆的 fleet ID
self.max_agent_num: int          # 固定 25
self.one_directional: bool       # SUMO 标定线路为 True

# sim_core/bus.py  class Bus
bus.bus_id: int                  # fleet 编号
bus.trip_id: int                 # 当前车次
bus.trip_id_list: list[int]      # 已执行的所有车次列表
bus.direction: bool              # True=S方向, False=X方向
bus.absolute_distance: float     # 绝对距离 (m)
bus.current_speed: float         # m/s
bus.passengers: np.ndarray       # 车内乘客数组, len(passengers) = 乘客数
bus.on_route: bool               # 是否在路上
bus.state: BusState              # 枚举: TRAVEL/HOLDING/DWELLING/WAITING_ACTION
bus.last_station: Station        # 上一站对象引用
bus.next_station: Station        # 下一站对象引用
bus.forward_headway: float       # 前车间距 (s)
bus.backward_headway: float      # 后车间距 (s)
bus.holding_time: float          # 驻站时间 (s)
bus.next_station_dis: float      # 距下一站 (m)
bus.last_station_dis: float      # 距上一站 (m)
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


```

> ⚠️ **【设计决策】宏观特征 z 最终方案：双快照 `z_t` / `z_{t+1}`（各 30 维，共 60 维）**
>
> - **废弃方案**：将 $T_1→T_2$ 期间所有快照压缩为统计分布（`z_temporal` 120 维）。将时序信息折叠为静态分布会损失语义；不压缩的变长序列在数据量不足时更难对 Real/Sim 做相似性比较。
> - **最终方案**：只保留决策时刻 ($T_1$) 和到站结算时刻 ($T_2$) 的两个 30 维快照向量 `z_t`、`z_{t+1}`。Discriminator 输入为 `[obs, act, next_obs, z_t, z_{t+1}]`，简洁且符合因果律。
> - **`aggregate_temporal_contexts` 函数已废弃，不再实现。**

> ⚠️ **【早截断等待策略】First-step 不触发截断**
>
> 每次从 snapshot reset 后，Ego Bus 必须完成一段完整的站间行驶才能凑齐 $z_{t+1}$。因此在 H2O+ 训练循环 `for t in range(max_steps)` 内，**仅当 `t > 0`** 才检查 `w < W_THRESHOLD`；`t == 0` 时无论 `w` 多低都强制放行，保证至少一步有效 Rollout 并避免因 Discriminator 冷启动噪声导致立即截断。


<!-- **新方案** : 输入 30 个数（空间图谱）。判别器能判断 “为什么第 3 段（十字路口）没有降速？” 或者 “为什么第 5 段（学校）没有很多人？”
结果: 这极大地增强了 H2O+ 区分 Sim/Real 的能力，迫使 Policy 在那些 SimBus 模拟不准的“特征路段”（红绿灯、拥堵点）更加保守，而在路况简单的路段更加自信。 -->

### 0.3 Phase 0 实现笔记 (实际文件与 API)

> **状态: 已完成.** 以下是实际的文件位置和 API 签名，可供后续信号控制等模块复用。

| 功能 | 文件 | 签名 |
|----------|------|-----------|
| 路网距离映射 (Edge map) | `common/data_utils.py` | `build_edge_linear_map(edge_xml, line_id)` 返回 `{edge_id: cum_dist}` |
| z 特征提取 | `common/data_utils.py` | `extract_structured_context(snapshot)` 返回 `np.ndarray(30,)` |
| 路线总长设置 | `common/data_utils.py` | `set_route_length(meters)` -- **必须**在调用 `extract_structured_context` 前调用 |
| SUMO 轻量快照 | `sumo_env/sumo_snapshot.py` | `bridge_to_snapshot(bridge, edge_map)` -- 仅用于提取 z 特征 |
| SUMO 全保真快照 | `sumo_env/sumo_snapshot.py` | `bridge_to_full_snapshot(bridge, edge_map)` 返回 `{line_id: SnapshotDict}` |
| z-only 判别器 | `common/data_utils.py` | `ZOnlyDiscriminator(context_dim=30)` |
| w 权重计算 | `common/data_utils.py` | `compute_z_importance_weight(D, z_t, z_t1)` |

> [!CAUTION]
> **注意事项 (Gotcha)**: `build_edge_linear_map` 解析的是 `a_sorted_busline_edge.xml` (预排序的公交 Edge 列表)，**不是** `.net.xml`。实际调用为 `build_edge_linear_map(edge_xml_path, line_id)`。`set_route_length()` 是全局单例 -- 如果不提前设置，空间分箱计算的分母将默认为 0，导致提取的 z 特征全为 0。

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

### 1.1.1 `SumoRLBridge` 实际 API 参考

> **注意**: 以上 1.1 是初始设计文档 (`SumoGymWrapper`).实际实现中用的是 `SumoRLBridge` (`sumo_env/rl_bridge.py`)，API 完全不同。

#### 初始化与生命周期

```python
from sumo_env.rl_bridge import SumoRLBridge

# root_dir 指向 SUMO_ruiguang/online_control/
bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=25000)
bridge.reset()    # 启动 libsumo 会话，加载路网，初始化所有对象

# 事件循环 (核心模式)
while True:
    events, done, _ = bridge.fetch_events()  # 进一步到下一个决策事件
    if done:
        break
    if not events:      # 可能没有车到站，继续
        continue
    for ev in events:
        bridge.apply_action(ev, hold_seconds=0.0)  # 应用动作
    # 在此多次调用 bridge_to_snapshot 或 bridge_to_full_snapshot

bridge.close()    # 关闭 libsumo 会话
```

#### `bridge` 对象内部属性速查 (快照函数需读取的字段)

```python
# 核心对象字典
bridge.bus_obj_dic:   {sumo_trip_id: BusObj}     # 所有活跃公交车对象
bridge.stop_obj_dic:  {stop_id: StopObj}         # 全网站点对象
bridge.fleet_obj_dic: {sumo_trip_id: FleetBus}   # Fleet 复用映射

# 活跃车辆 ID 列表 (用于遍历)
bridge.active_bus_ids: list[str]

# 位置/头距查找
bridge.line_stop_distances: {line_id: {stop_id: float}}  # 各站线性距离
bridge.line_headways:       {line_id: float}             # 平均发车间距
bridge.current_time:        float                        # 当前 SUMO 仿真时间 (s)

# BusObj 属性 (快照中用到的)
bus_obj.belong_line_id_s: str     # 所属线路 ID ("7X", "7S", ...)
bus_obj.current_stop_id:  str     # 当前站点 ID
bus_obj.bus_state_s:      str     # "Running" 或 "Stop"
bus_obj.current_load_n:   int     # 车内乘客数
bus_obj.direction_n:      int     # 方向

# FleetBus 属性
fb.fleet_id:   str               # 编号字符串 ("7X_0", "7X_1", ...), 末尾数字为 fleet_idx
fb.trip_ids:   list[str]         # 该 fleet 已执行的所有 trip ID
fb.on_route:   bool              # 是否在路上

# StopObj 属性
stop_obj.wait_passenger_num_n: int  # 站台等待人数
```

> [!CAUTION]
> **致命注意: libsumo 单会话限制**
>
> libsumo 全局只能有一个活跃会话。如果在任何地方 `import` 了会触发 `import traci` 的模块 (如 `rl_env.py` 中的 `SumoBusHoldingEnv`)，会静默导致 bridge 的 `reset()`/`fetch_events()` 失效（无输出，无事件，exit code 0）。快照测试脚本中**绝对不能** `from sumo_env.rl_env import SumoBusHoldingEnv`。

### 1.2 `collect_data_sumo.py` 实现

### 1.2.1 异步数据采集与时序对齐 (Asynchronous Data Alignment)

**核心挑战**: 公交系统不同于标准 Gym 环境，动作 ($a_t$) 与下一状态 ($s_{t+1}$) (并不是在t+1时刻的状态，只是约定俗成，实际上下一站上下乘客后的当前状态)之间存在较长的物理时间延迟，且不同车辆是异步到达的。

**解决方案**: 采用 **"Pending Cache (挂起缓存)"** 机制，跨时间步拼接完整的 Transition 元组。可以参考`sac_v2_bus.py`或 `sac_v2_bus_SUMO.py`中使用state_dict和action_dict并在state_dict的长度凑足后才存入buffer的办法。

**注意**:无论在offline环境还是online环境，都要采样相同的数据格式，即(s, a, r, s', snapshot_T1, snapshot_T2)，并记录来源是sim还是real

#### A. 时序逻辑定义
一条用于 Offline RL 和 H2O+ 的标准数据必须由两个时间点的事件共同构成：

*   **时刻 $T_1$ (上一站)**: 车辆到达站点 $k$。
    *   产生: `obs` ($s$), `action` ($a$), **`snapshot_T1`**（完整全网快照），**`z_t`**（30 维路网特征）。
    *   动作: **暂存 (Cache)** 这些数据，因为此时不知道奖励和下一状态。
    *   *注意: `snapshot_T1` 捕获 $T_1$ 时刻全网状态，用于后续 Reset；`z_t` 在新开单时立即提取。*
*   **时刻 $T_2$ (当前站)**: 车辆到达站点 $k+1$。
    *   产生: `next_obs` ($s'$), `reward` ($r$), **`z_t1`**（30 维路网特征）。
    *   动作: **结算 (Settle)** 上一站的缓存，生成完整 Tuple `(s, a, r, s', z_t, z_t1, snapshot_T1)` 并存入 Buffer。

> **【Pending Cache 放置策略】**
> - **第一版实现（首选）**：cache dict 维护在 **主循环脚本**（`collect_data_sumo.py`）里，风格参考 `sac_ensemble_SUMO_linear_penalty.py` 的 `state_dict`/`action_dict` 多级字典，易于调试。
> - **优化方向（后续重构）**：封装进 `BusSimEnv.stop()` 内部，`step()` 阻塞直到组装好完整 Transition 再返回，对外隐藏异步复杂度。首版暂不做，降低引入新 bug 的风险。

#### B. `collect_data_sumo.py` 核心实现逻辑

```python
# ===========================================================================
# 缓存字典，参考 sac_ensemble_SUMO_linear_penalty.py 风格
# Key = bus_id, Value = T1 时刻的决策上下文
# ===========================================================================
pending_transitions = {}  # {bus_id: {'obs', 'action', 'z_t', 'snapshot_T1', 'timestamp'}}

def on_simulation_step():
    arrived_buses = sumo_env.get_arrived_buses()

    # 若同一仿真步有多辆车同时到站，共享同一次 snapshot 采集（节省开销）
    if arrived_buses:
        shared_snapshot = capture_full_system_snapshot()  # T2 时刻全网快照
        z_t1 = extract_structured_context(shared_snapshot)  # 30 维, z_{t+1}
    
    for bus_id in arrived_buses:
        current_obs  = sumo_env.get_state(bus_id)
        current_time = sumo_env.get_time()

        # --- [结算逻辑: 若该车有挂起的上一站决策] ---
        if bus_id in pending_transitions:
            prev = pending_transitions.pop(bus_id)  # pop 即 GC，无残留
            reward = calculate_reward(prev['obs'], current_obs)  # 复用 bus.py

            replay_buffer.add(
                obs      = prev['obs'],
                action   = prev['action'],
                reward   = reward,
                next_obs = current_obs,
                terminal = False,
                z_t      = prev['z_t'],         # 30 维, T1 出发时路网
                z_t1     = z_t1,                # 30 维, T2 到站时路网（共享）
                snapshot = prev['snapshot_T1'], # 完整原始快照，用于 Reset
                source   = 'real',
            )
        # GC 说明：pending_transitions.pop() 是 O(1) 的隐式 GC。
        # 车辆 terminal 时不挂新决策，因此不会留下悬挂项。
        # 同一仿真步 2 辆车同时到站 → 各自独立结算 + 共享 z_t1，内存不会膨胀。

        # --- [新开单逻辑: 若车辆未到达终点] ---
        if not is_route_end(bus_id):
            snapshot_T1 = capture_full_system_snapshot(ego_id=bus_id)
            z_t         = extract_structured_context(snapshot_T1)  # 30 维, z_t

            action = behavior_policy.get_action(current_obs)
            sumo_env.apply_action(bus_id, action)

            pending_transitions[bus_id] = {
                'obs'         : current_obs,
                'action'      : action,
                'z_t'         : z_t,
                'snapshot_T1' : snapshot_T1,
                'timestamp'   : current_time,
            }
```

#### C. 对 H2O+ 的影响
这种采集方式确保了：
1.  **Reset 有效性**: 用 Buffer 中的 `snapshot_T1` 重置 SimBus 时，SimBus 回到 $T_1$ 时刻，Agent 看到正是 $s$，可以输出新动作产生反事实轨迹。
2.  **Critic 训练正确性**: $r$ 和 $s'$ 真实反映了在真实动力学下执行 $a$ 后的结果。
3.  **Discriminator 输入一致**: Real 和 Sim 侧的 `z_t`/`z_t1` 使用同一 `extract_structured_context()` 函数，且都依赖 `sumo_pos_to_linear()` 做坐标对齐，保证双边特征可比。

*   **输出**: `datasets/sumo_offline_full.hdf5`（含字段 `obs, action, reward, next_obs, z_t, z_t1, snapshot, source`）。

#### 验证：通过训练offline RL,用collect_data_sumo.py和SumoGymWrapper采集的数据训练offline RL，并通过收敛性确认以上工作的完成情况。

---

## Phase 2: 仿真环境改造 (Sim World - /LSTM-RL/env)

**目标**: 改造 `LSTM-RL/env`中的`sim.py`/`bus.py`等依赖，使其支持多线路仿真，和并使其支持"写入快照(Step)"和"读取快照(Reset)"。

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
    *   `info` 必须包含两个 snapshot key：`snapshot_T1`（Ego Bus 上一站出发时捕获）和 `snapshot_T2`（本站到达时捕获），供 H2O+ 训练循环提取 `z_t` 和 `z_t1`。
    
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

### 2.4 Phase 2 实现笔记

> **状态: 已完成.** 包含实际架构、全保真 Schema 以及记录的各种坑。

#### 核心架构

```
sim_core/sim.py
  class env_bus                        # 单线仿真基类
    restore_full_system_snapshot()     # 上帝模式注入 (实现在此，而非 BusSimEnv)
  class MultiLineEnv                   # 多线聚合器
    line_map: {line_id: env_bus}

envs/bus_sim_env.py
  class BusSimEnv(env_bus)             # 单线 + gym 适配
    capture_full_system_snapshot()     # 输出全保真 SnapshotDict
  class MultiLineSimEnv(MultiLineEnv)  # 多线 gym 适配
```

**关键设计**: `restore_full_system_snapshot` 实现在 `env_bus` (基类) 中，而不是 `BusSimEnv`。这是因为 `MultiLineSimEnv.line_map` 中持有的是 `env_bus` 实例，而非 `BusSimEnv` 实例。

#### 全保真 SnapshotDict (包含 14 个公交字段)

`restore_full_system_snapshot` 所需的每辆车字段：
`bus_id(int)`, `trip_id`, `trip_id_list`, `direction(bool)`, `absolute_distance`, `current_speed`, `holding_time`, `forward_headway`, `backward_headway`, `last_station_id(int)`, `next_station_id(int)`, `next_station_dis`, `last_station_dis`, `state(str: TRAVEL/HOLDING/DWELLING)`, `on_route(bool)`, `load(int)`。另外为了 `extract_structured_context` 的兼容性，还包含 `pos` 和 `speed` 别名。

站点字段：`station_id(int)`, `station_name(str)`, `direction(bool)`, `waiting_count(int)`, `pos(float)`。

顶层字段：`sim_time`, `current_time`, `launched_trips: [int]`。

#### SUMO 到 Sim 的快照重置模式 (Snapshot Reset)

```python
# 1. 在 T 时刻从 SUMO 捕获各线路的全保真快照
full_snap = bridge_to_full_snapshot(bridge, edge_map)  # {line_id: SnapshotDict}

# 2. 创建并初始化 sim
sim_env = MultiLineSimEnv(calib_path)
sim_env.reset()

# 3. 注入各线路快照 (仅针对在 SUMO 中有活跃车辆的线路)
for line_id, le in sim_env.line_map.items():
    if line_id in full_snap:
        le.restore_full_system_snapshot(full_snap[line_id])
```

> [!WARNING]
> **已记录的坑 (Documented Pitfalls - 后续信号控制复用时务必参考)**
>
> | # | 问题 | 根因 | 修复方案 |
> |---|---------|------------|-----|
> | 1 | SUMO bridge 静默退出 (code 0, 无事件输出) | `from sumo_env.rl_env import SumoBusHoldingEnv` 在模块级别触发了 `import traci` (=libsumo)，这会与 bridge 自身的 libsumo 会话冲突 | 移除未使用的 rl_env 导入，或改为延迟导入 |
> | 2 | MultiLineSimEnv 报错 `env.done` AttributeError | `self.done` 仅在首次调用 `step()` 后被赋值 (sim.py L513)，`reset()` 并未初始化该属性 | 使用 `step()` 返回值中的 `done`，而非直接访问 `env.done` 属性 |
> | 3 | `route.route_update` 报错 TypeError: 列表索引必须是整数 (float) | `restore_full_system_snapshot` 将 `current_time` 设置为 float 类型，导致 `current_time//3600` 返回 float | 在 route.py 中将索引强制转换为 `int(current_time//3600)` |
> | 4 | 线路环境 (env_bus) 缺少 `restore_full_system_snapshot` 方法 | `MultiLineSimEnv.line_map` 存储的是 `env_bus` 实例，而非 `BusSimEnv` | 将该方法下沉到 `env_bus` 基类中实现 |
> | 5 | 仅有 3/12 条线路在 SUMO 快照中被重置 | SUMO 是事件驱动的；在 T=1388s 时只有 7S/7X/122S 有到战决策事件 | `bridge_to_full_snapshot` 仅生成有活跃车辆的线路快照；未重置的线路按 t=0 默认运行 |
> | 6 | z 特征向量全为 0 | 在调用 `extract_structured_context` 前未调用 `set_route_length()` | 务必在脚本初始化阶段提前调用 `set_route_length()` |

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
    WARMUP_EPISODES = 20  # 或者是前 10% 的训练步数

    offline_buffer.load("sumo_offline_full.pkl")  # 含 z_t, z_t1, snapshot 字段
    online_buffer = ReplayBuffer()

    for episode in range(MAX_EPISODES):
        # --- A. 混合重置策略 ---
        if random.random() < P_RESET (e.g., 0.5):
            # Mode: Buffer Reset (解决 Drift)
            real_batch = offline_buffer.sample(1)
            snapshot   = real_batch['snapshot'][0]   # 完整 SnapshotDict (T1 时刻)
            obs        = sim_env.reset(snapshot=snapshot)
            max_steps  = H_ROLLOUT  # e.g., 20，短程修补
        else:
            # Mode: Standard (全程规划)
            obs       = sim_env.reset(snapshot=None)
            max_steps = FULL_LENGTH
        # z_t 和 z_t1 由 sim_env.step() 返回后在到站事件时提取，episode 开始时不预取

        # --- B. 数据交互与收集 ---
        for t in range(max_steps):
            action = agent.select_action(obs)  # Actor 只看 obs，不看 z（防止信息泄露）
            next_obs, reward, done, info = sim_env.step(action)

            # T2 时刻：Ego Bus 到达下一站后才能提取 z_t 和 z_t1
            z_t  = extract_structured_context(info['snapshot_T1'])  # 出发时(T1)路网, 30维
            z_t1 = extract_structured_context(info['snapshot_T2'])  # 到站时(T2)路网, 30维

            # 事后计算"真实度" w（仅用于 Buffer 存储和截断判断，不泄露给 Actor）
            with torch.no_grad():
                logit     = discriminator(obs, action, next_obs, z_t, z_t1)
                prob_real = torch.sigmoid(logit).item()
                w         = prob_real / (1.0 - prob_real + 1e-8)

            # 核心：动态截断 (Early Exit)
            # 【t>0 规则】Reset 后第一步强制放行：z_t1 刚刚凑齐，Discriminator 冷启动噪声大，
            # 保证至少一步有效 Rollout 进入 Buffer，防止 episode 立即终止。
            if w < W_THRESHOLD and episode > WARMUP_EPISODES and t > 0:
                online_buffer.add(obs, action, reward, next_obs,
                                  z_t=z_t, z_t1=z_t1,
                                  done=False, truncated=True, source='sim')
                break  # 提前跳出，重新从真实快照 Reset
            else:
                online_buffer.add(obs, action, reward, next_obs,
                                  z_t=z_t, z_t1=z_t1,
                                  done=done, truncated=False, source='sim')
                obs = next_obs

            if done: break

            # --- C. H2O+ 更新步骤 (每步或每N步) ---
            if ready_to_train:
                # 1. 采样（两侧 Buffer 均含 z_t, z_t1 字段）
                batch_real = offline_buffer.sample(BATCH_SIZE)
                batch_sim  = online_buffer.sample(BATCH_SIZE)

                # 2. 训练 Discriminator（参考 Appendix C）
                d_loss = update_discriminator(batch_real, batch_sim)

                # 3. 计算 Importance Weight（参考 H2O 论文）
                w = compute_weights(discriminator, batch_sim)

                # 4. 训练 Critic (带权重)
                # 简化版: Loss = w * (Q_sim - Target)^2 + (Q_real - Target)^2
                c_loss = update_critic(batch_real, batch_sim, w)

                # 5. 训练 Actor (标准 SAC，只在 Sim 数据上更新)
                a_loss = update_actor(batch_sim)
    ```
    
> ⚠️ **CRITICAL PROMPT FOR AI CODING ASSISTANTS (关于因果律与防信息泄露):**
> 在编写 Actor 动作选择与 Discriminator 推断逻辑时，必须严格区分“决策时刻 ($T_1$)”与“结算时刻 ($T_2$)”，严禁发生未来信息泄露 (Information Leakage)！
> 1. **动作决策 (Action Selection)**：在 $T_1$ 时刻（公交刚到站），Actor 网络 `policy(obs)` 只能且必须只能接收当前的局部微观观测 `obs_t`。此时 $T_2$ 时刻的路网快照 `z_t1` 尚未产生，绝不能将其作为 Actor 的输入。
> 2. **信度评估 (Weight Calculation)**：Discriminator 的计算 $w = D(obs_t, a_t, obs_{t+1}, z_t, z_{t+1})$ 必须是**事后诸葛亮 (Post-evaluation)**。只有当公交车在 $T_2$ 时刻到达下一站，凑齐了 `z_t`（出发时路网）和 `z_t1`（到站时路网）后，才能进行前向推断。
> 3. **变量边界**：计算出的信度权重 $w$ 仅用于两处：(A) 存入 Buffer，在更新 Critic 损失函数时作为 Importance Weight；(B) 决定是否触发 Early Truncation 放弃该条轨迹的后续 rollout。**绝对不能将 $w$、`z_t` 或 `z_t1` 泄露给当前的 Actor。**



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

### 4.1 已通过的验证 (快照发散测试)

两种方案均已通过，端到端地验证了 z 提取、判别器训练和 w 计算的正确性。

#### 方案 1: z 特征平行对比 (`test_snapshot_headtohead.py`)

SUMO 和 Sim 同时从 t=0 出发，采取零驻留策略。在匹配的 SUMO 决策时刻点提取 z，比较 z 的散度。

| 指标 (Metric) | 起始 (Start) | 终止 (End) | 结果 |
|--------|-------|-----|--------|
| L2 距离 | 12.4 | 187.4 | **通过 (PASS)** |
| 余弦相似度 (cos) | 0.34 | 0.006 | **通过 (PASS)** |
| w_SUMO / w_sim | 9.0 / 0.11 | -- | **通过 (PASS)** |

#### 方案 2: 全保真快照重置 (`test_snapshot_reset.py`)

SUMO 运行到 T=1388 (80 个事件)，捕获全保真快照并注入 Sim，两边以此为起点同时向前运行 100 个事件。

| 指标 (Metric) | 重置瞬间 (At Reset) | 终止 (End) | 结果 |
|--------|----------|-----|--------|
| z 保真度 | L2=1.14, cos=0.92 | -- | **极高 (HIGH)** |
| L2 距离 | 1.14 | 62.1 | **通过 (PASS)** |
| 余弦相似度 (cos) | 0.92 | 0.03 | **通过 (PASS)** |
| w_SUMO / w_sim | 9.0 / 0.11 | -- | **通过 (PASS)** |

> [!IMPORTANT]
> 方案 2 证实了**快照重置的保真度**：在重置瞬间余弦相似度高达 0.92，随后自然发散。这验证了 H2O+ 的核心假设：快照重置能有效消除仿真器的时间漂移。

```bash
# 如何运行验证脚本 (在 H2Oplus/bus_h2o/ 下)
python test_snapshot_headtohead.py --max_events 200 --plot   # 运行约 15s
python test_snapshot_reset.py --T_events 80 --K_events 100 --plot  # 运行约 20s
```

---

## Phase 5: JTT-Enhanced Targeted Snapshot Reset (JTT-增强定向快照重置)

> **依赖**: Phase 3 (H2O+ 训练循环) 已完成, 离线数据 (datasets_v2) 已就绪。
> **参考**: `reference/jtt/` (JTT 原始代码), `reference/jtt_targeted_reset_design.md` (设计路线书), *Just Train Twice: Improving Group Robustness without Training Group Information* (Liu et al., ICML 2021)。

### 5.0 核心洞察: JTT → H2O+ 的跨范式迁移

**JTT (监督学习)** 的核心是两阶段协议:
1. **Phase 1 (Identify)**: 以 ERM 训练 $T$ 个 epoch → 标记**错误集 E** (被误分类的样本)
2. **Phase 2 (Upweight)**: 从头训练, 对 E 中的样本加权 $\lambda$ 倍

> [!IMPORTANT]
> **H2O+ Bus (Sim2Real RL) 的关键创新**:
> 在监督学习中 (JTT), 你只能**加权**已有样本。
> 在 RL 中, 通过快照重置, 我们可以**生成全新的反事实轨迹** — 将仿真器重置到精确的脆弱状态, 用更新后的策略展开全新 rollout。这是 JTT 加权的严格增强 (Active Learning × JTT)。

| JTT 概念 | H2O+ 类比 | 实现方式 |
|-----------|-----------|----------|
| ERM 模型 (Phase 1) | Warmup 策略 | 前 K 个 epoch, 均匀随机重置 |
| 错误集 E | 脆弱转移集 F | `priority_score > threshold` |
| 加权 λ | 重置概率提升 | `P(reset to i) ∝ priority(i)` |
| 从头训练 (Phase 2) | 继续训练 | 同一网络, 定向重置 |
| Group Robustness | Sim-Robustness | 策略在脆弱区域精准改善 |

### 5.1 脆弱性信号定义 (Fragility Signals)

在 JTT 中, 错误集是二值的 `y_pred ≠ y_true`。在 RL 中没有标签, 我们需要代理信号来衡量**策略最脆弱的位置**。

#### 优先级评分: `p(i) = α·TD_err(i) + β·Q_disagree(i) + γ·disc_drift(i)`

| 信号 | 公式 | 直觉 |
|------|------|------|
| **TD Error** | `\|r + γQ(s',a') - Q(s,a)\|` | 价值函数最不准确之处 — "意外" |
| **Q Disagreement** | `\|Q1(s,a) - Q2(s,a)\|` | 双 Q 网络分歧 — 认知不确定性 |
| **Disc. Drift** | `1 - w(z_t, z_{t+1})` | 仿真动力学偏离真实之处 — 动力学间隙 |

> [!NOTE]
> 三个信号在 H2O+ 训练中**已经全部被计算**:
> - TD error: 在 `h2oplus_bus.py::train()` 中计算 Q-loss 时产出
> - Q disagreement: `qf1(s,a) - qf2(s,a)` 来自双 Q 网络
> - 判别器权重 w: 由 `ZOnlyDiscriminator` 在每个训练步计算
>
> 这意味着引入 JTT 的**边际计算开销几乎为零** — 仅需额外维护 `PriorityIndex` 数组。

### 5.2 PriorityIndex 组件设计

```python
# 新增文件: SimpleSAC/priority_index.py

class PriorityIndex:
    """维护离线数据每条转移的优先级评分。
    
    设计原则:
    - 仅追踪离线 (real) 数据, 因为只有离线数据有对应的快照可供重置
    - 使用 EMA 平滑以抵消单 batch 估计的噪声
    - 支持温度控制的概率采样, 从均匀 → 贪心的渐进聚焦
    """
    
    def __init__(self, n_offline: int, alpha=0.5, beta=0.3, gamma=0.2):
        self.n = n_offline
        self.alpha = alpha   # TD error 权重
        self.beta = beta     # Q 分歧权重
        self.gamma = gamma   # 判别器漂移权重
        
        # EMA 平滑的优先级信号 (per-transition)
        self.td_error = np.zeros(n_offline, dtype=np.float32)
        self.q_disagree = np.zeros(n_offline, dtype=np.float32)
        self.disc_drift = np.zeros(n_offline, dtype=np.float32)
        self.update_count = np.zeros(n_offline, dtype=np.int32)
        
    def update(self, indices: np.ndarray, td_err, q_dis, d_drift, ema=0.1):
        """更新采样到的离线转移的优先级评分。
        
        在每个训练 batch 后调用 — 评分通过 EMA 平滑以
        避免单 batch 估计的噪声锯齿。
        """
        for arr, new_val in [(self.td_error, td_err), 
                              (self.q_disagree, q_dis),
                              (self.disc_drift, d_drift)]:
            old = arr[indices]
            arr[indices] = (1 - ema) * old + ema * new_val
        self.update_count[indices] += 1
        
    def sample_reset_idx(self, temperature=1.0) -> int:
        """按优先级的概率采样一个离线索引用于快照重置。
        
        temperature=1.0: 标准优先级采样
        temperature→∞:   均匀随机 (退化为标准 H2O+)
        temperature→0:   始终选最高优先级 (贪心)
        """
        valid = self.update_count > 0
        if not valid.any():
            return np.random.randint(self.n)
            
        p = (self.alpha * self.td_error + 
             self.beta * self.q_disagree + 
             self.gamma * self.disc_drift)
        p = np.clip(p, 0, None)
        p[~valid] = 0
        
        # 温度缩放
        p = p ** (1.0 / max(temperature, 1e-4))
        p = p / (p.sum() + 1e-8)
        
        return np.random.choice(self.n, p=p)
    
    @property
    def priority_scores(self):
        """返回当前加权优先级评分向量 (用于日志和分析)。"""
        return (self.alpha * self.td_error + 
                self.beta * self.q_disagree + 
                self.gamma * self.disc_drift)
```

### 5.3 训练协议: 两阶段 (Two-Phase Protocol)

```
Phase 1: Warmup (Epoch 0..K):
  ├── 均匀随机快照重置 (与原 H2O+ 行为完全一致)
  ├── 累积每条离线转移的优先级评分 (TD error / Q-disagreement / disc drift)
  ├── 不触发 JTT 定向重置 (冷启动保护)
  └── 日志: mean_priority, priority_std, error_set_size

Phase 2: Targeted Reset (Epoch K..N):
  ├── 优先级加权的快照重置 (聚焦脆弱状态)
  ├── 持续更新优先级评分 (它们是演化的！)
  ├── 逐步退火温度: T = T_max * max(T_min/T_max, (1 - (epoch-K)/(N-K))^2)
  │   (起始时宽探索, 逐步聚焦)
  ├── 监控: 平均优先级是否下降? (策略在学习!)
  └── 日志: reset_target_distribution, priority_decay_rate

温度计划 (Temperature Schedule):
  T_max = 2.0  (Phase 2 初期, 宽泛探索)
  T_min = 0.3  (训练末期, 紧密聚焦)
  退火: T = T_max * max(T_min/T_max, (1 - (epoch-K)/(N-K))^2)
```

### 5.4 代码改造清单 (Modification Points)

#### A. `SimpleSAC/h2oplus_bus.py` — 提取脆弱性信号

在 `H2OPlusBus.train()` 的 **Main H2O+ phase** 分支中 (Line ~239), 计算 Q-loss 之后、反向传播之前, 新增优先级信号的收集:

```python
# h2oplus_bus.py::train(), 在 "Main H2O+ phase" 内
# ... 在计算完 real_q1_pred, real_q2_pred, real_td_target 之后 ...

# ── JTT: 提取脆弱性信号 (per-sample, 仅 real batch) ──────────
if self.priority_index is not None:
    with torch.no_grad():
        # 1. TD Error (绝对 Bellman 残差)
        td_err_1 = torch.abs(real_q1_pred.squeeze() - real_td_target).cpu().numpy()
        td_err_2 = torch.abs(real_q2_pred.squeeze() - real_td_target).cpu().numpy()
        td_error = 0.5 * (td_err_1 + td_err_2)
        
        # 2. Q Disagreement (认知不确定性)
        q_disagree = torch.abs(
            real_q1_pred.squeeze() - real_q2_pred.squeeze()
        ).cpu().numpy()
        
        # 3. Discriminator Drift (仿真偏离度)
        # real 数据的 w 应接近 1; 偏离越大说明该区域真实-仿真差距越大
        w_real = compute_z_importance_weight(
            self.discriminator, real_z_t, real_z_t1
        ).squeeze()
        disc_drift = (1.0 - w_real.clamp(0, 10) / 10.0).cpu().numpy()
        
    # 更新优先级索引 (batch_indices 需从 buffer.sample 返回)
    self.priority_index.update(
        real_batch["_indices"],  # 离线数据中的原始索引
        td_error, q_disagree, disc_drift
    )
```

> [!IMPORTANT]
> **`real_batch["_indices"]` 依赖**: `BusMixedReplayBuffer.sample()` 需要新增返回采样索引的功能。在 `scope="real"` 时, 将 `ind` 数组作为 `"_indices"` 字段附加到返回字典中。这是最小侵入式的改动 — 仅在 `sample()` 函数末尾添加一行 `batch["_indices"] = torch.LongTensor(ind).to(self.device)`。

#### B. `SimpleSAC/bus_sampler.py` — 定向重置

在 `BusStepSampler.sample()` 的 episode 起始部分, 修改 buffer reset 逻辑:

```python
# bus_sampler.py::sample(), episode start 部分
if use_buffer_reset:
    if (self._episode_count > self.warmup_episodes 
        and self.priority_index is not None):
        # ── JTT Phase 2: 定向重置到脆弱状态 ──
        idx = self.priority_index.sample_reset_idx(
            temperature=self.current_temperature
        )
        snapshot, _, _ = self.replay_buffer.sample_snapshot_by_idx(idx)
    else:
        # ── Phase 1: 均匀随机重置 (标准 H2O+) ──
        snapshot, _, _ = self.replay_buffer.sample_snapshot()
    
    self.env.reset(snapshot=snapshot)
    max_events = self.h_rollout
```

> [!CAUTION]
> **`sample_snapshot_by_idx(idx)` 依赖**: `BusMixedReplayBuffer` 需要新增一个方法, 根据指定的离线索引加载快照。当前的 `sample_snapshot()` 是均匀随机的; 新方法接受 `idx` 参数, 直接从 `self._snapshot_bytes[idx]` 加载。如果该索引没有有效快照 (None), 回退到最近邻有快照的索引。

#### C. `SimpleSAC/h2o+_bus_main.py` — 编排两阶段协议

```python
# h2o+_bus_main.py, FLAGS_DEF 新增:
jtt_warmup_epochs=50,         # Phase 1 → Phase 2 切换点 (K)
jtt_alpha=0.5,                # TD error 权重
jtt_beta=0.3,                 # Q disagreement 权重
jtt_gamma=0.2,                # discriminator drift 权重
jtt_temperature_max=2.0,      # 重置采样宽度 (起始)
jtt_temperature_min=0.3,      # 重置采样紧密度 (终止)
jtt_ema_decay=0.1,            # 优先级评分 EMA 平滑系数

# main() 中:
from priority_index import PriorityIndex

priority_index = PriorityIndex(
    n_offline=replay_buffer.fixed_dataset_size,
    alpha=FLAGS.jtt_alpha,
    beta=FLAGS.jtt_beta,
    gamma=FLAGS.jtt_gamma,
)

# 注入到 H2OPlusBus
h2o.priority_index = priority_index

# 注入到 BusStepSampler
train_sampler.priority_index = priority_index

# 训练循环中, 每 epoch 更新温度:
for epoch in trange(FLAGS.n_epochs):
    # 温度退火
    if epoch >= FLAGS.jtt_warmup_epochs:
        progress = (epoch - FLAGS.jtt_warmup_epochs) / max(
            FLAGS.n_epochs - FLAGS.jtt_warmup_epochs, 1
        )
        temp = FLAGS.jtt_temperature_max * max(
            FLAGS.jtt_temperature_min / FLAGS.jtt_temperature_max,
            (1.0 - progress) ** 2
        )
    else:
        temp = float('inf')  # 均匀随机 (Phase 1)
    train_sampler.current_temperature = temp
    
    # ... 原有的 rollout + train + eval 循环 ...
    
    # JTT 日志
    if epoch >= FLAGS.jtt_warmup_epochs:
        scores = priority_index.priority_scores
        valid_scores = scores[priority_index.update_count > 0]
        if len(valid_scores) > 0:
            threshold = np.percentile(valid_scores, 80)
            metrics["jtt/mean_priority"] = valid_scores.mean()
            metrics["jtt/priority_std"] = valid_scores.std()
            metrics["jtt/error_set_size"] = int((valid_scores > threshold).sum())
            metrics["jtt/temperature"] = temp
```

#### D. `SimpleSAC/bus_replay_buffer.py` — 最小扩展

需要两个改动:

```python
# 1. sample() 返回采样索引
def sample(self, batch_size, scope=None, type=None):
    # ... 原有逻辑 ...
    batch["_indices"] = torch.LongTensor(ind).to(self.device)  # 新增
    return batch

# 2. 新增 sample_snapshot_by_idx()
def sample_snapshot_by_idx(self, idx: int):
    """根据指定的离线索引加载快照 (JTT 定向重置用)。
    
    如果该索引没有有效快照, 回退到最近的有快照的索引。
    """
    if idx < len(self._snapshot_bytes) and self._snapshot_bytes[idx] is not None:
        snap_bytes = self._snapshot_bytes[idx]
    elif self._valid_snap_indices:
        # 回退: 找最近的有效快照索引
        distances = np.abs(np.array(self._valid_snap_indices) - idx)
        nearest = self._valid_snap_indices[np.argmin(distances)]
        snap_bytes = self._snapshot_bytes[nearest]
        idx = nearest
    else:
        raise RuntimeError("No valid snapshots in offline data")
    
    snapshot_dict = pickle.loads(snap_bytes)
    obs = self.state[idx].copy()
    z_t = self.z_t[idx].copy()
    return snapshot_dict, obs, z_t
```

### 5.5 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `jtt_warmup_epochs` (K) | 50 | Phase 1 均匀重置的 epoch 数 |
| `alpha` | 0.5 | TD error 在优先级评分中的权重 |
| `beta` | 0.3 | Q disagreement 的权重 |
| `gamma` | 0.2 | 判别器漂移的权重 |
| `temperature_max` | 2.0 | Phase 2 起始采样温度 (宽) |
| `temperature_min` | 0.3 | 训练末期采样温度 (窄) |
| `ema_decay` | 0.1 | 优先级 EMA 平滑系数 |

### 5.6 与 JTT 原始代码的对应关系

```
JTT 代码 (reference/jtt/)              →  H2O+ Bus 对应
─────────────────────────────────────      ──────────────────────────────────────
process_training.py:                       PriorityIndex.update():
  train_df["wrong_1_times"]                  td_error[], q_disagree[], disc_drift[]
  (二值: 对/错)                              (连续: EMA 平滑的三信号组合)

generate_downstream.py:                    h2o+_bus_main.py:
  up_weights = [20, 50, 100]                 温度退火: T_max=2.0 → T_min=0.3
  Subset(train_data, aug_indices * λ)        priority_index.sample_reset_idx(T)

run_expt.py:                               h2oplus_bus.py::train():
  Phase 2 ERM 从头训练                        Phase 2 继续训练 + 定向重置
  ConcatDataset + upsampled_points           不复制数据, 直接调整 P(reset)

loss.py: LossComputer:                     h2oplus_bus.py:
  compute_group_avg(losses, group_idx)       priority_index 按转移追踪 (非 group)
  adv_probs 在 group 间再平衡                 温度控制在转移间再平衡
```

### 5.7 验证计划 (Verification)

1. **优先级覆盖率**: 训练 K 个 epoch 后, 检查 `priority_index.update_count` — 应有 >80% 的离线转移被至少采样过一次。若覆盖不足, 增大 `real_batch_size` 或延长 warmup。

2. **优先级下降曲线**: Phase 2 开始后, `mean_priority` 应逐 epoch 下降 — 策略正在学习解决脆弱区域。如果 `mean_priority` 不降反升, 说明定向重置产生的新数据反而增加了该区域的 TD error, 需调低温度退火速率。

3. **重置分布可视化**: 记录每个 epoch 重置的离线索引, 绘制直方图。Phase 1 应为均匀分布, Phase 2 应逐步向少数高优先级索引集中。

4. **消融实验**:
   - **JTT vs Uniform**: 比较 JTT 定向重置 vs 原始均匀随机重置 (温度 = ∞) 的训练曲线与最终性能。
   - **信号消融**: 分别设 α=1,β=γ=0 (TD only) / β=1,α=γ=0 (Q-disagree only) / γ=1,α=β=0 (disc only), 对比哪个信号贡献最大。
   - **温度敏感性**: 固定 K, 对比不同 T_min (0.1, 0.3, 0.5, 1.0) 的效果。

### 5.8 Open Questions

> [!IMPORTANT]
> 1. **优先级评分范围**: 应对所有离线数据计算优先级, 还是仅限有有效快照的转移?
>    - 推荐: 对所有离线数据计算 (因为 TD error 和 Q-disagreement 不依赖快照), 但 `sample_reset_idx` 仅在有快照的子集中采样。
>
> 2. **两阶段 vs 连续退火**: Phase 1/2 的硬切换 vs 从第 0 个 epoch 就开始退火 (从 T=∞ 缓降)?
>    - 硬切换更安全 (冷启动时优先级信号噪声大), 但连续退火更平滑。首版用硬切换。
>
> 3. **快照可用性**: 当前 merged_all_v2.h5 不含 snapshot_T1 字段 (仅 z_t/z_t1)。定向重置需要:
>    - 方案 A: 重新采集含 snapshot 的 HDF5 数据 (推荐, 最稳健)
>    - 方案 B: 用 z_t 反向构造近似快照 (不精确, 损失重置保真度)
>    - 方案 C: 仅对有 snapshot 的 legacy 数据集启用 JTT, merged 数据集走 uniform reset

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

    real_obs      = real_batch['obs']
    real_act      = real_batch['action']
    real_next_obs = real_batch['next_obs']
    real_z_t      = real_batch['z_t']    # (Batch, 30) — T1 快照
    real_z_t1     = real_batch['z_t1']   # (Batch, 30) — T2 快照

    # [关键防御 1: 数据增强] 为真实路网特征注入 5% 高斯噪声
    # 迫使判别器学习"拥堵的空间分布"而非"绝对数值"
    noise_t  = torch.randn_like(real_z_t)  * 0.05
    noise_t1 = torch.randn_like(real_z_t1) * 0.05

    real_logits = discriminator(real_obs, real_act, real_next_obs,
                                real_z_t + noise_t, real_z_t1 + noise_t1)

    # [关键防御 2: 标签平滑] 真实数据标签设为 0.9，而非绝对的 1.0
    real_labels = torch.full_like(real_logits, 0.9)
    loss_real   = criterion(real_logits, real_labels)

    # ---------------------------------------------------------
    # 2. 仿真数据 (SimBus) 前向传播
    # ---------------------------------------------------------
    sim_batch = online_buffer.sample(batch_size)

    sim_obs      = sim_batch['obs']
    sim_act      = sim_batch['action']
    sim_next_obs = sim_batch['next_obs']
    sim_z_t      = sim_batch['z_t']      # (Batch, 30)
    sim_z_t1     = sim_batch['z_t1']     # (Batch, 30)

    sim_logits  = discriminator(sim_obs, sim_act, sim_next_obs, sim_z_t, sim_z_t1)

    # 仿真数据标签设为 0.1，而非绝对的 0.0
    sim_labels  = torch.full_like(sim_logits, 0.1)
    loss_sim    = criterion(sim_logits, sim_labels)

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