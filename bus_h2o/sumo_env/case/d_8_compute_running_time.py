# 功能：划分不同路段元素并且考虑交叉口延误，计算公交车站间运行时间
# 时间：2025.03.23
# 1) 在"d_8_compute_running_time_单周期同行OK"的基础上进行修改，那个计算复杂度高，而且若当前周期无法同行，程序还存在问题，所以本程序先更改为交叉口是 平均延误时间

try:
    import gurobipy
except ImportError:
    print("Warning: gurobipy module not found, optimization functionality may be limited")
    gurobipy = None
import copy

def get_sorted_busline_edge(edge_file):
    """获取各公交线路的路段ID构成的字典"""
    involved_tl_ID_l = []
    busline_tl_time_d = {}  # 保存各线路要经过的信号灯及对应相位时长及信号周期时长
    sorted_busline_edge_d = {}
    stop_edge_d = {}  # 保存各线路的公交站对应的路段ID
    edge_root = edge_file.getroot()
    for busline_child in edge_root:
        if busline_child.tag == "busline" and busline_child.get("id") not in sorted_busline_edge_d:
            sorted_busline_edge_d[busline_child.get("id")] = {}
            stop_edge_d[busline_child.get("id")] = {}
            busline_tl_time_d[busline_child.get("id")] = {}

            for element_child in busline_child:
                if element_child.tag == "element" and element_child.get("id") not in sorted_busline_edge_d[busline_child.get("id")]:
                    if element_child.get("type") == "w":
                        sorted_busline_edge_d[busline_child.get("id")][element_child.get("id")] = [element_child.get("type"), element_child.get("length"),
                                                                                                   element_child.get("dedicated_flag"), element_child.get("impacted_edge"),
                                                                                                   element_child.get("impacted_length")]
                    elif element_child.get("type") == "s":
                        sorted_busline_edge_d[busline_child.get("id")][element_child.get("id")] = [element_child.get("type"), element_child.get("length"),
                                                                                                   element_child.get("dedicated_flag"), element_child.get("stop_id")]
                        stop_edge_d[busline_child.get("id")][element_child.get("stop_id")] = element_child.get("id")
                    else:
                        sorted_busline_edge_d[busline_child.get("id")][element_child.get("id")] = [element_child.get("type"), element_child.get("length"),
                                                                                                   element_child.get("dedicated_flag"), element_child.get("tl"),
                                                                                                   element_child.get("phase_time"), element_child.get("cycle_time"),
                                                                                                   element_child.get("from_lane"), element_child.get("dir")]
                        if element_child.get("tl"):  # 若终点站过后的路段是以交叉口结尾的则不算，因为行程已经结束
                            involved_tl_ID_l.append(element_child.get("tl"))
                            busline_tl_time_d[busline_child.get("id")][element_child.get("tl")] = [element_child.get("phase_time"), element_child.get("cycle_time")]
    return sorted_busline_edge_d, list(set(involved_tl_ID_l)), stop_edge_d, busline_tl_time_d


def compute_tl_delay_time(transferred_trigger_time_var, trigger_time, bus_id, cur_edge_id, cur_edge_remaining_distance, tl_info_l, tl, w_l,
                          bus_saved_queue_rear_l, traffic_flow_speed, pass_tl_edge_var, MaxTime, GurobiModel):
    """计算交叉口延误子函数"""

    """计算延误模型等效触发时刻所在信号周期的红灯起始时刻以及绿灯起始时刻"""
    # 获取交叉口相位时长和信号周期时长
    tl_phase_time = tl_info_l[2]
    tl_cycle_time = tl_info_l[3]

    # 以trigger_time所处信号周期为第一个信号周期，则transferred_trigger_time_var时刻，所处信号周期为第n_transferred_cycle_var+1个
    n_transferred_cycle_var = GurobiModel.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, name=f"n_transferred_cycle_{bus_id}_{cur_edge_id}")
    GurobiModel.addConstr(n_transferred_cycle_var * tl_cycle_time <= transferred_trigger_time_var - (trigger_time - tl_info_l[5]),
                          name=f"n_transferred_cycle_<=_{bus_id}_{cur_edge_id}")
    GurobiModel.addConstr((n_transferred_cycle_var + 1) * tl_cycle_time >= transferred_trigger_time_var - (trigger_time - tl_info_l[5]),
                          name=f"n_transferred_cycle_>=_{bus_id}_{cur_edge_id}")

    # 相对trigger_time，计算transferred_trigger_time_var时刻，该交叉口的第一个有效红灯、绿灯相位起始时间
    tl1_red_start_var = GurobiModel.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=MaxTime, name=f"tl1_red_var_{bus_id}_{cur_edge_id}")
    tl1_green_start_var = GurobiModel.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=MaxTime, name=f"tl1_green_var_{bus_id}_{cur_edge_id}")
    GurobiModel.addConstr(tl1_red_start_var == trigger_time - tl_info_l[5] + n_transferred_cycle_var * tl_cycle_time, name=f"tl1_red_start_var_{bus_id}_{cur_edge_id}")
    GurobiModel.addConstr(tl1_green_start_var == tl1_red_start_var + tl_cycle_time - tl_phase_time, name=f"tl1_green_start_var_{bus_id}_{cur_edge_id}")

    """如果公交车还未到达队尾，则计算其到达队尾的时空信息"""

    # 计算该交叉口可以消散的最大排队长度
    ML = (-1) * w_l[1] * w_l[2] * tl_phase_time / (w_l[2] - w_l[1])

    n_arr_queue_rear_var = GurobiModel.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, name=f"n_arr_queue_rear_{bus_id}_{cur_edge_id}")  # 以等效触发时刻所处周期为第一个信号周期，计算公交车在第几个信号周期到达队尾
    d_arrive_queue_rear_var = GurobiModel.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=10000, name=f"d_arrive_queue_rear_var_{bus_id}_{cur_edge_id}")  # 计算公交车到达队尾的距离
    if not bus_saved_queue_rear_l[0]:  # 非空表明该公交车已经进行了排队
        """基于实际触发时刻trigger_time_var的相关信息，计算1）trigger_time_var所处信号周期的最大排队长度；2）该交叉口是否有排队剩余"""
        # 判断触发时刻trigger_time_var处于排队阶段还是离去阶段
        if tl_info_l[5] >= tl_info_l[3] - tl_info_l[2]:  # 表示触发时刻该交叉口处于红灯相位
            wt = w_l[0]
        else:  # 表示触发时刻该交叉口处于绿灯相位
            wt = w_l[2]

        # 计算对应交叉口在有效信号周期内的最大排队长度d_1_2
        # d_1_2 = GurobiModel.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=10000, name=f"d_1_2_{bus_id}_{cur_edge_id}")
        # t_1_2 = GurobiModel.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=trigger_time, ub=MaxTime, name=f"t_1_2_{bus_id}_{cur_edge_id}")
        # trigger_time_tl1_green_start = trigger_time + (tl_info_l[3] - tl_info_l[5]) - tl_info_l[2]
        # GurobiModel.addConstr(d_1_2 == w_l[1] * (wt * (trigger_time_tl1_green_start - trigger_time) - tl_info_l[6]) / (wt - w_l[1]), name=f"d_1_2==_{bus_id}_{cur_edge_id}")
        # GurobiModel.addConstr(t_1_2 == trigger_time_tl1_green_start - d_1_2 / w_l[1], name=f"t_1_2==_{bus_id}_{cur_edge_id}")
        # 判断该交叉口是否有排队剩余 # spillover_flag = 1 表示有排队剩余
        # spillover_flag = GurobiModel.addVar(vtype=gurobipy.GRB.BINARY, name=f"spillover_flag_{bus_id}_{cur_edge_id}")
        # GurobiModel.addConstr(ML <= d_1_2 * spillover_flag + 1E6 * (1 - spillover_flag), name=f"ML <=_{bus_id}_{cur_edge_id}")
        # GurobiModel.addConstr(ML >= d_1_2 * (1 - spillover_flag) - 1E6 * spillover_flag)


        # 计算对应交叉口在有效信号周期内的最大排队长度d_1_2，好像没有变量参与
        trigger_time_tl1_green_start = trigger_time + (tl_info_l[3] - tl_info_l[5]) - tl_info_l[2]  # 实际触发时刻所处信号周期的绿灯起始时间
        trigger_time_tl1_red_start = trigger_time - tl_info_l[5]  # 实际触发时刻所处信号周期的红灯起始时间
        # d_1_2 = w_l[1] * (wt * (trigger_time_tl1_green_start - trigger_time) - tl_info_l[6]) / (wt - w_l[1])
        d_1_2 = ML  # 假设所有交叉口都没有排队剩余！！！！！！！！！！！！！！！！！！！
        t_1_2 = trigger_time_tl1_green_start - d_1_2 / w_l[1]
        # 判断该交叉口是否有排队剩余  # spillover_flag = 1 表示有排队剩余  好像没有变量参与
        if ML < d_1_2:
            spillover_flag = 1   # LRG  等于1的时候有问题，暂时还没发现问题之处！！！！！！！！！！
        else:
            spillover_flag = 0

        """LRG !!!如果d_1_2大于cur_edge_remaining_distance，则手动修改cur_edge_remaining_distance为d_1_2+10(当前存在这个问题，理论上如果超过了，应当对上个路段的离开时间增加，因为排队延误已经影响至上个路段了)"""
        if cur_edge_remaining_distance < d_1_2:
            cur_edge_remaining_distance = d_1_2 + 10

        """等效的触发时刻所处信号周期的t_1_2取值"""
        transferred_t_1_2_var = GurobiModel.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=MaxTime, name=f"transferred_t_1_2_{bus_id}_{cur_edge_id}")
        GurobiModel.addConstr(transferred_t_1_2_var == t_1_2 + tl_cycle_time * n_transferred_cycle_var, name=f"transferred_t_1_2_var_{bus_id}_{cur_edge_id}")

        """计算公交车在第几个信号周期到达队尾"""
        # 公交车轨迹与各周期最大排队长度形成的直线方程的交点  距离交叉口的距离
        meeting_point_distance_var = GurobiModel.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=10000, name=f"meeting_point_distance_var_{bus_id}_{cur_edge_id}")
        GurobiModel.addConstr(meeting_point_distance_var * (w_l[0] * (transferred_t_1_2_var - tl1_red_start_var) - traffic_flow_speed * tl_cycle_time + ML) ==
                              (-1) * traffic_flow_speed * (tl_cycle_time * d_1_2 + (transferred_trigger_time_var - transferred_t_1_2_var) *
                                                           (w_l[0] * tl1_red_start_var - w_l[0] * transferred_t_1_2_var - ML)) +
                              cur_edge_remaining_distance * (w_l[0] * transferred_t_1_2_var - w_l[0] * tl1_red_start_var + ML)
                              , name=f"meeting_point_distance_var_{bus_id}_{cur_edge_id}")
        # GurobiModel.addConstr(meeting_point_distance_var == 100, name=f"meeting_point_distance_var_{bus_id}_{cur_edge_id}")

        # 无排队剩余时
        if spillover_flag == 0:
            GurobiModel.addConstr(n_arr_queue_rear_var * traffic_flow_speed * tl_cycle_time >= cur_edge_remaining_distance +
                                  traffic_flow_speed * (transferred_trigger_time_var - tl1_red_start_var), name=f"n_arr_queue_rear_var1>=_{bus_id}_{cur_edge_id}")
            GurobiModel.addConstr((n_arr_queue_rear_var - 1) * traffic_flow_speed * tl_cycle_time <= cur_edge_remaining_distance +
                                  traffic_flow_speed * (transferred_trigger_time_var - tl1_red_start_var), name=f"n_arr_queue_rear_var1<=_{bus_id}_{cur_edge_id}")
        else:  # 有排队剩余时
            GurobiModel.addConstr((n_arr_queue_rear_var - 1) * (ML + w_l[0] * (transferred_t_1_2_var - tl1_red_start_var)) >= d_1_2 - meeting_point_distance_var,
                                  name=f"n_arr_queue_rear_var2>=_{bus_id}_{cur_edge_id}")
            GurobiModel.addConstr((n_arr_queue_rear_var - 2) * (ML + w_l[0] * (transferred_t_1_2_var - tl1_red_start_var)) <= d_1_2 - meeting_point_distance_var,
                                  name=f"n_arr_queue_rear_var2>=_{bus_id}_{cur_edge_id}")

        # 计算公交车到达队尾时，距离交叉口的距离
        GurobiModel.addConstr(d_arrive_queue_rear_var * (traffic_flow_speed - w_l[0]) == traffic_flow_speed *
                              (w_l[0] * (transferred_t_1_2_var - transferred_trigger_time_var + n_arr_queue_rear_var * tl_cycle_time - tl_cycle_time) + d_1_2 -
                               spillover_flag * (n_arr_queue_rear_var - 1) * (ML + w_l[0] * transferred_t_1_2_var)) - w_l[0] * cur_edge_remaining_distance,
                              name=f"d_arrive_queue_rear_var1_{bus_id}_{cur_edge_id}")

    else:
        GurobiModel.addConstr(n_arr_queue_rear_var == 1, name=f"n_arr_queue_rear_var == 0_{bus_id}_{cur_edge_id}")
        GurobiModel.addConstr(d_arrive_queue_rear_var == tl_info_l[-1], name=f"d_arrive_queue_rear_var2_{bus_id}_{cur_edge_id}")

    """基于公交车到达队尾的信息，获取公交车通过交叉口的时间"""
    # 从公交车到达队尾开始，公交车在第几个信号周期通过交叉口
    n_pass_tl_var = GurobiModel.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, name=f"n_from_rear_pass_tl_var_{bus_id}_{cur_edge_id}")
    GurobiModel.addConstr((n_pass_tl_var - n_arr_queue_rear_var + 1) * ML >= d_arrive_queue_rear_var, name=f"n_from_rear_pass_tl_var>=_{bus_id}_{cur_edge_id}")
    GurobiModel.addConstr((n_pass_tl_var - n_arr_queue_rear_var) * ML <= d_arrive_queue_rear_var, name=f"n_from_rear_pass_tl_var<=_{bus_id}_{cur_edge_id}")

    # 公交车通过交叉口所处信号周期的绿灯相位开始时间
    tln_green_start_time_var = GurobiModel.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=MaxTime, name=f"tln_green_start_time_var_{bus_id}_{cur_edge_id}")
    GurobiModel.addConstr(tln_green_start_time_var == tl1_red_start_var + n_pass_tl_var * tl_cycle_time - tl_phase_time,
                          name=f"tln_green_start_time_var ==_{bus_id}_{cur_edge_id}")
    GurobiModel.addConstr((pass_tl_edge_var - tln_green_start_time_var) * (w_l[1] * traffic_flow_speed) == (w_l[1] - traffic_flow_speed) *
                          (d_arrive_queue_rear_var + (n_arr_queue_rear_var-n_pass_tl_var) * ML), name=f"pass_tl_var_{bus_id}_{cur_edge_id}")

    # 增加一个时间约束
    GurobiModel.addConstr(pass_tl_edge_var >= transferred_trigger_time_var - 1, name=f"pass_tl_edge_var>_{bus_id}_{cur_edge_id}")


def compute_station_interval_running_time(arrive_time_var, last_stop_depart_time_var, trigger_time, trigger_next_stop_edge_index, bus_id, cur_edge_id, cur_edge_remaining_distance, next_stop_id,
                                          next_stop_edge_id, sorted_busline_edge_d, w_l, bus_saved_queue_rear_l, decision_variable_inv_d,
                                          traffic_flow_speed, MaxTime, subsequent_tl_info_d, GurobiModel):
    """计算公交车站间运行时间"""
    interval_edge_depart_time = {}
    # 获取公交车由当前位置至下一个公交站，需要经过的所有edge元素
    busline_id = bus_id.split("_")[0]
    busline_edge_l = list(sorted_busline_edge_d[busline_id].keys())
    cur_edge_index = busline_edge_l.index(cur_edge_id)
    next_stop_edge_index = busline_edge_l.index(next_stop_edge_id)
    interval_edge_l = copy.deepcopy(busline_edge_l[cur_edge_index:next_stop_edge_index])

    # 计算多个路段交通流速度取均值之后的整体的交通流速度
    impacted_edge_mean_traffic_flow_speed = 0
    for temp_edge in interval_edge_l:
        impacted_edge_mean_traffic_flow_speed += traffic_flow_speed[busline_id][temp_edge]
    impacted_edge_mean_traffic_flow_speed /= len(interval_edge_l)

    # 决策速度应该不超过低于交通流速度
    GurobiModel.addConstr(decision_variable_inv_d[bus_id, next_stop_id] * impacted_edge_mean_traffic_flow_speed >= 1,
                          name=f"decision_variable<mean_traffic_flow_speed_{bus_id}_{next_stop_id}")

    # 计算公交车服务完各edge所需要的时间
    cur_edge_remaining_distance = float(cur_edge_remaining_distance)
    temp_edge = interval_edge_l[0]
    while temp_edge in interval_edge_l:
        # 表示触发时刻公交车处于车站，注意1）该部分是计算站间运行时间的，所以即使触发时刻公交车处于公交站，也只会在下一个路段元素开始计算；2）interval_edge_l以下一个公交站的前一个路段元素结尾）
        if sorted_busline_edge_d[busline_id][temp_edge][0] == "s":
            temp_edge = busline_edge_l[cur_edge_index + 1]

        elif sorted_busline_edge_d[busline_id][temp_edge][0] == "w":  # 表示公交车处于w类型的edge元素
            # 获取其可以合并至的edge的ID并创建变量
            impacted_edge = sorted_busline_edge_d[busline_id][temp_edge][-2]  # 当前路段可以合并至某个路段
            interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] = GurobiModel.addVar(
                vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=MaxTime,
                name=f"interval_edge_depart_time_{busline_id}_{bus_id}_{next_stop_id}_{impacted_edge}")

            temp_edge_length = float(sorted_busline_edge_d[busline_id][temp_edge][1])
            impacted_length = float(sorted_busline_edge_d[busline_id][temp_edge][-1])
            temp_edge_index = busline_edge_l.index(temp_edge)
            impacted_edge_index = busline_edge_l.index(sorted_busline_edge_d[busline_id][temp_edge][-2])

            # 后面合并的impacted_edge是w类型路段
            if sorted_busline_edge_d[busline_id][impacted_edge][0] == "w":
                # if sorted_busline_edge_d[busline_id][impacted_edge][2] == "1":  # 表示布设了公交专用道  (下面这里如果不是第一个，那么就不是trigger——time)
                if temp_edge == cur_edge_id:  # 表示在处理 触发时刻公交车处于的w类型路段
                    GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] == last_stop_depart_time_var +  # 注意这里控制速度没有细分到元素，还是站间速度
                                          decision_variable_inv_d[bus_id, next_stop_id] * (impacted_length - temp_edge_length + cur_edge_remaining_distance),
                                          name=f"interval_edge_depart_time_w1_{bus_id}_{temp_edge}")
                else:  # LRG interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]]感觉可以更换为某个信号灯路段或者公交站离开时间，因为前一个被打断的只能是信号灯和公交站这两种情况
                    GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] ==
                                          interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] +  # 注意这里控制速度没有细分到元素，还是站间速度
                                          decision_variable_inv_d[bus_id, next_stop_id] * impacted_length,
                                          name=f"interval_edge_depart_time_w2_{bus_id}_{temp_edge}")

                # else:  # 没有布设公交专用道，无法控制公交车速度，故采用交通流速度
                #     if temp_edge == cur_edge_id:  # 表示处理的第一个edge元素
                #         GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] == last_stop_depart_time_var +
                #                               (impacted_length - temp_edge_length + cur_edge_remaining_distance)/traffic_flow_speed[busline_id][temp_edge],
                #                               name=f"interval_edge_depart_time_w3_{bus_id}_{temp_edge}")  # 本质有多个路段，这里使用impacted_edge来近似
                #     else:  # LRG interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] 感觉可以更换为某个信号灯路段或者公交站离开时间，因为前一个被打断的只能是信号灯和公交站这两种情况
                #         GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] ==
                #                               interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] +
                #                               impacted_length/traffic_flow_speed[busline_id][temp_edge],
                #                               name=f"interval_edge_depart_time_w4_{bus_id}_{temp_edge}")

            # 后面合并的impacted_edge是i类型路段
            else:
                tl = sorted_busline_edge_d[busline_id][impacted_edge][3]
                # 在线路edge列表中，cur_edge_id的索引是否早于触发时刻公交车下一个服务车站的索引，只有早于的情况下，trigger_time才会等于等效触发时间transferred_trigger_time_var
                if temp_edge == cur_edge_id:
                    GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] == last_stop_depart_time_var +  # 注意这里控制速度没有细分到元素，还是站间速度
                                          (impacted_length - temp_edge_length + cur_edge_remaining_distance) * decision_variable_inv_d[bus_id, next_stop_id]+
                                          subsequent_tl_info_d[tl][7],  # subsequent_tl_info_d[tl][7]为交叉口改相位的平均延误时间
                                          name=f"interval_edge_depart_time_w5_{bus_id}_{temp_edge}")
                else:
                    GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] ==
                                          interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] +  # 注意这里控制速度没有细分到元素，还是站间速度
                                          impacted_length * decision_variable_inv_d[bus_id, next_stop_id] + subsequent_tl_info_d[tl][7],  # subsequent_tl_info_d[tl][7]为交叉口改相位的平均延误时间,
                                          name=f"interval_edge_depart_time_w6_{bus_id}_{temp_edge}")

            # impacted_edge_index = busline_edge_l.index(impacted_edge)
            temp_edge = busline_edge_l[impacted_edge_index + 1]

        # i类型的edge元素
        else:
            interval_edge_depart_time[busline_id, bus_id, next_stop_id, temp_edge] = GurobiModel.addVar(
                vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=MaxTime,
                name=f"interval_edge_depart_time_{busline_id}_{bus_id}_{next_stop_id}_{temp_edge}")

            tl = sorted_busline_edge_d[busline_id][temp_edge][3]
            if temp_edge == cur_edge_id:  # 表示在处理 触发时刻公交车处于的i类型路段
                GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, temp_edge] == last_stop_depart_time_var +  # 注意这里控制速度没有细分到元素，还是站间速度
                                      cur_edge_remaining_distance * decision_variable_inv_d[bus_id, next_stop_id] +
                                      subsequent_tl_info_d[tl][7],  # subsequent_tl_info_d[tl][7]为交叉口改相位的平均延误时间
                                      name=f"interval_edge_depart_time_t1_{bus_id}_{temp_edge}")
            else:
                temp_edge_length = float(sorted_busline_edge_d[busline_id][temp_edge][1])
                GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, temp_edge] ==
                                      interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] +  # 注意这里控制速度没有细分到元素，还是站间速度
                                      temp_edge_length * decision_variable_inv_d[bus_id, next_stop_id] +
                                      subsequent_tl_info_d[tl][7],  # subsequent_tl_info_d[tl][7]为交叉口改相位的平均延误时间
                                      name=f"interval_edge_depart_time_t2_{bus_id}_{temp_edge}")

            temp_edge_index = busline_edge_l.index(temp_edge)
            temp_edge = busline_edge_l[temp_edge_index + 1]

    # 站间路段运行过程结束
    GurobiModel.addConstr(arrive_time_var == interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[-1]],
                          name=f"arrive_time_var_{bus_id}_{temp_edge}")



def compute_station_interval_running_time_dedicatedlane(arrive_time_var, last_stop_depart_time_var, trigger_time, trigger_next_stop_edge_index, bus_id, cur_edge_id, cur_edge_remaining_distance, next_stop_id,
                                          next_stop_edge_id, sorted_busline_edge_d, w_l, bus_saved_queue_rear_l, decision_variable_inv_d,
                                          traffic_flow_speed, MaxTime, subsequent_tl_info_d, GurobiModel):
    """计算公交车站间运行时间"""
    interval_edge_depart_time = {}
    # 获取公交车由当前位置至下一个公交站，需要经过的所有edge元素
    busline_id = bus_id.split("_")[0]
    busline_edge_l = list(sorted_busline_edge_d[busline_id].keys())
    cur_edge_index = busline_edge_l.index(cur_edge_id)
    next_stop_edge_index = busline_edge_l.index(next_stop_edge_id)
    interval_edge_l = copy.deepcopy(busline_edge_l[cur_edge_index:next_stop_edge_index])

    # 计算多个路段交通流速度取均值之后的整体的交通流速度
    impacted_edge_mean_traffic_flow_speed = 0
    for temp_edge in interval_edge_l:
        impacted_edge_mean_traffic_flow_speed += traffic_flow_speed[busline_id][temp_edge]
    impacted_edge_mean_traffic_flow_speed /= len(interval_edge_l)

    # 决策速度应该不超过低于交通流速度
    GurobiModel.addConstr(decision_variable_inv_d[bus_id, next_stop_id] * impacted_edge_mean_traffic_flow_speed >= 1,
                          name=f"decision_variable<mean_traffic_flow_speed_{bus_id}_{next_stop_id}")

    # 计算公交车服务完各edge所需要的时间
    cur_edge_remaining_distance = float(cur_edge_remaining_distance)
    temp_edge = interval_edge_l[0]
    while temp_edge in interval_edge_l:
        # 表示触发时刻公交车处于车站，注意1）该部分是计算站间运行时间的，所以即使触发时刻公交车处于公交站，也只会在下一个路段元素开始计算；2）interval_edge_l以下一个公交站的前一个路段元素结尾）
        if sorted_busline_edge_d[busline_id][temp_edge][0] == "s":
            temp_edge = busline_edge_l[cur_edge_index + 1]

        elif sorted_busline_edge_d[busline_id][temp_edge][0] == "w":  # 表示公交车处于w类型的edge元素
            # 获取其可以合并至的edge的ID并创建变量
            impacted_edge = sorted_busline_edge_d[busline_id][temp_edge][-2]  # 当前路段可以合并至某个路段
            interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] = GurobiModel.addVar(
                vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=MaxTime,
                name=f"interval_edge_depart_time_{busline_id}_{bus_id}_{next_stop_id}_{impacted_edge}")

            temp_edge_length = float(sorted_busline_edge_d[busline_id][temp_edge][1])
            impacted_length = float(sorted_busline_edge_d[busline_id][temp_edge][-1])
            temp_edge_index = busline_edge_l.index(temp_edge)
            impacted_edge_index = busline_edge_l.index(sorted_busline_edge_d[busline_id][temp_edge][-2])

            # 后面合并的impacted_edge是w类型路段
            if sorted_busline_edge_d[busline_id][impacted_edge][0] == "w":
                if sorted_busline_edge_d[busline_id][impacted_edge][2] == "1":  # 表示布设了公交专用道  (下面这里如果不是第一个，那么就不是trigger——time)
                    if temp_edge == cur_edge_id:  # 表示在处理 触发时刻公交车处于的w类型路段
                        GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] == last_stop_depart_time_var +  # 注意这里控制速度没有细分到元素，还是站间速度
                                              decision_variable_inv_d[bus_id, next_stop_id] * (impacted_length - temp_edge_length + cur_edge_remaining_distance),
                                              name=f"interval_edge_depart_time_w1_{bus_id}_{temp_edge}")
                    else:  # LRG interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]]感觉可以更换为某个信号灯路段或者公交站离开时间，因为前一个被打断的只能是信号灯和公交站这两种情况
                        GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] ==
                                              interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] +  # 注意这里控制速度没有细分到元素，还是站间速度
                                              decision_variable_inv_d[bus_id, next_stop_id] * impacted_length,
                                              name=f"interval_edge_depart_time_w2_{bus_id}_{temp_edge}")

                else:  # 没有布设公交专用道，无法控制公交车速度，故采用交通流速度
                    if temp_edge == cur_edge_id:  # 表示处理的第一个edge元素
                        GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] == last_stop_depart_time_var +
                                              (impacted_length - temp_edge_length + cur_edge_remaining_distance)/impacted_edge_mean_traffic_flow_speed,
                                              name=f"interval_edge_depart_time_w3_{bus_id}_{temp_edge}")  # 本质有多个路段，这里使用impacted_edge来近似
                    else:  # LRG interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] 感觉可以更换为某个信号灯路段或者公交站离开时间，因为前一个被打断的只能是信号灯和公交站这两种情况
                        GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] ==
                                              interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] +
                                              impacted_length/impacted_edge_mean_traffic_flow_speed,
                                              name=f"interval_edge_depart_time_w4_{bus_id}_{temp_edge}")

            # 后面合并的impacted_edge是i类型路段
            else:
                tl = sorted_busline_edge_d[busline_id][impacted_edge][3]
                # 在线路edge列表中，cur_edge_id的索引是否早于触发时刻公交车下一个服务车站的索引，只有早于的情况下，trigger_time才会等于等效触发时间transferred_trigger_time_var
                if temp_edge == cur_edge_id:
                    GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] == last_stop_depart_time_var +  # 注意这里控制速度没有细分到元素，还是站间速度
                                          (impacted_length - temp_edge_length + cur_edge_remaining_distance)/impacted_edge_mean_traffic_flow_speed +
                                          subsequent_tl_info_d[tl][7],  # subsequent_tl_info_d[tl][7]为交叉口改相位的平均延误时间
                                          name=f"interval_edge_depart_time_w5_{bus_id}_{temp_edge}")
                else:
                    GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, impacted_edge] ==
                                          interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] +  # 注意这里控制速度没有细分到元素，还是站间速度
                                          impacted_length/impacted_edge_mean_traffic_flow_speed + subsequent_tl_info_d[tl][7],  # subsequent_tl_info_d[tl][7]为交叉口改相位的平均延误时间,
                                          name=f"interval_edge_depart_time_w6_{bus_id}_{temp_edge}")

            # impacted_edge_index = busline_edge_l.index(impacted_edge)
            temp_edge = busline_edge_l[impacted_edge_index + 1]

        # i类型的edge元素
        else:
            interval_edge_depart_time[busline_id, bus_id, next_stop_id, temp_edge] = GurobiModel.addVar(
                vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=MaxTime,
                name=f"interval_edge_depart_time_{busline_id}_{bus_id}_{next_stop_id}_{temp_edge}")

            tl = sorted_busline_edge_d[busline_id][temp_edge][3]
            if temp_edge == cur_edge_id:  # 表示在处理 触发时刻公交车处于的i类型路段
                GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, temp_edge] == last_stop_depart_time_var +  # 注意这里控制速度没有细分到元素，还是站间速度
                                      cur_edge_remaining_distance/impacted_edge_mean_traffic_flow_speed +
                                      subsequent_tl_info_d[tl][7],  # subsequent_tl_info_d[tl][7]为交叉口改相位的平均延误时间
                                      name=f"interval_edge_depart_time_t1_{bus_id}_{temp_edge}")
            else:
                temp_edge_length = float(sorted_busline_edge_d[busline_id][temp_edge][1])
                GurobiModel.addConstr(interval_edge_depart_time[busline_id, bus_id, next_stop_id, temp_edge] ==
                                      interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[interval_edge_l.index(temp_edge)-1]] +  # 注意这里控制速度没有细分到元素，还是站间速度
                                      temp_edge_length/impacted_edge_mean_traffic_flow_speed +
                                      subsequent_tl_info_d[tl][7],  # subsequent_tl_info_d[tl][7]为交叉口改相位的平均延误时间
                                      name=f"interval_edge_depart_time_t2_{bus_id}_{temp_edge}")

            temp_edge_index = busline_edge_l.index(temp_edge)
            temp_edge = busline_edge_l[temp_edge_index + 1]

    # 站间路段运行过程结束
    GurobiModel.addConstr(arrive_time_var == interval_edge_depart_time[busline_id, bus_id, next_stop_id, interval_edge_l[-1]],
                          name=f"arrive_time_var_{bus_id}_{temp_edge}")

