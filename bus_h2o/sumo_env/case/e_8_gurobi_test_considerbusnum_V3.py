# 功能： 基于e_8_gurobi_test修改的，进一步考虑 公交站只计算未来有限公交车的运行计划，之前e_8_gurobi_test_considerbusnum那个程序1）在考虑前1、3、5站时，全程Gurobi都求解不出来，导致结果都一样；2）如果只考虑前一些车站，剩余的车站应该直接在模型中去掉，而不仅仅是 是否考虑车头时距的区别
# 时间：2025年7月23日

import copy
import sys
import os

# 添加当前目录到Python路径以支持模块导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import a_8_subfunction_for_initial_info
try:
    import c_8_subfunction_for_gurobi_test
except ImportError:
    print("Warning: c_8_subfunction_for_gurobi_test module not found, some functionality may be limited")
    c_8_subfunction_for_gurobi_test = None

try:
    import gurobipy
except ImportError:
    print("Warning: gurobipy module not found, optimization functionality may be limited")
    gurobipy = None

import random
import re
from collections import defaultdict

# 设置种子为41、38
random.seed(41)


def Simplified_line_obj_dic(line_obj_dic, first_stop_num):

    simplified_line_obj_dic = copy.deepcopy(line_obj_dic)
    for busline in line_obj_dic:
        simplified_line_obj_dic[busline].distance_between_stop_d = dict(list(simplified_line_obj_dic[busline].distance_between_stop_d.items())[:first_stop_num])
        simplified_line_obj_dic[busline].stop_id_l = simplified_line_obj_dic[busline].stop_id_l[:first_stop_num]

    return simplified_line_obj_dic


def get_static_info(line_obj_dic):
    """获取公交线路、线路公交站等静态信息"""

    # simplified_line_obj_dic = Simplified_line_obj_dic(line_obj_dic, 18)

    bus_line_list = list(line_obj_dic.keys())

    station_dict = {}
    station_interval_dis_dict = {}
    for busline in bus_line_list:
        station_dict[busline] = line_obj_dic[busline].stop_id_l
        station_interval_dis_dict[busline] = line_obj_dic[busline].distance_between_stop_d

    """初始化公交站、公交车OD、OTD及乘客到达率"""
    line_station_od_otd_dict = a_8_subfunction_for_initial_info.get_busline_station_od_otd_dict(station_dict)
    # 对line_station_od_otd_dict中的部分OD进行删减，从而降低Gurobi模型求解复杂度
    scaled_line_station_od_otd_dict = a_8_subfunction_for_initial_info.get_scaled_line_station_od_otd_dict(line_station_od_otd_dict)

    bus_arrstation_od_otd_dict = a_8_subfunction_for_initial_info.get_bus_arrstation_od_otd(scaled_line_station_od_otd_dict, station_dict)
    # 前面都是指获得od/otd的列表，到这一步是用列表去换取乘客到达率
    od_otd_arr_rate_dict = a_8_subfunction_for_initial_info.get_arr_rate(scaled_line_station_od_otd_dict)

    """常量定义"""
    BusCap = 50  # 公交乘客容量  50更符合实际一些  # 20(20S)、50(90S)
    AveAlightingTime = 1.2  # 每位乘客平均上车时间（s/per）
    AveBoardingTime = 1.5  # 每位乘客平均下车时间（s/per）

    return bus_line_list, station_dict, station_interval_dis_dict, scaled_line_station_od_otd_dict, line_station_od_otd_dict, \
           bus_arrstation_od_otd_dict, od_otd_arr_rate_dict, BusCap, AveAlightingTime, AveBoardingTime


def dynamic_sort_key(pattern_str, default=0):
    escaped = re.escape(pattern_str)  # 转义特殊字符
    regex = re.compile(fr'{escaped}(\d+)')  # 动态生成正则

    def get_key(s):
        match = regex.search(s)
        return int(match.group(1)) if match else default
    return get_key


def Simplified_bus_and_stop_dict(origin_bus_obj_dic, origin_stop_obj_dic, first_stop_num, scaled_line_station_od_otd_dict):
    """对bus_obj_dic、stop_obj_dic进行first_stop_num处理，即每个公交车只考虑接下来的first_stop_num个公交站"""
    bus_obj_dic = copy.deepcopy(origin_bus_obj_dic)
    stop_obj_dic = copy.deepcopy(origin_stop_obj_dic)

    for bus_obj_id in bus_obj_dic:
        bus_obj_dic[bus_obj_id].unserved_stop_l = bus_obj_dic[bus_obj_id].unserved_stop_l[:min(first_stop_num, len(bus_obj_dic[bus_obj_id].unserved_stop_l))]

    # 清空stop_obj_dic中各公交站对象的unserved_bus_l、service_line_l
    scaled_stop_od = {}
    for stop_obj_id in stop_obj_dic:
        stop_obj_dic[stop_obj_id].unserved_bus_l = []
        stop_obj_dic[stop_obj_id].service_line_l = []
        # 在删减各车站部分OD的需求下，需要修改just_serve_data的OD数据
        scaled_od = {}
        if stop_obj_id in scaled_stop_od:  # 表明是共线站，在其他线路已经进行了处理
            stop_obj_dic[stop_obj_id].just_leave_data_l[-1] = copy.deepcopy(scaled_stop_od[stop_obj_id])
        else:
            for od_otd in stop_obj_dic[stop_obj_id].just_leave_data_l[-1]:
                if od_otd in scaled_line_station_od_otd_dict[stop_obj_dic[stop_obj_id].just_leave_data_l[1]][stop_obj_id]:
                    scaled_od[od_otd] = copy.deepcopy(stop_obj_dic[stop_obj_id].just_leave_data_l[-1][od_otd])
            stop_obj_dic[stop_obj_id].just_leave_data_l[-1] = copy.deepcopy(scaled_od)
            scaled_stop_od[stop_obj_id] = copy.deepcopy(scaled_od)

    # 基于修改后的bus_obj_dic信息更新stop_obj_dic中的unserved_bus_l、service_line_l
    # 会造成一个现象：在1）没有未服务公交车和2）不被考虑在内的公交站这两种情况下，对应的这两个属性均为空，也合理，因为这些不被后面的模型考虑在内
    for bus_obj_id in bus_obj_dic:
        bus_obj_unserved_stop_l = bus_obj_dic[bus_obj_id].unserved_stop_l
        for unserved_stop_id in bus_obj_unserved_stop_l:
            stop_obj_dic[unserved_stop_id].unserved_bus_l.append(bus_obj_id)
            if bus_obj_id.split("_")[0] not in stop_obj_dic[unserved_stop_id].service_line_l:
                stop_obj_dic[unserved_stop_id].service_line_l.append(bus_obj_id.split("_")[0])

    return bus_obj_dic, stop_obj_dic

# def Simplified_bus_and_stop_dict(bus_obj_dic, stop_obj_dic, station_dict, line_station_od_otd_dict, bus_arrstation_od_otd_dict):
#
#     simplified_bus_obj_dic = copy.deepcopy(bus_obj_dic)
#     for bus_obj_id in simplified_bus_obj_dic:
#         simplified_bus_obj_dic[bus_obj_id].unserved_stop_l = list(set(simplified_bus_obj_dic[bus_obj_id].unserved_stop_l) & set(station_dict[simplified_bus_obj_dic[bus_obj_id].belong_line_id_s]))
#         simplified_bus_obj_dic[bus_obj_id].unserved_stop_l = sorted(simplified_bus_obj_dic[bus_obj_id].unserved_stop_l, key=dynamic_sort_key(simplified_bus_obj_dic[bus_obj_id].belong_line_id_s))
#
#         simplified_bus_obj_dic[bus_obj_id].bus_passenger_d = {}
#         for od_id in bus_obj_dic[bus_obj_id].bus_passenger_d:
#             if od_id in bus_arrstation_od_otd_dict[simplified_bus_obj_dic[bus_obj_id].belong_line_id_s][simplified_bus_obj_dic[bus_obj_id].next_stop_id_s]:
#                 simplified_bus_obj_dic[bus_obj_id].bus_passenger_d[od_id] = bus_obj_dic[bus_obj_id].bus_passenger_d[od_id]
#
#     simplified_stop_obj_dic = copy.deepcopy(stop_obj_dic)
#     for stop_obj_id in stop_obj_dic:
#         if stop_obj_id == "122S20_406X18":
#             debug = 0
#         # 删除不考虑的公交站信息
#         if stop_obj_id not in station_dict[stop_obj_dic[stop_obj_id].just_leave_data_l[1]]:
#             if stop_obj_id == "122S20_406X18":
#                 debug = 0
#             del simplified_stop_obj_dic[stop_obj_id]
#             continue
#
#         # 删除站内乘客中涉及不考虑公交站的OD信息
#         simplified_stop_obj_dic[stop_obj_id].just_leave_data_l[-1] = {}
#         for od_id in stop_obj_dic[stop_obj_id].just_leave_data_l[-1]:
#             if od_id in line_station_od_otd_dict[stop_obj_dic[stop_obj_id].just_leave_data_l[1]][stop_obj_id]:
#                 simplified_stop_obj_dic[stop_obj_id].just_leave_data_l[-1][od_id] = stop_obj_dic[stop_obj_id].just_leave_data_l[-1][od_id]
#
#     # 删除不考虑的公交线路信息和未服务公交车信息
#     for stop_obj_id in simplified_stop_obj_dic:
#         serve_buslines = list(set(stop_obj_dic[stop_obj_id].service_line_l) & set(list(station_dict.keys())))
#         for busline in stop_obj_dic[stop_obj_id].service_line_l:
#             # 删除 现在公交时刻表中不考虑的公交线路
#             if busline not in serve_buslines:
#                 simplified_stop_obj_dic[stop_obj_id].service_line_l.remove(busline)
#                 continue
#             # 删除 现在公交时刻表中的公交线路不考虑的车站
#             if busline in serve_buslines and stop_obj_id not in station_dict[busline]:
#                 simplified_stop_obj_dic[stop_obj_id].service_line_l.remove(busline)
#                 for unserved_bus_id in stop_obj_dic[stop_obj_id].unserved_bus_l:
#                     if unserved_bus_id.split("_")[0] == busline:
#                         simplified_stop_obj_dic[stop_obj_id].unserved_bus_l.remove(unserved_bus_id)
#
#     return simplified_bus_obj_dic, simplified_stop_obj_dic
#


# **定义回调函数**
def my_callback(model, where):
    if where == gurobipy.GRB.Callback.MIP:
        best_obj = model.cbGet(gurobipy.GRB.Callback.MIP_OBJBST)  # 当前最优整数解
        best_bound = model.cbGet(gurobipy.GRB.Callback.MIP_OBJBND)  # 当前最优界
        if best_obj != gurobipy.GRB.INFINITY and best_bound != -gurobipy.GRB.INFINITY:  # 确保解是有限的
            gap = abs(best_bound - best_obj) / max(abs(best_obj), 1e-10)  # 计算 MIPGap
            print(f"当前 GAP: {gap:.5f}")  # 打印 GAP 以便调试
            if gap < 0.01:
                print("GAP 小于 0.01，终止求解！")
                model.terminate()  # 终止求解


def judge_stop_chuanche(stop_obj_d, specified_headway, trigger_time):
    """判断某个车站是否发生串车"""
    for stop_id in stop_obj_d:
        service_data_l = stop_obj_d[stop_id].service_data_l
        scaled_service_data_l = []
        # 筛选指定时间范围内的公交车数据
        for service_data in service_data_l:
            if service_data[2] < trigger_time - 300:  # 判断触发时刻前300s有没有发生串车
                scaled_service_data_l.append(service_data)
        # 划分为不同线路
        scaled_busline_service_data_d = {}
        for service_data in scaled_service_data_l:
            if service_data[1] not in scaled_busline_service_data_d:
                scaled_busline_service_data_d[service_data[1]] = [service_data]
            else:
                scaled_busline_service_data_d[service_data[1]].append(service_data)
        # 判断不同线路内部的公交车是否存在串车现象
        for busline in scaled_busline_service_data_d:
            busline_service_data_l = scaled_busline_service_data_d[busline]
            for index, service_data in enumerate(busline_service_data_l):
                if index < len(busline_service_data_l)-1:
                    if busline_service_data_l[index+1][2] - service_data[2] < specified_headway and service_data[0].split("_")[1] != "0":
                        print(f"{service_data[0]}与{busline_service_data_l[index+1][0]}在{stop_id}发生串车！！！")
                        debug = 0


def keep_first_n_buses(bus_list, n):
    """每条线路只保留前n个公交车"""
    route_to_buses = defaultdict(list)

    # 收集每条线路的非零车次（注意按 bus_id 的数字排序）
    for bus in bus_list:
        route, bus_id = bus.split('_')
        if bus_id != '0':
            route_to_buses[route].append((int(bus_id), bus))

    # 每条线路只保留前 n 个编号小的车次
    keep_set = set()
    for route, buses in route_to_buses.items():
        # 根据编号排序后取前 n 个
        sorted_buses = sorted(buses, key=lambda x: x[0])
        for _, bus in sorted_buses[:n]:
            keep_set.add(bus)

    # 按原始顺序筛选保留的车次
    return [bus for bus in bus_list if bus in keep_set]


def extract_busline_stop_number(data_list, prefix):
    """在s中提取busline_id 后面的数值，便于按照公交站先后顺序进行排序"""
    pattern = re.compile(re.escape(prefix) + r'(\d+)')

    def extract_number(s):
        match = pattern.search(s)
        return int(match.group(1)) if match else float('inf')  # 无匹配的排后面

    return sorted(data_list, key=extract_number)


def process_variable_for_stop_consider_bus_num(stop_consider_unserved_bus_num, initial_bus_obj_l, initial_stop_obj_l, station_interval_dis_dict, initial_stop_obj_id_l):
    """处理原始数据，引入考量因素：各公交站考虑未来几辆公交车的运行计划信息"""

    # 获取各公交站的公交车到站服务顺序
    initial_stop_bus_serve_seq_d = c_8_subfunction_for_gurobi_test.get_bus_sequence_on_stops_without_arrdep_contraints(initial_stop_obj_l, initial_bus_obj_l, station_interval_dis_dict, initial_stop_obj_id_l)
    # 各公交站只保留待服务的前n个公交车次
    stop_bus_serve_seq_d = {}
    for stop_id in initial_stop_bus_serve_seq_d:
        filtered = keep_first_n_buses(initial_stop_bus_serve_seq_d[stop_id], stop_consider_unserved_bus_num)
        stop_bus_serve_seq_d[stop_id] = copy.deepcopy(filtered)

    # 获取各公交站的前stop_consider_unserved_bus_num个
    stop_qualified_sorted_bus_d = {}
    for stop_id in stop_bus_serve_seq_d:
        if stop_id not in stop_qualified_sorted_bus_d:
            stop_qualified_sorted_bus_d[stop_id] = []
        else:
            print("error in process_variable_for_stop_consider_bus_num\n")
        stop_qualified_sorted_bus_d[stop_id] = copy.deepcopy(stop_bus_serve_seq_d[stop_id])

    # 处理公交站属性，如果公交车位于公交站服务一定数量的之外，则去掉
    stop_obj_l = []
    for temp_stop_obj in initial_stop_obj_l:
        stop_id = temp_stop_obj.stop_id_s  # 获取公交站名称
        modified_stop_obj = copy.deepcopy(temp_stop_obj)
        modified_stop_obj.unserved_bus_l = copy.deepcopy(stop_qualified_sorted_bus_d[stop_id])
        stop_obj_l.append(modified_stop_obj)

    # 将stop_consider_unserved_bus_num转换为以公交车名称为键，待服务公交站名称组成列表
    bus_stop_seq_d = {}
    for stop_name, bus_list in stop_bus_serve_seq_d.items():
        for bus in bus_list:
            if bus not in bus_stop_seq_d:
                bus_stop_seq_d[bus] = []
            bus_stop_seq_d[bus].append(stop_name)

    # 处理公交车属性，如果其待服务的某个公交站距离其较远，该公交车已经超出对应公交站待服务的一定数量，则去掉
    bus_obj_l = []
    for temp_bus_obj in initial_bus_obj_l:
        bus_id = temp_bus_obj.bus_id_s  # 获取公交车名称
        modified_bus_obj = copy.deepcopy(temp_bus_obj)
        modified_bus_obj.unserved_stop_l = copy.deepcopy(extract_busline_stop_number(bus_stop_seq_d[bus_id], temp_bus_obj.belong_line_id_s))
        bus_obj_l.append(modified_bus_obj)

    # 获取公交车对象名称列表和公交站对象名称列表
    bus_obj_id_l = []
    stop_obj_id_l = []
    for temp_bus_obj in bus_obj_l:
        bus_obj_id_l.append(temp_bus_obj.bus_id_s)
    for temp_stop_obj in stop_obj_l:
        stop_obj_id_l.append(temp_stop_obj.stop_id_s)

    debug = 0

    return bus_obj_l, stop_obj_l, bus_obj_id_l, stop_obj_id_l


"""将触发时刻前后的公交车到站时间整合起来，绘制各公交车完整的到站时间曲线"""
# # 个公交历史到站时间(s)
# bus_arrival_time_dict = {
#     "705S_1": {"705S01": "200", "705S02": "500", "705S03_7S03": "800", "705S04_7S04": "1000"},
#     "705S_2": {"705S01": "500", "705S02": "800", "705S03_7S03": "1100"},
#     "705S_3": {"705S01": "800", "705S02": "900", "705S03_7S03": "1105"},
#     "705S_4": {"705S01": "1000"},
# }
# complete_arrival_time_dict = plot_complete_arrival_time_result(arrival_time_variable_d, "arrival_time_", station_dict, bus_arrival_time_dict)
