# 功能：基于刘师兄的考虑乘客换乘的多线路公交速度控制模型，定义获取初始信息的相关函数等
# 时间：2025年1月2日


import re
import random
import copy

# 设置种子为41、38
random.seed(41)

"""获取字典d的嵌套层数"""
def get_dict_depth(d):
    if not isinstance(d, dict) or not d:
        return 0
    # 过滤出字典类型的值
    dict_values = [value for value in d.values() if isinstance(value, dict)]
    # 如果没有字典类型的值，返回 1（当前层）
    if not dict_values:
        return 1
    # 否则，递归计算字典嵌套的深度
    return 1 + max(get_dict_depth(value) for value in dict_values)


"""获取各线路的共线车站  公交线路与共站ID的嵌套字典格式  {705S:{705S03: [7S03], 705S04:[7S04]},7S:..}"""
def get_busline_shared_station(temp_station_dict):
    busline_shared_station_dict = {}
    for busline in temp_station_dict:
        busline_shared_station_dict[busline] = {}
        for station in temp_station_dict[busline]:
            if "_" in station and len(list(set([re.match(r'^(\d+[A-Z])', part).group(1) for part in station.split('_')]) & set(list(temp_station_dict.keys())))) > 1:  # 表示为共线公交站(默认otd三个公交站不可处于同一线路)
                split_station_list = station.split("_")
                for temp_split_station in split_station_list:
                    if re.match(r"([0-9]*[A-Z]+)([0-9]*)", temp_split_station).groups()[0] == busline:
                        split_station_list.remove(temp_split_station)
                        busline_shared_station_dict[busline][temp_split_station] = split_station_list
                        break
    return busline_shared_station_dict


"""基于共线公交站的名称获取涉及的公交线路 公交线路构成的列表 ["705S","7S"...]"""
def get_busline_from_station_ID(station_ID, temp_station_dict):
    shared_busline = []
    split_station_list = station_ID.split("_")
    for temp_split_station in split_station_list:
        shared_busline.append(re.match(r"([0-9]*[A-Z]+)([0-9]*)", temp_split_station).groups()[0])
    return list(set(shared_busline) & set(temp_station_dict.keys()))


"""形参station作为O，构造该站所有OTD组合  {公交线路1:{公交站1:{otd1,otd2,...}}}"""
def get_OTD_station(index_O, busline, temp_station_dict, line_station_od_otd, station, shared_busline):
    for station_T in temp_station_dict[busline][index_O + 1:]:
        if "_" in station_T and len(list(set([re.match(r'^(\d+[A-Z])', part).group(1) for part in station_T.split('_')]) & set(list(temp_station_dict.keys())))) > 1 and shared_busline != get_busline_from_station_ID(station_T, temp_station_dict):  # 查询共享线路不一致的共线车站作为OTD中的T
            split_station_list = station_T.split("_")  # 将"705S03_7S03"划分为列表["705S03","7S03"]
            for temp_split_station in split_station_list:
                shared_busline = re.match(r"([0-9]*[A-Z]+)([0-9]*)", temp_split_station).groups()[0]
                if shared_busline != busline and shared_busline in list(temp_station_dict.keys()):  # 选择与O所在线路不一致的公交线路，以挑选D（这里应该是不属于busline的其它线路）
                    station_T_index_in_shared_busline = temp_station_dict[shared_busline].index(
                        station_T)  # 查询共线车站在共享线路车站中的索引
                    for station_D in temp_station_dict[shared_busline][
                                     station_T_index_in_shared_busline + 1:]:  # 遍历共享线路中共线车站后的所有车站
                        if busline not in station_D:  # 筛选共享线路中与station_O所在线路无关的车站作为D
                            line_station_od_otd[busline][station].append(station+"-"+station_T+"-"+station_D)
    return line_station_od_otd


"""形参station作为T，构造该站所有TOD组合  {公交线路1:{公交站1:{otd1,otd2,...}}}"""
def get_TOD_station(busline, temp_station_dict, line_station_od_otd, station_T, shared_busline_list):
    for shared_busline in shared_busline_list:
        stationT_index_in_shared_busline = temp_station_dict[shared_busline].index(station_T)
        station_O_list = temp_station_dict[shared_busline][:stationT_index_in_shared_busline]
        for station_O in station_O_list:
            station_D_busline_list = list(set(shared_busline_list) - set(get_busline_from_station_ID(station_O, temp_station_dict)))  # 获取可以作为OTD中D的公交线路
            if not station_D_busline_list:
                continue  # 若为空，则表示O、T公交站包含的公交线路完全一致，不满足OTD要求
            else:
                for station_D_busline in station_D_busline_list:
                    stationD_index = temp_station_dict[station_D_busline].index(station_T)
                    station_D_list = temp_station_dict[station_D_busline][stationD_index+1:]  # 获取符合条件的公交线路中可以作为OTD中D的公交站列表
                    station_D_list = list(set(station_D_list) - set(temp_station_dict[shared_busline][stationT_index_in_shared_busline:]))
                    if not station_D_list:
                        continue
                    else:
                        for station_D in station_D_list:
                            if station_O+"-"+station_T+"-"+station_D not in line_station_od_otd[busline][station_T]:
                                line_station_od_otd[busline][station_T].append(station_O+"-"+station_T+"-"+station_D)
    return line_station_od_otd


"""基于各线路的公交站信息确定各站的od、otd {公交线路1:{公交站1:{od1,od2,otd1,...}}}"""
def get_busline_station_od_otd_dict(temp_station_dict):
    line_station_od_otd = {}
    # 确定OD
    for busline in temp_station_dict:
        line_station_od_otd[busline] = {}
        for index, station in enumerate(temp_station_dict[busline][:-1]):
            if station not in line_station_od_otd[busline]:
                line_station_od_otd[busline][station] = []
            # 添加该站涉及的本线路的OD
            for station_D_index in range(index+1, len(temp_station_dict[busline])):
                # 这里是在遍历该线路后续的公交站作为D
                line_station_od_otd[busline][station].append(station + "-" + temp_station_dict[busline][station_D_index])
            # 该公交站为共线公交站，需要增加该站其它线路的OD
            if "_" in station and len(list(set([re.match(r'^(\d+[A-Z])', part).group(1) for part in station.split('_')]) & set(list(temp_station_dict.keys())))) > 1:
                contributed_buslines = get_busline_from_station_ID(station, temp_station_dict)
                contributed_buslines.remove(busline)
                # 添加OD
                for contributed_busline in contributed_buslines:
                    if contributed_busline in list(temp_station_dict.keys()) and station in temp_station_dict[contributed_busline]:
                        station_index_contributed_busline = temp_station_dict[contributed_busline].index(station)
                        for station_D_index_contributed_busline in range(station_index_contributed_busline + 1, len(temp_station_dict[contributed_busline])):
                            if station + "-" + temp_station_dict[contributed_busline][station_D_index_contributed_busline] not in line_station_od_otd[busline][station]:
                                line_station_od_otd[busline][station].append(station + "-" +
                                                                             temp_station_dict[contributed_busline][station_D_index_contributed_busline])

    # # 确定OTD
    # for busline in temp_station_dict:
    #     for index_O, station in enumerate(temp_station_dict[busline]):
    #         if station == "122S15_311S21_406X13":
    #             debug = 0
    #         if "_" in station and len(list(set([re.match(r'^(\d+[A-Z])', part).group(1) for part in station.split('_')]) & set(list(temp_station_dict.keys())))) > 1:
    #             # 共线车站需考虑所有相关线路后续涉及到的中转站构造OTD
    #             split_station_list = station.split("_")
    #             shared_busline_list = []
    #             for temp_split_station in split_station_list:
    #                 if re.match(r"([0-9]*[A-Z]+)([0-9]*)", temp_split_station).groups()[0] in list(temp_station_dict.keys()):
    #                     shared_busline_list.append(re.match(r"([0-9]*[A-Z]+)([0-9]*)", temp_split_station).groups()[0])
    #             line_station_od_otd = get_OTD_station(index_O, busline, temp_station_dict, line_station_od_otd, station,
    #                                                   shared_busline_list)
    #             # 共线车站需要额外考虑作为中转站T构造OTD
    #             line_station_od_otd = get_TOD_station(busline, temp_station_dict, line_station_od_otd,
    #                                                   station, shared_busline_list)
    #         else:    # 非共线车站只需考虑本线路后续涉及到的中转站构造OTD
    #             line_station_od_otd = get_OTD_station(index_O, busline, temp_station_dict, line_station_od_otd, station,
    #                                                   [busline])

    return line_station_od_otd


"""对line_station_od_otd_dict中的部分OD进行删减，从而降低Gurobi模型求解复杂度"""
def get_scaled_line_station_od_otd_dict(line_station_od_otd_dict):

    station_od_d = {}
    scaled_line_station_od_otd_dict = {}
    for busline in line_station_od_otd_dict:
        scaled_line_station_od_otd_dict[busline] = {}
        for station in line_station_od_otd_dict[busline]:
            scaled_line_station_od_otd_dict[busline][station] = []

            # 该车站od信息第一次计算
            if station not in station_od_d:
                station_od_d[station] = []
                for index, od_otd in enumerate(line_station_od_otd_dict[busline][station]):
                    # 每个车站删减一半的OD，降低模型求解复杂度
                    if index % 8 == 0:
                        scaled_line_station_od_otd_dict[busline][station].append(od_otd)
                        station_od_d[station].append(od_otd)
            # 该车站为共线车站，已经在其他线路处理过了
            else:
                scaled_line_station_od_otd_dict[busline][station] = copy.deepcopy(station_od_d[station])

    return scaled_line_station_od_otd_dict


"""获取不同公交到达车站时，车内可能的od {公交线路1:{公交站1:{od1,od2,otd1,...}}}"""
def get_bus_arrstation_od_otd(line_station_od_otd_dict, station_dict):
    bus_arrstation_od_otd = {}
    for busline in station_dict:
        bus_arrstation_od_otd[busline] = {}
        for station in station_dict[busline][1:]:  # 遍历该线路第二个及后续车站记录公交到站前车内数据
            bus_arrstation_od_otd[busline][station] = []
            station_index = station_dict[busline].index(station)
            passed_station_list = station_dict[busline][:station_index]  # 获取该线路在该站之前的所有公交站
            for passed_station in passed_station_list:  # 遍历已经完成服务的所有公交站
                passed_station_od_otd_list = line_station_od_otd_dict[busline][passed_station]
                # 去除D为passed_station_list元素之一的od和otd
                filted_passed_station_od_otd_list = [item for item in passed_station_od_otd_list
                                                     if item.split("-")[-1] not in passed_station_list]
                # 去除D为其它线路公交站的od、OT均为passed_station_list元素之一的otd
                temp_filted_passed_station_od_otd_list = copy.deepcopy(filted_passed_station_od_otd_list)
                for temp_item in temp_filted_passed_station_od_otd_list:
                    if temp_item.count("-") == 1 and temp_item.split("-")[1] not in station_dict[busline]:
                        filted_passed_station_od_otd_list.remove(temp_item)
                    if temp_item.count("-") == 2 and temp_item.split("-")[0] in passed_station_list and \
                            temp_item.split("-")[1] in passed_station_list:
                        filted_passed_station_od_otd_list.remove(temp_item)
                bus_arrstation_od_otd[busline][station].append(filted_passed_station_od_otd_list)
            bus_arrstation_od_otd[busline][station] = sum(bus_arrstation_od_otd[busline][station], [])
            bus_arrstation_od_otd[busline][station] = list(dict.fromkeys(bus_arrstation_od_otd[busline][station]))  # 进行元素去重
    return bus_arrstation_od_otd


"""获取OD、OTD乘客到达率 (LRG 20250226新增修改)"""
def get_arr_rate(line_station_od_otd_dict):
    arr_rate_dict = {}
    for busline in line_station_od_otd_dict:
        for station in line_station_od_otd_dict[busline]:
            if station not in arr_rate_dict:
                arr_rate_dict[station] = {}
            for od_otd in line_station_od_otd_dict[busline][station]:
                if od_otd.count("-") == 1 and od_otd.startswith(station):  # 表示在处理以该站为O的OD数据
                    arr_rate_dict[station][od_otd] = str(random.uniform(0.00035, 0.00055))   # 定义随机数规则 str(round(random.uniform(0.035, 0.055), 6))
                # else:
                #     arr_rate_dict[station][od_otd] = str(random.uniform(0.035, 0.055))  # 定义随机数规则 str(0.035)
                    # arr_rate_dict[station][od_otd] = str(0)  # 定义随机数规则
    return arr_rate_dict


"""获取公交车剩余未服务公交站"""
def get_unserved_station(busline, station_dict, next_stop_id_s):
    index = station_dict[busline].index(next_stop_id_s)
    return station_dict[busline][index:]


"""车内乘客OD、OTD到站/离站更新，该函数基于公交车状态获取最近一次更新之后的车内OD、OTD，进行初始化  {od1:value,otd1:value,...}"""
def get_just_updated_passenger(bus_arrstation_od_otd_dict, busline, next_stop_id):
    just_updated_passenger = {}
    od_otd_list = bus_arrstation_od_otd_dict[busline][next_stop_id]
    for od_otd in od_otd_list:
        just_updated_passenger[od_otd] = random.random()  # 取0-1内随机数
    return just_updated_passenger


"""公交车到达/离开公交站时，公交站数据更新，获取公交站最近被更新的公交车ID、线路ID、乘客OD、OTD数据"""
def get_just_updated_service_data(line_station_od_otd_dict, already_infor, stop_id_s):
    od_otd_dict = {}
    busline = already_infor[1]
    od_otd_list = line_station_od_otd_dict[busline][stop_id_s]
    for od_otd in od_otd_list:
        od_otd_dict[od_otd] = random.random()  # 取0-1内随机数
    return already_infor+[od_otd_dict]


"""获取模型触发时刻，公交站还未服务的公交车列表 [公交车1,, 公交车2,...]"""
def get_unserved_bus(stop_id_s, bus_obj_list):
    unserved_bus_l = []
    for bus_obj in bus_obj_list:
        if stop_id_s in bus_obj.unserved_stop_l:
            unserved_bus_l.append(bus_obj.bus_id_s)
    return unserved_bus_l


"""三条公交线路存在两处共线区域的情况"""
def three_lines_with_two_contributed_areas():
    bus_line_list = ["705S", "7S", "211S"]  # 公交线路ID
    bus_dict = {"705S": ["705S_1", "705S_2", "705S_3", "705S_4"],  # 各线路公交ID  公交线路ID:[公交ID,...]
                "7S": ["7S_1", "7S_2", "7S_3", "7S_4"],
                "211S": ["211S_1", "211S_2", "211S_3", "211S_4"]}
    station_dict = {
        "705S": ["705S01", "705S02", "705S03_7S03", "705S04_7S04", "705S05", "705S06_211S02",   # 各线路公交站ID 公交线路ID:[公交站ID,...]
                 "705S07_211S03", "705S08"],
        "7S": ["7S01", "7S02", "705S03_7S03", "705S04_7S04", "7S05", "7S06"],
        "211S": ["211S01", "705S06_211S02", "705S07_211S03", "211S04", "211S05"]}

    station_interval_dis_dict = {"705S02": "500", "705S03_7S03": "600", "705S04_7S04": "600", "705S05": "600", "705S06_211S02": "700",  # 各车站与上游车站的距离 公交站：与下游站间距(m)
                                            "705S07_211S03": "500", "705S08": "400",
                                 "7S02": "400", "705S03_7S03": "500", "705S04_7S04": "600", "7S05": "400", "7S06": "600",
                                 "705S06_211S02": "700", "705S07_211S03": "500", "211S04": "400", "211S05": "300"}
    return bus_line_list, bus_dict, station_dict, station_interval_dis_dict


"""综合模型触发时刻站间运行和站内服务公交车数据  {公交线路1:[公交车1,...]}"""
def get_cur_bus_set(temp_interval_running_bus_dict, temp_station_serving_bus_dict):
    temp_cur_bus_dict = copy.deepcopy(temp_interval_running_bus_dict)
    for temp_busline in temp_station_serving_bus_dict:
        if temp_busline not in temp_cur_bus_dict:
            temp_cur_bus_dict[temp_busline] = temp_station_serving_bus_dict[temp_busline]
        else:
            temp_cur_bus_dict[temp_busline].append(temp_station_serving_bus_dict[temp_busline])
    return temp_cur_bus_dict


"""获取公交车到达下一个公交站所需经过的所有edge及对应的长度"""
def get_edges_between_next_stop(current_edge_id, remaining_distance, next_stop_id, sorted_busline_edge, busline_stop_edge):
    sorted_busline_edge_l = list(sorted_busline_edge.keys())
    current_edge_index = sorted_busline_edge_l.index(current_edge_id)

    edges_between_next_stop = {current_edge_id: remaining_distance}

    temp_edge_id = sorted_busline_edge_l[current_edge_index+1]
    while temp_edge_id != busline_stop_edge[next_stop_id]:
        # 如果是以交叉口结束的edge，那么直接保存该edge的ID和长度即可
        if sorted_busline_edge[temp_edge_id][0] == "i":
            edges_between_next_stop[temp_edge_id] = sorted_busline_edge[temp_edge_id][1]
            temp_edge_id = sorted_busline_edge_l[sorted_busline_edge_l.index(temp_edge_id)+1]
        else:  # 只可能是以公交站结尾的edge
            edges_between_next_stop[temp_edge_id] = sorted_busline_edge[temp_edge_id][-1]
            temp_edge_id = sorted_busline_edge_l[sorted_busline_edge_l.index(sorted_busline_edge[temp_edge_id][-2])+1]
    return edges_between_next_stop


"""获取各公交线路要经过的信控交叉口"""
def get_busline_tl(sorted_busline_edge_d):
    busline_tl_d = {}
    for busline in sorted_busline_edge_d:
        busline_tl_d[busline] = {}
        for edge in sorted_busline_edge_d[busline]:
            if sorted_busline_edge_d[busline][edge][0] == "i":
                tl_id = sorted_busline_edge_d[busline][edge][-3]
                tl_phase_time = sorted_busline_edge_d[busline][edge][-2]
                tl_cycle_time = sorted_busline_edge_d[busline][edge][-1]
                busline_tl_d[busline][tl_id] = [tl_phase_time, tl_cycle_time]
