import xml.etree.ElementTree as ET
import os

# 获取路径信息
current_dir = os.path.dirname(os.path.abspath(__file__))
online_control_dir = os.path.dirname(current_dir)
output_file_dir = os.path.join(online_control_dir, "output_file")


def save_lane_data(lane_obj_dic):
    # 创建XML根元素
    root = ET.Element("lane_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for lane_id in lane_obj_dic.keys():
        lane_obj = ET.SubElement(root, "lane_obj")
        lane_obj.set("lane_id_s", lane_obj_dic[lane_id].lane_id_s)
        lane_obj.set("length_n", str(lane_obj_dic[lane_id].length_n))
        lane_obj.set("from_junction_s", lane_obj_dic[lane_id].from_junction_s)
        lane_obj.set("to_junction_s", lane_obj_dic[lane_id].to_junction_s)
        lane_obj.text = "\n\t\t"
        for n in range(0, len(lane_obj_dic[lane_id].state_data_l)):
            data = ET.SubElement(lane_obj, "state_data")
            data.set("t_" + str(lane_obj_dic[lane_id].state_data_l[n][0]),
                     ' '.join([str(m) for m in lane_obj_dic[lane_id].state_data_l[n]]))
            if n == len(lane_obj_dic[lane_id].state_data_l) - 1:
                data.tail = "\n\t"
            else:
                data.tail = "\n\t\t"
        lane_obj.tail = "\n\t"

    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, "save_data_lane.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"车道对象 XML 文件已生成。")


def save_lane_data_multiexp(lane_obj_dic, index):
    # 创建XML根元素
    root = ET.Element("lane_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for lane_id in lane_obj_dic.keys():
        lane_obj = ET.SubElement(root, "lane_obj")
        lane_obj.set("lane_id_s", lane_obj_dic[lane_id].lane_id_s)
        lane_obj.set("length_n", str(lane_obj_dic[lane_id].length_n))
        lane_obj.set("from_junction_s", lane_obj_dic[lane_id].from_junction_s)
        lane_obj.set("to_junction_s", lane_obj_dic[lane_id].to_junction_s)
        lane_obj.text = "\n\t\t"
        for n in range(0, len(lane_obj_dic[lane_id].state_data_l)):
            data = ET.SubElement(lane_obj, "state_data")
            data.set("t_" + str(lane_obj_dic[lane_id].state_data_l[n][0]),
                     ' '.join([str(m) for m in lane_obj_dic[lane_id].state_data_l[n]]))
            if n == len(lane_obj_dic[lane_id].state_data_l) - 1:
                data.tail = "\n\t"
            else:
                data.tail = "\n\t\t"
        lane_obj.tail = "\n\t"

    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, f"{index}_save_data_lane.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"车道对象 XML 文件已生成。")


def save_stop_data(stop_obj_dic):
    # 创建XML根元素
    root = ET.Element("stop_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for stop_id in stop_obj_dic.keys():
        stop_obj = ET.SubElement(root, "stop_obj")
        stop_obj.set("stop_id_s", stop_obj_dic[stop_id].stop_id_s)
        stop_obj.set("at_edge_s", stop_obj_dic[stop_id].at_edge_s)
        stop_obj.set("at_lane_s", stop_obj_dic[stop_id].at_lane_s)
        stop_obj.set("service_line_l", ' '.join(stop_obj_dic[stop_id].service_line_l))
        stop_obj.text = "\n\t\t"
        for n in range(0, len(stop_obj_dic[stop_id].service_data_l)):
            data = ET.SubElement(stop_obj, "service_data")
            data.set("s_" + stop_obj_dic[stop_id].service_data_l[n][0],
                     ' '.join([str(m) for m in stop_obj_dic[stop_id].service_data_l[n]]))
            if n == len(stop_obj_dic[stop_id].service_data_l) - 1:
                data.tail = "\n\t"
            else:
                data.tail = "\n\t\t"
        stop_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, "save_data_stop.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"车站对象 XML 文件已生成。")


def save_stop_data_multiexp(stop_obj_dic, index):
    # 创建XML根元素
    root = ET.Element("stop_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for stop_id in stop_obj_dic.keys():
        stop_obj = ET.SubElement(root, "stop_obj")
        stop_obj.set("stop_id_s", stop_obj_dic[stop_id].stop_id_s)
        stop_obj.set("at_edge_s", stop_obj_dic[stop_id].at_edge_s)
        stop_obj.set("at_lane_s", stop_obj_dic[stop_id].at_lane_s)
        stop_obj.set("service_line_l", ' '.join(stop_obj_dic[stop_id].service_line_l))
        stop_obj.text = "\n\t\t"
        for n in range(0, len(stop_obj_dic[stop_id].service_data_l)):
            data = ET.SubElement(stop_obj, "service_data")
            data.set("s_" + stop_obj_dic[stop_id].service_data_l[n][0],
                     ' '.join([str(m) for m in stop_obj_dic[stop_id].service_data_l[n]]))
            if n == len(stop_obj_dic[stop_id].service_data_l) - 1:
                data.tail = "\n\t"
            else:
                data.tail = "\n\t\t"
        stop_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, f"{index}_save_data_stop.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"车站对象 XML 文件已生成。")


def save_signal_data(signal_obj_dic):
    # 创建XML根元素
    root = ET.Element("signal_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for signal_id in signal_obj_dic.keys():
        signal_obj = ET.SubElement(root, "signal_obj")
        signal_obj.set("signal_id_s", signal_obj_dic[signal_id].signal_id_s)
        signal_obj.text = "\n\t\t"
        for n in range(0, len(signal_obj_dic[signal_id].round_bus_state_l)):
            round_bus_state = ET.SubElement(signal_obj, "round_bus_state")
            round_bus_state.set("t_" + str(signal_obj_dic[signal_id].round_bus_state_l[n][0]),
                                ' '.join([str(m) for m in signal_obj_dic[signal_id].round_bus_state_l[n]]))
            round_bus_state.tail = "\n\t\t"
        for n in range(0, len(signal_obj_dic[signal_id].service_data_l)):
            service_data = ET.SubElement(signal_obj, "service_data")
            service_data.set("s_" + signal_obj_dic[signal_id].service_data_l[n][0],
                             ' '.join([str(m) for m in signal_obj_dic[signal_id].service_data_l[n]]))
            service_data.tail = "\n\t\t"
        for n in range(0, len(signal_obj_dic[signal_id].lane_queue_number_l)):
            lane_queue_number = ET.SubElement(signal_obj, "lane_queue_number")
            lane_queue_number.set("t_" + str(signal_obj_dic[signal_id].lane_queue_number_l[n][0]),
                                  ' '.join([";".join([str(k) for k in m]) for m in
                                            signal_obj_dic[signal_id].lane_queue_number_l[n][1:]]))
            if n == len(signal_obj_dic[signal_id].lane_queue_number_l) - 1:
                lane_queue_number.tail = "\n\t"
            else:
                lane_queue_number.tail = "\n\t\t"
        signal_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, "save_data_signal.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"信号灯对象 XML 文件已生成。")


def save_signal_data_multiexp(signal_obj_dic, index):
    # 创建XML根元素
    root = ET.Element("signal_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for signal_id in signal_obj_dic.keys():
        signal_obj = ET.SubElement(root, "signal_obj")
        signal_obj.set("signal_id_s", signal_obj_dic[signal_id].signal_id_s)
        signal_obj.text = "\n\t\t"
        for n in range(0, len(signal_obj_dic[signal_id].round_bus_state_l)):
            round_bus_state = ET.SubElement(signal_obj, "round_bus_state")
            round_bus_state.set("t_" + str(signal_obj_dic[signal_id].round_bus_state_l[n][0]),
                                ' '.join([str(m) for m in signal_obj_dic[signal_id].round_bus_state_l[n]]))
            round_bus_state.tail = "\n\t\t"
        for n in range(0, len(signal_obj_dic[signal_id].service_data_l)):
            service_data = ET.SubElement(signal_obj, "service_data")
            service_data.set("s_" + signal_obj_dic[signal_id].service_data_l[n][0],
                             ' '.join([str(m) for m in signal_obj_dic[signal_id].service_data_l[n]]))
            service_data.tail = "\n\t\t"
        for n in range(0, len(signal_obj_dic[signal_id].lane_queue_number_l)):
            lane_queue_number = ET.SubElement(signal_obj, "lane_queue_number")
            lane_queue_number.set("t_" + str(signal_obj_dic[signal_id].lane_queue_number_l[n][0]),
                                  ' '.join([";".join([str(k) for k in m]) for m in
                                            signal_obj_dic[signal_id].lane_queue_number_l[n][1:]]))
            if n == len(signal_obj_dic[signal_id].lane_queue_number_l) - 1:
                lane_queue_number.tail = "\n\t"
            else:
                lane_queue_number.tail = "\n\t\t"
        signal_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, f"{index}_save_data_signal.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"信号灯对象 XML 文件已生成。")


def save_line_data(line_obj_dic):
    # 创建XML根元素
    root = ET.Element("line_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for line_id in line_obj_dic.keys():
        line_obj = ET.SubElement(root, "line_obj")
        line_obj.set("line_id_s", line_obj_dic[line_id].line_id_s)
        line_obj.text = "\n\t\t"
        for n in range(0, len(line_obj_dic[line_id].state_data_l)):
            state_data = ET.SubElement(line_obj, "state_data")
            state_data.set("t_" + str(line_obj_dic[line_id].state_data_l[n][0]),
                           ' '.join([";".join([str(k) for k in m]) for m in line_obj_dic[line_id].state_data_l[n][1:]]))
            if n == len(line_obj_dic[line_id].state_data_l) - 1:
                state_data.tail = "\n\t"
            else:
                state_data.tail = "\n\t\t"
        line_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, "save_data_line.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"线路对象 XML 文件已生成。")


def save_line_data_multiexp(line_obj_dic, index):
    # 创建XML根元素
    root = ET.Element("line_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for line_id in line_obj_dic.keys():
        line_obj = ET.SubElement(root, "line_obj")
        line_obj.set("line_id_s", line_obj_dic[line_id].line_id_s)
        line_obj.text = "\n\t\t"
        for n in range(0, len(line_obj_dic[line_id].state_data_l)):
            state_data = ET.SubElement(line_obj, "state_data")
            state_data.set("t_" + str(line_obj_dic[line_id].state_data_l[n][0]),
                           ' '.join([";".join([str(k) for k in m]) for m in line_obj_dic[line_id].state_data_l[n][1:]]))
            if n == len(line_obj_dic[line_id].state_data_l) - 1:
                state_data.tail = "\n\t"
            else:
                state_data.tail = "\n\t\t"
        line_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, f"{index}_save_data_line.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"线路对象 XML 文件已生成。")


def save_bus_data(bus_obj_dic):
    # 创建XML根元素
    root = ET.Element("bus_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for bus_id in bus_obj_dic.keys():
        bus_obj = ET.SubElement(root, "bus_obj")
        bus_obj.set("bus_id_s", bus_obj_dic[bus_id].bus_id_s)
        bus_obj.set("start_time_n", str(bus_obj_dic[bus_id].start_time_n))
        bus_obj.set("belong_line_id_s", bus_obj_dic[bus_id].belong_line_id_s)
        bus_obj.text = "\n\t\t"
        bus_speed = ET.SubElement(bus_obj, "bus_speed")
        bus_speed.set("bus_speed_l", " ".join([str(n) for n in bus_obj_dic[bus_id].bus_speed_l]))
        bus_speed.tail = "\n\t\t"
        distance = ET.SubElement(bus_obj, "distance")
        distance.set("distance_l", " ".join([str(n) for n in bus_obj_dic[bus_id].distance_l]))
        distance.tail = "\n\t\t"
        arriver_stop_time = ET.SubElement(bus_obj, "arriver_stop_time")
        for stop_id in bus_obj_dic[bus_id].arriver_stop_time_d.keys():
            arriver_stop_time.set("s_" + stop_id, str(bus_obj_dic[bus_id].arriver_stop_time_d[stop_id]))
        arriver_stop_time.tail = "\n\t\t"
        depart_stop_time = ET.SubElement(bus_obj, "depart_stop_time")
        for stop_id in bus_obj_dic[bus_id].depart_stop_time_d.keys():
            depart_stop_time.set("s_" + stop_id, str(bus_obj_dic[bus_id].depart_stop_time_d[stop_id]))
        depart_stop_time.tail = "\n\t\t"
        arriver_timetable = ET.SubElement(bus_obj, "arriver_timetable")
        for stop_id in bus_obj_dic[bus_id].arriver_timetable_d.keys():
            arriver_timetable.set("s_" + stop_id, str(bus_obj_dic[bus_id].arriver_timetable_d[stop_id]))
        arriver_timetable.tail = "\n\t\t"
        alight_num = ET.SubElement(bus_obj, "alight_num")
        for stop_id in bus_obj_dic[bus_id].alight_num_d.keys():
            alight_num.set("s_" + stop_id, str(bus_obj_dic[bus_id].alight_num_d[stop_id]))
        alight_num.tail = "\n\t\t"
        want_board_num = ET.SubElement(bus_obj, "want_board_num")
        for stop_id in bus_obj_dic[bus_id].want_board_num_d.keys():
            want_board_num.set("s_" + stop_id, str(bus_obj_dic[bus_id].want_board_num_d[stop_id]))
        want_board_num.tail = "\n\t\t"
        board_num = ET.SubElement(bus_obj, "board_num")
        for stop_id in bus_obj_dic[bus_id].board_num_d.keys():
            board_num.set("s_" + stop_id, str(bus_obj_dic[bus_id].board_num_d[stop_id]))
        board_num.tail = "\n\t\t"
        strand_num = ET.SubElement(bus_obj, "strand_num")
        for stop_id in bus_obj_dic[bus_id].strand_num_d.keys():
            strand_num.set("s_" + stop_id, str(bus_obj_dic[bus_id].strand_num_d[stop_id]))
        strand_num.tail = "\n\t\t"
        arriver_signal_time = ET.SubElement(bus_obj, "arriver_signal_time")
        for signal_id in bus_obj_dic[bus_id].arriver_signal_time_d.keys():
            arriver_signal_time.set(signal_id, str(bus_obj_dic[bus_id].arriver_signal_time_d[signal_id]))
        arriver_signal_time.tail = "\n\t\t"
        depart_signal_time = ET.SubElement(bus_obj, "depart_signal_time")
        for signal_id in bus_obj_dic[bus_id].depart_signal_time_d.keys():
            depart_signal_time.set(signal_id, str(bus_obj_dic[bus_id].depart_signal_time_d[signal_id]))
        depart_signal_time.tail = "\n\t\t"
        no_stop_signal_id_list = ET.SubElement(bus_obj, "no_stop_signal_id_list")
        no_stop_signal_id_list.set("no_stop_signal_id_list_l", " ".join(bus_obj_dic[bus_id].no_stop_signal_id_list_l))
        no_stop_signal_id_list.tail = "\n\t\t"
        stop_signal_id_list = ET.SubElement(bus_obj, "stop_signal_id_list")
        stop_signal_id_list.set("stop_signal_id_list_l", " ".join(bus_obj_dic[bus_id].stop_signal_id_list_l))
        stop_signal_id_list.tail = "\n\t"
        bus_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, "save_data_bus.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"公交车对象 XML 文件已生成。")


def save_bus_data_multiexp(bus_obj_dic, index):
    # 创建XML根元素
    root = ET.Element("bus_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for bus_id in bus_obj_dic.keys():
        bus_obj = ET.SubElement(root, "bus_obj")
        bus_obj.set("bus_id_s", bus_obj_dic[bus_id].bus_id_s)
        bus_obj.set("start_time_n", str(bus_obj_dic[bus_id].start_time_n))
        bus_obj.set("belong_line_id_s", bus_obj_dic[bus_id].belong_line_id_s)
        bus_obj.text = "\n\t\t"
        bus_speed = ET.SubElement(bus_obj, "bus_speed")
        bus_speed.set("bus_speed_l", " ".join([str(n) for n in bus_obj_dic[bus_id].bus_speed_l]))
        bus_speed.tail = "\n\t\t"
        distance = ET.SubElement(bus_obj, "distance")
        distance.set("distance_l", " ".join([str(n) for n in bus_obj_dic[bus_id].distance_l]))
        distance.tail = "\n\t\t"
        arriver_stop_time = ET.SubElement(bus_obj, "arriver_stop_time")
        for stop_id in bus_obj_dic[bus_id].arriver_stop_time_d.keys():
            arriver_stop_time.set("s_" + stop_id, str(bus_obj_dic[bus_id].arriver_stop_time_d[stop_id]))
        arriver_stop_time.tail = "\n\t\t"
        depart_stop_time = ET.SubElement(bus_obj, "depart_stop_time")
        for stop_id in bus_obj_dic[bus_id].depart_stop_time_d.keys():
            depart_stop_time.set("s_" + stop_id, str(bus_obj_dic[bus_id].depart_stop_time_d[stop_id]))
        depart_stop_time.tail = "\n\t\t"
        arriver_timetable = ET.SubElement(bus_obj, "arriver_timetable")
        for stop_id in bus_obj_dic[bus_id].arriver_timetable_d.keys():
            arriver_timetable.set("s_" + stop_id, str(bus_obj_dic[bus_id].arriver_timetable_d[stop_id]))
        arriver_timetable.tail = "\n\t\t"
        alight_num = ET.SubElement(bus_obj, "alight_num")
        for stop_id in bus_obj_dic[bus_id].alight_num_d.keys():
            alight_num.set("s_" + stop_id, str(bus_obj_dic[bus_id].alight_num_d[stop_id]))
        alight_num.tail = "\n\t\t"
        want_board_num = ET.SubElement(bus_obj, "want_board_num")
        for stop_id in bus_obj_dic[bus_id].want_board_num_d.keys():
            want_board_num.set("s_" + stop_id, str(bus_obj_dic[bus_id].want_board_num_d[stop_id]))
        want_board_num.tail = "\n\t\t"
        board_num = ET.SubElement(bus_obj, "board_num")
        for stop_id in bus_obj_dic[bus_id].board_num_d.keys():
            board_num.set("s_" + stop_id, str(bus_obj_dic[bus_id].board_num_d[stop_id]))
        board_num.tail = "\n\t\t"
        strand_num = ET.SubElement(bus_obj, "strand_num")
        for stop_id in bus_obj_dic[bus_id].strand_num_d.keys():
            strand_num.set("s_" + stop_id, str(bus_obj_dic[bus_id].strand_num_d[stop_id]))
        strand_num.tail = "\n\t\t"
        arriver_signal_time = ET.SubElement(bus_obj, "arriver_signal_time")
        for signal_id in bus_obj_dic[bus_id].arriver_signal_time_d.keys():
            arriver_signal_time.set(signal_id, str(bus_obj_dic[bus_id].arriver_signal_time_d[signal_id]))
        arriver_signal_time.tail = "\n\t\t"
        depart_signal_time = ET.SubElement(bus_obj, "depart_signal_time")
        for signal_id in bus_obj_dic[bus_id].depart_signal_time_d.keys():
            depart_signal_time.set(signal_id, str(bus_obj_dic[bus_id].depart_signal_time_d[signal_id]))
        depart_signal_time.tail = "\n\t\t"
        no_stop_signal_id_list = ET.SubElement(bus_obj, "no_stop_signal_id_list")
        no_stop_signal_id_list.set("no_stop_signal_id_list_l", " ".join(bus_obj_dic[bus_id].no_stop_signal_id_list_l))
        no_stop_signal_id_list.tail = "\n\t\t"
        stop_signal_id_list = ET.SubElement(bus_obj, "stop_signal_id_list")
        stop_signal_id_list.set("stop_signal_id_list_l", " ".join(bus_obj_dic[bus_id].stop_signal_id_list_l))
        stop_signal_id_list.tail = "\n\t"
        bus_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, f"{index}_save_data_bus.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"公交车对象 XML 文件已生成。")


def save_passenger_data(passenger_obj_dic):
    # 创建XML根元素
    root = ET.Element("passenger_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for passenger_id in passenger_obj_dic.keys():
        passenger_obj = ET.SubElement(root, "passenger_obj")
        passenger_obj.set("passenger_id_s", passenger_obj_dic[passenger_id].passenger_id_s)
        passenger_obj.set("start_time_n", str(passenger_obj_dic[passenger_id].start_time_n))
        passenger_obj.set("start_stop_id_s", passenger_obj_dic[passenger_id].start_stop_id_s)
        passenger_obj.set("transfer_stop_id_s", passenger_obj_dic[passenger_id].transfer_stop_id_s)
        passenger_obj.set("end_stop_id_s", passenger_obj_dic[passenger_id].end_stop_id_s)
        passenger_obj.set("start_edge_id_s", passenger_obj_dic[passenger_id].start_edge_id_s)
        passenger_obj.set("transfer_edge_id_s", passenger_obj_dic[passenger_id].transfer_edge_id_s)
        passenger_obj.set("end_edge_id_s", passenger_obj_dic[passenger_id].end_edge_id_s)
        passenger_obj.text = "\n\t\t"
        travel_data = ET.SubElement(passenger_obj, "travel_data")
        travel_data.set("travel_data_l", " ".join([";".join([str(m) for m in n]) for n in
                                                   passenger_obj_dic[passenger_id].travel_data_l]))
        travel_data.tail = "\n\t"
        passenger_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, "save_data_passenger.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"乘客对象 XML 文件已生成。")


def save_passenger_data_multiexp(passenger_obj_dic, index):
    # 创建XML根元素
    root = ET.Element("passenger_obj_xml")
    # 添加换行和空格
    root.text = "\n\t"
    for passenger_id in passenger_obj_dic.keys():
        passenger_obj = ET.SubElement(root, "passenger_obj")
        passenger_obj.set("passenger_id_s", passenger_obj_dic[passenger_id].passenger_id_s)
        passenger_obj.set("start_time_n", str(passenger_obj_dic[passenger_id].start_time_n))
        passenger_obj.set("start_stop_id_s", passenger_obj_dic[passenger_id].start_stop_id_s)
        passenger_obj.set("transfer_stop_id_s", passenger_obj_dic[passenger_id].transfer_stop_id_s)
        passenger_obj.set("end_stop_id_s", passenger_obj_dic[passenger_id].end_stop_id_s)
        passenger_obj.set("start_edge_id_s", passenger_obj_dic[passenger_id].start_edge_id_s)
        passenger_obj.set("transfer_edge_id_s", passenger_obj_dic[passenger_id].transfer_edge_id_s)
        passenger_obj.set("end_edge_id_s", passenger_obj_dic[passenger_id].end_edge_id_s)
        passenger_obj.text = "\n\t\t"
        travel_data = ET.SubElement(passenger_obj, "travel_data")
        travel_data.set("travel_data_l", " ".join([";".join([str(m) for m in n]) for n in
                                                   passenger_obj_dic[passenger_id].travel_data_l]))
        travel_data.tail = "\n\t"
        passenger_obj.tail = "\n\t"
    # 创建XML文件并写入数据
    xml_tree = ET.ElementTree(root)
    xml_file = os.path.join(output_file_dir, f"{index}_save_data_passenger.add.xml")
    xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"乘客对象 XML 文件已生成。")

