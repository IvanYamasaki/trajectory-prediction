import math
import pickle
from reader.Reader import Reader # Removido o ponto
from reader.MessageType import MessageType # Removido o ponto

def add_robot_measurement(dic, elem, pkt):
    dic[elem.robot_id]['x'].append(elem.x)
    dic[elem.robot_id]['y'].append(elem.y)
    dic[elem.robot_id]['psi'].append(elem.orientation)
    dic[elem.robot_id]['time_c'].append(pkt.detection.t_capture)
    dic[elem.robot_id]['mask'].append(True)

def add_robot_element(dic, st, elem, pkt):
    if elem.robot_id not in dic:
        dic[elem.robot_id] = {'x': [], 'y': [], 'psi': [], 'time_c': [], 'mask': []}
    if elem.robot_id not in st:
        diff = -50
        if len(dic[elem.robot_id]['time_c']) > 0:
            diff = pkt.detection.t_capture - dic[elem.robot_id]['time_c'][-1]
        if diff == -50 or (0.01 < diff < 0.022):
            add_robot_measurement(dic, elem, pkt)
            st.add(elem.robot_id)
        elif diff >= 0.022:
            steps = math.floor(diff * 60)
            for k in range(steps):
                if (pkt.detection.t_capture - dic[elem.robot_id]['time_c'][-1] - (1 / 60)) < 0.01:
                    break
                dic[elem.robot_id]['x'].append(dic[elem.robot_id]['x'][-1])
                dic[elem.robot_id]['y'].append(dic[elem.robot_id]['y'][-1])
                dic[elem.robot_id]['psi'].append(dic[elem.robot_id]['psi'][-1])
                dic[elem.robot_id]['time_c'].append(dic[elem.robot_id]['time_c'][-1] + (1 / 60))
                dic[elem.robot_id]['mask'].append(False)
            add_robot_measurement(dic, elem, pkt)
            st.add(elem.robot_id)

def add_ball_measurement(dic, elem, pkt):
    dic['x'].append(elem.x)
    dic['y'].append(elem.y)
    dic['mask'].append(True)
    dic['time_c'].append(pkt.detection.t_capture)

def add_ball_element(dic, elem, pkt):
    diff = -50
    if len(dic['time_c']) > 0:
        diff = pkt.detection.t_capture - dic['time_c'][-1]
    if diff == -50 or (0.01 < diff < 0.022):
        add_ball_measurement(dic, elem, pkt)
    elif diff >= 0.022:
        steps = math.floor(diff * 60)
        for k in range(steps):
            if (pkt.detection.t_capture - dic['time_c'][-1] - (1 / 60)) < 0.01:
                break
            dic['x'].append(dic['x'][-1])
            dic['y'].append(dic['y'][-1])
            dic['time_c'].append(dic['time_c'][-1] + (1 / 60))
            dic['mask'].append(False)
        add_ball_measurement(dic, elem, pkt)

def process_log(path):
    reader = Reader('dataset/' + path + '.log')
    reader.read_header()
    collect = False
    i = 0
    robots_b, robots_y = {}, {}
    ball = {'x': [], 'y': [], 'time_c': [], 'mask': []}
    all_data = {'yellow': [], 'blue': [], 'ball': [], 'stop_id': []}

    while reader.has_next():
        msg_type = reader.decode_msg()
        if msg_type in [MessageType.MESSAGE_SSL_VISION_2010, MessageType.MESSAGE_SSL_VISION_2014, MessageType.MESSAGE_SSL_VISION_TRACKER_2020]:
            wrapper_packet = reader.get_wrapper_packet()
            if wrapper_packet is None or wrapper_packet.detection is None: continue
            if wrapper_packet.detection.t_capture == 0: continue

            if collect:
                if len(wrapper_packet.detection.robots_blue) > 0:
                    st = set()
                    for elem in wrapper_packet.detection.robots_blue:
                        add_robot_element(robots_b, st, elem, wrapper_packet)
                if len(wrapper_packet.detection.robots_yellow) > 0:
                    st = set()
                    for elem in wrapper_packet.detection.robots_yellow:
                        add_robot_element(robots_y, st, elem, wrapper_packet)
                if len(wrapper_packet.detection.balls) > 0:
                    add_ball_element(ball, wrapper_packet.detection.balls[0], wrapper_packet)

        elif msg_type == MessageType.MESSAGE_SSL_REFBOX_2013:
            command = reader.get_referee_packet().command
            if command in [2, 3]:
                collect = True
            elif collect:
                collect = False
                all_data['blue'].append(robots_b)
                all_data['yellow'].append(robots_y)
                all_data['ball'].append(ball)
                all_data['stop_id'].append(i)
                i += 1
                robots_b, robots_y = {}, {}
                ball = {'x': [], 'y': [], 'time_c': [], 'mask': []}

    with open('dataset/' + path + '.pkl', 'wb') as f:
        pickle.dump(all_data, f)