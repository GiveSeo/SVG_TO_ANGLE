import math
import matplotlib.pyplot as plt
from config import *
def split_lines(cmd_list, max_len):
    new_cmds = []
    prev = None

    for cmd in cmd_list:
        if cmd['cmd'] == 'M':
            x, y = cmd['points']
            new_cmds.append({
                "pen": 0,
                "x": x,
                "y": y
            })
            prev = (x, y)
        elif cmd['cmd'] == 'L':
            x, y = cmd['points']
            x0, y0 = prev
            dx = x - x0
            dy = y - y0
            dist = math.hypot(dx, dy)

            if dist <= max_len:
                new_cmds.append({"pen": 1, "x": x, "y": y})
                prev = (x, y)
            else:
                steps = math.ceil(dist / max_len)
                for i in range(1, steps + 1):
                    t = i / steps
                    nx = x0 + dx * t
                    ny = y0 + dy * t
                    new_cmds.append({"pen": 1, "x": nx, "y": ny})
                prev = (x, y)
    return new_cmds

# 역기구학
def point_to_angle(L1, L2, x, y):
    r2 = x*x + y*y

    # 도달 범위 체크
    if r2 > (L1 + L2)**2 or r2 < (L1 - L2)**2:
        return None

    cos_t2 = (r2 - L1*L1 - L2*L2) / (2 * L1 * L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    sin_t2 = math.sqrt(1.0 - cos_t2*cos_t2)
    theta2 = math.atan2(sin_t2, cos_t2)

    k1 = L1 + L2 * cos_t2
    k2 = L2 * sin_t2
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)
    return theta1, theta2

def point_list_to_angle_list(cmd_list, L1, L2):
    angle_cmds = []
    pre = None
    for cmd in cmd_list:
        res = point_to_angle(L1, L2, cmd['x'], cmd['y'])
        if res is None:
            angle_cmds.append({
                "pen": 0,
                "theta1": None,
                "theta2": None,
                "x": cmd['x'],
                "y": cmd['y']
            })
            continue

        t1, t2 = res
        cur_p1 = int(t1 / PULSE)
        cur_p2 = int(t2 / PULSE)

        if pre is None:
            angle_cmds.append({
                "pen": cmd['pen'],
                "theta1": cur_p1,
                "theta2": cur_p2,
                "x": cmd['x'],
                "y": cmd['y']
            })
        else:
            angle_cmds.append({
                "pen": cmd['pen'],
                "theta1": cur_p1 - pre[0],
                "theta2": cur_p2 - pre[1],
                "x": cmd['x'],
                "y": cmd['y']
            })

        pre = (cur_p1, cur_p2)
    return angle_cmds


def draw_2link_from_angle_list(angle_points, L1, L2,show_path=True, pause_sec=1):
    
    
    MAX_PULSE_PER_10MS = 100.0 # 10ms 당 1도를 최대 움직일 수 있음.
    effective_pause = pause_sec if pause_sec > 0 else 0.01
    max_pulse_per_frame = MAX_PULSE_PER_10MS * (effective_pause / 0.01)    
    #시각화 초기 설정
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.set_xlim(-L1-L2, L1+L2)
    ax.set_ylim(-L1-L2, L1+L2)
    ax.grid(True)
    ax.set_title("2-Link Sequential Motion")

    trace_line, = ax.plot([], [], '-', color='red', linewidth=2)
    arm_line, = ax.plot([], [], 'o-', color='black', linewidth=3)
    tip_x, tip_y = [], []

    cur_t1 = 0.0
    cur_t2 = 0.0
    initialized = False
    pen_down = False  # 지금 펜 내려가 있는지 여부

    for p in angle_points:
        pen = p.get("pen", 1)
        inc1 = p.get("theta1")
        inc2 = p.get("theta2")

        # 펜 상태 전환 처리
        if pen == 0:
            pen_down = False
        else:
            # 0 → 1로 올라온 순간: 라인만 끊어주고 기존건 살려둠
            if not pen_down:
                tip_x.append(None)  # 여기서 선 끊김
                tip_y.append(None)
                pen_down = True

        # 첫 점이면서 pen=0인 경우: 위치만 세팅
        if not initialized and pen == 0:
            cur_t1 = inc1
            cur_t2 = inc2
            initialized = True

            theta1 = (cur_t1 * PULSE)
            theta2 = (cur_t2 * PULSE)
            x1 = L1 * math.cos(theta1)
            y1 = L1 * math.sin(theta1)
            x2 = x1 + L2 * math.cos(theta1 + theta2)
            y2 = y1 + L2 * math.sin(theta1 + theta2)
            arm_line.set_data([0, x1, x2], [0, y1, y2])
            plt.pause(pause_sec)
            continue

        # 목표 각도
        target_t1 = cur_t1 + inc1
        target_t2 = cur_t2 + inc2

        # joint1 먼저 목표각에 도달
        while abs(cur_t1 - target_t1) > 0.001:
            diff = target_t1 - cur_t1
            step = math.copysign(min(abs(diff), max_pulse_per_frame), diff)
            cur_t1 += step

            theta1 = (cur_t1 * PULSE)
            theta2 = (cur_t2 * PULSE)
            x1 = L1 * math.cos(theta1)
            y1 = L1 * math.sin(theta1)
            x2 = x1 + L2 * math.cos(theta1 + theta2)
            y2 = y1 + L2 * math.sin(theta1 + theta2)

            if pen_down:
                tip_x.append(x2)
                tip_y.append(y2)
                if show_path:
                    trace_line.set_data(tip_x, tip_y)
            arm_line.set_data([0, x1, x2], [0, y1, y2])
            plt.pause(pause_sec)

        # joint2 다음 목표각에 도달
        while abs(cur_t2 - target_t2) > 0.001:
            diff = target_t2 - cur_t2
            step = math.copysign(min(abs(diff), max_pulse_per_frame), diff)
            cur_t2 += step

            theta1 = (cur_t1 * PULSE)
            theta2 = (cur_t2 * PULSE)
            x1 = L1 * math.cos(theta1)
            y1 = L1 * math.sin(theta1)
            x2 = x1 + L2 * math.cos(theta1 + theta2)
            y2 = y1 + L2 * math.sin(theta1 + theta2)

            if pen_down:
                tip_x.append(x2)
                tip_y.append(y2)
                if show_path:
                    trace_line.set_data(tip_x, tip_y)
            arm_line.set_data([0, x1, x2], [0, y1, y2])
            plt.pause(pause_sec)

        cur_t1 = target_t1
        cur_t2 = target_t2

    plt.show()
