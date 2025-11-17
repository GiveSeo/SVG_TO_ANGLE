from parse import *
import svg_parse 
from util import *
from config import *
import json

if __name__ == "__main__":

    # 구 버전 parsing
    cmds = parse_svg_file(SVG_PATH, L1, L2)
    # 신 버전 parsing
    #cmds = svg_parse.parse_svg_file(SVG_PATH, L1, L2)
    line_cmds = []
    for cmd in cmds:
        line_cmds.extend(split_lines(cmd, MAX_LENGTH))  # 최대 선 길이 max_length 단위로 쪼개기

    angle_cmds = point_list_to_angle_list(line_cmds, L1, L2)

    # 결과를 JSON 파일로 저장
    with open(OUTPUT_PATH, "w") as f:
        json.dump(angle_cmds, f, indent=4)

    # 시각화
    draw_2link_from_angle_list(angle_cmds, L1, L2, show_path=True, pause_sec=0.01)