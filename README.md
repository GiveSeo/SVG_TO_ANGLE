# svg to 2-Link Robot Angle Converter

이 프로젝트는 PATH 기반 SVG를 파싱하여 2링크 로봇팔이 그릴 수 있는 관절 각도 형태로 변환하는 프로젝트입니다.<br>
svgpathtools, xml.etree.ElementTree를 이용하여 SVG파싱을 진행해 좌표값을 얻어, 2DOF 역기구학을 통해 좌표를 각도로 변환하고, matplotlib을 통해 로봇팔의 움직임을 시각화합니다.<br>


## 파일 구조
├── config.py       # 설정 파일 (로봇팔 길이, 펄스 단위, SVG 파일 경로)<br>
├── parse.py        # SVG 파싱 및 곡선-직선 변환<br>
├── util.py         # 역기구학, 좌표 분할, 시각화 함수<br>
├── main.py         # 메인 실행 파일<br>
├── Mouse.svg       # 입력 SVG 파일 (예시)<br>
└── output.json  # 출력 JSON 파일 (생성됨)<br>

## 사용 방법
config.py의 파일에 다음 정보를 사용하고자 하는 로봇 정보와 svg 파일 이름과 일치시킵니다.<br>
```python
# 파일 이름
SVG_PATH = "Mouse.svg"
OUTPUT_PATH = "output.json"

# 펄스 단위
PULSE = 0.01

# 선 근사 시 오차 범위
MAX_ERR = 0.1
# 로봇팔 링크 길이
L1 = 20  # 첫 번째 링크 길이
L2 = 20  # 두 번째 링크 길이

#좌표 분해시 최대 길이 단위
MAX_LENGTH = min(L1, L2) / 100
```
이후, 프로젝트 안에 해당 svg 파일을 넣고<br>
```bash
python main.py
```
로 실행합니다.
실행 시, 좌표와 각도 정보가 담긴 output.json 파일과 matplotlib를 통해 시각화 창이 출력됩니다.<br><br>

출력 json 형식은<br>
```json
[
  {
    "pen": 0,           // 0: 펜 올림, 1: 펜 내림
    "theta1": 314,      // joint1 펄스 증분값
    "theta2": -157,     // joint2 펄스 증분값
    "x": 150.5,         // 목표 x 좌표
    "y": 200.3          // 목표 y 좌표
  }
]
```
다음과 같습니다.<br>
또한, 맨 처음 theta값은 증분값이 아닌 초기 pulse 값이라 생각하면 됩니다.<br>

## 주요 함수 설명

### parse.py
parse_svg_file(file_path ,L1,L2): svg파일을 파싱하고, svg 크기와 로봇팔 크기에 따라 전체 좌표를 스케일링합니다.<br>
cur_to_line(path, scale, max_err=0.5, max_depth=10): path에서 곡선을 직선으로 변환합니다. 이때, 직선 변환시 적응형으로 변환합니다.(곡선 기울기에 따라 직선 변환 수가 달라집니다.)<br>

### util.py
split_lines(cmd_list, max_len): 좌표 리스트를 max_len 단위보다 작거나 같게 여러개의 좌표로 분할합니다.<br>
point_to_angle(L1, L2, x, y): 역기구학을 통해 좌표 -> 각도 단위로 변환합니다.<br>
point_list_to_angle_list(cmd_list, L1, L2): 역기구학 함수를 활용하여, 좌표 리스트를 펄스 증분 리스트로 변환합니다. (1 펄스 = 0.01도)<br>
draw_2link_from_angle_list(angle_points, L1, L2,show_path=True, pause_sec=0.01): 로봇팔 움직임을 matplotlib을 통해 시각화 합니다. 그림은 joint1 -> joint2 가 각각 순차대로 움직입니다.(시각화 상 최대 오차를 확인하기 위해 각각의 모터가 따로 움직이는 방식으로 시각화 하게 되었습니다.)<br>


## 역기구학 알고리즘

좌표(x,y)와 로봇팔의 길이(L1,L2)를 통해 (각도1, 각도2)로 변환하는 알고리즘입니다.<br><br>

theta2 계산: cos(θ2) = (x² + y² - L1² - L2²) / (2·L1·L2)<br>
k1 = L1 + L2 * cos_t2 <br>
k2 = L2 * sin_t2 <br>
theta1 계산: θ1 = atan2(y, x) - atan2(k2, k1)<br>

## 기타 설정 시 주의사항

PULSE 단위, MAX_LENGTH, MAX_ERR에 따라서 그림의 정확도가 달라질 수 있습니다.(PULSE 단위의 각도가 낮을 수록, MAX_LENGTH가 짧을 수록, MAX_ERR가 낮을 수록 그림이 정확하게 그려집니다.)
