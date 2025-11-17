# all angles: degree
# device command: pulse (0.01deg = 1/PULSE)
# 10ms / 1 degree

# 파일 이름
SVG_PATH = "Mouse.svg"
OUTPUT_PATH = "output.json"

# 펄스 단위
PULSE = 0.01

MAX_ERR = 0.5
# 로봇팔 링크 길이
L1 = 100  # 첫 번째 링크 길이
L2 = 100  # 두 번째 링크 길이

#좌표 분해시 최대 길이 단위
MAX_LENGTH = min(L1, L2) / 100
