import math
import re
import xml.etree.ElementTree as ET

from svgpathtools import parse_path, Line
from config import *  # L1, L2, MAX_ERR 등 사용한다고 가정

def parse_len(v):
    # 단위 무시하고 숫자만 추출하는 함수
    if v is None:
        return 0
    return float(re.sub(r"[^0-9.+-eE]", "", v))

# 2x3 변환 행렬 관련 함수 - svg의 transform 속성에 따른 좌표 변환을 적용하기 위해 사용합니다.
#   [a, b, c, d, e, f]
#   x' = a*x + c*y + e
#   y' = b*x + d*y + f

# 항등 행렬 반환 (아무 효과도 없는 상태)
def identity_matrix():
    return [1, 0, 0, 1, 0, 0]

# 행렬 곱 함수, m1 적용 후 m2 적용
def multiply_matrix(m2, m1):
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return [
        a2 * a1 + c2 * b1,
        b2 * a1 + d2 * b1,
        a2 * c1 + c2 * d1,
        b2 * c1 + d2 * d1,
        a2 * e1 + c2 * f1 + e2,
        b2 * e1 + d2 * f1 + f2,
    ]

# transform 중 translate() 적용을 위한 행렬 만들기 함수 (평행 이동)
def translate_matrix(tx, ty):
    return [1, 0, 0, 1, tx, ty]

# transform 중 scale() 적용을 위한 행렬 만들기 함수
def scale_matrix(sx, sy=None):
    if sy is None:# sy가 없으면 sx와 같게 설정
        sy = sx
    return [sx, 0, 0, sy, 0, 0]

# transform 중 rotate(angle) 적용을 위한 행렬 만들기 함수 (원점 기준 회전)
def rotate_matrix(angle_deg):
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return [c, s, -s, c, 0, 0]


def matrix_from_svg(a, b, c, d, e, f):
    return [a, b, c, d, e, f]

# transform 속성을 파싱하는 함수(svgpathtools에서 transform 파싱을 지원하지 않기 때문에, 문자열 직접 파싱)
def parse_transform_attr(transform_str):
    M = identity_matrix()
    if not transform_str:
        return M

    pattern = r'(matrix|translate|scale|rotate)\s*\(([^)]*)\)'
    for m in re.finditer(pattern, transform_str):
        kind = m.group(1) #어느 옵션인지(ex) matrix, translate, scale.. )
        args = [float(x) for x in re.split(r'[ ,]+', m.group(2).strip()) if x]
        T = identity_matrix()

        if kind == 'matrix' and len(args) == 6:
            T = matrix_from_svg(*args)

        elif kind == 'translate':
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0.0
            T = translate_matrix(tx, ty)

        elif kind == 'scale':
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            T = scale_matrix(sx, sy)

        elif kind == 'rotate':
            # rotate(angle) 또는 rotate(angle, cx, cy)
            if len(args) == 1:
                angle = args[0]
                T = rotate_matrix(angle)
            elif len(args) == 3:
                angle, cx, cy = args
                R = rotate_matrix(angle)
                T = multiply_matrix(
                    translate_matrix(cx, cy),
                    multiply_matrix(R, translate_matrix(-cx, -cy))
                )
        # transform 옵션에 해당하는 행렬을 계속 구해 곱함
        M = multiply_matrix(T, M)
        

    return M

# x,y에 M 행렬 
def apply_matrix(x, y, M):
    a, b, c, d, e, f = M
    x2 = a * x + c * y + e
    y2 = b * x + d * y + f
    return x2, y2


# 선분과 점 사이 최단거리 계산 함수 (적응형 직선화에 필요한 함수)
def _dist_point_to_segment(px, py, x1, y1, x2, y2):
    """(x1,y1)-(x2,y2) 선분과 점 (px,py) 사이 거리"""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0:
        dx, dy = px - x1, py - y1
        return math.sqrt(dx * dx + dy * dy)

    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    projx = x1 + t * vx
    projy = y1 + t * vy
    dx, dy = px - projx, py - projy
    return math.sqrt(dx * dx + dy * dy)


def _sample_segment_adaptive(seg, t0, t1, max_err, depth, max_depth, out_local):
    p0 = seg.point(t0)
    p1 = seg.point(t1)
    x0, y0 = p0.real, p0.imag
    x1, y1 = p1.real, p1.imag

    tm = (t0 + t1) / 2.0
    pm = seg.point(tm)
    xm, ym = pm.real, pm.imag

    err = _dist_point_to_segment(xm, ym, x0, y0, x1, y1)

    if err <= max_err or depth >= max_depth:
        out_local.append((x1, y1))
    else:
        _sample_segment_adaptive(seg, t0, tm, max_err, depth + 1, max_depth, out_local)
        _sample_segment_adaptive(seg, tm, t1, max_err, depth + 1, max_depth, out_local)

# segment 하나를 점 리스트로 변환하는 함수
def segment_to_points_adaptive(seg, max_err=0.1, max_depth=10):
    points = []
    p0 = seg.point(0.0)
    points.append((p0.real, p0.imag))
    _sample_segment_adaptive(seg, 0.0, 1.0, max_err, 0, max_depth, points)
    return points


def cur_to_line_with_matrix(path, scale, transform_matrix, max_err=0.5, max_depth=10):
    """
    svgpathtools.Path → M/L 명령 리스트
    transform_matrix, scale 적용
    """
    cmd = []

    # 시작점
    start = path[0].start
    sx, sy = apply_matrix(start.real, start.imag, transform_matrix)
    cmd.append({
        'cmd': 'M',
        'points': [sx * scale, -sy * scale]
    })

    for seg in path:
        if isinstance(seg, Line):
            end = seg.end
            ex, ey = apply_matrix(end.real, end.imag, transform_matrix)
            cmd.append({
                'cmd': 'L',
                'points': [ex * scale, -ey * scale]
            })
        else:
            pts = segment_to_points_adaptive(seg, max_err=max_err, max_depth=max_depth)
            for (x, y) in pts[1:]:
                tx, ty = apply_matrix(x, y, transform_matrix)
                cmd.append({
                    'cmd': 'L',
                    'points': [tx * scale, -ty * scale]
                })

    return cmd


# -------------------------
# 기본 도형(rect, circle, ...) → 포인트 리스트
# -------------------------
def points_to_cmd(points, scale, transform_matrix):
    """
    points: [(x,y), ...]  (원본 SVG 좌표계)
    transform + scale 적용해서 M/L 명령으로 변환
    """
    if not points:
        return []

    out = []
    first = True
    for (x, y) in points:
        tx, ty = apply_matrix(x, y, transform_matrix)
        px = tx * scale
        py = -ty * scale

        if first:
            out.append({'cmd': 'M', 'points': [px, py]})
            first = False
        else:
            out.append({'cmd': 'L', 'points': [px, py]})
    return out


def rect_to_points(elem):
    x = parse_len(elem.get('x'))
    y = parse_len(elem.get('y'))
    w = parse_len(elem.get('width'))
    h = parse_len(elem.get('height'))
    # 시계 방향 + 닫기
    return [
        (x,     y),
        (x + w, y),
        (x + w, y + h),
        (x,     y + h),
        (x,     y),
    ]


def line_to_points(elem):
    x1 = parse_len(elem.get('x1'))
    y1 = parse_len(elem.get('y1'))
    x2 = parse_len(elem.get('x2'))
    y2 = parse_len(elem.get('y2'))
    return [(x1, y1), (x2, y2)]


def polyline_to_points(elem, close=False):
    points_attr = elem.get('points', '')
    pts = []
    # "x,y x,y ..." 파싱
    for pair in re.findall(
        r'([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        points_attr
    ):
        x = float(pair[0])
        y = float(pair[1])
        pts.append((x, y))
    if close and pts:
        pts.append(pts[0])
    return pts


def circle_to_points(elem, segments=32):
    cx = parse_len(elem.get('cx'))
    cy = parse_len(elem.get('cy'))
    r = parse_len(elem.get('r'))
    pts = []
    for i in range(segments + 1):
        t = 2 * math.pi * i / segments
        x = cx + r * math.cos(t)
        y = cy + r * math.sin(t)
        pts.append((x, y))
    return pts


def ellipse_to_points(elem, segments=32):
    cx = parse_len(elem.get('cx'))
    cy = parse_len(elem.get('cy'))
    rx = parse_len(elem.get('rx'))
    ry = parse_len(elem.get('ry'))
    pts = []
    for i in range(segments + 1):
        t = 2 * math.pi * i / segments
        x = cx + rx * math.cos(t)
        y = cy + ry * math.sin(t)
        pts.append((x, y))
    return pts


# -------------------------
# 기타 유틸
# -------------------------
def strip_ns(tag):
    """{namespace}tag → tag"""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


# -------------------------
# 메인: SVG 파싱
# path + 기본 도형 + use + transform 모두 처리
# -------------------------
def parse_svg_file(file_path, L1, L2, max_err=None):
    """
    file_path: SVG 파일 경로
    L1, L2: 로봇팔 링크 길이 (스케일 계산용)
    max_err: 곡선 근사 허용 오차 (없으면 config.MAX_ERR 사용)
    return: [
        [ {'cmd': 'M'/'L', 'points': [x,y]}, ... ],  # path 1
        [ ... ],                                    # path 2
        ...
    ]
    """
    if max_err is None:
        max_err = MAX_ERR

    tree = ET.parse(file_path)
    root = tree.getroot()

    # SVG 전체 크기 → 대각선 길이
    svg_w = parse_len(root.get('width'))
    svg_h = parse_len(root.get('height'))
    if (svg_w == 0 or svg_h == 0) and 'viewBox' in root.attrib:
        _, _, svg_w, svg_h = map(float, root.attrib['viewBox'].split())

    if svg_w == 0 or svg_h == 0:
        svg_w, svg_h = 1.0, 1.0

    svg_diagonal = math.sqrt(svg_w ** 2 + svg_h ** 2)
    arm_diagonal = math.sqrt(L1 ** 2 + L2 ** 2)
    if svg_diagonal == 0:
        svg_diagonal = 1.0
    scale = arm_diagonal / svg_diagonal

    cmds_all = []

    # id → element 매핑 (use에서 참조용)
    id_map = {}
    for el in root.iter():
        el_id = el.get('id')
        if el_id:
            id_map[el_id] = el

    def walk(node, parent_matrix):
        tag = strip_ns(node.tag)

        # ------- <use> : x,y + transform 같이 처리 -------
        if tag == 'use':
            href = node.get('{http://www.w3.org/1999/xlink}href') or node.get('href')
            tx = parse_len(node.get('x'))
            ty = parse_len(node.get('y'))

            # x,y를 translate로 보고 transform과 함께 합침
            tf_str = node.get('transform') or ""
            combined_tf = f"translate({tx},{ty}) {tf_str}".strip()

            node_M = parse_transform_attr(combined_tf)
            cur_M = multiply_matrix(node_M, parent_matrix)

            if href and href.startswith('#'):
                ref_id = href[1:]
                target = id_map.get(ref_id)
                if target is not None:
                    walk(target, cur_M)
            return

        # ------- 일반 노드: transform + 부모 행렬 -------
        node_M = parse_transform_attr(node.get('transform'))
        cur_M = multiply_matrix(node_M, parent_matrix)

        # 그룹류: 자식만 내려감
        if tag in ('svg', 'g', 'defs', 'symbol'):
            for child in node:
                walk(child, cur_M)
            return

        # path
        if tag == 'path':
            d = node.get('d')
            if not d:
                return
            path = parse_path(d)
            cmds = cur_to_line_with_matrix(path, scale, cur_M, max_err=max_err)
            cmds_all.append(cmds)
            return

        # rect / line / polyline / polygon / circle / ellipse
        if tag == 'rect':
            pts = rect_to_points(node)
            cmds_all.append(points_to_cmd(pts, scale, cur_M))
            return

        if tag == 'line':
            pts = line_to_points(node)
            cmds_all.append(points_to_cmd(pts, scale, cur_M))
            return

        if tag == 'polyline':
            pts = polyline_to_points(node, close=False)
            cmds_all.append(points_to_cmd(pts, scale, cur_M))
            return

        if tag == 'polygon':
            pts = polyline_to_points(node, close=True)
            cmds_all.append(points_to_cmd(pts, scale, cur_M))
            return

        if tag == 'circle':
            pts = circle_to_points(node, segments=64)
            cmds_all.append(points_to_cmd(pts, scale, cur_M))
            return

        if tag == 'ellipse':
            pts = ellipse_to_points(node, segments=64)
            cmds_all.append(points_to_cmd(pts, scale, cur_M))
            return

        # 그 외 타입들은 일단 무시
        return

    # 루트부터 시작
    walk(root, identity_matrix())

    return cmds_all
