import json, math, re, xml.etree.ElementTree as ET
from svgpathtools import svg2paths2, parse_path, Line, CubicBezier, QuadraticBezier, Arc, Path
import math
import json
import re
from config import *

def parse_svg_file(file_path ,L1,L2):
    svg,_,attrs = svg2paths2(file_path)
    # width, height 속성에서 크기 추출
    svg_w = parse_len(attrs.get('width'))
    svg_h = parse_len(attrs.get('height'))
    if (svg_w == 0 or svg_h == 0) and 'viewBox' in attrs: # width, height가 없으면 svg viewBox 속성에서 크기 추출
        _, _, svg_w, svg_h = map(float, attrs["viewBox"].split())
    svg_diagonal = math.sqrt(svg_w**2 + svg_h**2)
    scale =  math.sqrt(L1**2 + L2**2) / svg_diagonal # 좌표에 곱할 scale 계산
    cmds = []
    for path in svg:
        cmds.append(cur_to_line(path,scale,MAX_ERR))
    return cmds

def _dist_point_to_segment(px, py, x1, y1, x2, y2): # x1,y1 -> x2,y2 선분에 점 px,py까지의 거리
    # (x1,y1)-(x2,y2) 선분에 점 (px,py)까지의 거리
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        # 선분이 한 점일 때
        dx, dy = px - x1, py - y1
        return math.sqrt(dx*dx + dy*dy)
    t = (wx*vx + wy*vy) / seg_len2
    t = max(0.0, min(1.0, t))
    projx = x1 + t*vx
    projy = y1 + t*vy
    dx, dy = px - projx, py - projy
    return math.sqrt(dx*dx + dy*dy) # x1,y1 -> x2,y2 선분에 점 px,py까지의 거리

def _sample_segment_adaptive(seg, t0, t1, max_err, depth, max_depth, out):
    """seg: svgpathtools 세그먼트
       t0,t1: 이 세그먼트 구간
       out: 점을 (x,y)로 append하는 리스트
    """
    p0 = seg.point(t0)
    p1 = seg.point(t1)
    x0, y0 = p0.real, p0.imag
    x1, y1 = p1.real, p1.imag

    # 중간점
    tm = (t0 + t1) / 2.0
    pm = seg.point(tm)
    xm, ym = pm.real, pm.imag

    # 중간점이 직선에서 얼마나 벗어났는지
    err = _dist_point_to_segment(xm, ym, x0, y0, x1, y1)

    if err <= max_err or depth >= max_depth:
        # 이 구간은 직선으로 충분 → 끝점만 추가
        out.append((x1, y1))
    else:
        # 더 나눠
        _sample_segment_adaptive(seg, t0, tm, max_err, depth+1, max_depth, out)
        _sample_segment_adaptive(seg, tm, t1, max_err, depth+1, max_depth, out)

def segment_to_points_adaptive(seg, max_err=0.1, max_depth=10):
    """세그먼트 하나를 적응형으로 점 리스트로"""
    points = []
    p0 = seg.point(0.0)
    points.append((p0.real, p0.imag))
    _sample_segment_adaptive(seg, 0.0, 1.0, max_err, 0, max_depth, points)
    return points

def cur_to_line(path, scale, max_err=0.5, max_depth=10):
    cmd = []
    # 시작점
    start = path[0].start
    cmd.append({
        'cmd': 'M',
        'points': [start.real * scale, -start.imag * scale]
    })

    for seg in path:
        if isinstance(seg, Line):
            # 직선은 그냥 끝점 하나면 됨
            end = seg.end
            cmd.append({
                'cmd': 'L',
                'points': [end.real * scale, -end.imag * scale]
            })
        else:
            # 곡선류는 적응형으로
            pts = segment_to_points_adaptive(seg, max_err=max_err, max_depth=max_depth)
            # 첫 점은 이전 세그먼트 끝과 같으니까 생략하고 나머지 추가
            for (x, y) in pts[1:]:
                cmd.append({
                    'cmd': 'L',
                    'points': [x * scale, -y * scale]
                })

    return cmd
def parse_len(v): # attrs에서 추출하기 위해 사용하는 함수
    if v is None:
        return 0
    return float(re.sub(r"[^0-9.+-eE]", "", v))