# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import math
import json
import trimesh
from shapely.geometry import Polygon, MultiPolygon, LineString
import tempfile
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict, Any
from datetime import date, datetime
from pathlib import Path

# =========================
# App basics
# =========================
st.set_page_config(page_title="3DCP 슬라이서", layout="wide")

# ── 전역 CSS ──
st.markdown(
    '''
    <style>
    footer {visibility: hidden;}
    [data-testid="stFooter"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}

    .block-container { padding-top: 2.0rem; }
    .stTabs { margin-top: 1.0rem !important; padding-top: 0.2rem !important; }
    .stTabs { overflow: hidden !important; }
    .stTabs [data-baseweb="tab-list"] {
      margin-top: 0.6rem !important;
      display: grid !important;
      grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
      gap: 0.35rem !important;
      width: 100% !important;
      align-items: stretch !important;
      overflow: hidden !important;
    }
    .stTabs [data-baseweb="tab"] {
      width: 100% !important;
      max-width: 100% !important;
      min-width: 0 !important;
      height: auto !important;
      min-height: 48px !important;
      white-space: normal !important;
      word-break: break-word !important;
      overflow-wrap: anywhere !important;
      text-align: center !important;
      line-height: 1.2 !important;
      padding: 0.5rem 0.55rem !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      overflow: hidden !important;
      box-sizing: border-box !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
      white-space: normal !important;
      word-break: break-word !important;
      overflow-wrap: anywhere !important;
      margin: 0 !important;
      line-height: 1.2 !important;
      text-align: center !important;
      max-width: 100% !important;
    }
    @media (max-width: 1200px) {
      .stTabs [data-baseweb="tab-list"] {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
      }
    }
    @media (max-width: 760px) {
      .stTabs [data-baseweb="tab-list"] {
        grid-template-columns: minmax(0, 1fr) !important;
      }
    }

    .left-panel-scroll {
      position: sticky;
      top: 2.0rem;
      max-height: calc(100vh - 2rem);
      overflow-y: auto;
      overflow-x: hidden;
      padding-right: 8px;
      scrollbar-gutter: stable;
      min-height: 0;
    }
    .left-panel-scroll::-webkit-scrollbar,
    .right-panel::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    .left-panel-scroll::-webkit-scrollbar-thumb,
    .right-panel::-webkit-scrollbar-thumb {
      background: #c8c8c8;
      border-radius: 999px;
      border: 2px solid transparent;
      background-clip: padding-box;
    }
    .left-panel-scroll::-webkit-scrollbar-track,
    .right-panel::-webkit-scrollbar-track {
      background: #f3f3f3;
      border-radius: 999px;
    }
    .center-panel-scroll {
      position: sticky;
      top: 2.0rem;
      max-height: calc(100vh - 2rem);
      overflow-y: auto;
      overflow-x: hidden;
      min-height: 0;
      scrollbar-gutter: stable;
      padding-right: 4px;
    }
    .center-panel-scroll::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    .center-panel-scroll::-webkit-scrollbar-thumb {
      background: #c8c8c8;
      border-radius: 999px;
      border: 2px solid transparent;
      background-clip: padding-box;
    }
    .center-panel-scroll::-webkit-scrollbar-track {
      background: #f3f3f3;
      border-radius: 999px;
    }
    .right-panel {
      position: sticky;
      top: 2.0rem;
      max-height: calc(100vh - 2rem);
      overflow-y: auto;
      border-left: 1px solid #e6e6e6;
      padding-left: 12px;
      background: white;
      scrollbar-gutter: stable;
    }

    .sidebar-title {
      margin: 0.25rem 0 0.6rem 0;
      font-size: 1.5rem;
      font-weight: 700;
      line-height: 1.2;
    }

    .dims-block {
      white-space: pre-line;
      line-height: 1.3;
      font-variant-numeric: tabular-nums;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

st.sidebar.markdown("<div class='sidebar-title'>3DCP 슬라이서</div>", unsafe_allow_html=True)

EXTRUSION_K = 0.05

# 색상 팔레트
PATH_COLOR_DEFAULT = "#222222"
PATH_COLOR_LIGHT   = "#D0D0D0"
OFFSET_DARK_GRAY   = "#444444"
CAP_COLOR          = "rgba(220,0,0,0.95)"

def clamp(v, lo, hi):
    try:
        return lo if v < lo else hi if v > hi else v
    except Exception:
        return lo

# =========================
# Helpers (연산 로직)
# =========================
def ensure_open_ring(segment: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    seg = np.asarray(segment, dtype=float)
    if len(seg) >= 2 and np.linalg.norm(seg[0, :2] - seg[-1, :2]) <= tol:
        return seg[:-1]
    return seg

def trim_closed_ring_tail(segment: np.ndarray, trim_distance: float) -> np.ndarray:
    pts = np.asarray(segment, dtype=float)
    if len(pts) < 2 or trim_distance <= 0:
        return ensure_open_ring(pts)

    ring = ensure_open_ring(pts)
    n = len(ring)
    if n < 2:
        return ring

    lens = []
    for i in range(n):
        p = ring[i]
        q = ring[(i + 1) % n]
        lens.append(float(np.linalg.norm((q - p)[:2])))
    total = float(sum(lens))
    if total <= trim_distance:
        return ring

    target = total - trim_distance
    acc = 0.0
    out = [ring[0].copy()]
    i = 0
    while i < n and acc + lens[i] < target:
        acc += lens[i]
        out.append(ring[(i + 1) % n].copy())
        i += 1

    p = ring[i]; q = ring[(i + 1) % n]; d = lens[i]
    cut = p + ((target - acc) / d) * (q - p) if d > 0 else p.copy()
    out.append(cut)
    return np.asarray(out, dtype=float)

def simplify_segment(segment: np.ndarray, min_dist: float) -> np.ndarray:
    """
    전체를 등간격(min_dist)으로 쪼갠 뒤, 
    일직선상에 있는 중간 점들은 제거하여 
    직선은 양 끝점만 남기고, 곡선은 지정된 간격을 유지하도록 변경.
    """
    pts = np.asarray(segment, dtype=float)
    if len(pts) <= 2 or min_dist <= 0:
        return pts
    
    # 1. 전체 경로의 누적 길이 배열 생성
    s = _poly_arclen_s_xy(pts)
    total_length = float(s[-1])
    
    if total_length <= min_dist:
        return np.vstack([pts[0], pts[-1]])
    
    # 2. 등간격 배열 생성 및 리샘플링 (무조건 min_dist 간격으로 찍기)
    num_segments = max(1, int(np.round(total_length / min_dist)))
    s_targets = np.linspace(0.0, total_length, num_segments + 1)
    resampled_pts = _resample_polyline_by_s(pts, s_targets)
    
    # 3. 일직선(Collinear) 검사하여 직선 구간의 중간 점 제거
    if len(resampled_pts) <= 2:
        return resampled_pts
        
    out = [resampled_pts[0]]
    
    for i in range(1, len(resampled_pts) - 1):
        p_prev = out[-1]
        p_curr = resampled_pts[i]
        p_next = resampled_pts[i+1]
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        L1_L2 = float(np.linalg.norm(v1[:2]) * np.linalg.norm(v2[:2]))
        if L1_L2 < 1e-9:
            continue
            
        # 외적을 이용해 꺾인 각도의 사인(sin)값을 구함
        sin_angle = abs(v1[0]*v2[1] - v1[1]*v2[0]) / L1_L2
        
        # 꺾임이 거의 없는 직선(약 0.05도 이하)이면 점을 버리고, 곡선이면 살림
        if sin_angle > 1e-3:
            out.append(p_curr)
            
    out.append(resampled_pts[-1])
    return np.asarray(out, dtype=float)
def shift_to_nearest_start(segment, ref_point):
    """
    단순 점 검색이 아닌, 선분(Edge) 위에 수직 투영하여 가장 가까운 정확한 위치를 찾아
    새로운 시작점을 삽입하고 배열을 재배치합니다.
    """
    pts = np.asarray(segment, dtype=float)
    if len(pts) < 2:
        return pts, pts[0]
        
    ref = np.array(ref_point[:2], dtype=float)
    min_dist = float('inf')
    best_pt = None
    best_idx = 0
    
    # 모든 선분에 대해 가장 가까운 투영점 찾기
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i+1]
        
        v = p2[:2] - p1[:2]
        w = ref - p1[:2]
        
        c1 = float(np.dot(w, v))
        c2 = float(np.dot(v, v))
        
        if c1 <= 0:
            closest = p1.copy()
        elif c2 <= c1:
            closest = p2.copy()
        else:
            b = c1 / c2
            closest = p1 + b * (p2 - p1)
            
        d = float(np.linalg.norm(ref - closest[:2]))
        if d < min_dist:
            min_dist = d
            best_pt = closest
            best_idx = i
            
    # 찾은 정확한 좌표를 삽입하고 배열을 재배치
    if np.linalg.norm(best_pt - pts[best_idx]) < 1e-6:
        shift_idx = best_idx
        out = np.concatenate([pts[shift_idx:-1], pts[:shift_idx+1]], axis=0)
    elif np.linalg.norm(best_pt - pts[best_idx+1]) < 1e-6:
        shift_idx = best_idx + 1
        out = np.concatenate([pts[shift_idx:-1], pts[:shift_idx+1]], axis=0)
    else:
        out_pts = [best_pt.copy()]
        for j in range(best_idx + 1, len(pts) - 1):
            out_pts.append(pts[j].copy())
        for j in range(0, best_idx + 1):
            out_pts.append(pts[j].copy())
        out_pts.append(best_pt.copy())
        out = np.asarray(out_pts, dtype=float)
        
    return out, best_pt

def _poly_arclen_s_xy(poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly, dtype=float)
    if len(pts) < 2:
        return np.array([0.0], dtype=float)
    d = pts[1:] - pts[:-1]
    lens = np.linalg.norm(d[:, :2], axis=1)
    return np.concatenate([[0.0], np.cumsum(lens)])

def _resample_polyline_by_s(poly: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly, dtype=float)
    if len(pts) < 2:
        return pts.copy()

    s = _poly_arclen_s_xy(pts)
    total = float(s[-1])
    if total < 1e-9:
        return pts[[0]].copy()

    seg = pts[1:] - pts[:-1]
    lens = np.linalg.norm(seg[:, :2], axis=1)

    s_targets = np.clip(np.asarray(s_targets, dtype=float), 0.0, total)

    out = []
    j = 0
    for st_val in s_targets:
        while j < len(lens) - 1 and s[j+1] < st_val:
            j += 1
        d = float(lens[j])
        if d < 1e-12:
            out.append(pts[j].copy())
        else:
            t = (st_val - s[j]) / d
            out.append((1.0 - t) * pts[j] + t * pts[j+1])
    return np.asarray(out, dtype=float)

def _insert_corner_neighbors(poly: np.ndarray, d_mm: float = 5.0,
                             collinear_eps: float = 1e-12,
                             min_sep_eps: float = 1e-6) -> np.ndarray:
    pts = np.asarray(poly, dtype=float)
    n = len(pts)
    if n < 3 or d_mm <= 0:
        return pts.copy()

    out = [pts[0].copy()]

    for i in range(1, n - 1):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[i + 1]

        v1 = p1 - p0
        v2 = p2 - p1
        L1 = float(np.linalg.norm(v1[:2]))
        L2 = float(np.linalg.norm(v2[:2]))

        cross = v1[0]*v2[1] - v1[1]*v2[0]
        is_corner = abs(cross) > collinear_eps and L1 > 1e-9 and L2 > 1e-9

        if not is_corner:
            out.append(p1.copy())
            continue

        # before point
        if L1 > d_mm + min_sep_eps:
            u1 = v1 / L1
            p_before = p1 - u1 * d_mm
            if np.linalg.norm((p_before - out[-1])[:2]) > min_sep_eps:
                out.append(p_before)

        # corner itself
        if np.linalg.norm((p1 - out[-1])[:2]) > min_sep_eps:
            out.append(p1.copy())

        # after point
        if L2 > d_mm + min_sep_eps:
            u2 = v2 / L2
            p_after = p1 + u2 * d_mm
            if np.linalg.norm((p_after - out[-1])[:2]) > min_sep_eps:
                out.append(p_after)

    if np.linalg.norm((pts[-1] - out[-1])[:2]) > min_sep_eps:
        out.append(pts[-1].copy())

    return np.asarray(out, dtype=float)

def _resample_closed_polyline_xy(poly: np.ndarray, spacing: float) -> np.ndarray:
    pts = np.asarray(poly, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    if np.linalg.norm(pts[0, :2] - pts[-1, :2]) > 1e-9:
        pts = np.vstack([pts, pts[0]])
    s = _poly_arclen_s_xy(pts)
    total = float(s[-1])
    if total < 1e-9:
        return pts.copy()
    step = max(float(spacing), 1.0)
    count = max(8, int(np.ceil(total / step)))
    targets = np.linspace(0.0, total, count + 1)
    out = _resample_polyline_by_s(pts, targets)
    if np.linalg.norm(out[0, :2] - out[-1, :2]) > 1e-9:
        out = np.vstack([out, out[0]])
    return out


def _resample_linestring_min_spacing(coords_xy: np.ndarray, spacing: float) -> np.ndarray:
    coords_xy = np.asarray(coords_xy, dtype=float)
    if coords_xy.ndim != 2 or coords_xy.shape[0] < 2:
        return coords_xy
    spacing = max(float(spacing), 1e-6)
    try:
        line = LineString(coords_xy)
    except Exception:
        return coords_xy
    total = float(line.length)
    if not np.isfinite(total) or total <= 1e-9:
        return coords_xy
    dists = [0.0]
    cur = spacing
    while cur < total - 1e-9:
        dists.append(cur)
        cur += spacing
    if total > dists[-1] + 1e-9:
        dists.append(total)
    pts = []
    for d in dists:
        pt = line.interpolate(d)
        pts.append([pt.x, pt.y])
    out = np.asarray(pts, dtype=float)
    if out.shape[0] >= 2 and np.linalg.norm(out[-1] - out[0]) < 1e-9:
        out = out[:-1]
    return out

def _compress_collinear_open_path(poly: np.ndarray, collinear_eps: float = 1e-6) -> np.ndarray:
    pts = np.asarray(poly, dtype=float)
    if pts.ndim != 2 or len(pts) <= 2:
        return pts
    out = [pts[0].copy()]
    for i in range(1, len(pts) - 1):
        p_prev = out[-1]
        p_curr = pts[i]
        p_next = pts[i + 1]
        v1 = p_curr[:2] - p_prev[:2]
        v2 = p_next[:2] - p_curr[:2]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-9 or n2 <= 1e-9:
            continue
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
        if cross > collinear_eps:
            out.append(p_curr.copy())
    out.append(pts[-1].copy())
    return np.asarray(out, dtype=float)

def _compress_closed_ring_straights(poly: np.ndarray, collinear_eps: float = 1e-6) -> np.ndarray:
    pts = np.asarray(poly, dtype=float)
    if pts.ndim != 2 or len(pts) < 4:
        return pts
    ring = ensure_open_ring(pts)
    if len(ring) < 3:
        return pts
    keep = []
    n = len(ring)
    for i in range(n):
        p_prev = ring[(i - 1) % n]
        p_curr = ring[i]
        p_next = ring[(i + 1) % n]
        v1 = p_curr[:2] - p_prev[:2]
        v2 = p_next[:2] - p_curr[:2]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-9 or n2 <= 1e-9:
            continue
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
        if cross > collinear_eps:
            keep.append(p_curr.copy())
    if len(keep) < 3:
        keep = [ring[0].copy(), ring[len(ring)//2].copy(), ring[-1].copy()]
    out = np.asarray(keep, dtype=float)
    out = np.vstack([out, out[0]])
    return out
def _apply_fillet_to_path(seg3d_closed: np.ndarray, r_mm: float = 20.0, spacing_mm: float = 5.0) -> np.ndarray:
    seg3d_closed = np.asarray(seg3d_closed, dtype=float)
    if seg3d_closed.ndim != 2 or seg3d_closed.shape[0] < 4:
        return seg3d_closed
    if seg3d_closed.shape[1] < 3:
        seg3d_closed = np.c_[seg3d_closed[:, :2], np.zeros(len(seg3d_closed))]
    xy = seg3d_closed[:, :2]
    z_val = float(np.mean(seg3d_closed[:, 2]))
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-9:
        xy = np.vstack([xy, xy[0]])
    poly = Polygon(xy)
    if poly.is_empty or not poly.is_valid or poly.area <= 1e-9:
        return seg3d_closed
    try:
        rounded = poly.buffer(-r_mm, join_style=1).buffer(r_mm, join_style=1)
    except Exception:
        return seg3d_closed
    if rounded.is_empty:
        return seg3d_closed
    if isinstance(rounded, MultiPolygon):
        rounded = max(rounded.geoms, key=lambda g: g.area if not g.is_empty else -1.0)
    if rounded.is_empty or rounded.area <= 1e-9:
        return seg3d_closed
    coords = np.asarray(rounded.exterior.coords, dtype=float)
    if len(coords) < 4:
        return seg3d_closed
    dense = _resample_linestring_min_spacing(coords, spacing=float(spacing_mm))
    if len(dense) < 3:
        dense = coords[:-1]
    out = np.column_stack([dense[:, 0], dense[:, 1], np.full(len(dense), z_val, dtype=float)])
    if np.linalg.norm(out[0, :2] - out[-1, :2]) > 1e-9:
        out = np.vstack([out, out[0]])
    return _compress_closed_ring_straights(out)

def _offset_inward_closed_path(seg3d_closed: np.ndarray, offset_mm: float):
    seg3d_closed = np.asarray(seg3d_closed, dtype=float)
    if seg3d_closed.ndim != 2 or seg3d_closed.shape[0] < 4 or offset_mm <= 0:
        return seg3d_closed, False
    if seg3d_closed.shape[1] < 3:
        seg3d_closed = np.c_[seg3d_closed[:, :2], np.zeros(len(seg3d_closed))]
    xy = seg3d_closed[:, :2]
    z_val = float(np.mean(seg3d_closed[:, 2]))
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-9:
        xy = np.vstack([xy, xy[0]])
    try:
        poly = Polygon(xy)
    except Exception:
        return seg3d_closed, False
    if poly.is_empty or (not poly.is_valid) or poly.area <= 1e-9:
        return seg3d_closed, False
    try:
        inward = poly.buffer(-float(offset_mm), join_style=2)
    except Exception:
        return seg3d_closed, False
    if inward.is_empty:
        return seg3d_closed, True
    if isinstance(inward, MultiPolygon):
        inward = max(list(inward.geoms), key=lambda g: g.area if not g.is_empty else -1.0)
    if inward.is_empty or inward.area <= 1e-9:
        return seg3d_closed, True
    try:
        coords = np.asarray(inward.exterior.coords, dtype=float)
    except Exception:
        return seg3d_closed, True
    if len(coords) < 4:
        return seg3d_closed, True
    out = np.column_stack([coords[:, 0], coords[:, 1], np.full(len(coords), z_val, dtype=float)])
    return out, False

def _make_seam_at_midpoint(segment: np.ndarray) -> np.ndarray:
    pts = np.asarray(segment, dtype=float)
    if len(pts) < 2:
        return pts
    if np.linalg.norm(pts[0, :2] - pts[-1, :2]) < 1e-9:
        pts = pts[:-1]

    n = len(pts)
    lens = np.linalg.norm(pts[1:, :2] - pts[:-1, :2], axis=1)
    lens = np.append(lens, np.linalg.norm(pts[0, :2] - pts[-1, :2]))
    max_idx = int(np.argmax(lens))

    p1 = pts[max_idx]
    p2 = pts[(max_idx + 1) % n]
    mid = (p1 + p2) / 2.0

    out = [mid]
    for i in range(1, n + 1):
        out.append(pts[(max_idx + i) % n].copy())
    out.append(mid.copy())
    return np.asarray(out, dtype=float)

def _shift_ring_start_along_path(segment: np.ndarray, shift_dist: float) -> np.ndarray:
    """폐곡선의 시작점을 둘레를 따라 shift_dist 만큼 이동시킵니다. (코너에서 시작하는 것을 방지)"""
    pts = np.asarray(segment, dtype=float)
    if len(pts) < 2 or shift_dist <= 0:
        return pts
    
    # 닫힌 루프 확인 및 중복 끝점 제거
    if np.linalg.norm(pts[0, :2] - pts[-1, :2]) < 1e-9:
        pts = pts[:-1]
        
    n = len(pts)
    lens = np.linalg.norm(pts[1:, :2] - pts[:-1, :2], axis=1)
    lens = np.append(lens, np.linalg.norm(pts[0, :2] - pts[-1, :2]))
    total = float(np.sum(lens))
    
    shift_dist = shift_dist % total
    if shift_dist < 1e-5:
        return np.vstack([pts, pts[0]]) # 닫아서 반환
        
    acc = 0.0
    i = 0
    while i < n and acc + lens[i] < shift_dist:
        acc += lens[i]
        i += 1
        
    p = pts[i]
    q = pts[(i + 1) % n]
    d = lens[i]
    
    cut = p + ((shift_dist - acc) / d) * (q - p) if d > 0 else p.copy()
    
    out = [cut]
    for j in range(1, n + 1):
        idx = (i + j) % n
        out.append(pts[idx].copy())
        
    out.append(cut.copy())
    return np.asarray(out, dtype=float)

# =========================
# Plotly: STL (정적)
# =========================
def _apply_translation_to_mesh(mesh: trimesh.Trimesh, dx: float, dy: float, dz: float = 0.0) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation([float(dx), float(dy), float(dz)])
    return m


def _apply_rotation_about_centroid(mesh: trimesh.Trimesh, rz_deg: float = 0.0, rx_deg: float = 0.0, ry_deg: float = 0.0) -> trimesh.Trimesh:
    m = mesh.copy()
    c = np.asarray(m.bounding_box.centroid, dtype=float)
    T1 = trimesh.transformations.translation_matrix(-c)
    T2 = trimesh.transformations.translation_matrix(c)
    Rz = trimesh.transformations.rotation_matrix(np.deg2rad(float(rz_deg)), [0, 0, 1])
    Rx = trimesh.transformations.rotation_matrix(np.deg2rad(float(rx_deg)), [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(np.deg2rad(float(ry_deg)), [0, 1, 0])
    M = T2 @ (Rz @ Rx @ Ry) @ T1
    m.apply_transform(M)
    return m


def plot_trimesh(mesh: trimesh.Trimesh, height=820) -> go.Figure:
    v = mesh.vertices
    f = mesh.faces
    fig = go.Figure(data=[go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="#888888", opacity=0.6, flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.9, roughness=0.9, specular=0.1)
    )])
    fig.update_layout(
        scene=dict(aspectmode="data", camera=dict(projection=dict(type="orthographic"))),
        height=height, margin=dict(l=0, r=0, t=10, b=0)
    )
    return fig

def make_slice_z_values(mesh, z_int: float) -> List[float]:
    zmin = float(mesh.bounds[0, 2])
    zmax = float(mesh.bounds[1, 2])
    height = zmax - zmin
    if not np.isfinite(zmin) or not np.isfinite(zmax) or height <= 1e-9:
        return []
    zstep = max(float(z_int), 1e-9)
    first_z = zmin + 0.5 * zstep
    if first_z >= zmax - 1e-9:
        return [float(zmin + 0.5 * height)]
    zvalues = []
    z = first_z
    while z < zmax - 1e-9:
        zvalues.append(float(z))
        z += zstep
    return zvalues

def generate_gcode(mesh, z_int=30.0, feed=2000, ref_pt_user=(0.0, 0.0),
                   e_on=False, start_e_on=False, start_e_val=0.1, e0_on=False,
                   trim_dist=30.0, min_spacing=5.0, auto_start=False, m30_on=False,
                   seq_print=False, seq_group_inner=True, nozzle_width=0.0, enable_inward_offset=False,
                   skip_invalid_offset=True):
    g = ["; *** Generated by 3DCP Slicer ***", "G21", "G90"]
    if e_on:
        g.append("M83")

    if seq_print:
        raw_sub_meshes = mesh.split(only_watertight=False)
        raw_sub_meshes = sorted(raw_sub_meshes, key=lambda m: m.bounds[0][0])

        if seq_group_inner:
            groups = []
            for m in raw_sub_meshes:
                b1 = m.bounds
                placed = False
                for g_dict in groups:
                    b2 = g_dict['bounds']
                    if not (b1[1][0] < b2[0][0] - 1.0 or b1[0][0] > b2[1][0] + 1.0 or
                            b1[1][1] < b2[0][1] - 1.0 or b1[0][1] > b2[1][1] + 1.0):
                        g_dict['mesh'] = g_dict['mesh'] + m
                        g_dict['bounds'][0] = np.minimum(g_dict['bounds'][0], b1[0])
                        g_dict['bounds'][1] = np.maximum(g_dict['bounds'][1], b1[1])
                        placed = True
                        break
                if not placed:
                    groups.append({'mesh': m, 'bounds': [b1[0].copy(), b1[1].copy()]})
            submeshes = [g_dict['mesh'] for g_dict in groups]
        else:
            submeshes = raw_sub_meshes
        g[0] = "; *** Generated by 3DCP Slicer (Sequential Printing) ***"
    else:
        submeshes = [mesh]

    safe_z_clearance = float(mesh.bounds[1][2]) + 150.0
    prev_start_xy = None

    for subidx, submesh in enumerate(submeshes):
        if seq_print:
            g.append(f"\n; ==========================================")
            g.append(f"; Object {subidx + 1} of {len(submeshes)}")
            g.append(f"; ==========================================")

        z_values = make_slice_z_values(submesh, z_int)

        for zidx, z in enumerate(z_values):
            sec = submesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if sec is None: continue
            try:
                slice_2D, to_3D = sec.to_2D()
            except Exception: continue

            segments = []
            for seg in slice_2D.discrete:
                seg = np.array(seg)
                seg3d = (to_3D @ np.hstack([seg, np.zeros((len(seg), 1)), np.ones((len(seg), 1))]).T).T[:, :3]
                segments.append(seg3d)
            if not segments: continue

            g.append(f"\n; ---------- Z = {z:.2f} mm ----------")
            ref_pt_layer = prev_start_xy if (auto_start and prev_start_xy is not None) else np.array(ref_pt_user, dtype=float)

            for iseg, seg3d in enumerate(segments):
                seg3d_no_dup = ensure_open_ring(seg3d)
                
                closed_mid = _make_seam_at_midpoint(seg3d_no_dup)
                if enable_inward_offset and float(nozzle_width) > 0:
                    closed_mid, offset_inverted = _offset_inward_closed_path(closed_mid, float(nozzle_width))
                    if offset_inverted and skip_invalid_offset:
                        start_pt = closed_mid[0]
                        g.append(f"; Offset collapsed/inverted at Z={z:.2f}, print skipped")
                        g.append(f"G00 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} Z{z:.3f}")
                        continue
                simplified = simplify_segment(closed_mid, min_spacing)
                shifted, _ = shift_to_nearest_start(simplified, ref_point=ref_pt_layer)
                if st.session_state.get('enable_fillet', False):
                    r_val = float(st.session_state.get('fillet_r', 20.0))
                    shifted_closed = np.vstack([ensure_open_ring(shifted), ensure_open_ring(shifted)[0]])
                    rounded = _apply_fillet_to_path(shifted_closed, r_mm=r_val, spacing_mm=min_spacing)
                    shifted, _ = shift_to_nearest_start(rounded, ref_point=ref_pt_layer)
                simplified = trim_closed_ring_tail(shifted, trim_dist)

                start = simplified[0]

                if seq_print and subidx > 0 and zidx == 0 and iseg == 0:
                    g.append(f"; Moving to new object start at Safe Z")
                    g.append(f"G00 X{start[0]:.3f} Y{start[1]:.3f} Z{safe_z_clearance:.3f}")

                if iseg > 0:
                    g.append(f"G01 X{start[0]:.3f} Y{start[1]:.3f} Z{z:.3f}")

                g.append(f"G01 F{feed}")
                if start_e_on:
                    g.append(f"G01 X{start[0]:.3f} Y{start[1]:.3f} Z{z:.3f} E{start_e_val:.5f}")
                else:
                    g.append(f"G01 X{start[0]:.3f} Y{start[1]:.3f} Z{z:.3f}")

                for p1, p2 in zip(simplified[:-1], simplified[1:]):
                    dist = np.linalg.norm(p2[:2] - p1[:2])
                    if e_on:
                        g.append(f"G01 X{p2[0]:.3f} Y{p2[1]:.3f} E{dist * EXTRUSION_K:.5f}")
                    else:
                        g.append(f"G01 X{p2[0]:.3f} Y{p2[1]:.3f}")

                if e0_on: g.append("G01 E0")
                if iseg == 0: prev_start_xy = start[:2]

        if seq_print and subidx < len(submeshes) - 1:
            g.append(f"\n; Retracting to Safe Z before moving to next object")
            g.append(f"G00 Z{safe_z_clearance:.3f}")

    g.append(f"G01 F{feed}")
    if m30_on: g.append("M30")
    return "\n".join(g)

# =========================
# Slice path computation 
# =========================
def compute_slice_paths_with_travel(
    mesh, z_int=30.0, ref_pt_user=(0.0, 0.0), trim_dist=30.0, min_spacing=5.0,
    auto_start=False, e_on=False, seq_print=False, seq_group_inner=True,
    nozzle_width=0.0, enable_inward_offset=False, skip_invalid_offset=True
) -> List[Tuple[np.ndarray, Optional[np.ndarray], bool]]:

    all_items: List[Tuple[np.ndarray, Optional[np.ndarray], bool]] = []
    prev_layer_last_end: Optional[np.ndarray] = None
    prev_start_xy = None

    if seq_print:
        raw_sub_meshes = mesh.split(only_watertight=False)
        raw_sub_meshes = sorted(raw_sub_meshes, key=lambda m: m.bounds[0][0])

        if seq_group_inner:
            groups = []
            for m in raw_sub_meshes:
                b1 = m.bounds
                placed = False
                for g_dict in groups:
                    b2 = g_dict['bounds']
                    if not (b1[1][0] < b2[0][0] - 1.0 or b1[0][0] > b2[1][0] + 1.0 or
                            b1[1][1] < b2[0][1] - 1.0 or b1[0][1] > b2[1][1] + 1.0):
                        g_dict['mesh'] = g_dict['mesh'] + m
                        g_dict['bounds'][0] = np.minimum(g_dict['bounds'][0], b1[0])
                        g_dict['bounds'][1] = np.maximum(g_dict['bounds'][1], b1[1])
                        placed = True
                        break
                if not placed:
                    groups.append({'mesh': m, 'bounds': [b1[0].copy(), b1[1].copy()]})
            sub_meshes = [g_dict['mesh'] for g_dict in groups]
        else:
            sub_meshes = raw_sub_meshes
    else:
        sub_meshes = [mesh]

    safe_z_clearance = float(mesh.bounds[1][2]) + 50.0
    highest_sliced_z = None
    for _sub_mesh in sub_meshes:
        _z_values = make_slice_z_values(_sub_mesh, z_int)
        if _z_values:
            _top_z = float(max(_z_values))
            highest_sliced_z = _top_z if highest_sliced_z is None else max(highest_sliced_z, _top_z)

    for sub_idx, sub_mesh in enumerate(sub_meshes):
        z_values = make_slice_z_values(sub_mesh, z_int)

        for z in z_values:
            sec = sub_mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if sec is None:
                continue
            try:
                slice2D, to3D = sec.to_2D()
            except Exception:
                continue

            segments = []
            for seg in slice2D.discrete:
                seg = np.array(seg)
                seg3d = (to3D @ np.hstack([seg, np.zeros((len(seg), 1)), np.ones((len(seg), 1))]).T).T[:, :3]
                segments.append(seg3d)
            if not segments:
                continue

            ref_pt_layer = prev_start_xy if (auto_start and prev_start_xy is not None) else np.array(ref_pt_user, dtype=float)

            layer_polys: List[np.ndarray] = []
            for i_seg, seg3d in enumerate(segments):
                seg3d_no_dup = ensure_open_ring(seg3d)

                closed_mid = _make_seam_at_midpoint(seg3d_no_dup)
                if enable_inward_offset and float(nozzle_width) > 0:
                    closed_mid, offset_inverted = _offset_inward_closed_path(closed_mid, float(nozzle_width))
                    if offset_inverted and skip_invalid_offset:
                        start_pt = closed_mid[0]
                        g.append(f"; Offset collapsed/inverted at Z={z:.2f}, print skipped")
                        g.append(f"G00 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} Z{z:.3f}")
                        continue
                simplified = simplify_segment(closed_mid, min_spacing)
                shifted, _ = shift_to_nearest_start(simplified, ref_point=ref_pt_layer)
                if st.session_state.get('enable_fillet', False):
                    r_val = float(st.session_state.get('fillet_r', 20.0))
                    shifted_closed = np.vstack([ensure_open_ring(shifted), ensure_open_ring(shifted)[0]])
                    rounded = _apply_fillet_to_path(shifted_closed, r_mm=r_val, spacing_mm=min_spacing)
                    shifted, _ = shift_to_nearest_start(rounded, ref_point=ref_pt_layer)
                simplified = trim_closed_ring_tail(shifted, trim_dist)

                layer_polys.append(simplified.copy())
                if i_seg == 0:
                    prev_start_xy = simplified[0][:2]

            if not layer_polys:
                continue

            first_poly_start = layer_polys[0][0]
            if prev_layer_last_end is not None:
                prev_z = float(prev_layer_last_end[2])
                prev_was_top_group = (highest_sliced_z is not None) and (abs(prev_z - float(highest_sliced_z)) <= 1e-6)
                if prev_was_top_group:
                    safe_prev = prev_layer_last_end.copy(); safe_prev[2] = safe_z_clearance
                    safe_next = first_poly_start.copy(); safe_next[2] = safe_z_clearance
                    travel_up = np.vstack([prev_layer_last_end, safe_prev])
                    travel_xy = np.vstack([safe_prev, safe_next])
                    travel_down = np.vstack([safe_next, first_poly_start])
                    all_items.append((travel_up, np.array([0.0, 0.0]) if e_on else None, True))
                    all_items.append((travel_xy, np.array([0.0, 0.0]) if e_on else None, True))
                    all_items.append((travel_down, np.array([0.0, 0.0]) if e_on else None, True))
                else:
                    travel = np.vstack([prev_layer_last_end, first_poly_start])
                    all_items.append((travel, np.array([0.0, 0.0]) if e_on else None, True))

            for i_seg in range(len(layer_polys)):
                poly = layer_polys[i_seg]
                if e_on:
                    e_vals = [0.0]
                    total = 0.0
                    for p1, p2 in zip(poly[:-1], poly[1:]):
                        dist = np.linalg.norm(p2[:2] - p1[:2])
                        total += dist * EXTRUSION_K
                        e_vals.append(total)
                    all_items.append((poly, np.array(e_vals), False))
                else:
                    all_items.append((poly, None, False))

                if i_seg < len(layer_polys) - 1:
                    nxt = layer_polys[i_seg + 1]
                    travel_intra = np.vstack([poly[-1], nxt[0]])
                    all_items.append((travel_intra, np.array([0.0, 0.0]) if e_on else None, True))

            prev_layer_last_end = layer_polys[-1][-1]


    return all_items

def _collect_layer_z_values(segments: List[Tuple[np.ndarray, np.ndarray, bool, bool]], tol: float = 1e-6) -> List[float]:
    zs: List[float] = []
    for p1, p2, is_travel, is_extruding in segments:
        if not is_extruding:
            continue
        zmid = float((p1[2] + p2[2]) * 0.5)
        if not zs or abs(zmid - zs[-1]) > tol:
            zs.append(zmid)
    return zs


def _filter_segments_by_layer_range(segments: List[Tuple[np.ndarray, np.ndarray, bool, bool]], layer_start: int, layer_end: int, tol: float = 1e-6):
    if not segments:
        return []
    zs = _collect_layer_z_values(segments, tol=tol)
    if not zs:
        return []
    n_layers = len(zs)
    layer_start = max(1, min(int(layer_start), n_layers))
    layer_end = max(layer_start, min(int(layer_end), n_layers))
    z_lo = zs[layer_start - 1]
    z_hi = zs[layer_end - 1]
    out = []
    for p1, p2, is_travel, is_extruding in segments:
        zmid = float((p1[2] + p2[2]) * 0.5)
        if is_extruding:
            if z_lo - tol <= zmid <= z_hi + tol:
                out.append((p1, p2, is_travel, is_extruding))
        else:
            z1 = float(p1[2]); z2 = float(p2[2])
            if (z_lo - tol <= z1 <= z_hi + tol) or (z_lo - tol <= z2 <= z_hi + tol):
                out.append((p1, p2, is_travel, is_extruding))
    return out


def _build_buffers_from_segments_subset(segments, travel_mode: str = 'solid'):
    reset_anim_buffers()
    buf = st.session_state.paths_anim_buf
    for p1, p2, is_travel, _ in segments:
        if is_travel:
            if travel_mode == 'hidden':
                continue
            key = 'dot' if travel_mode == 'dotted' else 'solid'
        else:
            key = 'solid'
        buf[key]['x'].extend([float(p1[0]), float(p2[0]), None])
        buf[key]['y'].extend([float(p1[1]), float(p2[1]), None])
        buf[key]['z'].extend([float(p1[2]), float(p2[2]), None])
    buf['built_upto'] = len(segments)
    buf['stride'] = 1
def items_to_segments(items: List[Tuple[np.ndarray, Optional[np.ndarray], bool]], e_on: bool
) -> List[Tuple[np.ndarray, np.ndarray, bool, bool]]:
    segs: List[Tuple[np.ndarray, np.ndarray, bool, bool]] = []
    if not items:
        return segs
    for poly, e_vals, is_travel in items:
        if poly is None or len(poly) < 2:
            continue
        if e_on and e_vals is not None:
            for p1, p2, e1, e2 in zip(poly[:-1], poly[1:], e_vals[:-1], e_vals[1:]):
                is_extruding = (e2 - e1) > 1e-12 and (not is_travel)
                segs.append((p1, p2, is_travel, is_extruding))
        else:
            for p1, p2 in zip(poly[:-1], poly[1:]):
                is_extruding = (not is_travel)
                segs.append((p1, p2, is_travel, is_extruding))
    return segs

def compute_layer_length_for_index(
    segments: List[Tuple[np.ndarray, np.ndarray, bool, bool]],
    upto_idx: int
) -> Tuple[Optional[float], Optional[float]]:
    if not segments or upto_idx <= 0:
        return None, None

    N = len(segments)
    upto = min(max(int(upto_idx), 0), N)

    layer_z = None
    for i in range(upto - 1, -1, -1):
        p1, p2, is_travel, is_extruding = segments[i]
        if is_extruding:
            layer_z = float((p1[2] + p2[2]) * 0.5)
            break
    if layer_z is None:
        return None, None

    total_len = 0.0
    for p1, p2, is_travel, is_extruding in segments:
        if not is_extruding:
            continue
        zmid = float((p1[2] + p2[2]) * 0.5)
        if abs(zmid - layer_z) < 1e-6:
            total_len += float(np.linalg.norm(p2[:2] - p1[:2]))

    return layer_z, total_len if total_len > 0 else None

# === 누적 렌더 버퍼 ===
def reset_anim_buffers():
    st.session_state.paths_anim_buf = {
        "solid": {"x": [], "y": [], "z": []},
        "dot":   {"x": [], "y": [], "z": []},
        "off_l": {"x": [], "y": [], "z": []},
        "off_r": {"x": [], "y": [], "z": []},
        "caps":  {"x": [], "y": [], "z": []},
        "built_upto": 0,
        "stride": 1,
    }

def ensure_anim_buffers():
    if "paths_anim_buf" not in st.session_state or not isinstance(st.session_state.paths_anim_buf, dict):
        reset_anim_buffers()

def append_segments_to_buffers(segments, start_idx, end_idx, stride=1):
    buf = st.session_state.paths_anim_buf
    travel_mode = st.session_state.get("paths_travel_mode", "solid")
    for i in range(start_idx, end_idx, max(1, int(stride))):
        p1, p2, is_travel, _ = segments[i]
        if is_travel:
            if travel_mode == "hidden":
                continue
            elif travel_mode == "dotted":
                buf["dot"]["x"].extend([float(p1[0]), float(p2[0]), None])
                buf["dot"]["y"].extend([float(p1[1]), float(p2[1]), None])
                buf["dot"]["z"].extend([float(p1[2]), float(p2[2]), None])
            else:
                buf["solid"]["x"].extend([float(p1[0]), float(p2[0]), None])
                buf["solid"]["y"].extend([float(p1[1]), float(p2[1]), None])
                buf["solid"]["z"].extend([float(p1[2]), float(p2[2]), None])
        else:
            buf["solid"]["x"].extend([float(p1[0]), float(p2[0]), None])
            buf["solid"]["y"].extend([float(p1[1]), float(p2[1]), None])
            buf["solid"]["z"].extend([float(p1[2]), float(p2[2]), None])
    buf["built_upto"] = end_idx
    buf["stride"] = max(1, int(stride))

def rebuild_buffers_to(segments, upto, stride=1):
    reset_anim_buffers()
    if upto > 0:
        append_segments_to_buffers(segments, 0, upto, stride=stride)

def compute_offsets_into_buffers(
    segments, upto, half_width, include_travel_climb: bool = False, climb_z_thresh: float = 1e-6
):
    buf = st.session_state.paths_anim_buf
    buf["off_l"] = {"x": [], "y": [], "z": []}
    buf["off_r"] = {"x": [], "y": [], "z": []}
    if half_width <= 0 or upto <= 0:
        return

    prev_tan = None
    N = min(upto, len(segments))
    for i in range(N):
        p1, p2, is_travel, is_extruding = segments[i]
        use_this = False
        if (not is_travel) and is_extruding:
            use_this = True
        elif include_travel_climb:
            dz = float(p2[2] - p1[2])
            if abs(dz) > climb_z_thresh:
                use_this = True

        if not use_this:
            dx = float(p2[0] - p1[0]); dy = float(p2[1] - p1[1])
            nrm = (dx*dx + dy*dy) ** 0.5
            if nrm > 1e-12:
                prev_tan = (dx/nrm, dy/nrm)
            continue

        dx = float(p2[0] - p1[0]); dy = float(p2[1] - p1[1])
        nrm = (dx*dx + dy*dy) ** 0.5
        if nrm > 1e-12:
            tx, ty = dx/nrm, dy/nrm
            prev_tan = (tx, ty)
        else:
            if prev_tan is None:
                prev_tan = (1.0, 0.0)
            tx, ty = prev_tan

        nx, ny = -ty, tx

        l1 = (float(p1[0] + nx*half_width), float(p1[1] + ny*half_width), float(p1[2]))
        l2 = (float(p2[0] + nx*half_width), float(p2[1] + ny*half_width), float(p2[2]))
        r1 = (float(p1[0] - nx*half_width), float(p1[1] - ny*half_width), float(p1[2]))
        r2 = (float(p2[0] - nx*half_width), float(p2[1] - ny*half_width), float(p2[2]))

        buf["off_l"]["x"].extend([l1[0], l2[0], None]); buf["off_l"]["y"].extend([l1[1], l2[1], None]); buf["off_l"]["z"].extend([l1[2], l2[2], None])
        buf["off_r"]["x"].extend([r1[0], r2[0], None]); buf["off_r"]["y"].extend([r1[1], r2[1], None]); buf["off_r"]["z"].extend([r1[2], r2[2], None])

def add_global_endcaps_into_buffers(segments, upto, half_width, samples=32, store_caps=False):
    if half_width <= 0 or upto <= 0 or len(segments) == 0:
        return

    first_idx = None
    last_idx = None
    N = min(upto, len(segments))

    for i in range(N):
        p1, p2, is_travel, is_extruding = segments[i]
        if (not is_travel) and is_extruding:
            first_idx = i
            break
    for i in range(N - 1, -1, -1):
        p1, p2, is_travel, is_extruding = segments[i]
        if (not is_travel) and is_extruding:
            last_idx = i
            break
    if first_idx is None or last_idx is None:
        return

    buf = st.session_state.paths_anim_buf

    def _append_arc(center, t_unit, n_unit, z, sign_t, steps):
        s = sign_t * t_unit
        thetas = np.linspace(0.0, np.pi, int(max(8, steps)))
        xs = center[0] + half_width*(n_unit[0]*np.cos(thetas) + s[0]*np.sin(thetas))
        ys = center[1] + half_width*(n_unit[1]*np.cos(thetas) + s[1]*np.sin(thetas))
        zs = np.full_like(xs, float(z))

        if len(buf["off_l"]["x"]) > 0 and buf["off_l"]["x"][-1] is not None:
            buf["off_l"]["x"].append(None); buf["off_l"]["y"].append(None); buf["off_l"]["z"].append(None)
        buf["off_l"]["x"].extend(xs.tolist() + [None])
        buf["off_l"]["y"].extend(ys.tolist() + [None])
        buf["off_l"]["z"].extend(zs.tolist() + [None])

        if store_caps:
            if len(buf["caps"]["x"]) > 0 and buf["caps"]["x"][-1] is not None:
                buf["caps"]["x"].append(None); buf["caps"]["y"].append(None); buf["caps"]["z"].append(None)
            buf["caps"]["x"].extend(xs.tolist() + [None])
            buf["caps"]["y"].extend(ys.tolist() + [None])
            buf["caps"]["z"].extend(zs.tolist() + [None])

    p1, p2, _, _ = segments[first_idx]
    dx = float(p2[0] - p1[0]); dy = float(p2[1] - p1[1]); nrm = (dx*dx + dy*dy) ** 0.5
    if nrm > 1e-12:
        t_unit = np.array([dx/nrm, dy/nrm], dtype=float)
        n_unit = np.array([-t_unit[1], t_unit[0]], dtype=float)
        _append_arc((float(p1[0]), float(p1[1])), t_unit, n_unit, float(p1[2]), sign_t=-1.0, steps=samples)

    p1, p2, _, _ = segments[last_idx]
    dx = float(p2[0] - p1[0]); dy = float(p2[1] - p1[1]); nrm = (dx*dx + dy*dy) ** 0.5
    if nrm > 1e-12:
        t_unit = np.array([dx/nrm, dy/nrm], dtype=float)
        n_unit = np.array([-t_unit[1], t_unit[0]], dtype=float)
        _append_arc((float(p2[0]), float(p2[1])), t_unit, n_unit, float(p2[2]), sign_t=+1.0, steps=samples)

def make_base_fig(height=820) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=4, dash="solid", color=PATH_COLOR_DEFAULT),
                               showlegend=False))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=4, dash="dot", color=PATH_COLOR_DEFAULT),
                               showlegend=False))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=4, dash="solid", color=OFFSET_DARK_GRAY),
                               showlegend=False))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=4, dash="solid", color=OFFSET_DARK_GRAY),
                               showlegend=False))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=6, dash="solid", color=CAP_COLOR),
                               name="Caps Emphasis", showlegend=False))
    fig.update_layout(
        scene=dict(aspectmode="data", camera=dict(projection=dict(type="orthographic"))),
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        uirevision="keep", transition={'duration': 0}
    )
    return fig

def ensure_traces(fig: go.Figure, want=5):
    def add_solid():
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(width=4, dash="solid", color=PATH_COLOR_DEFAULT),
                                   showlegend=False))
    def add_dot():
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(width=4, dash="dot", color=PATH_COLOR_DEFAULT),
                                   showlegend=False))
    def add_off():
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(width=4, dash="solid", color=OFFSET_DARK_GRAY),
                                   showlegend=False))
    def add_caps():
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(width=6, dash="solid", color=CAP_COLOR),
                                   showlegend=False))
    while len(fig.data) < want:
        n = len(fig.data)
        if n == 0: add_solid()
        elif n == 1: add_dot()
        elif n in (2, 3): add_off()
        else: add_caps()

def update_fig_with_buffers(fig: go.Figure, show_offsets: bool, show_caps: bool, segments_for_hover=None):
    ensure_traces(fig, want=5)
    buf = st.session_state.paths_anim_buf

    apply_on = bool(st.session_state.get("apply_offsets_flag", False))
    center_color = PATH_COLOR_LIGHT if apply_on else PATH_COLOR_DEFAULT
    fig.data[0].line.color = center_color
    fig.data[1].line.color = center_color

    fig.data[2].line.color = OFFSET_DARK_GRAY; fig.data[2].line.width = 4
    fig.data[3].line.color = OFFSET_DARK_GRAY; fig.data[3].line.width = 4

    fig.data[0].x = buf["solid"]["x"]; fig.data[0].y = buf["solid"]["y"]; fig.data[0].z = buf["solid"]["z"]
    fig.data[1].x = buf["dot"]["x"];   fig.data[1].y = buf["dot"]["y"];   fig.data[1].z = buf["dot"]["z"]

    if segments_for_hover is not None:
        layer_map = _collect_layer_z_values(segments_for_hover)
        def _layer_no(zv):
            for idx, zz in enumerate(layer_map, start=1):
                if abs(float(zv) - float(zz)) < 1e-6:
                    return idx
            return None
        solid_cd = []
        dot_cd = []
        travel_mode = st.session_state.get("paths_travel_mode", "solid")
        for p1, p2, is_travel, is_extruding in segments_for_hover:
            zmid = float((p1[2] + p2[2]) * 0.5)
            lno = _layer_no(zmid)
            row = [[lno], [lno], [None]]
            if is_travel:
                if travel_mode == "hidden":
                    continue
                dot_cd.extend(row if travel_mode == "dotted" else [])
                solid_cd.extend(row if travel_mode != "dotted" else [])
            else:
                solid_cd.extend(row)
        fig.data[0].customdata = solid_cd
        fig.data[1].customdata = dot_cd
        fig.data[0].hovertemplate = "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<br>레이어=%{customdata[0]}<extra></extra>"
        fig.data[1].hovertemplate = "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<br>레이어=%{customdata[0]}<extra></extra>"

    if show_offsets:
        fig.data[2].x = buf["off_l"]["x"]; fig.data[2].y = buf["off_l"]["y"]; fig.data[2].z = buf["off_l"]["z"]
        fig.data[3].x = buf["off_r"]["x"]; fig.data[3].y = buf["off_r"]["y"]; fig.data[3].z = buf["off_r"]["z"]
    else:
        fig.data[2].x = []; fig.data[2].y = []; fig.data[2].z = []
        fig.data[3].x = []; fig.data[3].y = []; fig.data[3].z = []

    if show_caps:
        fig.data[4].x = buf["caps"]["x"]; fig.data[4].y = buf["caps"]["y"]; fig.data[4].z = buf["caps"]["z"]
    else:
        fig.data[4].x = []; fig.data[4].y = []; fig.data[4].z = []

# ======= 치수 계산 유틸 =======
def _bbox_from_buffer(buf_dict):
    try:
        xs = [float(v) for v in buf_dict["x"] if v is not None]
        ys = [float(v) for v in buf_dict["y"] if v is not None]
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return {"x_min": x_min, "x_max": x_max, "x_len": x_max - x_min,
                "y_min": y_min, "y_max": y_max, "y_len": y_max - y_min}
    except Exception:
        return None

def _last_z_from_buffer(buf_dict):
    try:
        for v in reversed(buf_dict.get("z", [])):
            if v is not None:
                return float(v)
    except Exception:
        pass
    return None

def _fmt_dims_block_html(title, bbox, z_single: Optional[float]) -> str:
    if bbox is None:
        return f"<div class='dims-block'><b>{title}</b>\nX=(-)\nY=(-)\nZ=(-)</div>"
    xm, xM, xl = bbox["x_min"], bbox["x_max"], bbox["x_len"]
    ym, yM, yl = bbox["y_min"], bbox["y_max"], bbox["y_len"]
    z_txt = "-" if z_single is None else f"{z_single:.3f}"
    return ("<div class='dims-block'>"
            f"<b>{title}</b>\n"
            f"X=({xm:.3f}→{xM:.3f}, Δ{xl:.3f})\n"
            f"Y=({ym:.3f}→{yM:.3f}, Δ{yl:.3f})\n"
            f"Z=({z_txt})"
            "</div>")

# =========================
# Session init
# =========================
if "mesh" not in st.session_state:
    st.session_state.mesh = None
if "paths_items" not in st.session_state:
    st.session_state.paths_items = None
if "gcode_text" not in st.session_state:
    st.session_state.gcode_text = None
if "base_name" not in st.session_state:
    st.session_state.base_name = "output"

if "show_rapid_panel" not in st.session_state:
    st.session_state.show_rapid_panel = False
if "rapid_rx" not in st.session_state:
    st.session_state.rapid_rx = 0.0
if "rapid_ry" not in st.session_state:
    st.session_state.rapid_ry = 0.0
if "rapid_rz" not in st.session_state:
    st.session_state.rapid_rz = 0.0
if "rapid_text" not in st.session_state:
    st.session_state.rapid_text = None

if "paths_scrub" not in st.session_state:
    st.session_state.paths_scrub = total_segments
if "paths_travel_mode" not in st.session_state:
    st.session_state.paths_travel_mode = "solid"
if "paths_scrub_input" not in st.session_state:
    st.session_state.paths_scrub_input = total_segments
if "paths_scrub_slider" not in st.session_state:
    st.session_state.paths_scrub_slider = total_segments
if "layer_view_start_input" not in st.session_state:
    st.session_state.layer_view_start_input = 1
if "layer_view_end_input" not in st.session_state:
    st.session_state.layer_view_end_input = 1


def _sync_scrub_from_slider():
    st.session_state.paths_scrub = int(st.session_state.get("paths_scrub_slider", 0))
    st.session_state.paths_scrub_input = int(st.session_state.paths_scrub)


def _sync_scrub_from_input(max_segments: int):
    v = int(clamp(st.session_state.get("paths_scrub_input", 0), 0, max_segments))
    st.session_state.paths_scrub = v
    st.session_state.paths_scrub_slider = v
    st.session_state.paths_scrub_input = v


def _sync_layer_inputs(max_layer_no: int):
    s = int(clamp(st.session_state.get("layer_view_start_input", 1), 1, max_layer_no))
    e_raw = int(st.session_state.get("layer_view_end_input", s))
    e = int(clamp(e_raw, s, max_layer_no))
    st.session_state["layer_view_start"] = s
    st.session_state["layer_view_end"] = e

if "ui_banner" not in st.session_state:
    st.session_state.ui_banner = None

if "main_view" not in st.session_state:
    st.session_state.main_view = "슬라이싱 경로 (3D)"

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

if "ext_const_enable_a1" not in st.session_state:
    st.session_state.ext_const_enable_a1 = False
if "ext_const_enable_a2" not in st.session_state:
    st.session_state.ext_const_enable_a2 = False
if "ext_use_a3" not in st.session_state:
    st.session_state.ext_use_a3 = False
if "ext_use_a4" not in st.session_state:
    st.session_state.ext_use_a4 = False

if "ext_const_xmin" not in st.session_state:
    st.session_state.ext_const_xmin = 0.0
if "ext_const_xmax" not in st.session_state:
    st.session_state.ext_const_xmax = 6000.0
if "ext_const_a1_at_xmin" not in st.session_state:
    st.session_state.ext_const_a1_at_xmin = 4000.0
if "ext_const_a1_at_xmax" not in st.session_state:
    st.session_state.ext_const_a1_at_xmax = 0.0

if "ext_const_ymin" not in st.session_state:
    st.session_state.ext_const_ymin = 0.0
if "ext_const_ymax" not in st.session_state:
    st.session_state.ext_const_ymax = 1000.0
if "ext_const_a2_at_ymin" not in st.session_state:
    st.session_state.ext_const_a2_at_ymin = 500.0
if "ext_const_a2_at_ymax" not in st.session_state:
    st.session_state.ext_const_a2_at_ymax = 0.0

if "ext_const_speed_mm_s" not in st.session_state:
    st.session_state.ext_const_speed_mm_s = 200.0
if "ext_const_eps_mm" not in st.session_state:
    st.session_state.ext_const_eps_mm = 0.5
if "ext_const_apply_print_only" not in st.session_state:
    st.session_state.ext_const_apply_print_only = False
if "ext_const_travel_interp" not in st.session_state:
    st.session_state.ext_const_travel_interp = True
if "extconsta2usestep" not in st.session_state:
    st.session_state.extconsta2usestep = False
if "extconsta2stepmm" not in st.session_state:
    st.session_state.extconsta2stepmm = 60.0
if "singularity_avoid_enable" not in st.session_state:
    st.session_state.singularity_avoid_enable = False
if "singularity_z_trigger" not in st.session_state:
    st.session_state.singularity_z_trigger = 0.0
if "singularity_lift_z" not in st.session_state:
    st.session_state.singularity_lift_z = 300.0
ensure_anim_buffers()

# =========================
# 접근 권한
# =========================
ALLOWED_WITH_EXPIRY = {"robotics5107": None, "kaist_aramco3D": "2026-12-31", "kmou*": "2026-12-31", "DY25-01D4-E5F6-G7H8-I9J0-K1L2": "2030-12-30"}
def check_key_valid(k: str):
    if not k or k not in ALLOWED_WITH_EXPIRY:
        return False, None, None, "유효하지 않은 키입니다."
    exp = ALLOWED_WITH_EXPIRY[k]
    if exp is None:
        return True, None, None, "만료일 없음"
    try:
        exp_date = date.fromisoformat(exp)
    except Exception:
        return False, None, None, "키 만료일 형식 오류(YYYY-MM-DD)."
    today = date.today()
    remaining = (exp_date - today).days
    if remaining < 0:
        return False, exp_date, remaining, f"만료일 경과: {exp_date.isoformat()}"
    elif remaining == 0:
        return True, exp_date, remaining, f"오늘 만료 ({exp_date.isoformat()})"
    else:
        return True, exp_date, remaining, f"만료일: {exp_date.isoformat()} · {remaining}일 남음"

KEY_OK, EXP_DATE, REMAINING, STATUS_TXT = check_key_valid(st.session_state.get("access_key", ""))
uploaded = st.sidebar.file_uploader("STL 업로드", type=["stl"], help="최대 업로드 용량: 200MB")
with st.sidebar.expander("STL 위치/회전 보정", expanded=False):
    move_x = st.number_input("이동 X(mm)", value=float(st.session_state.get("stl_move_x", 0.0)), step=10.0, format="%.3f")
    move_y = st.number_input("이동 Y(mm)", value=float(st.session_state.get("stl_move_y", 0.0)), step=10.0, format="%.3f")
    move_z = st.number_input("이동 Z(mm)", value=float(st.session_state.get("stl_move_z", 0.0)), step=10.0, format="%.3f")
    if st.button("이동 적용", use_container_width=True, key="apply_stl_move"):
        st.session_state["stl_move_x"] = float(move_x)
        st.session_state["stl_move_y"] = float(move_y)
        st.session_state["stl_move_z"] = float(move_z)
        st.session_state["stl_apply_move"] = True
        st.sidebar.success("STL 이동 좌표를 적용했습니다.")

    rot_x = st.number_input("회전 Rx(deg)", value=float(st.session_state.get("stl_rot_x", 0.0)), step=1.0, format="%.3f")
    rot_y = st.number_input("회전 Ry(deg)", value=float(st.session_state.get("stl_rot_y", 0.0)), step=1.0, format="%.3f")
    rot_z = st.number_input("회전 Rz(deg)", value=float(st.session_state.get("stl_rot_z", 0.0)), step=1.0, format="%.3f")
    if st.button("회전 적용", use_container_width=True, key="apply_stl_rot"):
        st.session_state["stl_rot_x"] = float(rot_x)
        st.session_state["stl_rot_y"] = float(rot_y)
        st.session_state["stl_rot_z"] = float(rot_z)
        st.session_state["stl_apply_rot"] = True
        st.sidebar.success("중심점 기준 STL 회전을 적용했습니다.")


# =========================
# 파라미터
# =========================
st.sidebar.header("기본 파라미터")
z_int = st.sidebar.number_input("레이어(Z) 간격 (mm)", 1.0, 1000.0, 15.0)
feed = st.sidebar.number_input("이송속도 (F)", 1, 100000, 2000)
ref_x = st.sidebar.number_input("시작기준좌표(X)", value=0.0)
ref_y = st.sidebar.number_input("시작기준좌표(Y)", value=0.0)

# [수정] 시작기준좌표 하단에 '시작점 고정' 옵션 배치
fix_start = st.sidebar.checkbox("시작점 고정", value=False)
# [수정] 자동 연결 UI는 제거하고, 시작점 고정 여부에 따라 내부적으로 자동 전환되도록 설정
actual_auto_start = not fix_start

st.sidebar.subheader("압출 옵션")
e_on = st.sidebar.checkbox("재료토출(E) 삽입")
start_e_on = st.sidebar.checkbox("연속 레이어 출력", value=False, disabled=not e_on)
start_e_val = st.sidebar.number_input("시작 E 값", value=0.1, disabled=not (e_on and start_e_on))
e0_on = st.sidebar.checkbox("루프 끝에 E0 추가", value=False, disabled=not e_on)

st.sidebar.subheader("경로처리")
seq_print = st.sidebar.checkbox("순차 출력 (1개씩)", value=False)
seq_group_inner = st.sidebar.checkbox("내부 폐구간 그룹화", value=True, disabled=not seq_print)


with st.sidebar.expander("코너 라운딩(R) 옵션", expanded=False):
    enable_fillet = st.checkbox("라운딩 적용", value=False, key="enable_fillet")
    fillet_r = st.number_input("R 반경 (mm)", min_value=0.0, max_value=1000.0, value=20.0, step=1.0, key="fillet_r", disabled=not enable_fillet)

trim_dist = st.sidebar.number_input("트림 거리(mm)", 0.0, 1000.0, 50.0)
min_spacing = st.sidebar.number_input("최소 점간격(mm)", 0.0, 1000.0, 5.0)
with st.sidebar.expander("노즐 직경/오프셋 옵션", expanded=False):
    nozzle_diameter = st.number_input("노즐 직경(mm)", min_value=0.0, max_value=1000.0, value=float(st.session_state.get("nozzle_diameter_mm", 0.0)), step=0.5, help="입력한 노즐 직경의 반지름(직경/2)만큼 안쪽으로 오프셋합니다.")
    st.session_state["nozzle_diameter_mm"] = float(nozzle_diameter)
    enable_inward_offset = st.checkbox("노즐 반지름만큼 안쪽 오프셋 출력", value=bool(st.session_state.get("enable_inward_offset", False)))
    st.session_state["enable_inward_offset"] = bool(enable_inward_offset)
    skip_invalid_offset = st.checkbox("역오프셋/소멸 구간은 출력하지 않고 종료", value=bool(st.session_state.get("skip_invalid_offset", True)))
    st.session_state["skip_invalid_offset"] = bool(skip_invalid_offset)

# [수정] 기존 UI에 있던 auto_start 체크박스 삭제 완료
m30_on = st.sidebar.checkbox("M30 추가", value=False)

slice_clicked = st.sidebar.button("모델 슬라이싱", use_container_width=True)
access_key = st.sidebar.text_input("라이센스키", type="password", key="access_key", help="라이센스키를 입력하면 코드생성 및 부가기능이 활성화됩니다.")
if st.session_state.get("access_key", ""):
    if KEY_OK:
        if EXP_DATE is None:
            st.sidebar.success(STATUS_TXT)
        else:
            d_mark = f"D-{REMAINING}" if REMAINING > 0 else "D-DAY"
            st.sidebar.info(f"{STATUS_TXT} ({d_mark})")
    else:
        st.sidebar.error(STATUS_TXT)
else:
    st.sidebar.info("G-code 생성과 R-code 생성을 사용하려면 라이센스키를 입력해 주세요. 인증 후 이 영역에 만료일 안내가 표시됩니다.")
gen_clicked = st.sidebar.button("G-code 생성", use_container_width=True, disabled=not KEY_OK)
if gen_clicked and not KEY_OK:
    st.sidebar.warning("라이센스키를 입력해야 코드생성 및 부가기능을 사용할 수 있습니다.")

if uploaded is not None:
    is_new_upload = (st.session_state.get("last_uploaded_name") != uploaded.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    mesh = trimesh.load_mesh(tmp_path)
    if not isinstance(mesh, trimesh.Trimesh):
        st.error("STL 파일에는 단일 메시만 포함되어야 합니다.")
        st.stop()
    extents = np.asarray(mesh.extents, dtype=float)
    max_extent = float(np.max(extents)) if extents.size else 0.0
    scale_to_mm = 1000.0 if (0.0 < max_extent <= 20.0) else 1.0
    if scale_to_mm != 1.0:
        mesh.apply_scale(scale_to_mm)
    scale_matrix = np.eye(4)
    scale_matrix[2, 2] = 1.0000001
    mesh.apply_transform(scale_matrix)

    if bool(st.session_state.get("stl_apply_move", False)):
        mesh = _apply_translation_to_mesh(
            mesh,
            dx=float(st.session_state.get("stl_move_x", 0.0)),
            dy=float(st.session_state.get("stl_move_y", 0.0)),
            dz=float(st.session_state.get("stl_move_z", 0.0)),
        )

    if bool(st.session_state.get("stl_apply_rot", False)):
        mesh = _apply_rotation_about_centroid(
            mesh,
            rz_deg=float(st.session_state.get("stl_rot_z", 0.0)),
            rx_deg=float(st.session_state.get("stl_rot_x", 0.0)),
            ry_deg=float(st.session_state.get("stl_rot_y", 0.0)),
        )

    st.session_state.mesh = mesh
    st.session_state.base_name = Path(uploaded.name).stem or "output"
    st.session_state.last_uploaded_name = uploaded.name
    if is_new_upload:
        st.session_state.main_view = "STL 미리보기"

if slice_clicked and st.session_state.mesh is not None:
    items = compute_slice_paths_with_travel(
        st.session_state.mesh, z_int=z_int, ref_pt_user=(ref_x, ref_y),
        trim_dist=trim_dist, min_spacing=min_spacing, auto_start=actual_auto_start,
        e_on=e_on, seq_print=seq_print, seq_group_inner=seq_group_inner,
        nozzle_width=float(st.session_state.get("nozzle_diameter_mm", 0.0)) * 0.5,
        enable_inward_offset=bool(st.session_state.get("enable_inward_offset", False)),
        skip_invalid_offset=bool(st.session_state.get("skip_invalid_offset", True))
    )
    st.session_state.paths_items = items
    st.session_state.main_view = "슬라이싱 경로 (3D)"
    segs = items_to_segments(items, e_on=e_on)
    max_seg = len(segs)
    st.session_state.paths_scrub = max_seg
    reset_anim_buffers()
    rebuild_buffers_to(segs, max_seg)
    if max_seg == 0:
        st.session_state.ui_banner = "슬라이싱 결과가 없습니다. STL 단위, Z 간격, 트림/레이어 폭, 최소 점 간격을 확인하세요."
    else:
        st.session_state.ui_banner = f"슬라이싱 완료: 세그먼트 {max_seg:,}개"

if KEY_OK and gen_clicked and st.session_state.mesh is not None:
    gcode_text = generate_gcode(
        st.session_state.mesh, z_int=z_int, feed=feed, ref_pt_user=(ref_x, ref_y),
        e_on=e_on, start_e_on=start_e_on, start_e_val=start_e_val, e0_on=e0_on,
        trim_dist=trim_dist, min_spacing=min_spacing, auto_start=actual_auto_start, m30_on=m30_on,
        seq_print=seq_print  
    )
    st.session_state.gcode_text = gcode_text
    st.session_state.ui_banner = "G-code ready"

if st.session_state.get("gcode_text"):
    base = st.session_state.get("base_name", "output")
    st.sidebar.download_button("G-code 저장", st.session_state.gcode_text,
                               file_name=f"{base}.gcode", mime="text/plain",
                               use_container_width=True)
# =========================
# Rapid(MODX) - Mapping Presets + Converter
# =========================
def _fmt_pos(v: float) -> str:
    if abs(v) < 5e-5: v = 0.0
    s = f"{v:+.1f}"
    sign = s[0]
    intpart, dec = s[1:].split(".")
    intpart = intpart.zfill(4)
    return f"{sign}{intpart}.{dec}"

def _fmt_ang(v: float) -> str:
    if abs(v) < 5e-5: v = 0.0
    s = f"{v:+.2f}"
    sign = s[0]
    intpart, dec = s[1:].split(".")
    intpart = intpart.zfill(3)
    return f"{sign}{intpart}.{dec}"

def _linmap(val: float, a0: float, a1: float, b0: float, b1: float) -> float:
    if abs(a1 - a0) < 1e-12: return float(b0)
    t = (val - a0) / (a1 - a0)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return float(b0 + t * (b1 - b0))

DEFAULT_PRESET = {
    "0": {
        "X": {"in": [0.0, 6500.0], "A3_out": [0.0, 500.0]},
        "Y": {"in": [0.0, 1000.0]},
        "Z": {"in": [0.0, 3000.0], "A4_out": [0.0, 1000.0]},
    },
    "90": {
        "X": {"in": [0.0, 6500.0]},
        "Y": {"in": [0.0, 1000.0], "A3_out": [0.0, 500.0]},
        "Z": {"in": [0.0, 3000.0], "A4_out": [0.0, 1000.0]},
    },
    "-90": {
        "X": {"in": [0.0, 6500.0]},
        "Y": {"in": [0.0, 1000.0], "A3_out": [0.0, 500.0]},
        "Z": {"in": [0.0, 3000.0], "A4_out": [0.0, 1000.0]},
    },
}

def _deepcopy_preset(p: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(p))

if "mapping_preset" not in st.session_state:
    st.session_state.mapping_preset = _deepcopy_preset(DEFAULT_PRESET)

PAD_LINE = '+0000.0,+0000.0,+0000.0,+000.00,+000.00,+000.00,+0000.0,+0000.0,+0000.0,+0000.0'
MAX_LINES = 64000

def _extract_xyz_lines_count(gcode_text: str) -> int:
    cnt = 0
    for raw in gcode_text.splitlines():
        t = raw.strip()
        if not (t.startswith(("G0","G00","G1","G01"))):
            continue
        has_xyz = any(p.startswith(("X","Y","Z")) for p in t.split())
        if has_xyz:
            cnt += 1
    return cnt

def _apply_const_speed_profile_on_nodes(
    nodes: List[Dict[str, Any]], axis_key: str, coord_key: str, coord_min: float, coord_max: float,
    axis_at_min: float, axis_at_max: float, speed_mm_s: float = 200.0, deadband_mm: float = 11.0,
    eps_mm: float = 0.5, apply_print_only: bool = False, travel_interp: bool = True, step_mm: float = 0.0, step_round: str = "floor"
) -> None:
    if not nodes or axis_key not in ("a1", "a2"): return
    n = len(nodes)
    if n == 0: return

    coord_min = float(coord_min)
    coord_max = float(coord_max)
    axis_at_min = float(axis_at_min)
    axis_at_max = float(axis_at_max)
    eps = float(max(0.0, eps_mm))
    span = float(coord_max - coord_min)
    span_abs = abs(span)

    if span_abs <= 1e-9:
        for nd in nodes: nd[axis_key] = float(axis_at_min)
        return

    def _at_min(c: float) -> bool: return c <= coord_min + eps
    def _at_max(c: float) -> bool: return c >= coord_max - eps

    def _snap_linear(c: float) -> float:
        if _at_min(c): return float(axis_at_min)
        if _at_max(c): return float(axis_at_max)
        t = (c - coord_min) / span_abs
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return float(axis_at_min + t * (axis_at_max - axis_at_min))

    def _snap_step(c: float) -> float:
        if _at_min(c): return float(axis_at_min)
        if _at_max(c): return float(axis_at_max)
        t = (c - coord_min) / span_abs
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        dt = float(step_mm) / span_abs if step_mm is not None else 0.0
        if dt <= 1e-12: return _snap_linear(c)

        if step_round == "round": tq = round(t / dt) * dt
        elif step_round == "ceil": tq = math.ceil(t / dt) * dt
        else: tq = math.floor(t / dt) * dt
        tq = 0.0 if tq < 0.0 else 1.0 if tq > 1.0 else tq
        return float(axis_at_min + tq * (axis_at_max - axis_at_min))

    use_step = (step_mm is not None) and (float(step_mm) > 0.0)
    snap_for_coord = _snap_step if use_step else _snap_linear
    extr_node = [bool(nd.get("extr", False)) for nd in nodes]

    for i in range(n):
        c = float(nodes[i].get(coord_key, 0.0))
        target_val = snap_for_coord(c)
        if apply_print_only and not extr_node[i]:
            if travel_interp: nodes[i][axis_key] = target_val
            else: nodes[i][axis_key] = nodes[i-1][axis_key] if i > 0 else target_val
        else: nodes[i][axis_key] = target_val

    if travel_interp and apply_print_only:
        active_node = extr_node
        if any(active_node):
            i = 0
            while i < n:
                if active_node[i]:
                    i += 1; continue
                t0 = i
                while i < n and not active_node[i]: i += 1
                t1 = i - 1

                prev_idx = t0 - 1 if (t0 - 1) >= 0 else None
                next_idx = i if i < n else None
                if prev_idx is None or next_idx is None:
                    base = float(nodes[prev_idx][axis_key]) if prev_idx is not None else (
                        float(nodes[next_idx][axis_key]) if next_idx is not None else float(axis_at_min)
                    )
                    for k in range(t0, t1 + 1): nodes[k][axis_key] = base
                    continue

                a0 = float(nodes[prev_idx][axis_key])
                a1 = float(nodes[next_idx][axis_key])
                total = max(1, (t1 - t0 + 1))
                for kk, k in enumerate(range(t0, t1 + 1)):
                    u = (kk + 1) / float(total + 1)
                    nodes[k][axis_key] = float(a0 + (a1 - a0) * u)

def convert_gcode_to_rapid(
    gcode_text: str, rx: float, ry: float, rz: float, preset: Dict[str, Any], swap_a3_a4: bool = False,
    enable_a1_const: bool = False, enable_a2_const: bool = False, enable_a3: bool = False, enable_a4: bool = False,
    x_min: float = 0.0, x_max: float = 6000.0, a1_at_xmin: float = 4000.0, a1_at_xmax: float = 0.0,
    y_min: float = 0.0, y_max: float = 1000.0, a2_at_ymin: float = 0.0, a2_at_ymax: float = 4000.0,
    speed_mm_s: float = 200.0, boundary_eps_mm: float = 0.5, apply_print_only: bool = False,
    travel_interp: bool = True, singularity_avoid: bool = False, singularity_z_trigger: float = 0.0, singularity_lift_z: float = 300.0,
) -> str:
    key = "0" if abs(rz - 0.0) < 1e-6 else ("90" if abs(rz - 90.0) < 1e-6 else ("-90" if abs(rz + 90.0) < 1e-6 else None))
    P = preset.get(key, {}) if key is not None else {}

    def gi(d: Dict[str, Any], path: list, default: float) -> float:
        try:
            cur = d
            for k in path[:-1]: cur = cur[k]
            return float(cur[path[-1]])
        except Exception: return float(default)

    x0, x1 = gi(P, ["X","in",0], 0.0), gi(P, ["X","in",1], 1.0)
    y0, y1 = gi(P, ["Y","in",0], 0.0), gi(P, ["Y","in",1], 1.0)
    z0, z1 = gi(P, ["Z","in",0], 0.0), gi(P, ["Z","in",1], 1.0)
    a4_0, a4_1 = gi(P, ["Z","A4_out",0], 0.0), gi(P, ["Z","A4_out",1], 0.0)

    a3_on_x = bool(enable_a3) and ("A3_out" in P.get("X", {}))
    a3_on_y = bool(enable_a3) and ("A3_out" in P.get("Y", {}))
    a3x_0, a3x_1 = (gi(P, ["X","A3_out",0], 0.0), gi(P, ["X","A3_out",1], 0.0)) if a3_on_x else (0.0, 0.0)
    a3y_0, a3y_1 = (gi(P, ["Y","A3_out",0], 0.0), gi(P, ["Y","A3_out",1], 0.0)) if a3_on_y else (0.0, 0.0)

    frx, fry, frz = _fmt_ang(rx), _fmt_ang(ry), _fmt_ang(rz)
    have_prev = False; prev_x = prev_y = prev_z = 0.0; prev_e = None; cur_a3 = 0.0
    singularity_a4_offset = 0.0; singularity_applied = False

    xs_out, ys_out, zs_out, raw_xs, raw_ys = [], [], [], [], []
    a1_list, a2_list, a3_list, a4_list, is_extruding_list = [], [], [], [], []

    for raw in gcode_text.splitlines():
        t = raw.strip()
        if not t or not t.startswith(("G0","G00","G1","G01")): continue
        cx, cy, cz = prev_x, prev_y, prev_z
        ce = None; has_any = False
        for p in t.split():
            if p.startswith("X"):
                try: cx = float(p[1:]); has_any = True
                except: pass
            elif p.startswith("Y"):
                try: cy = float(p[1:]); has_any = True
                except: pass
            elif p.startswith("Z"):
                try: cz = float(p[1:]); has_any = True
                except: pass
            elif p.startswith("E"):
                try: ce = float(p[1:])
                except: ce = None
        if not has_any: continue

        is_extruding = False
        if ce is not None and prev_e is not None:
            if (ce - prev_e) > 1e-12: is_extruding = True
        if ce is not None: prev_e = ce

        a4_nominal = _linmap(cz, z0, z1, a4_0, a4_1) if bool(enable_a4) else 0.0
        a4_abs = a4_nominal - singularity_a4_offset if bool(enable_a4) else 0.0
        x_out, y_out, z_out = cx, cy, cz - a4_abs
        curr_h_raw = float(z_out + a4_abs)

        if key == "0" and a3_on_x: cur_a3 = _linmap(cx, x0, x1, a3x_0, a3x_1); x_out = cx - cur_a3
        elif key == "90" and a3_on_y: cur_a3 = _linmap(cy, y0, y1, a3y_0, a3y_1); y_out = cy - cur_a3
        elif key == "-90" and a3_on_y: cur_a3 = _linmap(cy, y0, y1, a3y_0, a3y_1); y_out = cy + cur_a3
        else: cur_a3 = 0.0

        if (bool(enable_a4) and bool(singularity_avoid) and (not singularity_applied) and curr_h_raw >= float(singularity_z_trigger) - 1e-9):
            z_bump = float(singularity_lift_z)
            trigger_a4_after = float(a4_nominal) - (singularity_a4_offset + z_bump)
            if trigger_a4_after < 0.0:
                raise ValueError("싱귤러리티 회피 불가: A4 값이 음수가 됩니다. 강제 상승/하강 수치를 조정하세요.")
            singularity_a4_offset += z_bump
            singularity_applied = True
            a4_abs = a4_nominal - singularity_a4_offset
            z_out = cz - a4_abs

            if key == "0" and a3_on_x: cur_a3 = _linmap(cx, x0, x1, a3x_0, a3x_1); x_out = cx - cur_a3; y_out = cy
            elif key == "90" and a3_on_y: cur_a3 = _linmap(cy, y0, y1, a3y_0, a3y_1); x_out = cx; y_out = cy - cur_a3
            elif key == "-90" and a3_on_y: cur_a3 = _linmap(cy, y0, y1, a3y_0, a3y_1); x_out = cx; y_out = cy + cur_a3
            else: cur_a3 = 0.0; x_out = cx; y_out = cy

        have_prev = True
        raw_xs.append(float(cx)); raw_ys.append(float(cy))
        xs_out.append(float(x_out)); ys_out.append(float(y_out)); zs_out.append(float(z_out))
        a1_list.append(0.0); a2_list.append(0.0); a3_list.append(float(cur_a3)); a4_list.append(float(a4_abs))
        is_extruding_list.append(bool(is_extruding))

        if len(xs_out) >= MAX_LINES: break
        prev_x, prev_y, prev_z = cx, cy, cz

    if len(xs_out) == 0:
        lines_out = [PAD_LINE] * MAX_LINES
        ts = datetime.now().strftime("%Y-%m-%d %p %I:%M:%S")
        return (f"MODULE Converted\n!*** Generated {ts}\nVAR string sFileCount:=\"{MAX_LINES}\";\n"
                f"VAR string d3dpDynLoad{{{MAX_LINES}}}:=[\n" + ",\n".join(f'"{ln}"' for ln in lines_out) + "\n];\nENDMODULE\n")

    nodes = [{"x": xs_out[i], "y": ys_out[i], "z": zs_out[i], "raw_x": raw_xs[i], "raw_y": raw_ys[i],
              "a1": a1_list[i], "a2": a2_list[i], "a3": a3_list[i], "a4": a4_list[i], "extr": is_extruding_list[i]} for i in range(len(xs_out))]

    if bool(enable_a1_const):
        _apply_const_speed_profile_on_nodes(nodes, "a1", "raw_x", x_min, x_max, a1_at_xmin, a1_at_xmax, speed_mm_s, 11.0, boundary_eps_mm, apply_print_only, travel_interp)
    if bool(enable_a2_const):
        use_step = bool(st.session_state.get("extconsta2usestep", False))
        _apply_const_speed_profile_on_nodes(nodes, "a2", "raw_y", y_min, y_max, a2_at_ymin, a2_at_ymax, speed_mm_s, 11.0, boundary_eps_mm, apply_print_only, travel_interp,
                                            float(st.session_state.get("extconsta2stepmm", 0.0)) if use_step else 0.0)

    lines_out = []
    for nd in nodes:
        if len(lines_out) >= MAX_LINES: break
        x, y, z = _fmt_pos(nd["x"]), _fmt_pos(nd["y"]), _fmt_pos(nd["z"])
        a3_v, a4_v = (nd["a4"], nd["a3"]) if swap_a3_a4 else (nd["a3"], nd["a4"])
        a1s, a2s, a3s, a4s = _fmt_pos(nd["a1"]), _fmt_pos(nd["a2"]), _fmt_pos(a3_v), _fmt_pos(a4_v)
        lines_out.append(f"{x},{y},{z},{frx},{fry},{frz},{a1s},{a2s},{a3s},{a4s}")

    while len(lines_out) < MAX_LINES: lines_out.append(PAD_LINE)

    ts = datetime.now().strftime("%Y-%m-%d %p %I:%M:%S")
    header = ("MODULE Converted\n!*** Generated {0} by Gcode→RAPID converter.\n"
              "!*** data3dp: X(mm), Y(mm), Z(mm), Rx(deg), Ry(deg), Rz(deg), A1,A2,A3,A4\n").format(ts)
    open_decl = f'VAR string sFileCount:="{MAX_LINES}";\nVAR string d3dpDynLoad{{{MAX_LINES}}}:=[\n'
    body = ",\n".join(f'"{ln}"' for ln in lines_out) + "\n"
    return header + open_decl + body + "];\nENDMODULE\n"

# ---- 사이드바: Rapid UI ----
st.sidebar.markdown("---")
rapid_clicked = st.sidebar.button("R-code 생성", use_container_width=True, disabled=not KEY_OK)
if rapid_clicked and not KEY_OK:
    st.sidebar.warning("인증키를 입력해야 R-code 생성을 사용할 수 있습니다.")
if KEY_OK and rapid_clicked:
    st.session_state.show_rapid_panel = True

if KEY_OK and st.session_state.show_rapid_panel:
        with st.sidebar.expander("Rapid 설정", expanded=True):
            st.session_state.rapid_rx = st.number_input("Rx (deg)", value=float(st.session_state.rapid_rx), step=0.1, format="%.2f")
            st.session_state.rapid_ry = st.number_input("Ry (deg)", value=float(st.session_state.rapid_ry), step=0.1, format="%.2f")
            rz_preset = st.selectbox("Rz (deg) 프리셋", options=[0.00, 90.0, -90.0],
                                     index={0.00:0, 90.0:1, -90.0:2}.get(float(st.session_state.get("rapid_rz", 0.0)), 0))
            st.session_state.rapid_rz = float(rz_preset)

        with st.sidebar.expander("외부축 (A1/A2 등속 왕복 · 경계정지)", expanded=True):
            st.caption("X/Y 입력 범위를 실제 외부축 A1/A2 위치로 매핑합니다.")
            st.session_state.ext_const_apply_print_only = False
            st.session_state.ext_const_travel_interp = True
            st.session_state.extconsta2usestep = False

            st.markdown("**A1 설정 (X → A1)**")
            st.session_state.ext_const_enable_a1 = st.checkbox("A1 사용", value=bool(st.session_state.ext_const_enable_a1))
            cols = st.columns(2)
            st.session_state.ext_const_xmin = cols[0].number_input("X 입력 최소값 (mm)", value=float(st.session_state.ext_const_xmin), step=50.0, format="%.3f")
            st.session_state.ext_const_xmax = cols[1].number_input("X 입력 최대값 (mm)", value=float(st.session_state.ext_const_xmax), step=50.0, format="%.3f")
            cols2 = st.columns(2)
            st.session_state.ext_const_a1_at_xmin = cols2[0].number_input("A1 최소 위치", value=float(st.session_state.ext_const_a1_at_xmin), step=50.0, format="%.3f")
            st.session_state.ext_const_a1_at_xmax = cols2[1].number_input("A1 최대 위치", value=float(st.session_state.ext_const_a1_at_xmax), step=50.0, format="%.3f")

            st.markdown("**A2 설정 (Y → A2)**")
            st.session_state.ext_const_enable_a2 = st.checkbox("A2 사용", value=bool(st.session_state.ext_const_enable_a2))
            cols3 = st.columns(2)
            st.session_state.ext_const_ymin = cols3[0].number_input("Y 입력 최소값 (mm)", value=float(st.session_state.ext_const_ymin), step=50.0, format="%.3f")
            st.session_state.ext_const_ymax = cols3[1].number_input("Y 입력 최대값 (mm)", value=float(st.session_state.ext_const_ymax), step=50.0, format="%.3f")
            cols4 = st.columns(2)
            st.session_state.ext_const_a2_at_ymin = cols4[0].number_input("A2 최소 위치", value=float(st.session_state.ext_const_a2_at_ymin), step=50.0, format="%.3f")
            st.session_state.ext_const_a2_at_ymax = cols4[1].number_input("A2 최대 위치", value=float(st.session_state.ext_const_a2_at_ymax), step=50.0, format="%.3f")

            with st.expander("고급 설정", expanded=False):
                st.session_state.ext_const_speed_mm_s = st.number_input("축 기준 속도 (mm/s)", min_value=1.0, max_value=2000.0, value=float(st.session_state.ext_const_speed_mm_s), step=10.0, format="%.1f")
                st.session_state.ext_const_eps_mm = st.number_input("경계 허용값 eps (mm)", min_value=0.0, max_value=50.0, value=float(st.session_state.ext_const_eps_mm), step=0.1, format="%.2f")

            def edit_axis(title_key: str, axis_key: str):
                use_a3 = bool(st.session_state.get("ext_use_a3", False))
                use_a4 = bool(st.session_state.get("ext_use_a4", False))
                visible = False
                if axis_key == "Z" and use_a4: visible = True
                elif title_key == "0" and axis_key == "X" and use_a3: visible = True
                elif title_key in ("90", "-90") and axis_key == "Y" and use_a3: visible = True
                if not visible: return

                st.write(f"Rz {title_key} · {axis_key}")
                if title_key not in st.session_state.mapping_preset: st.session_state.mapping_preset[title_key] = {}
                if axis_key not in st.session_state.mapping_preset[title_key]: st.session_state.mapping_preset[title_key][axis_key] = {}

                PAX = st.session_state.mapping_preset[title_key][axis_key]
                base = DEFAULT_PRESET.get(title_key, {}).get(axis_key, {})
                if "in" not in PAX: PAX["in"] = list(base.get("in", [0.0, 0.0]))

                cols_in = st.columns(2)
                in0 = cols_in[0].number_input("in0", value=float(PAX["in"][0]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_in0")
                in1 = cols_in[1].number_input("in1", value=float(PAX["in"][1]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_in1")
                PAX["in"] = [float(in0), float(in1)]

                cols_out = st.columns(2)
                if axis_key in ("X", "Y"):
                    if "A3_out" not in PAX: PAX["A3_out"] = list(base.get("A3_out", [0.0, 0.0]))
                    a30 = cols_out[0].number_input("A3out0", value=float(PAX["A3_out"][0]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_a30")
                    a31 = cols_out[1].number_input("A3out1", value=float(PAX["A3_out"][1]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_a31")
                    PAX["A3_out"] = [float(a30), float(a31)]
                elif axis_key == "Z":
                    if "A4_out" not in PAX: PAX["A4_out"] = list(base.get("A4_out", [0.0, 0.0]))
                    a40 = cols_out[0].number_input("A4out0", value=float(PAX["A4_out"][0]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_a40")
                    a41 = cols_out[1].number_input("A4out1", value=float(PAX["A4_out"][1]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_a41")
                    PAX["A4_out"] = [float(a40), float(a41)]
                st.session_state.mapping_preset[title_key][axis_key] = PAX

            st.session_state.ext_use_a3 = st.checkbox("A3 사용", value=bool(st.session_state.get("ext_use_a3", False)), key="ext_use_a3_checkbox")
            st.session_state.ext_use_a4 = st.checkbox("A4 사용", value=bool(st.session_state.get("ext_use_a4", False)), key="ext_use_a4_checkbox")

            current_rz = float(st.session_state.get("rapid_rz", 0.0))
            if abs(current_rz - 0.0) < 1e-6: key_title = "0"
            elif abs(current_rz - 90.0) < 1e-6: key_title = "90"
            elif abs(current_rz + 90.0) < 1e-6: key_title = "-90"
            else: key_title = "0"

            st.markdown(f"---\n### Rz {key_title} 프리셋")
            edit_axis(key_title, "X"); edit_axis(key_title, "Y"); edit_axis(key_title, "Z")

            preset_json = json.dumps(st.session_state.mapping_preset, ensure_ascii=False, indent=2)
            st.download_button("매핑 프리셋 JSON 저장", preset_json, file_name="mapping_preset.json", mime="application/json", use_container_width=True)

        with st.sidebar.expander("싱귤러리티 회피", expanded=False):
            a4_enabled_now = bool(st.session_state.get("ext_use_a4", False))
            st.session_state.singularity_avoid_enable = st.checkbox("싱귤러리티 회피 사용", value=bool(st.session_state.get("singularity_avoid_enable", False)), disabled=not a4_enabled_now, help="A4 사용 시에만 적용됩니다.")
            st.session_state.singularity_z_trigger = st.number_input("싱귤러리티 발생 Z 높이 (mm)", value=float(st.session_state.get("singularity_z_trigger", 0.0)), step=10.0, format="%.3f", disabled=not (a4_enabled_now and st.session_state.singularity_avoid_enable))
            st.session_state.singularity_lift_z = st.number_input("강제 상승/하강 거리 (mm)", value=float(st.session_state.get("singularity_lift_z", 300.0)), step=10.0, format="%.3f", disabled=not (a4_enabled_now and st.session_state.singularity_avoid_enable))
            if not a4_enabled_now: st.info("A4 사용을 켜야 싱귤러리티 회피 옵션이 적용됩니다.")

            gtxt = st.session_state.get("gcode_text")
            over = False
            if gtxt is not None:
                xyz_count = _extract_xyz_lines_count(gtxt)
                over = (xyz_count > MAX_LINES)

            save_rapid_clicked = st.sidebar.button("Rapid 변환", use_container_width=True, disabled=(gtxt is None))
            if gtxt is None:
                st.sidebar.info("먼저 G-code 생성 버튼으로 G-code를 생성하세요.")
            elif over:
                st.sidebar.error("G-code가 64,000줄을 초과하여 Rapid 파일 변환할 수 없습니다.")
            elif save_rapid_clicked:
                try:
                    st.session_state.rapid_text = convert_gcode_to_rapid(
                        gtxt, rx=st.session_state.rapid_rx, ry=st.session_state.rapid_ry, rz=st.session_state.rapid_rz,
                        preset=st.session_state.mapping_preset, swap_a3_a4=False,
                        enable_a1_const=bool(st.session_state.ext_const_enable_a1), enable_a2_const=bool(st.session_state.ext_const_enable_a2),
                        enable_a3=bool(st.session_state.get("ext_use_a3", False)), enable_a4=bool(st.session_state.get("ext_use_a4", False)),
                        x_min=float(st.session_state.ext_const_xmin), x_max=float(st.session_state.ext_const_xmax),
                        a1_at_xmin=float(st.session_state.ext_const_a1_at_xmin), a1_at_xmax=float(st.session_state.ext_const_a1_at_xmax),
                        y_min=float(st.session_state.ext_const_ymin), y_max=float(st.session_state.ext_const_ymax),
                        a2_at_ymin=float(st.session_state.ext_const_a2_at_ymin), a2_at_ymax=float(st.session_state.ext_const_a2_at_ymax),
                        speed_mm_s=float(st.session_state.ext_const_speed_mm_s), boundary_eps_mm=float(st.session_state.ext_const_eps_mm),
                        apply_print_only=bool(st.session_state.ext_const_apply_print_only), travel_interp=bool(st.session_state.ext_const_travel_interp),
                        singularity_avoid=bool(st.session_state.get("singularity_avoid_enable", False)),
                        singularity_z_trigger=float(st.session_state.get("singularity_z_trigger", 0.0)),
                        singularity_lift_z=float(st.session_state.get("singularity_lift_z", 300.0)),
                    )
                    st.sidebar.success(f"Rapid(*.MODX) 변환 완료 (Rz={st.session_state.rapid_rz:.2f}°)")
                except ValueError as e:
                    st.session_state.rapid_text = None
                    st.sidebar.warning(str(e), icon="⚠️")

            if st.session_state.get("rapid_text"):
                base = st.session_state.get("base_name", "output")
                st.sidebar.download_button("Rapid 저장 (.modx)", st.session_state.rapid_text, file_name=f"{base}.modx", mime="text/plain", use_container_width=True)

# =========================
# Layout (Center + Right)
# =========================
center_col, right_col = st.columns([14, 3], gap="large")

segments = None
total_segments = 0
if st.session_state.get("paths_items") is not None:
    segments = items_to_segments(st.session_state.paths_items, e_on=e_on)
    total_segments = len(segments)

with right_col:
    st.markdown("<div class='right-panel'>", unsafe_allow_html=True)
    if st.session_state.get("ui_banner"): st.success(st.session_state.ui_banner)

    st.subheader("보기 옵션")
    apply_offsets = st.checkbox("레이어 폭 적용", value=bool(st.session_state.get("apply_offsets_flag", False)), help="트림/레이어 폭(mm)을 W로 사용하여 중심 경로와 좌/우 오프셋을 표시합니다.", disabled=(segments is None))
    st.session_state.apply_offsets_flag = bool(apply_offsets)

    include_z_climb = st.checkbox("Z 상승 오프셋 포함", value=True, help="Z가 변하는 travel 구간에도 오프셋을 표시합니다.", disabled=(segments is None or not apply_offsets))
    emphasize_caps = st.checkbox("캡 강조", value=False, help="시작/끝 반원 캡을 빨강/굵은 선으로 강조합니다.", disabled=(segments is None or not apply_offsets))

    if e_on:
        show_dotted = st.checkbox("비출력 이동 경로를 점선으로 표시", value=True, disabled=(segments is None))
        travel_mode = "dotted" if show_dotted else "hidden"
    else:
        st.checkbox("비출력 이동 경로를 점선으로 표시", value=False, disabled=True, help="E 값 삽입 OFF이면 비출력 이동 경로는 실선으로 표기")
        travel_mode = "solid"
    prev_mode = st.session_state.get("paths_travel_mode", "solid")
    st.session_state.paths_travel_mode = travel_mode

    dims_placeholder = st.empty()
    st.markdown("---")

    if segments is None or total_segments == 0:
        st.info("슬라이싱 후 진행 슬라이더가 나타납니다.")
    else:
        default_val = int(clamp(st.session_state.paths_scrub, 0, total_segments))
        if "paths_scrub_slider_initialized" not in st.session_state:
            st.session_state.paths_scrub_slider = default_val
            st.session_state.paths_scrub_slider_initialized = True
        if "paths_scrub_input_initialized" not in st.session_state:
            st.session_state.paths_scrub_input = default_val
            st.session_state.paths_scrub_input_initialized = True

        st.slider("진행(세그먼트)", 0, int(total_segments), key="paths_scrub_slider", step=1, help="해당 세그먼트까지 누적 표시", on_change=_sync_scrub_from_slider)
        st.number_input("행 번호", min_value=0, max_value=int(total_segments), key="paths_scrub_input", step=1, help="표시할 최종 세그먼트(행) 번호", on_change=_sync_scrub_from_input, args=(int(total_segments),))

        target = int(clamp(st.session_state.get("paths_scrub", default_val), 0, total_segments))

        layer_z_values = _collect_layer_z_values(segments)
        max_layer_no = len(layer_z_values)
        view_selected_layers_only = st.checkbox("선택 레이어만 보기", value=st.session_state.get("view_selected_layers_only", False), help="선택한 레이어 범위만 3D 경로에 표시합니다.")
        st.caption(f"총 레이어 수: {max_layer_no}")
        st.session_state["view_selected_layers_only"] = bool(view_selected_layers_only)
        if max_layer_no > 0:
            default_layer_start = int(clamp(st.session_state.get("layer_view_start", 1), 1, max_layer_no))
            default_layer_end = int(clamp(st.session_state.get("layer_view_end", default_layer_start), default_layer_start, max_layer_no))
            if "layer_view_start_input" not in st.session_state:
                st.session_state["layer_view_start_input"] = default_layer_start
            if "layer_view_end_input" not in st.session_state:
                st.session_state["layer_view_end_input"] = default_layer_end
            c_layer1, c_layer2 = st.columns(2)
            with c_layer1:
                st.number_input("시작 레이어", min_value=1, max_value=max_layer_no, key="layer_view_start_input", step=1, on_change=_sync_layer_inputs, args=(int(max_layer_no),))
            with c_layer2:
                st.number_input("끝 레이어", min_value=1, max_value=max_layer_no, key="layer_view_end_input", step=1, on_change=_sync_layer_inputs, args=(int(max_layer_no),))
            layer_view_start = int(clamp(st.session_state.get("layer_view_start_input", default_layer_start), 1, max_layer_no))
            layer_view_end = int(clamp(st.session_state.get("layer_view_end_input", layer_view_start), layer_view_start, max_layer_no))
            st.session_state["layer_view_start"] = layer_view_start
            st.session_state["layer_view_end"] = layer_view_end
            st.caption(f"레이어 범위: {layer_view_start} ~ {layer_view_end}")
        else:
            st.session_state["layer_view_start"] = 1
            st.session_state["layer_view_end"] = 1
            st.caption("레이어 범위: -")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- 계산/버퍼 구성 ----
if segments is not None and total_segments > 0:
    target = int(clamp(st.session_state.paths_scrub, 0, total_segments))
    DRAW_LIMIT = 150000
    draw_stride = 1
    selected_only = bool(st.session_state.get("view_selected_layers_only", False))
    layer_view_start = int(st.session_state.get("layer_view_start", 1))
    layer_view_end = int(st.session_state.get("layer_view_end", layer_view_start))

    if selected_only:
        visible_segments = _filter_segments_by_layer_range(segments, layer_view_start, layer_view_end)
        _build_buffers_from_segments_subset(visible_segments, travel_mode=st.session_state.paths_travel_mode)
        st.session_state.paths_scrub = target
        if bool(st.session_state.get("apply_offsets_flag", False)):
            half_w = float(trim_dist) * 0.5
            compute_offsets_into_buffers(visible_segments, len(visible_segments), half_w, include_travel_climb=bool(include_z_climb), climb_z_thresh=1e-9)
            st.session_state.paths_anim_buf["caps"] = {"x": [], "y": [], "z": []}
            add_global_endcaps_into_buffers(visible_segments, len(visible_segments), half_width=half_w, samples=32, store_caps=bool(emphasize_caps))
        if visible_segments:
            total_len = sum(float(np.linalg.norm(p2[:2] - p1[:2])) for (p1, p2, is_travel, is_extruding) in visible_segments if is_extruding)
            st.markdown(f"**선택 레이어 총 길이:** {total_len/1000:.3f} m")
    else:
        built = st.session_state.paths_anim_buf["built_upto"]
        prev_stride = st.session_state.paths_anim_buf.get("stride", 1)
        mode_changed = (prev_mode != st.session_state.paths_travel_mode)

        if mode_changed or (draw_stride != prev_stride) or (target < built): rebuild_buffers_to(segments, target, stride=draw_stride)
        elif target > built: append_segments_to_buffers(segments, built, target, stride=draw_stride)

        st.session_state.paths_scrub = target

        if bool(st.session_state.get("apply_offsets_flag", False)):
            half_w = float(trim_dist) * 0.5
            compute_offsets_into_buffers(segments, target, half_w, include_travel_climb=bool(include_z_climb), climb_z_thresh=1e-9)
            st.session_state.paths_anim_buf["caps"] = {"x": [], "y": [], "z": []}
            add_global_endcaps_into_buffers(segments, target, half_width=half_w, samples=32, store_caps=bool(emphasize_caps))

        if segments is not None and target > 0:
            total_len = sum([float(np.linalg.norm(p2[:2] - p1[:2])) for i, (p1, p2, is_travel, is_extruding) in enumerate(segments[:target]) if is_extruding])
            st.markdown(f"**누적 레이어 총 길이:** {total_len/1000:.3f} m")
else:
    if "paths_anim_buf" in st.session_state:
        st.session_state.paths_anim_buf["off_l"] = {"x": [], "y": [], "z": []}
        st.session_state.paths_anim_buf["off_r"] = {"x": [], "y": [], "z": []}
        st.session_state.paths_anim_buf["caps"]  = {"x": [], "y": [], "z": []}

# ---- 중앙: 뷰 전환 ----
with center_col:
    st.markdown('<div class="left-panel-scroll center-panel-scroll">', unsafe_allow_html=True)
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
        white-space: nowrap;
        padding-left: 0.6rem;
        padding-right: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)
    v1, v2, v3, _sp = st.columns([1.05, 1.0, 0.95, 5.0])
    with v1:
        if st.button("슬라이싱 경로 (3D)", use_container_width=True):
            st.session_state.main_view = "슬라이싱 경로 (3D)"
    with v2:
        if st.button("STL 미리보기", use_container_width=True):
            st.session_state.main_view = "STL 미리보기"
    with v3:
        if st.button("G-code 뷰어", use_container_width=True):
            st.session_state.main_view = "G-code 뷰어"

    current_view = st.session_state.get("main_view", "슬라이싱 경로 (3D)")
    st.caption(f"현재 보기: {current_view}")

    if current_view == "슬라이싱 경로 (3D)":
        if segments is not None and total_segments > 0:
            if "paths_base_fig" not in st.session_state:
                st.session_state.paths_base_fig = make_base_fig(height=820)
            fig = st.session_state.paths_base_fig
            segments_hover_src = visible_segments if bool(st.session_state.get("view_selected_layers_only", False)) else segments[:int(clamp(st.session_state.paths_scrub, 0, total_segments))]
            update_fig_with_buffers(fig, show_offsets=bool(st.session_state.get("apply_offsets_flag", False)), show_caps=bool(emphasize_caps), segments_for_hover=segments_hover_src)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            tm = st.session_state.paths_travel_mode
            if not e_on:
                travel_lbl = "비출력 이동: 실선 (E 값 삽입 OFF)"
            else:
                travel_lbl = "비출력 이동: 점선" if tm == "dotted" else ("비출력 이동: 숨김" if tm == "hidden" else "비출력 이동: 실선")
            st.caption(
                f"세그먼트 총 {total_segments:,} | 현재 {st.session_state.paths_scrub:,}"
                + (f" | 오프셋: ON (W/2 = {float(trim_dist)*0.5:.2f} mm)" if st.session_state.get('apply_offsets_flag', False) else "")
                + (" | 캡 강조" if (st.session_state.get('apply_offsets_flag', False) and emphasize_caps) else "")
                + (f" | {travel_lbl}")
                + (f" | 표시 간격: ×{st.session_state.paths_anim_buf.get('stride',1)}" if st.session_state.paths_anim_buf.get('stride',1) > 1 else "")
            )
        else:
            st.info("슬라이싱 결과가 없으면 경로가 표시되지 않습니다.")

    elif current_view == "STL 미리보기":
        if st.session_state.mesh is not None:
            st.plotly_chart(plot_trimesh(st.session_state.mesh, height=820), use_container_width=True, key="stl_chart", config={"displayModeBar": False})
        else:
            st.info("STL 업로드 후 미리보기가 표시됩니다.")

    elif current_view == "G-code 뷰어":
        if st.session_state.gcode_text:
            st.text_area("G-code", st.session_state.gcode_text, height=820)
        else:
            st.info("G-code 생성 후 표시됩니다.")

