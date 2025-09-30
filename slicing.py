# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import math
import trimesh
import tempfile
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from datetime import date, datetime
from pathlib import Path

# =========================
# App basics
# =========================
st.set_page_config(page_title="3DCP Slicer", layout="wide")

# ── 전역 CSS (UI만 수정) ──
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    [data-testid="stFooter"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}

    /* 전체 상단 패딩(탭 포함 컨텐츠가 너무 위로 붙지 않게) */
    .block-container { padding-top: 2.0rem; }

    /* 탭 자체 상단 여백 */
    .stTabs { margin-top: 1.0rem !important; padding-top: 0.2rem !important; }
    .stTabs [data-baseweb="tab-list"] { margin-top: 0.6rem !important; }

    /* 우측 컨트롤 패널: 독립 스크롤 + sticky */
    .right-panel {
      position: sticky;
      top: 2.0rem;
      max-height: calc(100vh - 2rem);
      overflow-y: auto;
      border-left: 1px solid #e6e6e6;
      padding-left: 12px;
      background: white;
    }

    /* 좌측 사이드바 타이틀(조금 키움: 1~2단계) */
    .sidebar-title {
      margin: 0.25rem 0 0.6rem 0;
      font-size: 1.35rem;
      font-weight: 700;
      line-height: 1.2;
    }

    /* 외부치수: 줄바꿈 확실히 */
    .dims-block {
      white-space: pre-line;  /* \\n을 실제 줄바꿈으로 렌더 */
      line-height: 1.3;
      font-variant-numeric: tabular-nums;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 중앙 큰 타이틀 제거, 좌측 사이드바에만 작은 타이틀(조금 키움)
st.sidebar.markdown("<div class='sidebar-title'>3DCP Slicer</div>", unsafe_allow_html=True)

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
    """
    폐루프 길이를 기준으로 꼬리에서 trim_distance 제거 (open polyline 반환).
    """
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
    """XY 기준 RDP 간소화(끝점 보존)."""
    pts = np.asarray(segment, dtype=float)
    if len(pts) <= 2 or min_dist <= 0:
        return pts
    eps = float(min_dist) / 2.0

    def _perp_dist_xy(p, a, b) -> float:
        ab = b[:2] - a[:2]
        ap = p[:2] - a[:2]
        denom = np.dot(ab, ab)
        if denom <= 1e-18:
            return np.linalg.norm(ap)
        t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
        proj = a[:2] + t * ab
        return np.linalg.norm(p[:2] - proj)

    def _rdp_xy(points: np.ndarray, eps_val: float) -> np.ndarray:
        if len(points) <= 2:
            return points
        a, b = points[0], points[-1]
        dmax, idx = -1.0, -1
        for i in range(1, len(points) - 1):
            d = _perp_dist_xy(points[i], a, b)
            if d > dmax:
                dmax, idx = d, i
        if dmax <= eps_val:
            return np.vstack([a, b])
        left = _rdp_xy(points[: idx + 1], eps_val)
        right = _rdp_xy(points[idx:], eps_val)
        return np.vstack([left[:-1], right])

    return _rdp_xy(pts, eps)

def shift_to_nearest_start(segment, ref_point):
    idx = np.argmin(np.linalg.norm(segment[:, :2] - ref_point, axis=1))
    return np.concatenate([segment[idx:], segment[:idx]], axis=0), segment[idx]

# =========================
# Plotly: STL (정적)
# =========================
def plot_trimesh(mesh: trimesh.Trimesh, height=820) -> go.Figure:
    v = mesh.vertices
    f = mesh.faces
    fig = go.Figure(data=[go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="#888888", opacity=0.6, flatshading=True
    )])
    fig.update_layout(scene=dict(aspectmode="data"),
                      height=height, margin=dict(l=0, r=0, t=10, b=0))
    return fig

# =========================
# G-code generator
# =========================
def generate_gcode(mesh, z_int=30.0, feed=2000, ref_pt_user=(0.0, 0.0),
                   e_on=False, start_e_on=False, start_e_val=0.1, e0_on=False,
                   trim_dist=30.0, min_spacing=3.0, auto_start=False, m30_on=False):
    g = ["; *** Generated by 3DCP Slicer ***", "G21", "G90"]
    if e_on:
        g.append("M83")

    z_max = mesh.bounds[1, 2]
    z_values = list(np.arange(z_int, z_max + 0.001, z_int))
    if abs(z_max - z_values[-1]) > 1e-3:
        z_values.append(z_max)
    z_values.append(z_max + 0.01)

    prev_start_xy = None
    for z in z_values:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
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

        g.append(f"\n; ---------- Z = {z:.2f} mm ----------")

        if auto_start and prev_start_xy is not None:
            ref_pt_layer = prev_start_xy
        else:
            ref_pt_layer = np.array(ref_pt_user, dtype=float)

        for i_seg, seg3d in enumerate(segments):
            seg3d_no_dup = ensure_open_ring(seg3d)
            shifted, _ = shift_to_nearest_start(seg3d_no_dup, ref_point=ref_pt_layer)
            trimmed = trim_closed_ring_tail(shifted, trim_dist)
            simplified = simplify_segment(trimmed, min_spacing)

            if i_seg > 0:
                s = simplified[0]
                g.append(f"G01 X{s[0]:.3f} Y{s[1]:.3f} Z{z:.3f}")

            start = simplified[0]
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

            if e0_on:
                g.append("G01 E0")

            if i_seg == 0:
                prev_start_xy = start[:2]  # auto_start 기준

    g.append(f"G01 F{feed}")
    if m30_on:
        g.append("M30")
    return "\n".join(g)

# =========================
# Slice path computation
# =========================
def compute_slice_paths_with_travel(
    mesh,
    z_int=30.0,
    ref_pt_user=(0.0, 0.0),
    trim_dist=30.0,
    min_spacing=3.0,
    auto_start=False,
    e_on=False
) -> List[Tuple[np.ndarray, Optional[np.ndarray], bool]]:
    z_max = mesh.bounds[1, 2]
    z_values = list(np.arange(z_int, z_max + 0.001, z_int))
    if abs(z_max - z_values[-1]) > 1e-3:
        z_values.append(z_max)
    z_values.append(z_max + 0.01)

    all_items: List[Tuple[np.ndarray, Optional[np.ndarray], bool]] = []
    prev_layer_last_end: Optional[np.ndarray] = None
    prev_start_xy = None

    for z in z_values:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
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

        # 레이어 기준 시작점
        if auto_start and prev_start_xy is not None:
            ref_pt_layer = prev_start_xy
        else:
            ref_pt_layer = np.array(ref_pt_user, dtype=float)

        layer_polys: List[np.ndarray] = []
        for i_seg, seg3d in enumerate(segments):
            seg3d_no_dup = ensure_open_ring(seg3d)
            shifted, _ = shift_to_nearest_start(seg3d_no_dup, ref_point=ref_pt_layer)
            trimmed = trim_closed_ring_tail(shifted, trim_dist)
            simplified = simplify_segment(trimmed, min_spacing)
            layer_polys.append(simplified.copy())
            if i_seg == 0:
                prev_start_xy = simplified[0][:2]

        if not layer_polys:
            continue

        # 레이어 간 travel
        first_poly_start = layer_polys[0][0]
        if prev_layer_last_end is not None:
            travel = np.vstack([prev_layer_last_end, first_poly_start])
            all_items.append((travel, np.array([0.0, 0.0]) if e_on else None, True))

        # 같은 레이어 내: 경로 + 다음 폐구간으로 이동
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

# === items -> segments ===
def items_to_segments(items: List[Tuple[np.ndarray, Optional[np.ndarray], bool]],
                      e_on: bool
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
    segments,
    upto,
    half_width,
    include_travel_climb: bool = False,
    climb_z_thresh: float = 1e-6
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

    # Start cap
    p1, p2, _, _ = segments[first_idx]
    dx = float(p2[0] - p1[0]); dy = float(p2[1] - p1[1]); nrm = (dx*dx + dy*dy) ** 0.5
    if nrm > 1e-12:
        t_unit = np.array([dx/nrm, dy/nrm], dtype=float)
        n_unit = np.array([-t_unit[1], t_unit[0]], dtype=float)
        _append_arc((float(p1[0]), float(p1[1])), t_unit, n_unit, float(p1[2]), sign_t=-1.0, steps=samples)

    # End cap
    p1, p2, _, _ = segments[last_idx]
    dx = float(p2[0] - p1[0]); dy = float(p2[1] - p1[1]); nrm = (dx*dx + dy*dy) ** 0.5
    if nrm > 1e-12:
        t_unit = np.array([dx/nrm, dy/nrm], dtype=float)
        n_unit = np.array([-t_unit[1], t_unit[0]], dtype=float)
        _append_arc((float(p2[0]), float(p2[1])), t_unit, n_unit, float(p2[2]), sign_t=+1.0, steps=samples)

def make_base_fig(height=820) -> go.Figure:
    fig = go.Figure()
    # 0: solid (중심 경로 실선)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=3, dash="solid", color=PATH_COLOR_DEFAULT),
                               showlegend=False))
    # 1: dot (여정 점선)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=3, dash="dot", color=PATH_COLOR_DEFAULT),
                               showlegend=False))
    # 2: offset left (좌)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=3, dash="solid", color=OFFSET_DARK_GRAY),
                               showlegend=False))
    # 3: offset right (우)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=3, dash="solid", color=OFFSET_DARK_GRAY),
                               showlegend=False))
    # 4: caps emphasized (빨강)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                               line=dict(width=6, dash="solid", color=CAP_COLOR),
                               name="Caps Emphasis", showlegend=False))
    fig.update_layout(scene=dict(aspectmode="data"),
                      height=height, margin=dict(l=0, r=0, t=10, b=0),
                      uirevision="keep", transition={'duration': 0})
    return fig

def ensure_paths_fig(height=820):
    fig = st.session_state.get("paths_base_fig")
    ok = False
    try:
        ok = isinstance(fig, go.Figure) and (len(fig.data) >= 5)
    except Exception:
        ok = False
    if not ok:
        st.session_state.paths_base_fig = make_base_fig(height)

def ensure_traces(fig: go.Figure, want=5):
    def add_solid():
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(width=3, dash="solid", color=PATH_COLOR_DEFAULT),
                                   showlegend=False))
    def add_dot():
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(width=3, dash="dot", color=PATH_COLOR_DEFAULT),
                                   showlegend=False))
    def add_off():
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                   line=dict(width=3, dash="solid", color=OFFSET_DARK_GRAY),
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

def update_fig_with_buffers(fig: go.Figure, show_offsets: bool, show_caps: bool):
    ensure_traces(fig, want=5)
    buf = st.session_state.paths_anim_buf

    # 중심 경로 컬러/굵기 동적 설정
    apply_on = bool(st.session_state.get("apply_offsets_flag", False))
    center_color = PATH_COLOR_LIGHT if apply_on else PATH_COLOR_DEFAULT
    fig.data[0].line.color = center_color
    fig.data[1].line.color = center_color

    # 오프셋 트레이스(진한 회색 + 굵기 3)
    fig.data[2].line.color = OFFSET_DARK_GRAY; fig.data[2].line.width = 3
    fig.data[3].line.color = OFFSET_DARK_GRAY; fig.data[3].line.width = 3

    # 메인 좌표 업데이트
    fig.data[0].x = buf["solid"]["x"]; fig.data[0].y = buf["solid"]["y"]; fig.data[0].z = buf["solid"]["z"]
    fig.data[1].x = buf["dot"]["x"];   fig.data[1].y = buf["dot"]["y"];   fig.data[1].z = buf["dot"]["z"]

    # offsets
    if show_offsets:
        fig.data[2].x = buf["off_l"]["x"]; fig.data[2].y = buf["off_l"]["y"]; fig.data[2].z = buf["off_l"]["z"]
        fig.data[3].x = buf["off_r"]["x"]; fig.data[3].y = buf["off_r"]["y"]; fig.data[3].z = buf["off_r"]["z"]
    else:
        fig.data[2].x = []; fig.data[2].y = []; fig.data[2].z = []
        fig.data[3].x = []; fig.data[3].y = []; fig.data[3].z = []

    # caps 강조(빨강)
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
        return {
            "x_min": x_min, "x_max": x_max, "x_len": x_max - x_min,
            "y_min": y_min, "y_max": y_max, "y_len": y_max - y_min,
        }
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

# ▶ HTML 줄바꿈 보장 (외부치수)
def _fmt_dims_block_html(title, bbox, z_single: Optional[float]) -> str:
    if bbox is None:
        return f"<div class='dims-block'><b>{title}</b>\nX=(-)\nY=(-)\nZ=(-)</div>"
    xm, xM, xl = bbox["x_min"], bbox["x_max"], bbox["x_len"]
    ym, yM, yl = bbox["y_min"], bbox["y_max"], bbox["y_len"]
    z_txt = "-" if z_single is None else f"{z_single:.3f}"
    return (
        "<div class='dims-block'>"
        f"<b>{title}</b>\n"
        f"X=({xm:.3f}→{xM:.3f}, Δ{xl:.3f})\n"
        f"Y=({ym:.3f}→{yM:.3f}, Δ{yl:.3f})\n"
        f"Z=({z_txt})"
        "</div>"
    )

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
    st.session_state.paths_scrub = 0
if "paths_travel_mode" not in st.session_state:
    st.session_state.paths_travel_mode = "solid"

# 우측 메시지(슬라이스/G-code 완료) 전용
if "ui_banner" not in st.session_state:
    st.session_state.ui_banner = None

ensure_anim_buffers()

# =========================
# Sidebar (Access)
# =========================
st.sidebar.header("Access")
ALLOWED_WITH_EXPIRY = {"robotics5107": None, "kmou*": "2026-12-31", "0703": "2026-12-31"}
access_key = st.sidebar.text_input("Access Key", type="password", key="access_key")

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

KEY_OK, EXP_DATE, REMAINING, STATUS_TXT = check_key_valid(access_key)
if access_key:
    if KEY_OK:
        if EXP_DATE is None:
            st.sidebar.success(STATUS_TXT)
        else:
            d_mark = f"D-{REMAINING}" if REMAINING > 0 else "D-DAY"
            st.sidebar.info(f"{STATUS_TXT} ({d_mark})")
    else:
        st.sidebar.error(STATUS_TXT)
else:
    st.sidebar.warning("Access Key를 입력하세요.")

uploaded = st.sidebar.file_uploader("Upload STL", type=["stl"], disabled=not KEY_OK)

# =========================
# Sidebar (Parameters)
# =========================
st.sidebar.header("Parameters")
z_int = st.sidebar.number_input("Z interval (mm)", 1.0, 1000.0, 15.0)
feed = st.sidebar.number_input("Feedrate (F)", 1, 100000, 2000)
ref_x = st.sidebar.number_input("Reference X", value=0.0)
ref_y = st.sidebar.number_input("Reference Y", value=0.0)

st.sidebar.subheader("Extrusion options")
e_on = st.sidebar.checkbox("Insert E values")
start_e_on = st.sidebar.checkbox("Continuous Layer Printing", value=False, disabled=not e_on)
start_e_val = st.sidebar.number_input("Start E value", value=0.1, disabled=not (e_on and start_e_on))
e0_on = st.sidebar.checkbox("Add E0 at loop end", value=False, disabled=not e_on)

st.sidebar.subheader("Path processing")
trim_dist = st.sidebar.number_input("Trim/Layer Width (mm)", 0.0, 1000.0, 50.0)
min_spacing = st.sidebar.number_input("Minimum point spacing (mm)", 0.0, 1000.0, 3.0)
auto_start = st.sidebar.checkbox("Start next layer near previous start")
m30_on = st.sidebar.checkbox("Append M30 at end", value=False)

b1 = st.sidebar.container()
b2 = st.sidebar.container()
slice_clicked = b1.button("Slice Model", use_container_width=True)
gen_clicked = b2.button("Generate G-Code", use_container_width=True)

# =========================
# Load mesh on upload
# =========================
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    mesh = trimesh.load_mesh(tmp_path)
    if not isinstance(mesh, trimesh.Trimesh):
        st.error("STL must contain a single mesh")
        st.stop()
    # Z 축 미세 확장 (최대 Z 절단면 인식 보정)
    scale_matrix = np.eye(4)
    scale_matrix[2, 2] = 1.0000001
    mesh.apply_transform(scale_matrix)
    st.session_state.mesh = mesh
    st.session_state.base_name = Path(uploaded.name).stem or "output"

# =========================
# Actions
# =========================
if KEY_OK and slice_clicked and st.session_state.mesh is not None:
    items = compute_slice_paths_with_travel(
        st.session_state.mesh,
        z_int=z_int,
        ref_pt_user=(ref_x, ref_y),
        trim_dist=trim_dist,
        min_spacing=min_spacing,
        auto_start=auto_start,
        e_on=e_on
    )
    st.session_state.paths_items = items

    # 첫 화면에서 '최대'로 보이게
    segs = items_to_segments(items, e_on=e_on)
    max_seg = len(segs)
    st.session_state.paths_scrub = max_seg

    reset_anim_buffers()
    rebuild_buffers_to(segs, max_seg)

    # 우측 View Options 상단 배너
    st.session_state.ui_banner = "Slicing complete"

if KEY_OK and gen_clicked and st.session_state.mesh is not None:
    gcode_text = generate_gcode(
        st.session_state.mesh, z_int=z_int, feed=feed, ref_pt_user=(ref_x, ref_y),
        e_on=e_on, start_e_on=start_e_on, start_e_val=start_e_val, e0_on=e0_on,
        trim_dist=trim_dist, min_spacing=min_spacing, auto_start=auto_start, m30_on=m30_on
    )
    st.session_state.gcode_text = gcode_text
    # 중앙 성공 메시지 대신 우측 배너로만 표시
    st.session_state.ui_banner = "G-code ready"

if st.session_state.get("gcode_text"):
    base = st.session_state.get("base_name", "output")
    st.sidebar.download_button("G-code 저장", st.session_state.gcode_text,
                               file_name=f"{base}.gcode", mime="text/plain",
                               use_container_width=True)

# =========================
# Rapid(MODX)
# =========================
def _fmt_pos(v: float) -> str:
    s = f"{v:+.1f}"; sign = s[0]; intpart, dec = s[1:].split("."); intpart = intpart.zfill(4); return f"{sign}{intpart}.{dec}"
def _fmt_ang(v: float) -> str:
    s = f"{v:+.2f}"; sign = s[0]; intpart, dec = s[1: ].split("."); intpart = intpart.zfill(3); return f"{sign}{intpart}.{dec}"

# ▶ 부가축 포맷 업데이트 + ExtrudeFlag(마지막 11번째 필드)
PAD_LINE = '+0000.0,+0000.0,+0000.0,+000.00,+000.00,+000.00,+0000.0,+000.0,+0000.0,+000.0,0'
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

def gcode_to_cone1500_module(gcode_text: str, rx: float, ry: float, rz: float) -> str:
    lines_out = []
    cur_x = cur_y = cur_z = 0.0
    frx, fry, frz = _fmt_ang(rx), _fmt_ang(ry), _fmt_ang(rz)
    # ✅ 부가축 포맷(요청하신 형태)
    tail = "+0000.0,+000.0,+0000.0,+000.0"

    for raw in gcode_text.splitlines():
        t = raw.strip()
        if not t or not t.startswith(("G0","G00","G1","G01")):
            continue

        parts = t.split()
        has_xyz = False
        e_val = None

        for p in parts:
            c = p[:1]
            if c == "X":
                try: cur_x = float(p[1:]); has_xyz = True
                except: pass
            elif c == "Y":
                try: cur_y = float(p[1:]); has_xyz = True
                except: pass
            elif c == "Z":
                try: cur_z = float(p[1:]); has_xyz = True
                except: pass
            elif c == "E":
                try: e_val = float(p[1:])
                except: e_val = None

        if not has_xyz:
            continue

        fx, fy, fz = _fmt_pos(cur_x), _fmt_pos(cur_y), _fmt_pos(cur_z)
        # ✅ 이 줄에서 마지막 0/1 붙임
        flag = 1 if (e_val is not None and e_val > 0.0) else 0
        lines_out.append(f'{fx},{fy},{fz},{frx},{fry},{frz},{tail},{flag}')

        if len(lines_out) >= MAX_LINES:
            break

    # ✅ 패딩도 11필드 유지
    while len(lines_out) < MAX_LINES:
        lines_out.append(PAD_LINE)

    ts = datetime.now().strftime("%Y-%m-%d %p %I:%M:%S")
    header = ("MODULE Converted\n"
              "! ... \n")
    cnt_str = str(MAX_LINES)
    open_decl = f'VAR string sFileCount:="{cnt_str}";\nVAR string d3dpDynLoad{{{cnt_str}}}:=[\n'
    body = ""
    for i, ln in enumerate(lines_out):
        q = f'"{ln}"'
        body += (q + ",\n") if i < len(lines_out) - 1 else (q + "\n")
    close_decl = "];\nENDMODULE\n"
    return header + open_decl + body + close_decl


st.sidebar.markdown("---")
if KEY_OK:
    if st.sidebar.button("Generate Rapid", use_container_width=True):
        st.session_state.show_rapid_panel = True
    if st.session_state.show_rapid_panel:
        with st.sidebar.expander("Rapid Settings", expanded=True):
            st.session_state.rapid_rx = st.number_input("Rx (deg)", value=float(st.session_state.rapid_rx), step=0.1, format="%.2f")
            st.session_state.rapid_ry = st.number_input("Ry (deg)", value=float(st.session_state.rapid_ry), step=0.1, format="%.2f")
            st.session_state.rapid_rz = st.number_input("Rz (deg)", value=float(st.session_state.rapid_rz), step=0.1, format="%.2f")

            gtxt = st.session_state.get("gcode_text")
            over = None
            if gtxt is not None:
                xyz_count = _extract_xyz_lines_count(gtxt)
                over = (xyz_count > MAX_LINES)

            save_rapid_clicked = st.button("Save Rapid", use_container_width=True, disabled=(gtxt is None))
            if gtxt is None:
                st.info("먼저 Generate G-Code로 G-code를 생성하세요.")
            elif over:
                st.error("G-code가 64,000줄을 초과하여 Rapid 파일 변환할 수 없습니다.")
            elif save_rapid_clicked:
                st.session_state.rapid_text = gcode_to_cone1500_module(
                    gtxt, rx=st.session_state.rapid_rx, ry=st.session_state.rapid_ry, rz=st.session_state.rapid_rz
                )
                st.success("Rapid(MODX, cone1500 형식) 변환 완료")

            if st.session_state.get("rapid_text"):
                base = st.session_state.get("base_name", "output")
                st.download_button(
                    "Rapid 저장 (.modx)",
                    st.session_state.rapid_text,
                    file_name=f"{base}.modx",
                    mime="text/plain",
                    use_container_width=True
                )

# =========================
# Center + Right 레이아웃
# =========================
# ► 오른쪽 폭을 기존보다 약 3/5 축소 느낌([14,3] ≈ 17.6%)
center_col, right_col = st.columns([14, 3], gap="large")

# 현재 슬라이스 세그먼트
segments = None
total_segments = 0
if st.session_state.get("paths_items") is not None:
    segments = items_to_segments(st.session_state.paths_items, e_on=e_on)
    total_segments = len(segments)

# ---- 우측 스크롤 패널 ----
with right_col:
    st.markdown("<div class='right-panel'>", unsafe_allow_html=True)

    # 우측 상단 배너 (Slicing complete / G-code ready)
    if st.session_state.get("ui_banner"):
        st.success(st.session_state.ui_banner)

    st.subheader("View Options")
    apply_offsets = st.checkbox(
        "Apply layer width",
        value=bool(st.session_state.get("apply_offsets_flag", False)),
        help="Trim/Layer Width (mm)를 W로 사용하여 중심 경로와 좌/우 오프셋을 표시합니다.",
        disabled=(segments is None)
    )
    st.session_state.apply_offsets_flag = bool(apply_offsets)

    include_z_climb = st.checkbox(
        "Include Z-climb offsets",
        value=True,
        help="Z가 변하는 travel 구간에도 오프셋을 표시합니다.",
        disabled=(segments is None or not apply_offsets)
    )

    emphasize_caps = st.checkbox(
        "Emphasize caps",
        value=False,
        help="시작/끝 반원 캡을 빨강/굵은 선으로 강조합니다.",
        disabled=(segments is None or not apply_offsets)
    )

    if e_on:
        show_dotted = st.checkbox("Show dotted travel lines", value=True, disabled=(segments is None))
        travel_mode = "dotted" if show_dotted else "hidden"
    else:
        st.checkbox("Show dotted travel lines", value=False, disabled=True,
                    help="Insert E values OFF이면 travel은 실선으로 표기")
        travel_mode = "solid"
    prev_mode = st.session_state.get("paths_travel_mode", "solid")
    st.session_state.paths_travel_mode = travel_mode

    dims_placeholder = st.empty()
    st.markdown("---")

    if segments is None or total_segments == 0:
        st.info("슬라이싱 후 진행 슬라이더가 나타납니다.")
        scrub = None
        scrub_num = None
    else:
        default_val = int(clamp(st.session_state.paths_scrub, 0, total_segments))
        scrub = st.slider("진행(segments)", 0, int(total_segments), int(default_val), 1,
                          help="해당 세그먼트까지 누적 표시")
        scrub_num = st.number_input("행 번호", 0, int(total_segments),
                                    int(default_val), 1,
                                    help="표시할 최종 세그먼트(행) 번호")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- 계산/버퍼 구성 ----
if segments is not None and total_segments > 0:
    default_val = int(clamp(st.session_state.paths_scrub, 0, total_segments))
    target = default_val
    if scrub is not None and scrub != default_val:
        target = int(scrub)
    if scrub_num is not None and scrub_num != default_val:
        target = int(scrub_num)
    target = int(clamp(target, 0, total_segments))

    DRAW_LIMIT = 15000
    draw_stride = max(1, math.ceil(max(1, target) / DRAW_LIMIT))

    built = st.session_state.paths_anim_buf["built_upto"]
    prev_stride = st.session_state.paths_anim_buf.get("stride", 1)
    mode_changed = (prev_mode != st.session_state.paths_travel_mode)

    if mode_changed or (draw_stride != prev_stride) or (target < built):
        rebuild_buffers_to(segments, target, stride=draw_stride)
    elif target > built:
        append_segments_to_buffers(segments, built, target, stride=draw_stride)

    st.session_state.paths_scrub = target

    # 오프셋 + 전역 캡 + 외부치수(줄바꿈 보장)
    if apply_offsets:
        half_w = float(trim_dist) * 0.5
        compute_offsets_into_buffers(
            segments, target, half_w,
            include_travel_climb=bool(include_z_climb), climb_z_thresh=1e-9
        )
        st.session_state.paths_anim_buf["caps"] = {"x": [], "y": [], "z": []}
        add_global_endcaps_into_buffers(
            segments, target, half_width=half_w, samples=32, store_caps=bool(emphasize_caps)
        )
        bbox_r = _bbox_from_buffer(st.session_state.paths_anim_buf["off_r"])
        z_r = _last_z_from_buffer(st.session_state.paths_anim_buf["off_r"])
        dims_html = _fmt_dims_block_html("외부치수", bbox_r, z_r)
        dims_placeholder.markdown(dims_html, unsafe_allow_html=True)
    else:
        st.session_state.paths_anim_buf["off_l"] = {"x": [], "y": [], "z": []}
        st.session_state.paths_anim_buf["off_r"] = {"x": [], "y": [], "z": []}
        st.session_state.paths_anim_buf["caps"]  = {"x": [], "y": [], "z": []}
        emphasize_caps = False
        dims_placeholder.markdown("_Offsets OFF_")

# ---- 중앙: 탭 뷰어 ----
with center_col:
    tab_paths, tab_stl, tab_gcode = st.tabs(["Sliced Paths (3D)", "STL Preview", "G-code Viewer"])

    with tab_paths:
        if segments is not None and total_segments > 0:
            ensure_paths_fig(height=820)
            fig = st.session_state.paths_base_fig
            update_fig_with_buffers(
                fig,
                show_offsets=bool(st.session_state.get("apply_offsets_flag", False)),
                show_caps=bool(emphasize_caps)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            tm = st.session_state.paths_travel_mode
            if not e_on:
                travel_lbl = "Travel: solid (Insert E OFF)"
            else:
                travel_lbl = "Travel: dotted" if tm == "dotted" else ("Travel: hidden" if tm == "hidden" else "Travel: solid")
            st.caption(
                f"세그먼트 총 {total_segments:,} | 현재 {st.session_state.paths_scrub:,}"
                + (f" | Offsets: ON (W/2 = {float(trim_dist)*0.5:.2f} mm)" if st.session_state.get('apply_offsets_flag', False) else "")
                + (" | Caps 강조" if (st.session_state.get('apply_offsets_flag', False) and emphasize_caps) else "")
                + (f" | {travel_lbl}")
                + (f" | Viz stride: ×{st.session_state.paths_anim_buf.get('stride',1)}"
                   if st.session_state.paths_anim_buf.get('stride',1) > 1 else "")
            )
        else:
            st.info("슬라이싱을 실행하세요.")

    with tab_stl:
        if st.session_state.get("mesh") is not None:
            st.plotly_chart(
                plot_trimesh(st.session_state.mesh, height=820),
                use_container_width=True,
                key="stl_chart",
                config={"displayModeBar": False}
            )
        else:
            st.info("STL을 업로드하세요.")

    with tab_gcode:
        if st.session_state.get("gcode_text"):
            st.code(st.session_state.gcode_text, language="gcode")
        else:
            st.info("G-code를 생성하세요.")

# 키가 없거나 만료 시 안내
if not KEY_OK:
    st.warning("유효한 Access Key를 입력해야 프로그램이 작동합니다. (업로드/슬라이싱/G-code 버튼 비활성화)")
