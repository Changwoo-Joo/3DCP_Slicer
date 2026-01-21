# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import math
import json
import trimesh
import tempfile
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict, Any
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

    .block-container { padding-top: 2.0rem; }
    .stTabs { margin-top: 1.0rem !important; padding-top: 0.2rem !important; }
    .stTabs [data-baseweb="tab-list"] { margin-top: 0.6rem !important; }

    .right-panel {
      position: sticky;
      top: 2.0rem;
      max-height: calc(100vh - 2rem);
      overflow-y: auto;
      border-left: 1px solid #e6e6e6;
      padding-left: 12px;
      background: white;
    }

    .sidebar-title {
      margin: 0.25rem 0 0.6rem 0;
      font-size: 1.35rem;
      font-weight: 700;
      line-height: 1.2;
    }

    .dims-block {
      white-space: pre-line;
      line-height: 1.3;
      font-variant-numeric: tabular-nums;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
        color="#888888", opacity=0.6, flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.9, roughness=0.9, specular=0.1)
    )])
    fig.update_layout(
        scene=dict(aspectmode="data", camera=dict(projection=dict(type="orthographic"))),
        height=height, margin=dict(l=0, r=0, t=10, b=0)
    )
    return fig

# =========================
# G-code generator
# =========================
def generate_gcode(mesh, z_int=30.0, feed=2000, ref_pt_user=(0.0, 0.0),
                   e_on=False, start_e_on=False, start_e_val=0.1, e0_on=False,
                   trim_dist=30.0, min_spacing=5.0, auto_start=False, m30_on=False):
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
                prev_start_xy = start[:2]

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
    min_spacing=5.0,
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

        ref_pt_layer = prev_start_xy if (auto_start and prev_start_xy is not None) else np.array(ref_pt_user, dtype=float)

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

        first_poly_start = layer_polys[0][0]
        if prev_layer_last_end is not None:
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

# === items -> segments ===
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

# === (NEW) 현재 행 기준 레이어 전체 길이 계산 ===
def compute_layer_length_for_index(
    segments: List[Tuple[np.ndarray, np.ndarray, bool, bool]],
    upto_idx: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    upto_idx 번째 세그먼트까지 기준으로:
      - 그 구간에서 마지막 '압출 세그먼트'가 속한 Z 레이어를 찾고
      - 해당 Z 레이어 전체(모든 압출 세그먼트)의 XY 길이 합을 계산.
    반환: (레이어 Z값, 전체 길이[mm]) / 없으면 (None, None)
    """
    if not segments or upto_idx <= 0:
        return None, None

    N = len(segments)
    upto = min(max(int(upto_idx), 0), N)

    # 1) upto 구간 내 마지막 압출 세그먼트의 레이어 Z 찾기
    layer_z = None
    for i in range(upto):
        p1, p2, is_travel, is_extruding = segments[i]
        if is_extruding:
            layer_z = float((p1[2] + p2[2]) * 0.5)
    if layer_z is None:
        return None, None

    # 2) 해당 레이어 Z의 전체 XY 길이 합산
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

def update_fig_with_buffers(fig: go.Figure, show_offsets: bool, show_caps: bool):
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
    st.session_state.paths_scrub = 0
if "paths_travel_mode" not in st.session_state:
    st.session_state.paths_travel_mode = "solid"

if "ui_banner" not in st.session_state:
    st.session_state.ui_banner = None

# --- (NEW) External axis profile defaults ---
if "ext_profile_enable_a1" not in st.session_state:
    st.session_state.ext_profile_enable_a1 = False
if "ext_profile_enable_a2" not in st.session_state:
    st.session_state.ext_profile_enable_a2 = False
if "ext_profile_lead_start_mm" not in st.session_state:
    st.session_state.ext_profile_lead_start_mm = 100.0
if "ext_profile_lead_end_mm" not in st.session_state:
    st.session_state.ext_profile_lead_end_mm = 100.0
if "ext_profile_deadband_a1" not in st.session_state:
    st.session_state.ext_profile_deadband_a1 = 0.0
if "ext_profile_deadband_a2" not in st.session_state:
    st.session_state.ext_profile_deadband_a2 = 0.0
if "ext_profile_print_only" not in st.session_state:
    st.session_state.ext_profile_print_only = False

ensure_anim_buffers()

# =========================
# Access
# =========================
st.sidebar.header("Access")
ALLOWED_WITH_EXPIRY = {"robotics5107": None, "kaist_aramco3D": "2026-12-31", "kmou*": "2026-12-31", "DY25-01D4-E5F6-G7H8-I9J0-K1L2": "2030-12-30"}
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
# Parameters
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
min_spacing = st.sidebar.number_input("Minimum point spacing (mm)", 0.0, 1000.0, 5.0)
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
    # Z 아주 미세 확장 (절단면 인식)
    scale_matrix = np.eye(4)
    scale_matrix[2, 2] = 1.0000001
    mesh.apply_transform(scale_matrix)
    st.session_state.mesh = mesh
    st.session_state.base_name = Path(uploaded.name).stem or "output"

# =========================
# Slicing Actions
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
    segs = items_to_segments(items, e_on=e_on)
    max_seg = len(segs)
    st.session_state.paths_scrub = max_seg
    reset_anim_buffers()
    rebuild_buffers_to(segs, max_seg)
    st.session_state.ui_banner = "Slicing complete"

if KEY_OK and gen_clicked and st.session_state.mesh is not None:
    gcode_text = generate_gcode(
        st.session_state.mesh, z_int=z_int, feed=feed, ref_pt_user=(ref_x, ref_y),
        e_on=e_on, start_e_on=start_e_on, start_e_val=start_e_val, e0_on=e0_on,
        trim_dist=trim_dist, min_spacing=min_spacing, auto_start=auto_start, m30_on=m30_on
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
    s = f"{v:+.1f}"; sign = s[0]; intpart, dec = s[1:].split("."); intpart = intpart.zfill(4); return f"{sign}{intpart}.{dec}"
def _fmt_ang(v: float) -> str:
    s = f"{v:+.2f}"; sign = s[0]; intpart, dec = s[1:].split("."); intpart = intpart.zfill(3); return f"{sign}{intpart}.{dec}"

def _linmap(val: float, a0: float, a1: float, b0: float, b1: float) -> float:
    if abs(a1 - a0) < 1e-12:
        return float(b0)
    t = (val - a0) / (a1 - a0)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return float(b0 + t * (b1 - b0))

# ---- 교정된 기본 프리셋 ----
DEFAULT_PRESET = {
    "0": {
        # 0°: X -> A1, A4 / Y -> A2 / Z -> A3
        "X": {"in": [0.0, 0.0],     "A1_out": [0.0,   0.0], "A4_out": [0.0, 0.0]},
        "Y": {"in": [0.0, 0.0], "A2_out": [ 0.0,   0.0]},
        "Z": {"in": [0.0, 0.0],     "A3_out": [   0.0, 0.0]},
    },
    "90": {
        # +90°: X -> A1 / Y -> A2, A4 / Z -> A3
        "X": {"in": [0.0, 6500.0],  "A1_out": [ 4000.0, 0.0]},
        "Y": {"in": [0.0, 1000.0],  "A2_out": [ 500.0,   0.0], "A4_out": [0.0, 500.0]},
        "Z": {"in": [0.0, 3000.0],  "A3_out": [   0.0, 1000.0]},
    },
    "-90": {
        # −90°: X -> A1 / Y -> A2(0→500), A4 / Z -> A3
        "X": {"in": [0.0, 6500.0], "A1_out": [ 4000.0, 0.00]},
        "Y": {"in": [0.0, 1000.0], "A2_out": [ 500.0, 0.0], "A4_out": [0.0, 500.0]},
        "Z": {"in": [0.0, 3000.0], "A3_out": [   0.0, 1000.0]},
    },
}

def _deepcopy_preset(p: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(p))

# 세션에 프리셋 보관
if "mapping_preset" not in st.session_state:
    st.session_state.mapping_preset = _deepcopy_preset(DEFAULT_PRESET)

def map_axes_from_xyz_with_preset(x: float, y: float, z: float, rz: float, preset: Dict[str, Any]):
    """A4_out 이 어느 축(X/Y)에 있든 해당 축 입력구간으로 선형 매핑."""
    key = "0" if abs(rz - 0.0) < 1e-6 else ("90" if abs(rz - 90.0) < 1e-6 else ("-90" if abs(rz + 90.0) < 1e-6 else None))
    if key is None or key not in preset:
        return 0.0, 0.0, 0.0, 0.0
    P = preset[key]

    def gi(d: Dict[str, Any], path: List[Any], default: float) -> float:
        cur = d
        try:
            for k in path[:-1]:
                cur = cur[k]
            return float(cur[path[-1]])
        except Exception:
            return float(default)

    # 입력 구간
    x0 = gi(P, ["X", "in", 0], 0.0); x1 = gi(P, ["X", "in", 1], 1.0)
    y0 = gi(P, ["Y", "in", 0], 0.0); y1 = gi(P, ["Y", "in", 1], 1.0)
    z0 = gi(P, ["Z", "in", 0], 0.0); z1 = gi(P, ["Z", "in", 1], 1.0)

    # 출력 구간
    a1_0 = gi(P, ["X", "A1_out", 0], 0.0); a1_1 = gi(P, ["X", "A1_out", 1], 0.0)
    a2_0 = gi(P, ["Y", "A2_out", 0], 0.0); a2_1 = gi(P, ["Y", "A2_out", 1], 0.0)
    a3_0 = gi(P, ["Z", "A3_out", 0], 0.0); a3_1 = gi(P, ["Z", "A3_out", 1], 0.0)

    # A4는 X 또는 Y 중 "A4_out"이 존재하는 축을 자동 선택
    a4_axis = None
    if "A4_out" in P.get("Y", {}):
        a4_axis = ("Y", y, y0, y1, gi(P, ["Y", "A4_out", 0], 0.0), gi(P, ["Y", "A4_out", 1], 0.0))
    elif "A4_out" in P.get("X", {}):
        a4_axis = ("X", x, x0, x1, gi(P, ["X", "A4_out", 0], 0.0), gi(P, ["X", "A4_out", 1], 0.0))

    a1 = _linmap(x, x0, x1, a1_0, a1_1)
    a2 = _linmap(y, y0, y1, a2_0, a2_1)
    a3 = _linmap(z, z0, z1, a3_0, a3_1)
    if a4_axis is None:
        a4 = 0.0
    else:
        _, val, i0, i1, o0, o1 = a4_axis
        a4 = _linmap(val, i0, i1, o0, o1)
        # A4 절대 매핑 값도 범위 클램프 (안전)
        lo, hi = (o0, o1) if o0 <= o1 else (o1, o0)
        a4 = lo if a4 < lo else hi if a4 > hi else a4

    return a1, a2, a3, a4

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

# =========================
# (NEW) External axis helpers (A1/A2 lead/lag + deadband)
# =========================
def _apply_hysteresis_deadband(seq: List[float], db: float) -> List[float]:
    """Deadband + hysteresis. db<=0이면 그대로 반환."""
    if seq is None:
        return []
    if db is None or float(db) <= 0.0 or len(seq) == 0:
        return list(seq)
    db = float(db)
    out = []
    cmd = float(seq[0])
    for v in seq:
        v = float(v)
        d = v - cmd
        if abs(d) <= db:
            out.append(cmd)
        else:
            cmd = v - (db if d > 0 else -db)
            out.append(cmd)
    return out

def _timewarp_profile_by_distance(
    s: np.ndarray,
    v: np.ndarray,
    lead_start_mm: float,
    lead_end_mm: float
) -> np.ndarray:
    """
    거리축 s(0..L)에서 값 v(s)를
      - 초반 lead_start_mm 동안 v(0) hold
      - 말단 lead_end_mm 동안 v(L) hold
      - 중간은 s_eff로 time-warp 해서 v를 보존(형상 유지)
    """
    if s is None or v is None:
        return v
    if len(s) < 2:
        return v
    lead_start_mm = float(max(0.0, lead_start_mm))
    lead_end_mm = float(max(0.0, lead_end_mm))
    if (lead_start_mm + lead_end_mm) <= 1e-9:
        return v

    # s 단조 증가 보장(정지 점 포함 가능)
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    L = float(s[-1] - s[0])
    if L <= 1e-9:
        return v

    D1 = min(lead_start_mm, L)
    D2 = min(lead_end_mm, L)
    if D1 + D2 >= 0.95 * L:
        # 너무 크면 자동 축소
        scale = (0.95 * L) / max(1e-9, (D1 + D2))
        D1 *= scale
        D2 *= scale

    # s를 0 기준으로
    s0 = s[0]
    ss = s - s0
    # s_eff 계산
    denom = max(1e-9, (L - D1 - D2))
    stretch = L / denom
    s_eff = np.empty_like(ss)
    for i, si in enumerate(ss):
        if si <= D1:
            s_eff[i] = 0.0
        elif si >= (L - D2):
            s_eff[i] = L
        else:
            s_eff[i] = (si - D1) * stretch

    # np.interp는 x가 증가해야 함. ss는 증가(동일값 가능).
    # 동일값이 있을 때 대비해 "유니크 ss"로 압축
    ss_list = ss.tolist()
    v_list = v.tolist()
    ss_u = [ss_list[0]]
    v_u = [v_list[0]]
    for si, vi in zip(ss_list[1:], v_list[1:]):
        if abs(si - ss_u[-1]) < 1e-12:
            # 같은 s면 마지막 값으로 덮어쓰기
            v_u[-1] = vi
        else:
            ss_u.append(si)
            v_u.append(vi)
    ss_u = np.asarray(ss_u, dtype=float)
    v_u = np.asarray(v_u, dtype=float)

    # s_eff도 0..L 범위로 클램프
    s_eff = np.clip(s_eff, 0.0, float(ss_u[-1]))
    return np.interp(s_eff, ss_u, v_u)

def gcode_to_cone1500_module(
    gcode_text: str,
    rx: float,
    ry: float,
    rz: float,
    preset: Dict[str, Any],
    swap_a3_a4: bool = False,
    # --- (NEW) External axis profile options ---
    enable_a1_profile: bool = False,
    enable_a2_profile: bool = False,
    lead_start_mm: float = 100.0,
    lead_end_mm: float = 100.0,
    deadband_a1: float = 0.0,
    deadband_a2: float = 0.0,
    profile_print_only: bool = False
) -> str:
    """
    A4만 '비례 분해(증분 누적)'로 동작.
    추가: 출력 좌표 보정
      - Z' = Z - A3(절대값)
      - Rz=90:  Y' = Y - A4  (A2는 Y'로 산출)
      - Rz=0:   X' = X - A4  (A1은 X'로 산출)
      - Rz=-90: Y' = Y - (A4_max - A4)  (A2는 Y'로 산출; A4 범위 0~max 가정)

    (NEW) External axis load reduction:
      - A1/A2에 대해 "start hold / end hold(lead/lag)" 프로파일을 거리 기반 time-warp로 적용 가능
      - A1/A2에 대해 deadband(+hysteresis)로 미세 지그재그 억제 가능
    """
    key = "0" if abs(rz - 0.0) < 1e-6 else ("90" if abs(rz - 90.0) < 1e-6 else ("-90" if abs(rz + 90.0) < 1e-6 else None))
    P = preset.get(key, {}) if key is not None else {}

    def gi(d: Dict[str, Any], path: list, default: float) -> float:
        try:
            cur = d
            for k in path[:-1]:
                cur = cur[k]
            return float(cur[path[-1]])
        except Exception:
            return float(default)

    # 입력 스팬
    x0, x1 = gi(P, ["X","in",0], 0.0), gi(P, ["X","in",1], 1.0)
    y0, y1 = gi(P, ["Y","in",0], 0.0), gi(P, ["Y","in",1], 1.0)
    z0, z1 = gi(P, ["Z","in",0], 0.0), gi(P, ["Z","in",1], 1.0)

    # 출력 스팬
    a1_0, a1_1 = gi(P, ["X","A1_out",0], 0.0), gi(P, ["X","A1_out",1], 0.0)
    a2_0, a2_1 = gi(P, ["Y","A2_out",0], 0.0), gi(P, ["Y","A2_out",1], 0.0)
    a3_0, a3_1 = gi(P, ["Z","A3_out",0], 0.0), gi(P, ["Z","A3_out",1], 0.0)

    # A4 부착 위치와 출력 스팬
    a4_on_x = "A4_out" in P.get("X", {})
    a4_on_y = "A4_out" in P.get("Y", {})
    a4x_0, a4x_1 = (gi(P, ["X","A4_out",0], 0.0), gi(P, ["X","A4_out",1], 0.0)) if a4_on_x else (0.0, 0.0)
    a4y_0, a4y_1 = (gi(P, ["Y","A4_out",0], 0.0), gi(P, ["Y","A4_out",1], 0.0)) if a4_on_y else (0.0, 0.0)

    def _prop_split_local(delta: float, in0: float, in1: float, out0: float, out1: float) -> float:
        span_in = abs(float(in1) - float(in0))
        span_out = abs(float(out1) - float(out0))
        total = span_in + span_out
        if total <= 1e-12 or span_out <= 1e-12:
            return 0.0
        same_dir = ((in1 >= in0 and out1 >= out0) or (in1 <= in0 and out1 <= out0))
        ext_part = float(delta) * (span_out / total)
        if not same_dir:
            ext_part = -ext_part
        return ext_part

    frx, fry, frz = _fmt_ang(rx), _fmt_ang(ry), _fmt_ang(rz)

    # ---- 1) G-code 파싱 & 1차 계산(원래 로직 유지) ----
    have_prev = False
    prev_x = prev_y = prev_z = 0.0
    prev_e = None
    cur_a4 = 0.0  # 누적 A4

    # 저장 버퍼(후처리용)
    xs_out: List[float] = []
    ys_out: List[float] = []
    zs_out: List[float] = []
    a1_des: List[float] = []
    a2_des: List[float] = []
    a3_list: List[float] = []
    a4_list: List[float] = []
    is_extruding_list: List[bool] = []

    for raw in gcode_text.splitlines():
        t = raw.strip()
        if not t or not t.startswith(("G0","G00","G1","G01")):
            continue

        # 좌표 파싱(없으면 이전 유지)
        cx, cy, cz = prev_x, prev_y, prev_z
        ce = None
        has_any = False
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
        if not has_any:
            continue

        # E 기반 extruding 판정(있을 때만)
        is_extruding = False
        if ce is not None and prev_e is not None:
            if (ce - prev_e) > 1e-12:
                is_extruding = True
        # prev_e 업데이트는 모션라인 기준으로만
        if ce is not None:
            prev_e = ce

        # --- A3(절대) 먼저 산출 (원본 Z 기반) ---
        a3_abs = _linmap(cz, z0, z1, a3_0, a3_1)

        # --- A4: 첫 점은 절대 앵커, 이후 증분 누적 ---
        if not have_prev:
            if a4_on_x:
                cur_a4 = _linmap(cx, x0, x1, a4x_0, a4x_1)
            elif a4_on_y:
                cur_a4 = _linmap(cy, y0, y1, a4y_0, a4y_1)
            else:
                cur_a4 = 0.0
            have_prev = True
        else:
            dx, dy = cx - prev_x, cy - prev_y
            if a4_on_x and abs(dx) > 0:
                cur_a4 += _prop_split_local(dx, x0, x1, a4x_0, a4x_1)
            elif a4_on_y and abs(dy) > 0:
                cur_a4 += _prop_split_local(dy, y0, y1, a4y_0, a4y_1)

        # A4 범위 클램프
        if a4_on_x:
            lo, hi = (a4x_0, a4x_1) if a4x_0 <= a4x_1 else (a4x_1, a4x_0)
        elif a4_on_y:
            lo, hi = (a4y_0, a4y_1) if a4y_0 <= a4y_1 else (a4y_1, a4y_0)
        else:
            lo, hi = (0.0, 0.0)
        cur_a4 = lo if cur_a4 < lo else hi if cur_a4 > hi else cur_a4

        # --- 출력 좌표 보정 ---
        x_out, y_out, z_out = cx, cy, cz - a3_abs  # Z' = Z - A3

        if key == "90":
            # Y' = Y - A4
            y_out = cy - cur_a4
        elif key == "0":
            # X' = X - A4
            x_out = cx - cur_a4
        elif key == "-90":
            # Y' = Y - (A4_max - A4)
            a4_max = max(a4y_0, a4y_1) if a4_on_y else 0.0
            y_out = cy - (a4_max - cur_a4)

        # --- A1/A2는 보정된 축으로 계산, A3는 원본 Z 기반 값 유지 ---
        a1 = _linmap(x_out, x0, x1, a1_0, a1_1)
        a2 = _linmap(y_out, y0, y1, a2_0, a2_1)
        a3 = a3_abs
        a4 = cur_a4

        xs_out.append(float(x_out))
        ys_out.append(float(y_out))
        zs_out.append(float(z_out))
        a1_des.append(float(a1))
        a2_des.append(float(a2))
        a3_list.append(float(a3))
        a4_list.append(float(a4))
        is_extruding_list.append(bool(is_extruding))

        if len(xs_out) >= MAX_LINES:
            break

        prev_x, prev_y, prev_z = cx, cy, cz

    # 모션이 없으면 패딩만
    if len(xs_out) == 0:
        lines_out = [PAD_LINE] * MAX_LINES
        ts = datetime.now().strftime("%Y-%m-%d %p %I:%M:%S")
        header = ("MODULE Converted\n"
                  "!******************************************************************************************************************************\n"
                  "!*\n"
                  f"!*** Generated {ts} by Gcode→RAPID converter.\n"
                  "!\n"
                  "!*** data3dp: X(mm), Y(mm), Z(mm), Rx(deg), Ry(deg), Rz(deg), A1,A2,A3,A4\n"
                  "!\n"
                  "!******************************************************************************************************************************\n")
        cnt_str = str(len(lines_out))
        open_decl = f'VAR string sFileCount:="{cnt_str}";\nVAR string d3dpDynLoad{{{cnt_str}}}:=[\n'
        body = ""
        for i, ln in enumerate(lines_out):
            q = f'"{ln}"'
            body += (q + ",\n") if i < len(lines_out) - 1 else (q + "\n")
        close_decl = "];\nENDMODULE\n"
        return header + open_decl + body + close_decl

    # ---- 2) (NEW) A1/A2 리드/래그 + "홀드 포인트(추가 라인)" 삽입 ----
    #  - 축이 실제로 이동해야 하는 구간(해당 축 값이 유의미하게 단조 증가/감소)만 블록으로 잡고,
    #    블록 시작/끝에서 lead_start_mm / lead_end_mm 만큼은 축을 고정(hold)하도록 축 값을 거리기반으로 재분배한다.
    #  - 그리고 그 hold 경계가 실제 데이터 라인으로도 존재하도록, 폴리라인 중간에 보간 포인트를 삽입한다.
    #    (즉, 로봇은 끝점까지 가지만 축은 '조금 일찍' 도착/출발하도록 시간차를 만든다.)
    #  - print_only=True 이고 E 정보가 전혀 없는 파일은(모두 False) 전체를 printing으로 간주하여 적용한다.

    # 2-0) 노드 리스트 구성
    nodes = []
    n0 = len(xs_out)
    for i in range(n0):
        nodes.append({
            "x": float(xs_out[i]),
            "y": float(ys_out[i]),
            "z": float(zs_out[i]),
            "a1": float(a1_des[i]),
            "a2": float(a2_des[i]),
            "a3": float(a3_list[i]),
            "a4": float(a4_list[i]),
            "extr": bool(is_extruding_list[i]) if i < len(is_extruding_list) else False,
        })

    def _hysteresis_deadband(vals, band):
        """이전 값 기준으로 band 이내 변화는 무시(히스테리시스)."""
        if not vals:
            return []
        if band <= 0:
            return list(vals)
        out = [float(vals[0])]
        last = out[0]
        b = float(band)
        for v in vals[1:]:
            fv = float(v)
            if abs(fv - last) >= b:
                last = fv
            out.append(last)
        return out

    def _xy_dist(nA, nB):
        dx = float(nB["x"]) - float(nA["x"])
        dy = float(nB["y"]) - float(nA["y"])
        return math.hypot(dx, dy)

    def _interp_node(nA, nB, t):
        """nA->nB 선형 보간 (모든 필드)."""
        t = float(t)
        u = 1.0 - t
        return {
            "x": u * float(nA["x"]) + t * float(nB["x"]),
            "y": u * float(nA["y"]) + t * float(nB["y"]),
            "z": u * float(nA["z"]) + t * float(nB["z"]),
            "a1": u * float(nA["a1"]) + t * float(nB["a1"]),
            "a2": u * float(nA["a2"]) + t * float(nB["a2"]),
            "a3": u * float(nA["a3"]) + t * float(nB["a3"]),
            "a4": u * float(nA["a4"]) + t * float(nB["a4"]),
            "extr": bool(nA.get("extr", False)) or bool(nB.get("extr", False)),
        }

    def _split_at_distances(sub_nodes, s_targets, tol=1e-6):
        """
        sub_nodes(길이>=2) 폴리라인에서, 시작점으로부터 거리 s_targets(mm) 지점에 노드를 삽입하여 반환.
        s_targets 는 오름차순 가정.
        """
        if len(sub_nodes) < 2:
            return list(sub_nodes)

        # 누적거리
        s = [0.0]
        for i in range(1, len(sub_nodes)):
            s.append(s[-1] + _xy_dist(sub_nodes[i - 1], sub_nodes[i]))
        L = s[-1]
        if L <= 1e-9:
            return list(sub_nodes)

        # 기존 점과 너무 가까우면 삽입 안 함
        targets = []
        for st in s_targets:
            st = float(st)
            if st <= tol or st >= L - tol:
                continue
            # 기존 s와의 최소거리 체크
            if min(abs(st - si) for si in s) <= tol:
                continue
            targets.append(st)
        if not targets:
            return list(sub_nodes)

        out = []
        ti = 0
        for i in range(len(sub_nodes) - 1):
            out.append(sub_nodes[i])
            s0, s1 = s[i], s[i + 1]
            segL = s1 - s0
            if segL <= 1e-12:
                continue
            while ti < len(targets) and s0 < targets[ti] < s1:
                t = (targets[ti] - s0) / segL
                out.append(_interp_node(sub_nodes[i], sub_nodes[i + 1], t))
                ti += 1
        out.append(sub_nodes[-1])
        return out

    def _find_axis_blocks(nodes_local, axis_key, band, print_only):
        """
        axis_key 축 값이 '유의미하게' 단조 증가/감소하는 구간들을 (start_idx,end_idx) 로 반환.
        """
        n = len(nodes_local)
        if n < 2:
            return []

        mask = [True] * n
        if print_only:
            mask = [bool(nd.get("extr", False)) for nd in nodes_local]
            if not any(mask):
                # E 정보가 없으면 전체를 printing으로 간주
                mask = [True] * n

        axis_vals = [float(nd[axis_key]) for nd in nodes_local]
        base = _hysteresis_deadband(axis_vals, band if band > 0 else 0.0)

        eps_move = max(0.01, float(band) * 0.5) if band > 0 else 0.01

        blocks = []
        i = 1
        while i < n:
            if not mask[i]:
                i += 1
                continue
            da = base[i] - base[i - 1]
            if abs(da) <= eps_move:
                i += 1
                continue

            sign = 1 if da > 0 else -1
            start = i - 1
            j = i
            while j < n:
                if not mask[j]:
                    break
                daj = base[j] - base[j - 1]
                if abs(daj) <= eps_move:
                    break
                if (daj > 0) != (sign > 0):
                    # 방향 반전(지그재그) → 블록 종료
                    break
                j += 1

            end = j - 1
            if end > start:
                blocks.append((start, end))
            i = max(j, end + 1)

        return blocks

    def _apply_axis_profile_with_inserts(nodes_local, axis_key, enable, lead_start, lead_end, band, print_only):
        """
        축별 프로파일을 적용하고, lead_start/lead_end 경계에 포인트를 삽입하여 라인으로도 나타나게 함.
        """
        if not enable:
            return nodes_local
        if len(nodes_local) < 2:
            return nodes_local

        lead_start = max(0.0, float(lead_start))
        lead_end = max(0.0, float(lead_end))

        blocks = _find_axis_blocks(nodes_local, axis_key, band, print_only)
        if not blocks:
            return nodes_local

        offset = 0
        for (s_idx, e_idx) in blocks:
            s2 = s_idx + offset
            e2 = e_idx + offset
            if e2 <= s2:
                continue
            sub = nodes_local[s2:e2 + 1]
            if len(sub) < 2:
                continue

            # 블록 길이
            s_cum = [0.0]
            for k in range(1, len(sub)):
                s_cum.append(s_cum[-1] + _xy_dist(sub[k - 1], sub[k]))
            L = s_cum[-1]
            if L <= 1e-9:
                continue

            # 너무 짧으면(lead_start+lead_end >= L) 프로파일/삽입 생략
            core = L - lead_start - lead_end
            if core <= 1e-6:
                continue

            # 삽입 대상 거리
            targets = []
            if lead_start > 0.0:
                targets.append(lead_start)
            if lead_end > 0.0:
                targets.append(L - lead_end)
            targets = sorted(set(targets))

            v0 = float(sub[0][axis_key])
            v1 = float(sub[-1][axis_key])

            sub2 = _split_at_distances(sub, targets, tol=1e-4)

            # 새 누적거리
            s2_cum = [0.0]
            for k in range(1, len(sub2)):
                s2_cum.append(s2_cum[-1] + _xy_dist(sub2[k - 1], sub2[k]))
            L2 = s2_cum[-1]
            if L2 <= 1e-9:
                continue

            # (보정) 삽입 후에도 코어 길이는 동일 개념으로 사용
            core2 = L2 - lead_start - lead_end
            if core2 <= 1e-6:
                core2 = 1e-6

            # 축 값 할당(hold + linear)
            for k, nd in enumerate(sub2):
                sk = s2_cum[k]
                if sk <= lead_start + 1e-9:
                    nd[axis_key] = v0
                elif sk >= (L2 - lead_end) - 1e-9:
                    nd[axis_key] = v1
                else:
                    t = (sk - lead_start) / core2
                    nd[axis_key] = v0 + t * (v1 - v0)

            # 반영
            nodes_local = nodes_local[:s2] + sub2 + nodes_local[e2 + 1:]
            offset += (len(sub2) - len(sub))

        return nodes_local

    # A1 / A2 프로파일(+홀드포인트) 적용
    nodes = _apply_axis_profile_with_inserts(
        nodes, "a1", enable_a1_profile, lead_start_mm, lead_end_mm, deadband_a1, profile_print_only
    )
    nodes = _apply_axis_profile_with_inserts(
        nodes, "a2", enable_a2_profile, lead_start_mm, lead_end_mm, deadband_a2, profile_print_only
    )

    # ---- 3) A1/A2 deadband (최종 히스테리시스) ----
    if deadband_a1 > 0:
        a1_vals = [nd["a1"] for nd in nodes]
        a1_db = _hysteresis_deadband(a1_vals, deadband_a1)
        for nd, vv in zip(nodes, a1_db):
            nd["a1"] = vv

    if deadband_a2 > 0:
        a2_vals = [nd["a2"] for nd in nodes]
        a2_db = _hysteresis_deadband(a2_vals, deadband_a2)
        for nd, vv in zip(nodes, a2_db):
            nd["a2"] = vv

    # ---- 4) 출력 문자열 생성 ----
    lines_out = []
    if not nodes:
        lines_out = [PAD_LINE] * MAX_LINES
    else:
        for nd in nodes:
            if len(lines_out) >= MAX_LINES:
                break
            # NOTE: frx/fry/frz 변수는 위에서 각도 문자열(Rx/Ry/Rz)로 이미 사용 중이므로
            #       여기서는 좌표/외부축 값을 _fmt_pos()로 포맷한다.
            x = _fmt_pos(float(nd["x"]))
            y = _fmt_pos(float(nd["y"]))
            z = _fmt_pos(float(nd["z"]))

            if swap_a3_a4:
                a3_v = nd["a4"]
                a4_v = nd["a3"]
            else:
                a3_v = nd["a3"]
                a4_v = nd["a4"]

            a1 = _fmt_pos(float(nd["a1"]))
            a2 = _fmt_pos(float(nd["a2"]))
            a3 = _fmt_pos(float(a3_v))
            a4 = _fmt_pos(float(a4_v))

            lines_out.append(f"{x},{y},{z},{frx},{fry},{frz},{a1},{a2},{a3},{a4}")

        # 라인 수 부족 시 마지막 라인으로 패딩(로봇 재생/슬라이더용)
        # NOTE: do NOT pad to MAX_LINES; export only actual converted points (<= MAX_LINES).
        if len(lines_out) == 0:
            lines_out.append(PAD_LINE)
    ts = datetime.now().strftime("%Y-%m-%d %p %I:%M:%S")
    header = ("MODULE Converted\n"
              "!******************************************************************************************************************************\n"
              "!*\n"
              f"!*** Generated {ts} by Gcode→RAPID converter.\n"
              "!\n"
              "!*** data3dp: X(mm), Y(mm), Z(mm), Rx(deg), Ry(deg), Rz(deg), A1,A2,A3,A4\n"
              "!*** A1/A2 optional distance-profile(lead/lag) + deadband; A3 from original Z; Z' = Z-A3; A4 = proportional-split & clamped.\n"
              "!\n"
              "!******************************************************************************************************************************\n")
    cnt_str = str(len(lines_out))
    open_decl = f'VAR string sFileCount:="{cnt_str}";\nVAR string d3dpDynLoad{{{cnt_str}}}:=[\n'
    body = ""
    for i, ln in enumerate(lines_out):
        q = f'"{ln}"'
        body += (q + ",\n") if i < len(lines_out) - 1 else (q + "\n")
    close_decl = "];\nENDMODULE\n"
    return header + open_decl + body + close_decl

# ---- 사이드바: Rapid UI ----
st.sidebar.markdown("---")
if KEY_OK:
    if st.sidebar.button("Generate Rapid", use_container_width=True):
        st.session_state.show_rapid_panel = True

    if st.session_state.show_rapid_panel:
        with st.sidebar.expander("Rapid Settings", expanded=True):
            st.session_state.rapid_rx = st.number_input("Rx (deg)", value=float(st.session_state.rapid_rx), step=0.1, format="%.2f")
            st.session_state.rapid_ry = st.number_input("Ry (deg)", value=float(st.session_state.rapid_ry), step=0.1, format="%.2f")

            rz_preset = st.selectbox("Rz (deg) preset", options=[0.00, 90.0, -90.0],
                                     index={0.00:0, 90.0:1, -90.0:2}.get(float(st.session_state.get("rapid_rz", 0.0)), 0))
            st.session_state.rapid_rz = float(rz_preset)

        # ---- (NEW) External Axis Profile UI ----
        with st.sidebar.expander("External Axis Profile (A1/A2 부하완화)", expanded=False):
            st.caption("A1/A2의 시작/정지 타이밍을 로봇과 분리(lead/lag)하고, 미세 지그재그를 deadband로 억제합니다.")
            st.session_state.ext_profile_enable_a1 = st.checkbox(
                "Enable A1 profile (lead/lag + deadband)",
                value=bool(st.session_state.ext_profile_enable_a1),
            )
            st.session_state.ext_profile_enable_a2 = st.checkbox(
                "Enable A2 profile (lead/lag + deadband)",
                value=bool(st.session_state.ext_profile_enable_a2),
            )
            st.session_state.ext_profile_lead_start_mm = st.number_input(
                "Start hold (mm)  — 로봇이 먼저 움직이는 거리",
                min_value=0.0, max_value=100000.0, value=float(st.session_state.ext_profile_lead_start_mm), step=10.0, format="%.1f"
            )
            st.session_state.ext_profile_lead_end_mm = st.number_input(
                "End hold (mm)  — 로봇이 멈추기 전에 외부축이 먼저 멈추는 거리",
                min_value=0.0, max_value=100000.0, value=float(st.session_state.ext_profile_lead_end_mm), step=10.0, format="%.1f"
            )
            st.session_state.ext_profile_deadband_a1 = st.number_input(
                "A1 deadband (mm) — 이 범위 지그재그는 A1 고정",
                min_value=0.0, max_value=100000.0, value=float(st.session_state.ext_profile_deadband_a1), step=10.0, format="%.1f"
            )
            st.session_state.ext_profile_deadband_a2 = st.number_input(
                "A2 deadband (mm) — 이 범위 지그재그는 A2 고정",
                min_value=0.0, max_value=100000.0, value=float(st.session_state.ext_profile_deadband_a2), step=10.0, format="%.1f"
            )
            st.session_state.ext_profile_print_only = st.checkbox(
                "Apply profile only when E increases (printing only)",
                value=bool(st.session_state.ext_profile_print_only),
                help="G-code에 E가 있고, E가 증가하는 구간만 프로파일을 적용합니다. E가 없으면 적용 구간이 없을 수 있습니다."
            )

        # ---- Mapping Presets UI ----
        with st.sidebar.expander("Mapping Presets (편집/저장/불러오기)", expanded=False):
            st.caption("각 Rz 프리셋(0, +90, -90)에 대해 X/Y/Z 입력 구간과 A1/A2/A3/A4 출력 구간을 편집하세요.")

            # 불러오기
            up_json = st.file_uploader("Load preset JSON", type=["json"], key="mapping_preset_loader")
            if up_json is not None:
                try:
                    loaded = json.loads(up_json.read().decode("utf-8"))
                    if isinstance(loaded, dict) and all(k in loaded for k in ["0","90","-90"]):
                        st.session_state.mapping_preset = loaded
                        st.success("프리셋을 불러왔습니다.")
                    else:
                        st.error("프리셋 형식이 올바르지 않습니다. (keys: '0','90','-90')")
                except Exception as e:
                    st.error(f"프리셋 로드 실패: {e}")

            # 편집 폼
            def edit_axis(title_key: str, axis_key: str):
                st.write(f"**Rz = {title_key}° — {axis_key}**")
                cols = st.columns(4)
                P = st.session_state.mapping_preset[title_key][axis_key]

                # 입력 구간
                in0 = cols[0].number_input(f"{axis_key}.in[0]", value=float(P["in"][0]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_in0")
                in1 = cols[1].number_input(f"{axis_key}.in[1]", value=float(P["in"][1]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_in1")
                P["in"] = [float(in0), float(in1)]

                # 출력 (축별로 다름)
                if axis_key == "X":
                    # A1
                    a1_0 = cols[2].number_input("A1_out[0]", value=float(P.get("A1_out", [0.0,0.0])[0]), step=50.0, format="%.1f", key=f"{title_key}_X_a10")
                    a1_1 = cols[3].number_input("A1_out[1]", value=float(P.get("A1_out", [0.0,0.0])[1]), step=50.0, format="%.1f", key=f"{title_key}_X_a11")
                    P["A1_out"] = [float(a1_0), float(a1_1)]

                    cols2 = st.columns(2)
                    a4_0 = cols2[0].number_input("A4_out[0] (X)", value=float(P.get("A4_out", [0.0,0.0])[0] if "A4_out" in P else 0.0), step=50.0, format="%.1f", key=f"{title_key}_X_a40")
                    a4_1 = cols2[1].number_input("A4_out[1] (X)", value=float(P.get("A4_out", [0.0,0.0])[1] if "A4_out" in P else 0.0), step=50.0, format="%.1f", key=f"{title_key}_X_a41")
                    if "A4_out" in P:
                        P["A4_out"] = [float(a4_0), float(a4_1)]

                elif axis_key == "Y":
                    a2_0 = cols[2].number_input("A2_out[0]", value=float(P.get("A2_out", [0.0,0.0])[0]), step=50.0, format="%.1f", key=f"{title_key}_Y_a20")
                    a2_1 = cols[3].number_input("A2_out[1]", value=float(P.get("A2_out", [0.0,0.0])[1]), step=50.0, format="%.1f", key=f"{title_key}_Y_a21")
                    P["A2_out"] = [float(a2_0), float(a2_1)]

                    cols2 = st.columns(2)
                    a4_0 = cols2[0].number_input("A4_out[0] (Y)", value=float(P.get("A4_out", [0.0,0.0])[0] if "A4_out" in P else 0.0), step=50.0, format="%.1f", key=f"{title_key}_Y_a40")
                    a4_1 = cols2[1].number_input("A4_out[1] (Y)", value=float(P.get("A4_out", [0.0,0.0])[1] if "A4_out" in P else 0.0), step=50.0, format="%.1f", key=f"{title_key}_Y_a41")
                    if "A4_out" in P:
                        P["A4_out"] = [float(a4_0), float(a4_1)]

                else:  # Z
                    a3_0 = cols[2].number_input("A3_out[0]", value=float(P.get("A3_out", [0.0,0.0])[0]), step=50.0, format="%.1f", key=f"{title_key}_Z_a30")
                    a3_1 = cols[3].number_input("A3_out[1]", value=float(P.get("A3_out", [0.0,0.0])[1]), step=50.0, format="%.1f", key=f"{title_key}_Z_a31")
                    P["A3_out"] = [float(a3_0), float(a3_1)]

            for key_title in ["0", "90", "-90"]:
                st.markdown(f"---\n**Rz = {key_title}°**")
                edit_axis(key_title, "X")
                edit_axis(key_title, "Y")
                edit_axis(key_title, "Z")

            # 저장(다운로드)
            preset_json = json.dumps(st.session_state.mapping_preset, ensure_ascii=False, indent=2)
            st.download_button("Save preset JSON", preset_json, file_name="mapping_preset.json", mime="application/json", use_container_width=True)

        # ---- 저장 버튼 ----
        gtxt = st.session_state.get("gcode_text")
        over = None
        if gtxt is not None:
            xyz_count = _extract_xyz_lines_count(gtxt)
            over = (xyz_count > MAX_LINES)

        save_rapid_clicked = st.sidebar.button("Save Rapid (.modx)", use_container_width=True, disabled=(gtxt is None))
        if gtxt is None:
            st.sidebar.info("먼저 Generate G-Code로 G-code를 생성하세요.")
        elif over:
            st.sidebar.error("G-code가 64,000줄을 초과하여 Rapid 파일 변환할 수 없습니다.")
        elif save_rapid_clicked:
            st.session_state.rapid_text = gcode_to_cone1500_module(
                gtxt,
                rx=st.session_state.rapid_rx,
                ry=st.session_state.rapid_ry,
                rz=st.session_state.rapid_rz,
                preset=st.session_state.mapping_preset,
                swap_a3_a4=True,
                enable_a1_profile=bool(st.session_state.ext_profile_enable_a1),
                enable_a2_profile=bool(st.session_state.ext_profile_enable_a2),
                lead_start_mm=float(st.session_state.ext_profile_lead_start_mm),
                lead_end_mm=float(st.session_state.ext_profile_lead_end_mm),
                deadband_a1=float(st.session_state.ext_profile_deadband_a1),
                deadband_a2=float(st.session_state.ext_profile_deadband_a2),
                profile_print_only=bool(st.session_state.ext_profile_print_only),
            )
            st.sidebar.success(
                f"Rapid(*.MODX) 변환 완료 (Rz={st.session_state.rapid_rz:.2f}°)"
            )

        if st.session_state.get("rapid_text"):
            base = st.session_state.get("base_name", "output")
            st.sidebar.download_button(
                "Rapid 저장 (.modx)",
                st.session_state.rapid_text,
                file_name=f"{base}.modx",
                mime="text/plain",
                use_container_width=True
            )

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

    # 우측 슬라이더 / 행번호 + 레이어 길이 표시
    if segments is None or total_segments == 0:
        st.info("슬라이싱 후 진행 슬라이더가 나타납니다.")
        scrub = None
        scrub_num = None
    else:
        # 현재 저장된 값 기준 default 지정
        default_val = int(clamp(st.session_state.paths_scrub, 0, total_segments))

        scrub = st.slider("진행(segments)", 0, int(total_segments), int(default_val), 1,
                          help="해당 세그먼트까지 누적 표시")
        scrub_num = st.number_input("행 번호", 0, int(total_segments),
                                    int(default_val), 1,
                                    help="표시할 최종 세그먼트(행) 번호")

        # 슬라이더/행번호 입력을 통합해서 target 결정
        target = default_val
        if scrub != default_val:
            target = int(scrub)
        if scrub_num != default_val:
            target = int(scrub_num)

        target = int(clamp(target, 0, total_segments))
        st.session_state.paths_scrub = target

        # ---- (NEW) 현재 행 기준 레이어 전체 길이 표시 ----
        layer_z, layer_len = compute_layer_length_for_index(segments, target)
        if layer_z is not None and layer_len is not None:
            st.caption(
                f"현재 레이어 전체 길이: Z = {layer_z:.2f} mm · "
                f"{layer_len:.1f} mm (≈ {layer_len/1000:.3f} m)"
            )
        else:
            st.caption("현재 레이어 길이: -")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- 계산/버퍼 구성 ----
if segments is not None and total_segments > 0:
    # 위에서 결정된 값을 그대로 사용
    target = int(clamp(st.session_state.paths_scrub, 0, total_segments))

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

    if bool(st.session_state.get("apply_offsets_flag", False)):
        half_w = float(trim_dist) * 0.5
        compute_offsets_into_buffers(segments, target, half_w, include_travel_climb=bool(include_z_climb), climb_z_thresh=1e-9)
        st.session_state.paths_anim_buf["caps"] = {"x": [], "y": [], "z": []}
        add_global_endcaps_into_buffers(segments, target, half_width=half_w, samples=32, store_caps=bool(emphasize_caps))
        bbox_r = _bbox_from_buffer(st.session_state.paths_anim_buf["off_r"])
        z_r = _last_z_from_buffer(st.session_state.paths_anim_buf["off_r"])
        dims_html = _fmt_dims_block_html("외부치수", bbox_r, z_r)
        dims_placeholder.markdown(dims_html, unsafe_allow_html=True)

    if segments is not None and target > 0:
        total_len = 0.0
        for i in range(target):
            p1, p2, is_travel, is_extruding = segments[i]
            if is_extruding:  # E>0 구간만 길이로 인정
                total_len += float(np.linalg.norm(p2[:2] - p1[:2]))

        st.markdown(f"**누적 레이어 총 길이:** {total_len/1000:.3f} m")

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
            if "paths_base_fig" not in st.session_state:
                st.session_state.paths_base_fig = make_base_fig(height=820)
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
