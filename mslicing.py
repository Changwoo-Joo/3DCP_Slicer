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

# ── 전역 CSS ──
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
    if not segments or upto_idx <= 0:
        return None, None

    N = len(segments)
    upto = min(max(int(upto_idx), 0), N)

    layer_z = None
    for i in range(upto):
        p1, p2, is_travel, is_extruding = segments[i]
        if is_extruding:
            layer_z = float((p1[2] + p2[2]) * 0.5)
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

# --- A1/A2 Constant-Speed defaults ---
if "ext_const_enable_a1" not in st.session_state:
    st.session_state.ext_const_enable_a1 = True
if "ext_const_enable_a2" not in st.session_state:
    st.session_state.ext_const_enable_a2 = True

# 기본값(네 범위 기준)
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
    st.session_state.ext_const_a2_at_ymin = 0.0
if "ext_const_a2_at_ymax" not in st.session_state:
    st.session_state.ext_const_a2_at_ymax = 4000.0

if "ext_const_speed_mm_s" not in st.session_state:
    st.session_state.ext_const_speed_mm_s = 200.0
if "ext_const_eps_mm" not in st.session_state:
    st.session_state.ext_const_eps_mm = 0.5
if "ext_const_apply_print_only" not in st.session_state:
    st.session_state.ext_const_apply_print_only = False
if "ext_const_travel_interp" not in st.session_state:
    st.session_state.ext_const_travel_interp = True

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
    if abs(v) < 5e-5:
        v = 0.0
    s = f"{v:+.1f}"
    sign = s[0]
    intpart, dec = s[1:].split(".")
    intpart = intpart.zfill(4)
    return f"{sign}{intpart}.{dec}"

def _fmt_ang(v: float) -> str:
    if abs(v) < 5e-5:
        v = 0.0
    s = f"{v:+.2f}"
    sign = s[0]
    intpart, dec = s[1:].split(".")
    intpart = intpart.zfill(3)
    return f"{sign}{intpart}.{dec}"

def _linmap(val: float, a0: float, a1: float, b0: float, b1: float) -> float:
    if abs(a1 - a0) < 1e-12:
        return float(b0)
    t = (val - a0) / (a1 - a0)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return float(b0 + t * (b1 - b0))

DEFAULT_PRESET = {
    "0": {
        "X": {"in": [0.0, 6500.0], "A4_out": [0.0, 500.0]},
        "Y": {"in": [0.0, 1000.0]},
        "Z": {"in": [0.0, 3000.0], "A3_out": [0.0, 1000.0]},
    },
    "90": {
        "X": {"in": [0.0, 6500.0]},
        "Y": {"in": [0.0, 1000.0], "A4_out": [0.0, 500.0]},
        "Z": {"in": [0.0, 3000.0], "A3_out": [0.0, 1000.0]},
    },
    "-90": {
        "X": {"in": [0.0, 6500.0]},
        "Y": {"in": [0.0, 1000.0], "A4_out": [0.0, 500.0]},
        "Z": {"in": [0.0, 3000.0], "A3_out": [0.0, 1000.0]},
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

# =========================
# ✅ (FIX) A1/A2 Constant-Speed profile (robust)
#   - 경계(xmin/xmax, ymin/ymax) 없어도 고정되지 않음
#   - A1: raw_x의 |ΔX|만으로 진행 (Y로는 절대 진행 X)
#   - A2: raw_y의 |ΔY|만으로 진행
#   - 경계에 "있을 때"는 정확히 스냅/홀드, 경계에서 "벗어나는 순간"은 바로 진행
# =========================
def _apply_const_speed_profile_on_nodes(
    nodes: List[Dict[str, Any]],
    axis_key: str,
    coord_key: str,
    coord_min: float,
    coord_max: float,
    axis_at_min: float,
    axis_at_max: float,
    speed_mm_s: float = 200.0,
    eps_mm: float = 0.5,
    apply_print_only: bool = False,
    travel_interp: bool = True
) -> None:
    if not nodes or axis_key not in ("a1", "a2"):
        return
    n = len(nodes)
    if n == 0:
        return

    coord_min = float(coord_min)
    coord_max = float(coord_max)
    axis_at_min = float(axis_at_min)
    axis_at_max = float(axis_at_max)
    eps = float(max(0.0, eps_mm))

    span = float(coord_max - coord_min)
    span_abs = abs(span)
    if span_abs <= 1e-9:
        for nd in nodes:
            nd[axis_key] = float(axis_at_min)
        return

    # axis per mm (속도 파라미터는 출력 포지션에선 상쇄되지만, 인터페이스 호환 위해 유지)
    axis_per_mm = (axis_at_max - axis_at_min) / float(span_abs)

    def _coord(i: int) -> float:
        return float(nodes[i][coord_key])

    # 경계 판정: <= / >= 로 잡아서 오차에 강하게
    def _at_min(c: float) -> bool:
        return c <= coord_min + eps

    def _at_max(c: float) -> bool:
        return c >= coord_max - eps

    def _snap_axis_for_coord(c: float) -> float:
        if _at_min(c):
            return float(axis_at_min)
        if _at_max(c):
            return float(axis_at_max)
        # 내부면 선형 맵(시작점이 내부에서 시작할 때 점프 방지)
        t = (c - coord_min) / span_abs
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return float(axis_at_min + t * (axis_at_max - axis_at_min))

    # apply_print_only이면 "다음 노드가 extruding(True)일 때"만 진행(세그먼트 기준)
    extr_node = [bool(nd.get("extr", False)) for nd in nodes]

    # 초기값/방향 설정
    c0 = _coord(0)
    nodes[0][axis_key] = _snap_axis_for_coord(c0)

    # dir_mode: "fwd"(coord_min→coord_max로 가는 동안 axis_at_min→axis_at_max), "bwd"(반대)
    if _at_min(c0):
        dir_mode = "fwd"
    elif _at_max(c0):
        dir_mode = "bwd"
    else:
        # 첫 유효 Δcoord로 방향 추정
        dir_mode = "fwd"
        for j in range(1, n):
            dc = _coord(j) - c0
            if abs(dc) > 1e-9:
                dir_mode = "fwd" if dc > 0 else "bwd"
                break

    # 진행
    for i in range(n - 1):
        ci = _coord(i)
        cj = _coord(i + 1)

        # 현재가 경계면 방향 강제
        if _at_min(ci):
            nodes[i][axis_key] = float(axis_at_min)
            dir_mode = "fwd"
        elif _at_max(ci):
            nodes[i][axis_key] = float(axis_at_max)
            dir_mode = "bwd"

        ai = float(nodes[i][axis_key])

        # 이 스텝( i -> i+1 )에서 진행할지 여부
        active = True
        if apply_print_only:
            active = bool(extr_node[i + 1])

        dcoord = float(cj - ci)
        if (not active) or abs(dcoord) <= 1e-12:
            aj = ai
        else:
            # ✅ 핵심: 방향은 유지, 크기는 |Δcoord|만 반영
            step = axis_per_mm * abs(dcoord)
            if dir_mode == "fwd":
                aj = ai + step
            else:
                aj = ai - step

        # 다음 노드 경계 스냅 + 방향 갱신
        if _at_min(cj):
            aj = float(axis_at_min)
            dir_mode = "fwd"
        elif _at_max(cj):
            aj = float(axis_at_max)
            dir_mode = "bwd"

        # clamp
        lo = min(axis_at_min, axis_at_max)
        hi = max(axis_at_min, axis_at_max)
        if aj < lo: aj = lo
        if aj > hi: aj = hi

        nodes[i + 1][axis_key] = float(aj)

    # travel interpolation (옵션)
    if travel_interp and apply_print_only:
        # travel 구간(비활성)들을 앞/뒤 active 사이로 선형 보간
        active_node = [bool(extr_node[i]) for i in range(n)]
        if any(active_node):
            i = 0
            while i < n:
                if active_node[i]:
                    i += 1
                    continue
                t0 = i
                while i < n and (not active_node[i]):
                    i += 1
                t1 = i - 1
                prev_idx = t0 - 1 if t0 - 1 >= 0 else None
                next_idx = i if i < n else None
                if prev_idx is None or next_idx is None:
                    base = float(nodes[prev_idx][axis_key]) if prev_idx is not None else float(nodes[next_idx][axis_key]) if next_idx is not None else float(axis_at_min)
                    for k in range(t0, t1 + 1):
                        nodes[k][axis_key] = base
                    continue
                a0 = float(nodes[prev_idx][axis_key])
                a1 = float(nodes[next_idx][axis_key])
                total = max(1, (t1 - t0 + 1))
                for kk, k in enumerate(range(t0, t1 + 1)):
                    u = (kk + 1) / float(total + 1)
                    nodes[k][axis_key] = a0 + (a1 - a0) * float(u)

    # 마지막으로 경계 강제(보간이 경계를 덮어쓰지 않도록)
    for i in range(n):
        c = _coord(i)
        if _at_min(c):
            nodes[i][axis_key] = float(axis_at_min)
        elif _at_max(c):
            nodes[i][axis_key] = float(axis_at_max)

# =========================
# Rapid Converter (UPDATED)
# =========================
# =========================
# Rapid Converter (UPDATED)
# =========================
def _fmt_pos(v: float) -> str:
    if abs(v) < 5e-5:
        v = 0.0
    s = f"{v:+.1f}"
    sign = s[0]
    intpart, dec = s[1:].split(".")
    intpart = intpart.zfill(4)
    return f"{sign}{intpart}.{dec}"

def _fmt_ang(v: float) -> str:
    if abs(v) < 5e-5:
        v = 0.0
    s = f"{v:+.2f}"
    sign = s[0]
    intpart, dec = s[1:].split(".")
    intpart = intpart.zfill(3)
    return f"{sign}{intpart}.{dec}"

def _linmap(val: float, a0: float, a1: float, b0: float, b1: float) -> float:
    if abs(a1 - a0) < 1e-12:
        return float(b0)
    t = (val - a0) / (a1 - a0)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return float(b0 + t * (b1 - b0))

DEFAULT_PRESET = {
    "0": {
        "X": {"in": [0.0, 6500.0], "A4_out": [0.0, 500.0]},
        "Y": {"in": [0.0, 1000.0]},
        "Z": {"in": [0.0, 3000.0], "A3_out": [0.0, 1000.0]},
    },
    "90": {
        "X": {"in": [0.0, 6500.0]},
        "Y": {"in": [0.0, 1000.0], "A4_out": [0.0, 500.0]},
        "Z": {"in": [0.0, 3000.0], "A3_out": [0.0, 1000.0]},
    },
    "-90": {
        "X": {"in": [0.0, 6500.0]},
        "Y": {"in": [0.0, 1000.0], "A4_out": [0.0, 500.0]},
        "Z": {"in": [0.0, 3000.0], "A3_out": [0.0, 1000.0]},
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

# === A1/A2 Constant-Speed profile ===
def _apply_const_speed_profile_on_nodes(
    nodes: List[Dict[str, Any]],
    axis_key: str,
    coord_key: str,
    coord_min: float,
    coord_max: float,
    axis_at_min: float,
    axis_at_max: float,
    speed_mm_s: float = 200.0,
    eps_mm: float = 0.5,
    apply_print_only: bool = False,
    travel_interp: bool = True
) -> None:
    if not nodes or axis_key not in ("a1", "a2"):
        return
    n = len(nodes)
    if n == 0:
        return

    coord_min = float(coord_min)
    coord_max = float(coord_max)
    axis_at_min = float(axis_at_min)
    axis_at_max = float(axis_at_max)
    eps = float(max(0.0, eps_mm))

    span = float(coord_max - coord_min)
    span_abs = abs(span)
    if span_abs <= 1e-9:
        for nd in nodes:
            nd[axis_key] = float(axis_at_min)
        return

    axis_per_mm = (axis_at_max - axis_at_min) / float(span_abs)

    def _coord(i: int) -> float:
        return float(nodes[i][coord_key])

    def _at_min(c: float) -> bool:
        return c <= coord_min + eps

    def _at_max(c: float) -> bool:
        return c >= coord_max - eps

    def _snap_axis_for_coord(c: float) -> float:
        if _at_min(c):
            return float(axis_at_min)
        if _at_max(c):
            return float(axis_at_max)
        t = (c - coord_min) / span_abs
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return float(axis_at_min + t * (axis_at_max - axis_at_min))

    extr_node = [bool(nd.get("extr", False)) for nd in nodes]

    c0 = _coord(0)
    nodes[0][axis_key] = _snap_axis_for_coord(c0)

    if _at_min(c0):
        dir_mode = "fwd"
    elif _at_max(c0):
        dir_mode = "bwd"
    else:
        dir_mode = "fwd"
        for j in range(1, n):
            dc = _coord(j) - c0
            if abs(dc) > 1e-9:
                dir_mode = "fwd" if dc > 0 else "bwd"
                break

    for i in range(n - 1):
        ci = _coord(i)
        cj = _coord(i + 1)

        if _at_min(ci):
            nodes[i][axis_key] = float(axis_at_min)
            dir_mode = "fwd"
        elif _at_max(ci):
            nodes[i][axis_key] = float(axis_at_max)
            dir_mode = "bwd"

        ai = float(nodes[i][axis_key])

        active = True
        if apply_print_only:
            active = bool(extr_node[i + 1])

        dcoord = float(cj - ci)
        if (not active) or abs(dcoord) <= 1e-12:
            aj = ai
        else:
            step = axis_per_mm * abs(dcoord)
            if dir_mode == "fwd":
                aj = ai + step
            else:
                aj = ai - step

        if _at_min(cj):
            aj = float(axis_at_min)
            dir_mode = "fwd"
        elif _at_max(cj):
            aj = float(axis_at_max)
            dir_mode = "bwd"

        lo = min(axis_at_min, axis_at_max)
        hi = max(axis_at_min, axis_at_max)
        if aj < lo: aj = lo
        if aj > hi: aj = hi

        nodes[i + 1][axis_key] = float(aj)

    if travel_interp and apply_print_only:
        active_node = [bool(extr_node[i]) for i in range(n)]
        if any(active_node):
            i = 0
            while i < n:
                if active_node[i]:
                    i += 1
                    continue
                t0 = i
                while i < n and (not active_node[i]):
                    i += 1
                t1 = i - 1
                prev_idx = t0 - 1 if t0 - 1 >= 0 else None
                next_idx = i if i < n else None
                if prev_idx is None or next_idx is None:
                    base = float(nodes[prev_idx][axis_key]) if prev_idx is not None else float(nodes[next_idx][axis_key]) if next_idx is not None else float(axis_at_min)
                    for k in range(t0, t1 + 1):
                        nodes[k][axis_key] = base
                    continue
                a0 = float(nodes[prev_idx][axis_key])
                a1 = float(nodes[next_idx][axis_key])
                total = max(1, (t1 - t0 + 1))
                for kk, k in enumerate(range(t0, t1 + 1)):
                    u = (kk + 1) / float(total + 1)
                    nodes[k][axis_key] = a0 + (a1 - a0) * float(u)

    for i in range(n):
        c = _coord(i)
        if _at_min(c):
            nodes[i][axis_key] = float(axis_at_min)
        elif _at_max(c):
            nodes[i][axis_key] = float(axis_at_max)

# === 노드 재샘플링: gap 내 포인트 없을 때만 step 거리 앞뒤 보조 포인트 추가 ===
def densify_nodes_with_gap(
    nodes: List[Dict[str, Any]],
    gap_mm: float = 30.0,
    step_mm: float = 5.0,
) -> List[Dict[str, Any]]:
    if len(nodes) < 3:
        return nodes

    gap = float(gap_mm)
    step = float(step_mm)

    def _dist_xy(a, b) -> float:
        return float(np.linalg.norm(
            np.array([a["x"], a["y"]]) - np.array([b["x"], b["y"]])
        ))

    out: List[Dict[str, Any]] = []
    out.append(nodes[0])

    for i in range(1, len(nodes) - 1):
        cur = nodes[i]

        has_near_before = False
        has_near_after  = False

        acc = 0.0
        j = i - 1
        while j >= 0 and acc < gap:
            d = _dist_xy(nodes[j + 1], nodes[j])
            acc += d
            if acc < gap:
                has_near_before = True
                break
            j -= 1

        acc = 0.0
        j = i + 1
        while j < len(nodes) - 1 and acc < gap:
            d = _dist_xy(nodes[j], nodes[j + 1])
            acc += d
            if acc < gap:
                has_near_after = True
                break
            j += 1

        if has_near_before or has_near_after:
            out.append(cur)
            continue

        p_cur  = np.array([cur["x"],  cur["y"],  cur["z"]],  dtype=float)

        prev_nd = nodes[i - 1]
        p_prev  = np.array([prev_nd["x"], prev_nd["y"], prev_nd["z"]], dtype=float)
        dir_prev = p_prev - p_cur
        len_prev = float(np.linalg.norm(dir_prev[:2]))
        if len_prev > 1e-9:
            t = min(step / len_prev, 1.0)
            p_front = p_cur + dir_prev * t
            nd_front = {
                "x": float(p_front[0]),
                "y": float(p_front[1]),
                "z": float(p_front[2]),
                "extr": bool(prev_nd.get("extr", False) or cur.get("extr", False)),
            }
            out.append(nd_front)

        out.append(cur)

        next_nd = nodes[i + 1]
        p_next  = np.array([next_nd["x"], next_nd["y"], next_nd["z"]], dtype=float)
        dir_next = p_next - p_cur
        len_next = float(np.linalg.norm(dir_next[:2]))
        if len_next > 1e-9:
            t = min(step / len_next, 1.0)
            p_back = p_cur + dir_next * t
            nd_back = {
                "x": float(p_back[0]),
                "y": float(p_back[1]),
                "z": float(p_back[2]),
                "extr": bool(cur.get("extr", False) or next_nd.get("extr", False)),
            }
            out.append(nd_back)

    out.append(nodes[-1])
    return out

def gcode_to_rapid_module(
    gcode_text: str,
    rx: float,
    ry: float,
    rz: float,
    preset: Dict[str, Any],
    swap_a3_a4: bool = False,
    enable_a1_const: bool = True,
    enable_a2_const: bool = True,
    x_min: float = 0.0,
    x_max: float = 6000.0,
    a1_at_xmin: float = 4000.0,
    a1_at_xmax: float = 0.0,
    y_min: float = 0.0,
    y_max: float = 1000.0,
    a2_at_ymin: float = 0.0,
    a2_at_ymax: float = 4000.0,
    speed_mm_s: float = 200.0,
    boundary_eps_mm: float = 0.5,
    apply_print_only: bool = False,
    travel_interp: bool = True,
) -> str:
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

    x0, x1 = gi(P, ["X","in",0], 0.0), gi(P, ["X","in",1], 1.0)
    y0, y1 = gi(P, ["Y","in",0], 0.0), gi(P, ["Y","in",1], 1.0)
    z0, z1 = gi(P, ["Z","in",0], 0.0), gi(P, ["Z","in",1], 1.0)

    a3_0, a3_1 = gi(P, ["Z","A3_out",0], 0.0), gi(P, ["Z","A3_out",1], 0.0)

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

    have_prev = False
    prev_x = prev_y = prev_z = 0.0
    prev_e = None
    cur_a4 = 0.0

    xs_out: List[float] = []
    ys_out: List[float] = []
    zs_out: List[float] = []
    a3_list: List[float] = []
    a4_list: List[float] = []
    is_extruding_list: List[bool] = []

    for raw in gcode_text.splitlines():
        t = raw.strip()
        if not t or not t.startswith(("G0","G00","G1","G01")):
            continue

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

        is_extruding = False
        if ce is not None and prev_e is not None:
            if (ce - prev_e) > 1e-12:
                is_extruding = True
        if ce is not None:
            prev_e = ce

        a3_abs = _linmap(cz, z0, z1, a3_0, a3_1)

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

        if a4_on_x:
            lo, hi = (a4x_0, a4x_1) if a4x_0 <= a4x_1 else (a4x_1, a4x_0)
        elif a4_on_y:
            lo, hi = (a4y_0, a4y_1) if a4y_0 <= a4y_1 else (a4y_1, a4y_0)
        else:
            lo, hi = (0.0, 0.0)
        cur_a4 = lo if cur_a4 < lo else hi if cur_a4 > hi else cur_a4

        x_out, y_out, z_out = cx, cy, cz - a3_abs
        if key == "90":
            y_out = cy - cur_a4
        elif key == "0":
            x_out = cx - cur_a4
        elif key == "-90":
            a4_max = max(a4y_0, a4y_1) if a4_on_y else 0.0
            y_out = cy - (a4_max - cur_a4)

        xs_out.append(float(x_out))
        ys_out.append(float(y_out))
        zs_out.append(float(z_out))
        a3_list.append(float(a3_abs))
        a4_list.append(float(cur_a4))
        is_extruding_list.append(bool(is_extruding))

        if len(xs_out) >= MAX_LINES:
            break

        prev_x, prev_y, prev_z = cx, cy, cz

    # G-code 샘플 → nodes
    nodes: List[Dict[str, Any]] = []
    for cx, cy, cz, a3, a4, extr in zip(xs_out, ys_out, zs_out, a3_list, a4_list, is_extruding_list):
        nodes.append({
            "x": float(cx),
            "y": float(cy),
            "z": float(cz),
            "a3": float(a3),
            "a4": float(a4),
            "extr": bool(extr),
        })

    # 노드 재샘플링 (gap 30mm, step 5mm)
    nodes = densify_nodes_with_gap(
        nodes,
        gap_mm=30.0,
        step_mm=5.0,
    )

    # A1/A2 등속 프로파일
    if enable_a1_const:
        _apply_const_speed_profile_on_nodes(
            nodes=nodes,
            axis_key="a1",
            coord_key="x",
            coord_min=x_min,
            coord_max=x_max,
            axis_at_min=a1_at_xmin,
            axis_at_max=a1_at_xmax,
            speed_mm_s=speed_mm_s,
            eps_mm=boundary_eps_mm,
            apply_print_only=apply_print_only,
            travel_interp=travel_interp,
        )

    if enable_a2_const:
        _apply_const_speed_profile_on_nodes(
            nodes=nodes,
            axis_key="a2",
            coord_key="y",
            coord_min=y_min,
            coord_max=y_max,
            axis_at_min=a2_at_ymin,
            axis_at_max=a2_at_ymax,
            speed_mm_s=speed_mm_s,
            eps_mm=boundary_eps_mm,
            apply_print_only=apply_print_only,
            travel_interp=travel_interp,
        )

    # Rapid MODULE 문자열 생성
    lines: List[str] = []
    lines.append("MODULE Converted")
    lines.append("  CONST pos pPath{}:=[".format(len(nodes)))

    for i, nd in enumerate(nodes):
        fx = _fmt_pos(nd["x"])
        fy = _fmt_pos(nd["y"])
        fz = _fmt_pos(nd["z"])
        fa1 = _fmt_ang(float(nd.get("a1", 0.0)))
        fa2 = _fmt_ang(float(nd.get("a2", 0.0)))
        fa3 = _fmt_ang(float(nd.get("a3", 0.0)))
        fa4 = _fmt_ang(float(nd.get("a4", 0.0)))

        line = f"    [{fx},{fy},{fz},{fa1},{fa2},{fa3},{fa4},+0000.0,+0000.0,+0000.0]"
        if i < len(nodes) - 1:
            line += ","
        lines.append(line)

    lines.append("  ];")
    lines.append("ENDMODULE")

    return "\n".join(lines)

if not KEY_OK:
    st.warning("유효한 Access Key를 입력해야 프로그램이 작동합니다. (업로드/슬라이싱/G-code 버튼 비활성화)")
