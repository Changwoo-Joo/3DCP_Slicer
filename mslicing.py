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
    for st in s_targets:
        while j < len(lens) - 1 and s[j+1] < st:
            j += 1
        d = float(lens[j])
        if d < 1e-12:
            out.append(pts[j].copy())
        else:
            t = (st - s[j]) / d
            out.append((1.0 - t) * pts[j] + t * pts[j+1])
    return np.asarray(out, dtype=float)

def densify_sparse_corners(poly: np.ndarray,
                           step_mm: float = 5.0,
                           window_mm: float = 30.0,
                           collinear_eps: float = 1e-12) -> np.ndarray:
    pts = np.asarray(poly, dtype=float)
    if len(pts) < 3 or step_mm <= 0 or window_mm <= 0:
        return pts.copy()

    s = _poly_arclen_s_xy(pts)
    total = float(s[-1])
    if total < 1e-9:
        return pts.copy()

    s_targets = [0.0, total]

    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i-1]
        v2 = pts[i+1] - pts[i]

        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) <= collinear_eps:
            continue

        gap_prev = float(s[i] - s[i-1])
        gap_next = float(s[i+1] - s[i])
        if not (gap_prev > window_mm or gap_next > window_mm):
            continue

        si = float(s[i])
        a = max(0.0, si - window_mm)
        b = min(total, si + window_mm)

        n = int(max(2, math.floor((b - a) / step_mm) + 1))
        ss = a + step_mm * np.arange(n, dtype=float)
        ss = ss[ss <= b + 1e-6]
        s_targets.extend(ss.tolist())

    s_targets = np.unique(np.round(np.asarray(s_targets, dtype=float), 6))
    s_targets.sort()
    return _resample_polyline_by_s(pts, s_targets)

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

        if L1 > d_mm + min_sep_eps:
            u1 = v1 / L1
            p_before = p1 - u1 * d_mm
            if np.linalg.norm((p_before - out[-1])[:2]) > min_sep_eps:
                out.append(p_before)

        if np.linalg.norm((p1 - out[-1])[:2]) > min_sep_eps:
            out.append(p1.copy())

        if L2 > d_mm + min_sep_eps:
            u2 = v2 / L2
            p_after = p1 + u2 * d_mm
            if np.linalg.norm((p_after - out[-1])[:2]) > min_sep_eps:
                out.append(p_after)

    if np.linalg.norm((pts[-1] - out[-1])[:2]) > min_sep_eps:
        out.append(pts[-1].copy())

    return np.asarray(out, dtype=float)

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
                   trim_dist=30.0, min_spacing=5.0, auto_start=False, m30_on=False, enable_corner=False, corner_d=5.0):
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
            if enable_corner:
               simplified = _insert_corner_neighbors(simplified, d_mm=float(corner_d))

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
    mesh, z_int=30.0, ref_pt_user=(0.0, 0.0), trim_dist=30.0,
    min_spacing=5.0, auto_start=False, e_on=False, enable_corner=False, corner_d=5.0
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
            if enable_corner:
               simplified = _insert_corner_neighbors(simplified, d_mm=float(corner_d))
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
enable_corner = st.sidebar.checkbox(
    "Enable corner neighbor points",
    value=True,
    key="enable_corner_points",
)

corner_d = st.sidebar.number_input(
    "Corner neighbor distance (mm)",
    0.0, 1000.0, 5.0, 1.0,
    key="corner_neighbor_distance_mm",
    disabled=not enable_corner,
)

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
        e_on=e_on,
        enable_corner=enable_corner,
        corner_d=corner_d
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
        trim_dist=trim_dist, min_spacing=min_spacing, auto_start=auto_start, m30_on=m30_on,
        enable_corner=enable_corner, corner_d=corner_d
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
# ✅ (FIX) A1/A2 Constant-Speed profile (Absolute Mapping)
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
    deadband_mm: float = 11.0,
    eps_mm: float = 0.5,
    apply_print_only: bool = False,
    travel_interp: bool = True,
    step_mm: float = 0.0,
    step_round: str = "floor",
) -> None:

    if not nodes or axis_key not in ('a1', 'a2'):
        return

    n = len(nodes)
    if n == 0:
        return

    coord_min = float(coord_min)
    coord_max = float(coord_max)
    axis_at_min = float(axis_at_min)
    axis_at_max = float(axis_at_max)

    span = float(coord_max - coord_min)
    span_abs = abs(span)
    if span_abs < 1e-9:
        for nd in nodes:
            nd[axis_key] = float(axis_at_min)
        return

    use_step = (step_mm is not None and float(step_mm) > 0.0)
    dt = float(step_mm) / span_abs if use_step else 0.0

    def _map(c: float) -> float:
        t = (c - coord_min) / span
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

        if use_step and dt > 1e-12:
            if step_round == "round":
                t_q = round(t / dt) * dt
            elif step_round == "ceil":
                t_q = math.ceil(t / dt) * dt
            else:
                t_q = math.floor(t / dt) * dt
            
            t_q = 0.0 if t_q < 0.0 else (1.0 if t_q > 1.0 else t_q)
            return float(axis_at_min + t_q * (axis_at_max - axis_at_min))
        
        return float(axis_at_min + t * (axis_at_max - axis_at_min))

    extr_node = [bool(nd.get("extr", False)) for nd in nodes]

    for i in range(n):
        c_val = float(nodes[i].get(coord_key, 0.0))
        if apply_print_only and not extr_node[i]:
            nodes[i][axis_key] = None
        else:
            nodes[i][axis_key] = _map(c_val)

    # 1. 빈값(None) 채우기
    if nodes[0][axis_key] is None:
        for k in range(n):
            if nodes[k][axis_key] is not None:
                nodes[0][axis_key] = nodes[k][axis_key]
                break
        else:
            nodes[0][axis_key] = float(axis_at_min)

    if nodes[-1][axis_key] is None:
        for k in range(n - 1, -1, -1):
            if nodes[k][axis_key] is not None:
                nodes[-1][axis_key] = nodes[k][axis_key]
                break
        else:
            nodes[-1][axis_key] = float(axis_at_min)

    if apply_print_only:
        i = 0
        while i < n:
            if nodes[i][axis_key] is not None:
                i += 1
                continue
            
            t0 = i
            while i < n and nodes[i][axis_key] is None:
                i += 1
            t1 = i - 1

            prev_idx = t0 - 1
            next_idx = i if i < n else t1
            v_start = nodes[prev_idx][axis_key]
            v_end = nodes[next_idx][axis_key]
            
            steps = (t1 - t0) + 2
            for k in range(t0, t1 + 1):
                frac = (k - prev_idx) / steps
                nodes[k][axis_key] = v_start + frac * (v_end - v_start)

    # ==========================================
    # ✅ (NEW) 후진(역주행) 방지 및 보간 로직 적용
    # ==========================================
    vals = [nd[axis_key] for nd in nodes]
    n_vals = len(vals)
    if n_vals == 0:
        return

    is_increasing = (float(axis_at_max) >= float(axis_at_min))

    mono_vals = [vals[0]]
    for v in vals[1:]:
        if is_increasing:
            mono_vals.append(max(mono_vals[-1], v))
        else:
            mono_vals.append(min(mono_vals[-1], v))

    i = 0
    while i < n_vals - 1:
        if mono_vals[i] == mono_vals[i+1]:
            j = i + 1
            while j < n_vals and mono_vals[j] == mono_vals[i]:
                j += 1
            
            if j < n_vals:
                v_start = mono_vals[i]
                v_end = mono_vals[j]
                steps = j - i
                for k in range(i + 1, j):
                    mono_vals[k] = v_start + (v_end - v_start) * ((k - i) / float(steps))
            
            i = j
        else:
            i += 1

    for idx in range(n_vals):
        nodes[idx][axis_key] = mono_vals[idx]
def gcode_to_cone1500_module(
    gcode_text: str,
    preset: Dict[str, Any],
    rx: float, ry: float, rz: float,
    apply_const: bool,
    const_speed_mm_s: float,
    const_deadband_mm: float,
    const_eps_mm: float,
    apply_print_only: bool,
    travel_interp: bool
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
        return delta * (span_out / total)

    parsed = []
    lines = gcode_text.splitlines()

    cur_x, cur_y, cur_z = 0.0, 0.0, 0.0
    cur_e = 0.0

    for raw in lines:
        t = raw.strip()
        idx = t.find(';')
        if idx >= 0:
            t = t[:idx].strip()
        if not t:
            continue
        if t.startswith(("G0", "G00", "G1", "G01")):
            parts = t.split()
            has_xyz = any(p.startswith(("X","Y","Z")) for p in parts)
            if not has_xyz:
                continue

            nx, ny, nz = cur_x, cur_y, cur_z
            ne = cur_e
            for p in parts:
                if p.startswith('X'): nx = float(p[1:])
                elif p.startswith('Y'): ny = float(p[1:])
                elif p.startswith('Z'): nz = float(p[1:])
                elif p.startswith('E'): ne = float(p[1:])

            extr = False
            if ne > cur_e + 1e-9:
                extr = True

            parsed.append({
                "x": nx, "y": ny, "z": nz, "extr": extr
            })
            cur_x, cur_y, cur_z, cur_e = nx, ny, nz, ne

    n_pts = len(parsed)
    if n_pts == 0:
        return "MODULE Converted\nENDMODULE\n"

    for nd in parsed:
        cx, cy, cz = nd["x"], nd["y"], nd["z"]
        dx = cx - x0; dy = cy - y0; dz = cz - z0

        if a4_on_x:
            a4_val = _prop_split_local(dx, x0, x1, a4x_0, a4x_1)
        elif a4_on_y:
            a4_val = _prop_split_local(dy, y0, y1, a4y_0, a4y_1)
        else:
            a4_val = 0.0

        a3_val = _prop_split_local(dz, z0, z1, a3_0, a3_1)

        nd["a3"] = a3_0 + a3_val
        nd["a4"] = (a4x_0 if a4_on_x else a4y_0) + a4_val
        nd["a1"] = 0.0
        nd["a2"] = 0.0

    if apply_const:
        if st.session_state.get("ext_const_enable_a1", True):
            _apply_const_speed_profile_on_nodes(
                nodes=parsed, axis_key="a1", coord_key="x",
                coord_min=st.session_state.get("ext_const_xmin", 0.0),
                coord_max=st.session_state.get("ext_const_xmax", 6000.0),
                axis_at_min=st.session_state.get("ext_const_a1_at_xmin", 4000.0),
                axis_at_max=st.session_state.get("ext_const_a1_at_xmax", 0.0),
                speed_mm_s=const_speed_mm_s, deadband_mm=const_deadband_mm, eps_mm=const_eps_mm,
                apply_print_only=apply_print_only, travel_interp=travel_interp,
                step_mm=st.session_state.get("ext_const_step_mm", 0.0),
                step_round=st.session_state.get("ext_const_step_round", "floor")
            )
        else:
            for nd in parsed: nd["a1"] = 0.0

        if st.session_state.get("ext_const_enable_a2", True):
            _apply_const_speed_profile_on_nodes(
                nodes=parsed, axis_key="a2", coord_key="y",
                coord_min=st.session_state.get("ext_const_ymin", 0.0),
                coord_max=st.session_state.get("ext_const_ymax", 1000.0),
                axis_at_min=st.session_state.get("ext_const_a2_at_ymin", 0.0),
                axis_at_max=st.session_state.get("ext_const_a2_at_ymax", 4000.0),
                speed_mm_s=const_speed_mm_s, deadband_mm=const_deadband_mm, eps_mm=const_eps_mm,
                apply_print_only=apply_print_only, travel_interp=travel_interp,
                step_mm=st.session_state.get("ext_const_step_mm", 0.0),
                step_round=st.session_state.get("ext_const_step_round", "floor")
            )
        else:
            for nd in parsed: nd["a2"] = 0.0

    lines_out = []
    frx, fry, frz = _fmt_ang(rx), _fmt_ang(ry), _fmt_ang(rz)

    for nd in parsed:
        x = _fmt_pos(float(nd["x"]))
        y = _fmt_pos(float(nd["y"]))
        z = _fmt_pos(float(nd["z"]))
        if apply_const:
            a3_v = nd.get("a3", 0.0)
            a4_v = nd.get("a4", 0.0)
        else:
            a3_v = nd["a3"]
            a4_v = nd["a4"]

        a1s = _fmt_pos(float(nd["a1"]))
        a2s = _fmt_pos(float(nd["a2"]))
        a3s = _fmt_pos(float(a3_v))
        a4s = _fmt_pos(float(a4_v))

        lines_out.append(f"{x},{y},{z},{frx},{fry},{frz},{a1s},{a2s},{a3s},{a4s}")

    while len(lines_out) < MAX_LINES:
        lines_out.append(PAD_LINE)

    ts = datetime.now().strftime("%Y-%m-%d %p %I:%M:%S")
    header = ("MODULE Converted\n"
              "!******************************************************************************************************************************\n"
              "!*\n"
              f"!*** Generated {ts} by Gcode→RAPID converter.\n"
              "!\n"
              "!*** data3dp: X(mm), Y(mm), Z(mm), Rx(deg), Ry(deg), Rz(deg), A1,A2,A3,A4\n"
              "!*** Number of extracted points: " + str(n_pts) + "\n"
              "!******************************************************************************************************************************\n"
              "  PERS string data3dp{64000} := [\n")

    out_arr = []
    for i, l in enumerate(lines_out):
        is_last = (i == len(lines_out) - 1)
        suffix = "\n" if is_last else ",\n"
        out_arr.append(f'    "{l}"{suffix}')

    footer = "  ];\nENDMODULE\n"
    return header + "".join(out_arr) + footer

# =========================
# Right Panel: Visualization Settings & Rapid Exporter
# =========================
with st.sidebar:
    st.markdown("<hr style='margin: 0.5rem 0'>", unsafe_allow_html=True)
    if st.button("▶ RAPID Export Panel"):
        st.session_state.show_rapid_panel = not st.session_state.show_rapid_panel

if st.session_state.get("ui_banner"):
    st.toast(st.session_state.ui_banner, icon="✅")
    st.session_state.ui_banner = None

left_col, center_col, right_col = st.columns([1, 10, 3], gap="medium")

# ---- 좌측: 렌더링/뷰포트 설정 ----
with left_col:
    if st.session_state.mesh is not None:
        st.subheader("Viewport & Viz")
        max_idx = len(st.session_state.paths_anim_buf.get("solid", {}).get("x", [])) // 3
        if "paths_items" in st.session_state and st.session_state.paths_items:
            segs = items_to_segments(st.session_state.paths_items, e_on=e_on)
            max_idx = len(segs)
        else:
            segs = None

        if max_idx > 0:
            scrub = st.slider("Scrub Layers / Segments", 0, max_idx, st.session_state.paths_scrub)
            if scrub != st.session_state.paths_scrub:
                st.session_state.paths_scrub = scrub

            st.radio("Travel Lines Mode", ["solid", "dotted", "hidden"], key="paths_travel_mode")

            st.checkbox("Apply Offsets (Width/2)", value=False, key="apply_offsets_flag")
            emphasize_caps = st.checkbox("Emphasize Caps (if Offsets ON)", value=False)
            dims_placeholder = st.empty()

        else:
            st.info("No segments to scrub.")
            segs = None
            dims_placeholder = st.empty()

# ---- 계산/버퍼 구성 ----
if segs is not None and max_idx > 0:
    target = int(clamp(st.session_state.paths_scrub, 0, max_idx))

    DRAW_LIMIT = 15000
    draw_stride = max(1, math.ceil(max(1, target) / DRAW_LIMIT))

    built = st.session_state.paths_anim_buf["built_upto"]
    prev_stride = st.session_state.paths_anim_buf.get("stride", 1)

    if (draw_stride != prev_stride) or (target < built):
        rebuild_buffers_to(segs, target, stride=draw_stride)
    elif target > built:
        append_segments_to_buffers(segs, built, target, stride=draw_stride)

    st.session_state.paths_scrub = target

    if bool(st.session_state.get("apply_offsets_flag", False)):
        half_w = float(trim_dist) * 0.5
        compute_offsets_into_buffers(segs, target, half_w, include_travel_climb=False, climb_z_thresh=1e-9)
        st.session_state.paths_anim_buf["caps"] = {"x": [], "y": [], "z": []}
        add_global_endcaps_into_buffers(segs, target, half_width=half_w, samples=32, store_caps=bool(emphasize_caps))
        bbox_r = _bbox_from_buffer(st.session_state.paths_anim_buf["off_r"])
        z_r = _last_z_from_buffer(st.session_state.paths_anim_buf["off_r"])
        dims_html = _fmt_dims_block_html("외부치수", bbox_r, z_r)
        dims_placeholder.markdown(dims_html, unsafe_allow_html=True)

    layer_z, layer_len = compute_layer_length_for_index(segs, target)
    if layer_z is not None and layer_len is not None:
        total_len = sum(np.linalg.norm(p2[:2] - p1[:2]) for p1, p2, is_travel, is_extruding in segs[:target] if is_extruding)
        st.markdown(f"**누적 총 길이:** {total_len/1000:.3f} m")

# ---- 중앙: 탭 뷰어 ----
with center_col:
    tab_paths, tab_stl, tab_gcode = st.tabs(["Sliced Paths (3D)", "STL Preview", "G-code Viewer"])

    with tab_paths:
        if segs is not None and max_idx > 0:
            if "paths_base_fig" not in st.session_state:
                st.session_state.paths_base_fig = make_base_fig(height=820)
            fig = st.session_state.paths_base_fig
            update_fig_with_buffers(fig, show_offsets=bool(st.session_state.get("apply_offsets_flag", False)), show_caps=bool(emphasize_caps))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("슬라이싱을 실행하세요.")

    with tab_stl:
        if st.session_state.get("mesh") is not None:
            st.plotly_chart(plot_trimesh(st.session_state.mesh, height=820), use_container_width=True, key="stl_chart", config={"displayModeBar": False})
        else:
            st.info("STL을 업로드하세요.")

    with tab_gcode:
        if st.session_state.get("gcode_text"):
            st.code(st.session_state.gcode_text, language="gcode")
        else:
            st.info("G-code를 생성하세요.")

# ---- 우측: RAPID Export Panel ----
with right_col:
    if st.session_state.show_rapid_panel:
        st.markdown("<div class='right-panel'>", unsafe_allow_html=True)
        st.subheader("G-code → RAPID (.modx)")
        st.info("생성된 G-code를 배열 형태의 RAPID 데이터로 변환합니다.")

        st.markdown("**1. Orientation (deg)**")
        rx = st.number_input("Rx", value=st.session_state.rapid_rx, step=1.0)
        ry = st.number_input("Ry", value=st.session_state.rapid_ry, step=1.0)
        rz = st.number_input("Rz", value=st.session_state.rapid_rz, step=90.0)

        st.markdown("---")
        st.markdown("**2. Constant Speed (A1/A2)**")
        apply_const = st.checkbox("Enable Variable Extrusion (A1/A2)", value=True)
        if apply_const:
            st.checkbox("Enable A1 (X-axis)", key="ext_const_enable_a1")
            st.checkbox("Enable A2 (Y-axis)", key="ext_const_enable_a2")

            c1, c2 = st.columns(2)
            c1.number_input("X min (mm)", value=0.0, key="ext_const_xmin")
            c2.number_input("X max (mm)", value=6000.0, key="ext_const_xmax")
            c1.number_input("A1 at X min", value=4000.0, key="ext_const_a1_at_xmin")
            c2.number_input("A1 at X max", value=0.0, key="ext_const_a1_at_xmax")

            c1, c2 = st.columns(2)
            c1.number_input("Y min (mm)", value=0.0, key="ext_const_ymin")
            c2.number_input("Y max (mm)", value=1000.0, key="ext_const_ymax")
            c1.number_input("A2 at Y min", value=0.0, key="ext_const_a2_at_ymin")
            c2.number_input("A2 at Y max", value=4000.0, key="ext_const_a2_at_ymax")

            st.checkbox("Travel Interpolation", value=True, key="ext_const_travel_interp", help="Extrusion=0 구간 보간")
            const_speed_mm_s = st.number_input("Target TCP Speed (mm/s)", value=200.0)
            const_deadband_mm = st.number_input("Deadband (mm)", value=11.0)
            const_eps_mm = st.number_input("Epsilon (mm)", value=0.5)

        else:
            const_speed_mm_s = 200.0
            const_deadband_mm = 11.0
            const_eps_mm = 0.5

        if st.button("Convert to .modx", type="primary", use_container_width=True):
            if not st.session_state.gcode_text:
                st.error("먼저 G-code를 생성하세요.")
            else:
                lines_count = _extract_xyz_lines_count(st.session_state.gcode_text)
                if lines_count > MAX_LINES:
                    st.error(f"좌표가 {lines_count}개로 MAX_LINES({MAX_LINES})를 초과합니다.")
                else:
                    rapid_modx = gcode_to_cone1500_module(
                        gcode_text=st.session_state.gcode_text,
                        preset=st.session_state.mapping_preset,
                        rx=rx, ry=ry, rz=rz,
                        apply_const=apply_const,
                        const_speed_mm_s=const_speed_mm_s,
                        const_deadband_mm=const_deadband_mm,
                        const_eps_mm=const_eps_mm,
                        apply_print_only=st.session_state.get("ext_const_apply_print_only", False),
                        travel_interp=st.session_state.get("ext_const_travel_interp", True)
                    )
                    st.session_state.rapid_text = rapid_modx
                    st.session_state.rapid_rx = rx
                    st.session_state.rapid_ry = ry
                    st.session_state.rapid_rz = rz
                    st.success("RAPID 변환 완료!")

        if st.session_state.get("rapid_text"):
            dl_name = f"{st.session_state.base_name}_data3dp.modx"
            st.download_button("Download .modx", st.session_state.rapid_text,
                               file_name=dl_name, mime="text/plain", use_container_width=True)
            with st.expander("Preview .modx (앞부분)"):
                lines_prev = st.session_state.rapid_text.splitlines()[:25]
                st.code("\n".join(lines_prev) + "\n...", language="rapid")

        st.markdown("</div>", unsafe_allow_html=True)

