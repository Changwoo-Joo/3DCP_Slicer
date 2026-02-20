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

        # 각도 임계값 없이: '완전 일직선'만 제외
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) <= collinear_eps:
            continue

        # 코너에서 앞/뒤 인접 점 간격이 window_mm보다 큰 경우에만 발동
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
    disabled=not enable_corner,   # (선택) OFF면 입력 비활성
)

trim_dist = st.sidebar.number_input("Trim/Layer Width (mm)", 0.0, 1000.0, 50.0)
min_spacing = st.sidebar.number_input("Minimum point spacing (mm)", 0.0, 1000.0, 5.0)
corner_d = st.sidebar.number_input("Corner neighbor distance (mm)", 0.0, 1000.0, 5.0, 1.0)
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
# ✅ (FIX) A1/A2 Constant-Speed profile (Absolute Mapping)
#   - 누적 이동거리가 아닌 1:1 절대 좌표(Absolute Mapping) 방식으로 변경
#   - 물결(Wavy) 패턴에서도 좌표가 뒤로 가면 A축 값도 1:1로 되돌아감 (드리프트/누적현상 완벽해결)
#   - 스텝(Step) 모드 완벽 지원
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
    step_round: str = "floor"
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
    
    span = float(coord_max - coord_min)
    span_abs = abs(span)
    
    # Min/Max가 동일하거나 설정 오류인 경우 최소값으로 고정
    if span_abs <= 1e-9:
        for nd in nodes:
            nd[axis_key] = float(axis_at_min)
        return

    use_step = (step_mm is not None) and (float(step_mm) > 0.0)
    dt = float(step_mm) / span_abs if use_step else 0.0

    # 1:1 절대 좌표 매핑 내부 함수
    def _get_mapped_value(c: float) -> float:
        # 1. 현재 좌표가 전체 구간(span) 중 어디쯤 있는지 비율(t) 계산 (0.0 ~ 1.0)
        t = (c - coord_min) / span
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        
        # 2. 스텝 모드(계단형) 적용
        if use_step and dt > 1e-12:
            if step_round == "round":
                tq = round(t / dt) * dt
            elif step_round == "ceil":
                tq = math.ceil(t / dt) * dt
            else:  # "floor"
                tq = math.floor(t / dt) * dt
            
            # 양자화 후 범위 이탈 방지
            tq = 0.0 if tq < 0.0 else (1.0 if tq > 1.0 else tq)
            return float(axis_at_min + tq * (axis_at_max - axis_at_min))
        else:
            # 3. 선형 연속 모드 적용
            return float(axis_at_min + t * (axis_at_max - axis_at_min))

    extr_node = [bool(nd.get("extr", False)) for nd in nodes]
    
    # 1차 패스: 절대 좌표를 기반으로 모든 노드에 A축 값 1:1 직접 매핑
    for i in range(n):
        nd = nodes[i]
        c_val = float(nd.get(coord_key, 0.0))
        
        active = True
        if apply_print_only:
            active = extr_node[i]
            
        if active or not apply_print_only:
            nd[axis_key] = _get_mapped_value(c_val)
        else:
            # 비활성(Travel) 구간은 나중에 보간하기 위해 임시로 None 할당
            nd[axis_key] = None

    # 2차 패스: Travel 구간 처리 (apply_print_only == True 인 경우)
    if apply_print_only:
        # 양끝단 예외 처리 (첫 노드나 끝 노드가 travel일 경우 붕 뜨는 것 방지)
        if nodes[0][axis_key] is None:
            first_val = float(axis_at_min)
            for k in range(n):
                if nodes[k][axis_key] is not None:
                    first_val = nodes[k][axis_key]
                    break
            nodes[0][axis_key] = first_val
            
        if nodes[-1][axis_key] is None:
            last_val = float(axis_at_min)
            for k in range(n - 1, -1, -1):
                if nodes[k][axis_key] is not None:
                    last_val = nodes[k][axis_key]
                    break
            nodes[-1][axis_key] = last_val

        # 빈 공간 보간 (Travel Interpolation)
        if travel_interp:
            i = 0
            while i < n:
                if nodes[i][axis_key] is not None:
                    i += 1
                    continue
                
                # None(Travel) 구간 시작점 찾기
                t0 = i
                while i < n and nodes[i][axis_key] is None:
                    i += 1
                t1 = i - 1
                
                prev_idx = t0 - 1
                next_idx = i if i < n else None
                
                val_start = nodes[prev_idx][axis_key]
                val_end = nodes[next_idx][axis_key] if next_idx is not None else val_start
                
                steps = t1 - t0 + 2
                for k in range(t0, t1 + 1):
                    fraction = (k - prev_idx) / steps
                    nodes[k][axis_key] = val_start + fraction * (val_end - val_start)
        else:
            # 보간 안 함 (Hold 이전 값)
            curr_val = nodes[0][axis_key]
            for i in range(n):
                if nodes[i][axis_key] is None:
                    nodes[i][axis_key] = curr_val
                else:
                    curr_val = nodes[i][axis_key]



# =========================
# Rapid Converter (UPDATED)
# =========================
def gcode_to_cone1500_module(
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
    raw_xs: List[float] = []
    raw_ys: List[float] = []
    a1_list: List[float] = []
    a2_list: List[float] = []
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

        # E 기반 extruding 판정
        is_extruding = False
        if ce is not None and prev_e is not None:
            if (ce - prev_e) > 1e-12:
                is_extruding = True
        if ce is not None:
            prev_e = ce

        # A3(절대)
        a3_abs = _linmap(cz, z0, z1, a3_0, a3_1)

        # A4(기존 유지: 분해축)
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

        # clamp A4
        if a4_on_x:
            lo, hi = (a4x_0, a4x_1) if a4x_0 <= a4x_1 else (a4x_1, a4x_0)
        elif a4_on_y:
            lo, hi = (a4y_0, a4y_1) if a4y_0 <= a4y_1 else (a4y_1, a4y_0)
        else:
            lo, hi = (0.0, 0.0)
        cur_a4 = lo if cur_a4 < lo else hi if cur_a4 > hi else cur_a4

        # 좌표 보정 (기존 유지)
        x_out, y_out, z_out = cx, cy, cz - a3_abs
        if key == "90":
            y_out = cy - cur_a4
        elif key == "0":
            x_out = cx - cur_a4
        elif key == "-90":
            a4_max = max(a4y_0, a4y_1) if a4_on_y else 0.0
            y_out = cy - (a4_max - cur_a4)

        # 저장
        raw_xs.append(float(cx))
        raw_ys.append(float(cy))
        xs_out.append(float(x_out))
        ys_out.append(float(y_out))
        zs_out.append(float(z_out))
        a1_list.append(0.0)
        a2_list.append(0.0)
        a3_list.append(float(a3_abs))
        a4_list.append(float(cur_a4))
        is_extruding_list.append(bool(is_extruding))

        if len(xs_out) >= MAX_LINES:
            break

        prev_x, prev_y, prev_z = cx, cy, cz

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
        cnt_str = str(MAX_LINES)
        open_decl = f'VAR string sFileCount:="{cnt_str}";\nVAR string d3dpDynLoad{{{cnt_str}}}:=[\n'
        body = ""
        for i, ln in enumerate(lines_out):
            q = f'"{ln}"'
            body += (q + ",\n") if i < len(lines_out) - 1 else (q + "\n")
        close_decl = "];\nENDMODULE\n"
        return header + open_decl + body + close_decl

    # nodes
    nodes = []
    for i in range(len(xs_out)):
        nodes.append({
            "x": float(xs_out[i]),
            "y": float(ys_out[i]),
            "z": float(zs_out[i]),
            "raw_x": float(raw_xs[i]),
            "raw_y": float(raw_ys[i]),
            "a1": float(a1_list[i]),
            "a2": float(a2_list[i]),
            "a3": float(a3_list[i]),
            "a4": float(a4_list[i]),
            "extr": bool(is_extruding_list[i]),
        })
    # (수정됨) A2용 좌표: S = Y(출력) + A4
    for nd in nodes:
        nd["coord_s"] = float(nd.get("y", 0.0)) + float(nd.get("a4", 0.0))

    # ✅ A1/A2 const-speed: raw_x/raw_y 기반, 블록 경계 없어도 계속 진행
    if bool(enable_a1_const):
        _apply_const_speed_profile_on_nodes(
            nodes=nodes,
            axis_key="a1",
            coord_key="raw_x",
            coord_min=float(x_min),
            coord_max=float(x_max),
            axis_at_min=float(a1_at_xmin),
            axis_at_max=float(a1_at_xmax),
            speed_mm_s=float(speed_mm_s),
            eps_mm=float(boundary_eps_mm),
            apply_print_only=bool(apply_print_only),
            travel_interp=bool(travel_interp),
        )
    else:
        for nd in nodes:
            nd["a1"] = 0.0


    # --- A2 constant-speed profile (FIXED: always use coords = raw_y + a4) ---
    if bool(enable_a2_const):
        use_step = bool(st.session_state.get("ext_const_a2_use_step", False))

        # coords 필드 생성 (raw_y + a4)
        for nd in nodes:
            nd["coords"] = float(nd.get("raw_y", 0.0)) + float(nd.get("a4", 0.0))

        if use_step:
            _apply_const_speed_profile_on_nodes(
                nodes=nodes,
                axis_key="a2",
                coord_key="coords",
                coord_min=float(y_min),
                coord_max=float(y_max),
                axis_at_min=float(a2_at_ymin),
                axis_at_max=float(a2_at_ymax),
                speed_mm_s=float(speed_mm_s),
                eps_mm=float(boundary_eps_mm),
                apply_print_only=bool(apply_print_only),
                travel_interp=bool(travel_interp),
                step_mm=float(st.session_state.get("ext_const_a2_step_mm", 60.0)),
                step_round="floor",
            )
        else:
            _apply_const_speed_profile_on_nodes(
                nodes=nodes,
                axis_key="a2",
                coord_key="raw_y",
                coord_min=float(y_min),
                coord_max=float(y_max),
                axis_at_min=float(a2_at_ymin),
                axis_at_max=float(a2_at_ymax),
                speed_mm_s=float(speed_mm_s),
                eps_mm=float(boundary_eps_mm),
                apply_print_only=bool(apply_print_only),
                travel_interp=bool(travel_interp),
            )
    else:
        for nd in nodes:
            nd["a2"] = 0.0



    lines_out = []
    for nd in nodes:
        if len(lines_out) >= MAX_LINES:
            break

        x = _fmt_pos(float(nd["x"]))
        y = _fmt_pos(float(nd["y"]))
        z = _fmt_pos(float(nd["z"]))

        if swap_a3_a4:
            a3_v = nd["a4"]
            a4_v = nd["a3"]
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
              "!*** A1: raw_x |ΔX| based + boundary snap/hold (leave-boundary resumes)\n"
              "!*** A2: raw_y |ΔY| based + boundary snap/hold (leave-boundary resumes)\n"
              "!\n"
              "!******************************************************************************************************************************\n")
    cnt_str = str(MAX_LINES)
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

        with st.sidebar.expander("External Axis (A1/A2 등속 왕복 · 경계정지)", expanded=True):
            st.caption("A1은 raw X, A2는 raw Y 기준. A1은 |ΔX|로만, A2는 |ΔY|로만 진행합니다. "
                       "X가 xmin/xmax에 '있을 때'는 A1 고정, 경계에서 '벗어나면' 바로 진행합니다.")

            st.session_state.ext_const_speed_mm_s = st.number_input(
                "Axis speed base (mm/s)",
                min_value=1.0, max_value=2000.0,
                value=float(st.session_state.ext_const_speed_mm_s),
                step=10.0, format="%.1f"
            )
            st.session_state.ext_const_eps_mm = st.number_input(
                "Boundary snap/hold eps (mm)",
                min_value=0.0, max_value=50.0,
                value=float(st.session_state.ext_const_eps_mm),
                step=0.1, format="%.2f"
            )
            st.session_state.ext_const_apply_print_only = st.checkbox(
                "Apply printing-only blocks (E 증가 구간만)",
                value=bool(st.session_state.ext_const_apply_print_only)
            )
            # (추가) A2를 (Y + A4) 기준으로 계단식 적용
            if "extconsta2usestep" not in st.session_state:
                st.session_state.extconsta2usestep = True
            if "extconsta2stepmm" not in st.session_state:
                st.session_state.extconsta2stepmm = 60.0
            
            st.session_state.extconsta2usestep = st.checkbox(
                "A2 step mode uses (Y + A4)",
                value=bool(st.session_state.extconsta2usestep),
            )
            
            st.session_state.extconsta2stepmm = st.number_input(
                "A2 step length (Y+A4) mm",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.extconsta2stepmm),
                step=1.0,
                format="%.3f",
                disabled=not bool(st.session_state.extconsta2usestep),
            )

            st.session_state.ext_const_travel_interp = st.checkbox(
                "Interpolate across travel blocks",
                value=bool(st.session_state.ext_const_travel_interp)
            )

            st.markdown("**A1 (raw X → A1)**")
            st.session_state.ext_const_enable_a1 = st.checkbox(
                "Enable A1 constant-speed profile",
                value=bool(st.session_state.ext_const_enable_a1)
            )
            cols = st.columns(2)
            st.session_state.ext_const_xmin = cols[0].number_input("Xmin (mm)", value=float(st.session_state.ext_const_xmin), step=50.0, format="%.3f")
            st.session_state.ext_const_xmax = cols[1].number_input("Xmax (mm)", value=float(st.session_state.ext_const_xmax), step=50.0, format="%.3f")
            cols2 = st.columns(2)
            st.session_state.ext_const_a1_at_xmin = cols2[0].number_input("A1 @ Xmin", value=float(st.session_state.ext_const_a1_at_xmin), step=50.0, format="%.3f")
            st.session_state.ext_const_a1_at_xmax = cols2[1].number_input("A1 @ Xmax", value=float(st.session_state.ext_const_a1_at_xmax), step=50.0, format="%.3f")

            st.markdown("**A2 (raw Y → A2)**")
            st.session_state.ext_const_enable_a2 = st.checkbox(
                "Enable A2 constant-speed profile",
                value=bool(st.session_state.ext_const_enable_a2)
            )
            cols3 = st.columns(2)
            st.session_state.ext_const_ymin = cols3[0].number_input("Ymin (mm)", value=float(st.session_state.ext_const_ymin), step=50.0, format="%.3f")
            st.session_state.ext_const_ymax = cols3[1].number_input("Ymax (mm)", value=float(st.session_state.ext_const_ymax), step=50.0, format="%.3f")
            cols4 = st.columns(2)
            st.session_state.ext_const_a2_at_ymin = cols4[0].number_input("A2 @ Ymin", value=float(st.session_state.ext_const_a2_at_ymin), step=50.0, format="%.3f")
            st.session_state.ext_const_a2_at_ymax = cols4[1].number_input("A2 @ Ymax", value=float(st.session_state.ext_const_a2_at_ymax), step=50.0, format="%.3f")

        with st.sidebar.expander("Mapping Presets (편집/저장/불러오기)", expanded=False):
            st.caption("Rz 프리셋(0, +90, -90)에 대해 X/Y/Z 입력 구간과 A3/A4 출력 구간을 편집하세요.")

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

            def edit_axis(title_key: str, axis_key: str):
                st.write(f"**Rz = {title_key}° — {axis_key}**")
                cols = st.columns(4)
                PAX = st.session_state.mapping_preset[title_key].get(axis_key, {})
                if "in" not in PAX:
                    PAX["in"] = [0.0, 0.0]
                in0 = cols[0].number_input(f"{axis_key}.in[0]", value=float(PAX["in"][0]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_in0")
                in1 = cols[1].number_input(f"{axis_key}.in[1]", value=float(PAX["in"][1]), step=50.0, format="%.1f", key=f"{title_key}_{axis_key}_in1")
                PAX["in"] = [float(in0), float(in1)]

                if axis_key == "X":
                    cols2 = st.columns(2)
                    if "A4_out" in PAX:
                        a4_0 = cols2[0].number_input("A4_out[0] (X)", value=float(PAX.get("A4_out", [0.0,0.0])[0]), step=50.0, format="%.1f", key=f"{title_key}_X_a40")
                        a4_1 = cols2[1].number_input("A4_out[1] (X)", value=float(PAX.get("A4_out", [0.0,0.0])[1]), step=50.0, format="%.1f", key=f"{title_key}_X_a41")
                        PAX["A4_out"] = [float(a4_0), float(a4_1)]
                    else:
                        cols2[0].info("A4_out(X) 미사용 프리셋")

                elif axis_key == "Y":
                    cols2 = st.columns(2)
                    if "A4_out" in PAX:
                        a4_0 = cols2[0].number_input("A4_out[0] (Y)", value=float(PAX.get("A4_out", [0.0,0.0])[0]), step=50.0, format="%.1f", key=f"{title_key}_Y_a40")
                        a4_1 = cols2[1].number_input("A4_out[1] (Y)", value=float(PAX.get("A4_out", [0.0,0.0])[1]), step=50.0, format="%.1f", key=f"{title_key}_Y_a41")
                        PAX["A4_out"] = [float(a4_0), float(a4_1)]
                    else:
                        cols2[0].info("A4_out(Y) 미사용 프리셋")

                else:
                    a3_0 = cols[2].number_input("A3_out[0]", value=float(PAX.get("A3_out", [0.0,0.0])[0]), step=50.0, format="%.1f", key=f"{title_key}_Z_a30")
                    a3_1 = cols[3].number_input("A3_out[1]", value=float(PAX.get("A3_out", [0.0,0.0])[1]), step=50.0, format="%.1f", key=f"{title_key}_Z_a31")
                    PAX["A3_out"] = [float(a3_0), float(a3_1)]

                st.session_state.mapping_preset[title_key][axis_key] = PAX

            for key_title in ["0", "90", "-90"]:
                st.markdown(f"---\n**Rz = {key_title}°**")
                edit_axis(key_title, "X")
                edit_axis(key_title, "Y")
                edit_axis(key_title, "Z")

            preset_json = json.dumps(st.session_state.mapping_preset, ensure_ascii=False, indent=2)
            st.download_button("Save preset JSON", preset_json, file_name="mapping_preset.json", mime="application/json", use_container_width=True)

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
                enable_a1_const=bool(st.session_state.ext_const_enable_a1),
                enable_a2_const=bool(st.session_state.ext_const_enable_a2),
                x_min=float(st.session_state.ext_const_xmin),
                x_max=float(st.session_state.ext_const_xmax),
                a1_at_xmin=float(st.session_state.ext_const_a1_at_xmin),
                a1_at_xmax=float(st.session_state.ext_const_a1_at_xmax),
                y_min=float(st.session_state.ext_const_ymin),
                y_max=float(st.session_state.ext_const_ymax),
                a2_at_ymin=float(st.session_state.ext_const_a2_at_ymin),
                a2_at_ymax=float(st.session_state.ext_const_a2_at_ymax),
                speed_mm_s=float(st.session_state.ext_const_speed_mm_s),
                boundary_eps_mm=float(st.session_state.ext_const_eps_mm),
                apply_print_only=bool(st.session_state.ext_const_apply_print_only),
                travel_interp=bool(st.session_state.ext_const_travel_interp),
            )
            st.sidebar.success(f"Rapid(*.MODX) 변환 완료 (Rz={st.session_state.rapid_rz:.2f}°)")

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

    if segments is None or total_segments == 0:
        st.info("슬라이싱 후 진행 슬라이더가 나타납니다.")
    else:
        default_val = int(clamp(st.session_state.paths_scrub, 0, total_segments))

        scrub = st.slider("진행(segments)", 0, int(total_segments), int(default_val), 1,
                          help="해당 세그먼트까지 누적 표시")
        scrub_num = st.number_input("행 번호", 0, int(total_segments),
                                    int(default_val), 1,
                                    help="표시할 최종 세그먼트(행) 번호")

        target = default_val
        if scrub != default_val:
            target = int(scrub)
        if scrub_num != default_val:
            target = int(scrub_num)

        target = int(clamp(target, 0, total_segments))
        st.session_state.paths_scrub = target

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
            if is_extruding:
                total_len += float(np.linalg.norm(p2[:2] - p1[:2]))
        st.markdown(f"**누적 레이어 총 길이:** {total_len/1000:.3f} m")
else:
    if "paths_anim_buf" in st.session_state:
        st.session_state.paths_anim_buf["off_l"] = {"x": [], "y": [], "z": []}
        st.session_state.paths_anim_buf["off_r"] = {"x": [], "y": [], "z": []}
        st.session_state.paths_anim_buf["caps"]  = {"x": [], "y": [], "z": []}

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

if not KEY_OK:
    st.warning("유효한 Access Key를 입력해야 프로그램이 작동합니다. (업로드/슬라이싱/G-code 버튼 비활성화)")
