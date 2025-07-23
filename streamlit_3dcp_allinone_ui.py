
import streamlit as st
import numpy as np
import os
import tempfile
import re
import chardet
import plotly.graph_objects as go
import trimesh
from stl import mesh as stlmesh

st.set_page_config(layout="wide")
st.title("🏗️ 3DCP 통합 플랫폼: STL 변환 → G-code 생성 → 시각화")

# ===================== STL BACKEND =====================
def load_stl(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        path = tmp.name
    return stlmesh.Mesh.from_file(path)

def apply_transform(stl_mesh, axis, angle_deg, dx, dy, dz):
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.identity(3)
    if axis == "X":
        rot_matrix = np.array([[1, 0, 0],
                               [0, np.cos(angle_rad), -np.sin(angle_rad)],
                               [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == "Y":
        rot_matrix = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                               [0, 1, 0],
                               [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == "Z":
        rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                               [np.sin(angle_rad), np.cos(angle_rad), 0],
                               [0, 0, 1]])
    stl_mesh.vectors = np.dot(stl_mesh.vectors, rot_matrix.T)
    stl_mesh.x += dx
    stl_mesh.y += dy
    stl_mesh.z += dz
    return stl_mesh

def apply_scale(stl_mesh, axis, target_length):
    vectors = stl_mesh.vectors
    axis_index = "XYZ".index(axis)
    min_val = np.min(vectors[:, :, axis_index])
    max_val = np.max(vectors[:, :, axis_index])
    current_length = max_val - min_val
    if current_length == 0:
        return stl_mesh
    scale_factor = target_length / current_length
    stl_mesh.vectors *= scale_factor
    return stl_mesh

def render_mesh(stl_mesh):
    x, y, z = [], [], []
    I, J, K = [], [], []
    idx = 0
    for vec in stl_mesh.vectors:
        for vertex in vec:
            x.append(vertex[0])
            y.append(vertex[1])
            z.append(vertex[2])
        I.append(idx)
        J.append(idx + 1)
        K.append(idx + 2)
        idx += 3

    mesh3d = go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color='lightblue',
        opacity=0.5
    )

    fig = go.Figure(data=[mesh3d])
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=30, b=0))
    return fig

def save_stl_bytes(stl_mesh):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        stl_mesh.save(tmp.name)
        tmp.flush()
        path = tmp.name
    with open(path, "rb") as f:
        return f.read()

# ===================== G-CODE GENERATOR =====================
def trim_segment_end(segment, trim_distance=30.0):
    segment = np.array(segment)
    total_len = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
    if total_len <= trim_distance:
        return segment
    trimmed = [segment[0]]
    acc = 0.0
    for i in range(1, len(segment)):
        p1, p2 = segment[i - 1], segment[i]
        d = np.linalg.norm(p2 - p1)
        if acc + d >= total_len - trim_distance:
            r = (total_len - trim_distance - acc) / d
            trimmed.append(p1 + (p2 - p1) * r)
            break
        trimmed.append(p2)
        acc += d
    return np.array(trimmed)

def simplify_segment(segment, min_dist):
    simplified = [segment[0]]
    for pt in segment[1:-1]:
        if np.linalg.norm(pt[:2] - simplified[-1][:2]) >= min_dist:
            simplified.append(pt)
    simplified.append(segment[-1])
    return np.array(simplified)

def shift_to_nearest_start(segment, ref_point):
    idx = np.argmin(np.linalg.norm(segment[:, :2] - ref_point, axis=1))
    return np.concatenate([segment[idx:], segment[1:idx + 1]], axis=0), segment[idx]

def generate_gcode(mesh,
                   z_int=30.0,
                   feed=2000,
                   ref_pt_user=(0.0, 0.0),
                   e_on=False,
                   start_e_on=False,
                   start_e_val=0.1,
                   e0_on=False,
                   trim_dist=30.0,
                   min_spacing=3.0,
                   auto_start=False,
                   m30_on=False):
    extrusion_k = 0.05
    g = ["G21", "G90"]
    if e_on:
        g.append("M83")

    z_max = mesh.bounds[1, 2]
    prev_start_xy = None

    for z in np.arange(z_int, int(z_max) + 1, z_int):
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        slice2D, to3D = sec.to_2D()
        segments = []
        for seg in slice2D.discrete:
            seg = np.array(seg)
            seg3d = (to3D @ np.hstack([seg, np.zeros((len(seg), 1)), np.ones((len(seg), 1))]).T).T[:, :3]
            segments.append(seg3d)
        if not segments:
            continue

        if auto_start and prev_start_xy is not None:
            dists = [np.linalg.norm(s[0][:2] - prev_start_xy) for s in segments]
            first_idx = int(np.argmin(dists))
            segments = segments[first_idx:] + segments[:first_idx]
            ref_pt_layer = prev_start_xy
        else:
            ref_pt_layer = np.array(ref_pt_user)

        for i_seg, seg3d in enumerate(segments):
            shifted, _ = shift_to_nearest_start(seg3d, ref_pt_layer)
            trimmed = trim_segment_end(shifted, trim_dist)
            simplified = simplify_segment(trimmed, min_spacing)
            start = simplified[0]

            g.append(f"G01 F{feed}")
            if start_e_on:
                g.append(f"G01 X{start[0]:.3f} Y{start[1]:.3f} Z{z:.3f} E{start_e_val:.5f}")
            else:
                g.append(f"G01 X{start[0]:.3f} Y{start[1]:.3f} Z{z:.3f}")

            for p1, p2 in zip(simplified[:-1], simplified[1:]):
                dist = np.linalg.norm(p2[:2] - p1[:2])
                if e_on:
                    g.append(f"G01 X{p2[0]:.3f} Y{p2[1]:.3f} E{dist * extrusion_k:.5f}")
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

# ===================== UI =====================
uploaded = st.file_uploader("📂 STL 파일 업로드", type=["stl"])
if uploaded:
    stl_mesh = load_stl(uploaded.read())
    st.success("✅ STL 파일 로드 완료")

    with st.expander("🌀 회전 / 이동"):
        axis = st.selectbox("회전축", ["X", "Y", "Z"])
        angle = st.number_input("회전 각도", 0.0)
        dx = st.number_input("X 이동", 0.0)
        dy = st.number_input("Y 이동", 0.0)
        dz = st.number_input("Z 이동", 0.0)
        if st.button("적용 (Transform)"):
            stl_mesh = apply_transform(stl_mesh, axis, angle, dx, dy, dz)

    with st.expander("📏 스케일 조정"):
        scale_axis = st.selectbox("스케일 기준 축", ["X", "Y", "Z"])
        target_length = st.number_input("최종 길이", 100.0)
        if st.button("적용 (Scale)"):
            stl_mesh = apply_scale(stl_mesh, scale_axis, target_length)

    st.plotly_chart(render_mesh(stl_mesh), use_container_width=True)

    with st.expander("🧠 G-code 설정"):
        z_int = st.number_input("Z 간격", 1.0, 1000.0, 30.0)
        feed = st.number_input("Feedrate", 1, 100000, 2000)
        ref_x = st.number_input("기준점 X", value=0.0)
        ref_y = st.number_input("기준점 Y", value=0.0)
        e_on = st.checkbox("E 값 사용")
        start_e_on = st.checkbox("시작점 E 사용", disabled=not e_on)
        start_e_val = st.number_input("시작 E 값", value=0.1, disabled=not (e_on and start_e_on))
        e0_on = st.checkbox("E0 추가", value=False, disabled=not e_on)
        trim_dist = st.number_input("트리밍 거리", 0.0, 1000.0, 30.0)
        min_spacing = st.number_input("최소 점 간격", 0.0, 1000.0, 3.0)
        auto_start = st.checkbox("이전 레이어 시작점에 가깝게 시작")
        m30_on = st.checkbox("M30 명령 추가")

if st.button("🚀 G-code 생성"):
    stl_bytes = save_stl_bytes(stl_mesh)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(stl_bytes)
        tmp.flush()

        # ✅ Trimesh 로드 및 검증
        try:
            mesh = trimesh.load_mesh(tmp.name)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump().sum()
            if not isinstance(mesh, trimesh.Trimesh) or mesh.bounds is None:
                raise ValueError("유효하지 않은 Trimesh 객체입니다.")
        except Exception as e:
            st.error(f"❌ STL 메쉬 로드 중 오류 발생: {e}")
            st.stop()
        gcode = generate_gcode(
            mesh,
            z_int=z_int,
            feed=feed,
            ref_pt_user=(ref_x, ref_y),
            e_on=e_on,
            start_e_on=start_e_on,
            start_e_val=start_e_val,
            e0_on=e0_on,
            trim_dist=trim_dist,
            min_spacing=min_spacing,
            auto_start=auto_start,
            m30_on=m30_on
        )
        st.download_button("💾 G-code 다운로드", gcode, file_name="output.gcode", mime="text/plain")
