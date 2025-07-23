
import streamlit as st
import numpy as np
import tempfile
import trimesh
from stl import mesh as stl_mesh
from streamlit_gcodeviewer import show_gcode_viewer

def generate_gcode(mesh, nozzle_diameter, layer_height, e_on, trim_distance, min_distance, start_near_previous, m30_on):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]
    # Dummy G-code generator for placeholder
    return f"; G-code generated from {z_min:.2f} to {z_max:.2f}"

def parse_gcode_from_text(gcode_text):
    coords = []
    is_extrudes = []
    f_value = 0
    for line in gcode_text.splitlines():
        if line.startswith("G1"):
            parts = line.split()
            x = y = z = None
            e = 0
            for p in parts:
                if p.startswith("X"):
                    x = float(p[1:])
                elif p.startswith("Y"):
                    y = float(p[1:])
                elif p.startswith("Z"):
                    z = float(p[1:])
                elif p.startswith("E"):
                    e = float(p[1:])
            if x is not None and y is not None and z is not None:
                coords.append([x, y, z])
                is_extrudes.append(e > 0)
    return np.array(coords), is_extrudes, f_value

st.title("3DCP 통합 툴")
uploaded_stl = st.file_uploader("STL 파일 업로드", type=["stl"])
if uploaded_stl:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(uploaded_stl.read())
        mesh = trimesh.load_mesh(tmp.name)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

    nozzle_diameter = st.number_input("노즐 지름", value=30.0)
    layer_height = st.number_input("레이어 높이", value=15.0)
    e_on = st.checkbox("E0 추가", value=True)
    trim_distance = st.number_input("트리밍 거리", value=30.0)
    min_distance = st.number_input("최소 점 간격", value=3.0)
    start_near_previous = st.checkbox("이전 레이어 시작점에 가깝게 시작", value=True)
    m30_on = st.checkbox("M30 명령 추가", value=True)

    if st.button("🚀 G-code 생성"):
        gcode = generate_gcode(mesh, nozzle_diameter, layer_height, e_on, trim_distance, min_distance, start_near_previous, m30_on)
        st.download_button("💾 G-code 다운로드", gcode, file_name="output.gcode", mime="text/plain")

        coords, is_extrudes, f_value = parse_gcode_from_text(gcode)
        show_gcode_viewer(coords, is_extrudes, f_value)
