
import streamlit as st
import numpy as np
import trimesh
import tempfile
from stl_backend import save_stl_bytes
from streamlit_gcodeviewer import show_gcode_viewer
from streamlit_app import generate_gcode
from streamlit_viewer_app import show_stl_viewer

def parse_gcode_from_text(gcode_text):
    coords = []
    is_extrudes = []
    f_value = 0.0

    for line in gcode_text.splitlines():
        if line.startswith(";") or not line.strip():
            continue
        parts = line.strip().split()
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
            elif p.startswith("F"):
                f_value = float(p[1:])
        if x is not None and y is not None and z is not None:
            coords.append([x, y, z])
            is_extrudes.append(e > 0)
    return np.array(coords), is_extrudes, f_value

st.title("3DCP All-in-One UI")

uploaded_file = st.file_uploader("STL 파일 업로드", type=["stl"])
if uploaded_file:
    file_bytes = uploaded_file.read()
    stl_mesh = trimesh.load_mesh(file_bytes, file_type='stl')
    if isinstance(stl_mesh, trimesh.Scene):
        stl_mesh = stl_mesh.dump().sum()

    st.subheader("🔍 STL 미리보기")
    show_stl_viewer(stl_mesh)

    if st.button("🚀 G-code 생성"):
        stl_bytes = save_stl_bytes(stl_mesh)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(stl_bytes)
            tmp.flush()
            mesh = trimesh.load_mesh(tmp.name)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump().sum()

        gcode = generate_gcode(
            mesh,
            z_layer_height=15.0,
            nozzle_diameter=30.0,
            start_xy=(0, 0),
            trim_distance=30.0,
            min_distance=5.0,
            extrusion_value=0.1,
            start_e_value=0.1,
            m30_on=True
        )

        st.download_button("💾 G-code 다운로드", gcode, file_name="output.gcode", mime="text/plain")

        # G-code 시각화
        coords, is_extrudes, f_value = parse_gcode_from_text(gcode)
        st.subheader("🌀 G-code 경로 시각화")
        show_gcode_viewer(coords, is_extrudes, f_value)
