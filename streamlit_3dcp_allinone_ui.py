
import streamlit as st
import tempfile
import numpy as np
import trimesh
import re
import plotly.graph_objects as go

def parse_gcode_from_text(gcode_text):
    coords = []
    is_extrudes = []
    f_value = 0
    last_pos = {'X': None, 'Y': None, 'Z': None}
    for line in gcode_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        x = re.search(r'[Xx]([-+]?[0-9]*\.?[0-9]+)', line)
        y = re.search(r'[Yy]([-+]?[0-9]*\.?[0-9]+)', line)
        z = re.search(r'[Zz]([-+]?[0-9]*\.?[0-9]+)', line)
        e = re.search(r'[Ee]([-+]?[0-9]*\.?[0-9]+)', line)
        f = re.search(r'[Ff]([-+]?[0-9]*\.?[0-9]+)', line)
        if x:
            last_pos['X'] = float(x.group(1))
        if y:
            last_pos['Y'] = float(y.group(1))
        if z:
            last_pos['Z'] = float(z.group(1))
        if f:
            f_value = float(f.group(1))
        if None not in last_pos.values():
            coords.append([last_pos['X'], last_pos['Y'], last_pos['Z']])
            is_extrudes.append(float(e.group(1)) > 0 if e else False)
    return np.array(coords), is_extrudes, f_value

def show_gcode_viewer(gcode_text, z_height=9999):
    coords, is_extrudes, _ = parse_gcode_from_text(gcode_text)
    fig = go.Figure()
    for i in range(1, len(coords)):
        if coords[i][2] > z_height or coords[i-1][2] > z_height:
            continue
        x, y, z = zip(coords[i-1], coords[i])
        color = 'red' if is_extrudes[i] else 'gray'
        width = 4 if is_extrudes[i] else 2
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=color, width=width),
            showlegend=False
        ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=700
    )
    st.plotly_chart(fig)

# --- Streamlit App Entry ---
st.title("3DCP 통합 툴")
uploaded_gcode = st.file_uploader("G-code 파일 업로드", type=["gcode"])
if uploaded_gcode is not None:
    gcode_str = uploaded_gcode.read().decode("utf-8")
    st.text_area("G-code 내용", gcode_str, height=200)
    z_max = st.slider("Z 최대 시각화 높이", 0, 1000, 999)
    show_gcode_viewer(gcode_str, z_height=z_max)
