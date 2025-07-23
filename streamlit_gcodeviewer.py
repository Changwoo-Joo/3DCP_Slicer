
import numpy as np
import streamlit as st
import plotly.graph_objects as go

def show_gcode_viewer(coords, is_extrudes, f_value):
    if coords is None or len(coords) == 0:
        st.warning("G-code 좌표가 없습니다.")
        return

    fig = go.Figure()

    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    z_vals = coords[:, 2]

    color = ["red" if e else "blue" for e in is_extrudes]

    fig.add_trace(go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines+markers',
        marker=dict(size=2, color=color),
        line=dict(color='gray', width=2),
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title="G-code 경로 시각화",
        width=800,
        height=600
    )

    st.plotly_chart(fig)
