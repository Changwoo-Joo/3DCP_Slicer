import streamlit as st
from stl_backend import (
    load_stl,
    apply_transform,
    apply_scale,
    render_mesh,
    save_stl_bytes,
)

st.set_page_config(page_title="STL Viewer & Transformer", layout="wide")
st.title("STL Viewer & Transformer (Streamlit Cloud Ver.)")

uploaded = st.file_uploader("Upload STL file", type=["stl"])

# 세션 상태 초기화
if 'mesh' not in st.session_state:
    st.session_state.mesh = None
if 'updated' not in st.session_state:
    st.session_state.updated = False
if 'last_fig' not in st.session_state:
    st.session_state.last_fig = None

# 파일 업로드 처리
if uploaded is not None:
    st.session_state.mesh = load_stl(uploaded.read())
    st.session_state.updated = True

# 변환 메뉴
if st.session_state.mesh is not None:
    st.subheader("🌀 Transform (Rotation & Translation)")
    axis = st.selectbox("Rotation Axis", ["X", "Y", "Z"])
    angle = st.number_input("Rotation Angle (degrees)", value=0.0)
    dx = st.number_input("Shift X", value=0.0)
    dy = st.number_input("Shift Y", value=0.0)
    dz = st.number_input("Shift Z", value=0.0)

    if st.button("Apply Transform"):
        st.session_state.mesh = apply_transform(st.session_state.mesh, axis, angle, dx, dy, dz)
        st.session_state.updated = True

    st.subheader("📏 Axis-Based Scale")
    scale_axis = st.selectbox("Scale 기준 축", ["X", "Y", "Z"])
    target_length = st.number_input("해당 축의 최종 길이 (mm)", value=100.0)

    if st.button("Apply Axis-Based Scaling"):
        st.session_state.mesh = apply_scale(st.session_state.mesh, scale_axis, target_length)
        st.session_state.updated = True

    st.subheader("📊 Preview")
    if st.session_state.updated:
        fig = render_mesh(st.session_state.mesh)
        st.session_state.last_fig = fig
        st.session_state.updated = False
    else:
        fig = st.session_state.last_fig

    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "📥 Download Transformed STL",
        data=save_stl_bytes(st.session_state.mesh),
        file_name="transformed.stl",
        mime="application/sla"
    )
