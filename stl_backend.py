import numpy as np
from stl import mesh
import io
import tempfile
import plotly.graph_objects as go

def load_stl(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        path = tmp.name
    return mesh.Mesh.from_file(path)

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
        return stl_mesh  # 스케일 불가
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

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)

    xlen = xmax - xmin
    ylen = ymax - ymin
    zlen = zmax - zmin

    title_text = (
        f"X: {xmin:.2f} ~ {xmax:.2f} ({xlen:.2f}mm), "
        f"Y: {ymin:.2f} ~ {ymax:.2f} ({ylen:.2f}mm), "
        f"Z: {zmin:.2f} ~ {zmax:.2f} ({zlen:.2f}mm)"
    )

    mesh3d = go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color='lightblue',
        opacity=0.5,
        name='STL Model'
    )

    fig = go.Figure(data=[mesh3d])
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig

def save_stl_bytes(stl_mesh):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        stl_mesh.save(tmp.name)
        tmp.flush()
        path = tmp.name

    with open(path, "rb") as f:
        data = f.read()

    return io.BytesIO(data)
