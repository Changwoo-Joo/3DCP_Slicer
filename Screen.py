from vedo import Plotter, Line, settings, Text2D, Axes
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog

settings.default_backend = "vtk"

def parse_gcode(file_path):
    coords = []
    is_extrudes = []
    f_value = 0
    last_pos = {'X': None, 'Y': None, 'Z': None}

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp949', errors='ignore') as file:
            lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        x = re.search(r'[Xx]([-+]?[0-9]*\.?[0-9]+)', line)
        y = re.search(r'[Yy]([-+]?[0-9]*\.?[0-9]+)', line)
        z = re.search(r'[Zz]([-+]?[0-9]*\.?[0-9]+)', line)
        e = re.search(r'[Ee]([-+]?[0-9]*\.?[0-9]+)', line)
        f = re.search(r'[Ff]([-+]?[0-9]*\.?[0-9]+)', line)
        if x: last_pos['X'] = float(x.group(1))
        if y: last_pos['Y'] = float(y.group(1))
        if z: last_pos['Z'] = float(z.group(1))
        if f: f_value = float(f.group(1))
        if None not in last_pos.values():
            coords.append([last_pos['X'], last_pos['Y'], last_pos['Z']])
            is_extrudes.append(float(e.group(1)) > 0 if e else False)
    return np.array(coords), is_extrudes, f_value

def compute_total_distance(coords):
    diffs = np.diff(coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

def show_gcode_3d(coords, is_extrudes, f_value):
    vp = Plotter(title="G-code Viewer", axes=0, bg='white', size=(1400, 1000))
    all_lines = []

    for i in range(1, len(coords)):
        p1, p2 = coords[i - 1], coords[i]
        color = "blue" if is_extrudes[i] else "gray"
        lw = 2 if is_extrudes[i] else 1
        alpha = 1 if is_extrudes[i] else 0.4
        line = Line(p1, p2, lw=lw, c=color, alpha=alpha)
        line.off()
        all_lines.append(line)

    xmin, ymin, zmin = np.min(coords, axis=0)
    xmax, ymax, zmax = np.max(coords, axis=0)

    total_distance = compute_total_distance(coords)
    travel_time = total_distance / f_value if f_value > 0 else 0

    axes = Axes(xrange=(xmin, xmax), yrange=(ymin, ymax), zrange=(zmin, zmax))

    info_text = (
        f"Total segments: {len(all_lines)}\n"
        f"Blue = E>0 (extrusion)\nGray = travel\n"
        f"Total distance: {total_distance:.2f} mm\n"
        f"F value: {f_value:.1f} mm/min\n"
        f"Estimated time: {travel_time:.2f} min"
    )
    info = Text2D(info_text, pos='top-left', c='black', bg='lightyellow')
    max_label = Text2D(f"X={xmax:.1f}, Y={ymax:.1f}, Z={zmax:.1f}", pos='top-right', c='black', bg='white', font='Calco')

    def update_scene(fraction):
        n = int(fraction * len(all_lines))
        for i, line in enumerate(all_lines):
            if i < n:
                line.on()
            else:
                line.off()
        vp.render()

    def slider_callback(widget, event):
        update_scene(widget.value)

    vp.show(*all_lines, axes, info, max_label, resetcam=True, interactive=False)

    vp.add_slider(
        slider_callback,
        xmin=0,
        xmax=1,
        value=1,
        pos=[(0.25, 0.05), (0.75, 0.05)],
        title="G-code Progress",
        show_value=False
    )

    update_scene(1)
    vp.interactive().close()

def select_and_view_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("G-code files", "*.gcode *.nc"), ("All files", "*.*")])
    if not file_path:
        print("파일을 선택하지 않았습니다.")
        return
    coords, is_extrudes, f_value = parse_gcode(file_path)
    if len(coords) < 2:
        print("G-code 내 유효 좌표가 부족합니다.")
        return
    show_gcode_3d(coords, is_extrudes, f_value)

if __name__ == "__main__":
    select_and_view_file()
