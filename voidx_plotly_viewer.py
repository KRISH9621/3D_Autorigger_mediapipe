from builtins import print

import open3d as o3d
import json
import plotly.graph_objects as go
import numpy as np

print("VOIDX ENGINE: Generating Inline HTML 3D Viewer...")

# 1. Load the AI Blob using Open3D
try:
    mesh = o3d.io.read_triangle_mesh('male_t_pose.glb')
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
except Exception as e:
    print(f"Failed to load mesh: {e}")
    exit()

# 2. Load the DNA
try:
    with open('VOIDX_ULTIMATE_DNA.json', 'r') as f:
        full_dna = json.load(f)
        anchors = full_dna['pose']
except Exception as e:
    print(f"Failed to load JSON: {e}")
    exit()

fig = go.Figure()

fig.add_trace(go.Mesh3d(
    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    opacity=0.35, color='#00BFFF', name='AI Blob Surface', hoverinfo='none'       
))

joint_names = []
jx, jy, jz = [], [], []

for name, pos in anchors.items():
    joint_names.append(name)
    jx.append(pos['x']); jy.append(pos['y']); jz.append(pos['z'])

fig.add_trace(go.Scatter3d(
    x=jx, y=jy, z=jz,
    mode='markers+text', text=joint_names, textposition="top center",
    marker=dict(size=3, color='red', symbol='circle'), name='Joints'
))


 # --- THE ULTIMATE WIREFRAME ---
bone_connections = [
    # Core Body ONLY (No Face)
    ('pelvis', 'hip_L'), ('pelvis', 'hip_R'), ('hip_L', 'knee_L'), ('hip_R', 'knee_R'), 
    ('knee_L', 'ankle_L'), ('knee_R', 'ankle_R'), ('pelvis', 'neck'), ('neck', 'shoulder_L'), ('neck', 'shoulder_R'),
    ('shoulder_L', 'elbow_L'), ('shoulder_R', 'elbow_R'), ('elbow_L', 'wrist_L'), ('elbow_R', 'wrist_R'),
    ('ankle_L', 'heel_L'), ('ankle_L', 'foot_index_L'), ('heel_L', 'foot_index_L'),
    ('ankle_R', 'heel_R'), ('ankle_R', 'foot_index_R'), ('heel_R', 'foot_index_R')
]


for joint_a, joint_b in bone_connections:
    if joint_a in anchors and joint_b in anchors:
        fig.add_trace(go.Scatter3d(
            x=[anchors[joint_a]['x'], anchors[joint_b]['x']],
            y=[anchors[joint_a]['y'], anchors[joint_b]['y']],
            z=[anchors[joint_a]['z'], anchors[joint_b]['z']],
            mode='lines', line=dict(color='#32CD32', width=4), 
            name=f'Bone: {joint_a}-{joint_b}', hoverinfo='none'
        ))

fig.update_layout(
    title='VoidX Telemetry: Unified Inspection',
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=40), template="plotly_dark"         
)
fig.write_html("debug_rig_viewer.html", include_plotlyjs="cdn")
print("SUCCESS: debug_rig_viewer.html is generated.")