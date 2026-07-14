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
    opacity=0.15, color='#00BFFF', name='AI Blob Surface', hoverinfo='none'       
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

# Left Hand Micro-Web
left_fingers = [
    ('wrist_L', 'thumb_cmc_L'), ('thumb_cmc_L', 'thumb_mcp_L'), ('thumb_mcp_L', 'thumb_ip_L'), ('thumb_ip_L', 'thumb_tip_L'),
    ('wrist_L', 'index_mcp_L'), ('index_mcp_L', 'index_pip_L'), ('index_pip_L', 'index_dip_L'), ('index_dip_L', 'index_tip_L'),
    ('wrist_L', 'middle_mcp_L'), ('middle_mcp_L', 'middle_pip_L'), ('middle_pip_L', 'middle_dip_L'), ('middle_dip_L', 'middle_tip_L'),
    ('wrist_L', 'ring_mcp_L'), ('ring_mcp_L', 'ring_pip_L'), ('ring_pip_L', 'ring_dip_L'), ('ring_dip_L', 'ring_tip_L'),
    ('wrist_L', 'pinky_mcp_L'), ('pinky_mcp_L', 'pinky_pip_L'), ('pinky_pip_L', 'pinky_dip_L'), ('pinky_dip_L', 'pinky_tip_L')
]

# Right Hand Micro-Web
right_fingers = [
    ('wrist_R', 'thumb_cmc_R'), ('thumb_cmc_R', 'thumb_mcp_R'), ('thumb_mcp_R', 'thumb_ip_R'), ('thumb_ip_R', 'thumb_tip_R'),
    ('wrist_R', 'index_mcp_R'), ('index_mcp_R', 'index_pip_R'), ('index_pip_R', 'index_dip_R'), ('index_dip_R', 'index_tip_R'),
    ('wrist_R', 'middle_mcp_R'), ('middle_mcp_R', 'middle_pip_R'), ('middle_pip_R', 'middle_dip_R'), ('middle_dip_R', 'middle_tip_R'),
    ('wrist_R', 'ring_mcp_R'), ('ring_mcp_R', 'ring_pip_R'), ('ring_pip_R', 'ring_dip_R'), ('ring_dip_R', 'ring_tip_R'),
    ('wrist_R', 'pinky_mcp_R'), ('pinky_mcp_R', 'pinky_pip_R'), ('pinky_pip_R', 'pinky_dip_R'), ('pinky_dip_R', 'pinky_tip_R')
]

bone_connections.extend(left_fingers)
bone_connections.extend(right_fingers)

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