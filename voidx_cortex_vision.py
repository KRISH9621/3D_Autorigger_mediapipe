import open3d as o3d
import cv2
import mediapipe as mp
import numpy as np
import json
import math
import os

print("3D_Autorigger_mediapipe: Igniting Semantic Vision Matrix...")

# 1. Initialize the MediaPipe ...
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True, 
    model_complexity=2, 
    refine_face_landmarks=True,
    min_detection_confidence=0.5
)

# 2. Load the 3D model
mesh = o3d.io.read_triangle_mesh("male_t_pose.glb")
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.9, 0.8, 0.7]) # Skin tone 

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=1024, height=1024)
vis.add_geometry(mesh)

opt = vis.get_render_option()
opt.background_color = np.asarray([0.0, 0.0, 0.0])

# --- THE ORTHOGRAPHIC CAMERA ---
ctr = vis.get_view_control()
bbox = mesh.get_axis_aligned_bounding_box()
mesh_center = bbox.get_center()

# 1. Frame the mesh properly
ctr.set_lookat(mesh_center)
ctr.set_front([0, 0, -1])
ctr.set_up([0, 1, 0])
ctr.set_zoom(0.8)

# 2. Extract camera parameters
params = ctr.convert_to_pinhole_camera_parameters()
fx = params.intrinsic.intrinsic_matrix[0, 0]
fy = params.intrinsic.intrinsic_matrix[1, 1]

# 3. Scale Focal Length AND Distance by 20x 
# This mathematically flattens the perspective cone into a parallel cylinder
scale_factor = 20.0
params.intrinsic.set_intrinsics(1024, 1024, fx * scale_factor, fy * scale_factor, 512.0, 512.0)
    # --- FIX START ---
    # 1. Creating a writable copy of the matrix
new_extrinsic = params.extrinsic.copy()
    
    # 2. Modifying the copy (scale the translation components)
new_extrinsic[0, 3] *= scale_factor
new_extrinsic[1, 3] *= scale_factor
new_extrinsic[2, 3] *= scale_factor
    
    # 3. Update the parameter object with the new matrix
params.extrinsic = new_extrinsic
    # --- FIX END ---
ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
# ------------------------------------

# ========================================================
# 3. THE 3D_Autorigger_mediapipe SEMANTIC JURISDICTION ZONES (CORE BODY ONLY)
# ========================================================
SEMANTIC_ZONES = {
    # FRONT-FACING BONES (0° +/- 45°)
    "nose": [315, 0, 45], 
    "shoulder_L": [315, 0, 45], "shoulder_R": [315, 0, 45],
    "hip_L": [315, 0, 45], "hip_R": [315, 0, 45],
    "knee_L": [315, 0, 45], "knee_R": [315, 0, 45],

    # RIGHT-SIDE BONES (Include 0 and 180 for X-Axis calculation)
    "wrist_R": [0, 45, 90, 135, 180], 
    "elbow_R": [0, 45, 90, 135, 180], 
    "ankle_R": [0, 45, 90, 135, 180],

    # LEFT-SIDE BONES (Include 0 and 180 for X-Axis calculation)
    "wrist_L": [0, 180, 225, 270, 315], 
    "elbow_L": [0, 180, 225, 270, 315], 
    "ankle_L": [0, 180, 225, 270, 315]
}

# WE ONLY TRACK THE CORE 15 JOINTS. NO FACE, NO FINGERS, NO TOES.
# This prevents hallucinated extremities from corrupting the global scale_factor.
joint_names = [
    "nose", None, None, None, None, None, None,             # 0-6 (Face ignored)
    None, None, None, None,                                 # 7-10 (Ears/Mouth ignored)
    "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",       # 11-14
    "wrist_L", "wrist_R",                                   # 15-16
    None, None, None, None, None, None,                     # 17-22 (Fingers ignored)
    "hip_L", "hip_R", "knee_L", "knee_R",                   # 23-26
    "ankle_L", "ankle_R",                                   # 27-28
    None, None, None, None                                  # 29-32 (Heel/Toes ignored)
]
fusion_vault = {} 
angles = [0, 45, 90, 135, 180, 225, 270, 315]

print("Scanning via Semantic Jurisdictions (Orthographic Mode)...")

for angle in angles:
    mesh_copy = mesh.normalize_normals()
    R = mesh_copy.get_rotation_matrix_from_xyz((0, math.radians(angle), 0))
    mesh_copy.rotate(R, center=(0,0,0))
    
    vis.clear_geometries()
    vis.add_geometry(mesh_copy)
    vis.poll_events()
    vis.update_renderer()
    

    temp_img_path = f"temp_cortex_{angle}.png"
    vis.capture_screen_image(temp_img_path)
    
    # Save the front view for the hand engine
    if angle == 0:
        vis.capture_screen_image("DEBUG_PERFECT_FRONT.png")
        
    image = cv2.imread(temp_img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx >= len(joint_names): continue
            
            bone_name = joint_names[idx]
            
            # Skip the None values (face, fingers, toes)
            if bone_name is None: continue 
            
            allowed_angles = SEMANTIC_ZONES.get(bone_name, angles)
            if angle not in allowed_angles:
                continue

            screen_x = landmark.x - 0.5
            screen_y = -(landmark.y - 0.5) 
            
            if bone_name not in fusion_vault:
                fusion_vault[bone_name] = []
                
            if landmark.visibility > 0.1:
                fusion_vault[bone_name].append({
                    "angle": angle, "x": screen_x, "y": screen_y, "v": landmark.visibility
                })
                
    os.remove(temp_img_path)

vis.destroy_window()

# ========================================================
# 4. DECOUPLED AXIS PROJECTION (The Mathematical Fix)
# ========================================================
print("Triangulating via Decoupled Axis Projection (with Oblique Z-Extraction)...")
final_dna = {"pose": {}}

for bone_name, observations in fusion_vault.items():
    if not observations: continue
    
    y_values = []
    x_values = []
    
    # Step A: Extract absolute X and Y from front-facing/reliable angles first
    for obs in observations:
        if obs['angle'] in [0, 360]:
            x_values.append(obs['x'])
        y_values.append(obs['y'])
        
    true_x = sum(x_values) / len(x_values) if x_values else None
    true_y = sum(y_values) / len(y_values) if y_values else None
    
    # Step B: Extract Z using trigonometric inversion across ALL non-parallel angles
    z_values = []
    for obs in observations:
        angle = obs['angle']
        screen_x = obs['x']
        
        # Exclude 0 and 180 because sin(0) = 0 (causes division by zero)
        if angle not in [0, 180, 360] and true_x is not None:
            rad = math.radians(angle)
            # screen_x = X*cos(angle) + Z*sin(angle)  -->  Z = (screen_x - X*cos(angle)) / sin(angle)
            extracted_z = (screen_x - (true_x * math.cos(rad))) / math.sin(rad)
            z_values.append(extracted_z)

    true_z = sum(z_values) / len(z_values) if z_values else None
    
    final_dna["pose"][bone_name] = {"x": true_x, "y": true_y, "z": true_z}
# ========================================================
# 4.5 KINEMATIC INHERITANCE (Fix for corrupted hands/toes)
# ========================================================
kinematic_parents = {
    "wrist_L": "elbow_L", "wrist_R": "elbow_R",
    "elbow_L": "shoulder_L", "elbow_R": "shoulder_R",
    "foot_index_L": "ankle_L", "foot_index_R": "ankle_R",
    "heel_L": "ankle_L", "heel_R": "ankle_R",
    "ankle_L": "knee_L", "ankle_R": "knee_R",
    "knee_L": "hip_L", "knee_R": "hip_R"
}

# Run 3 times to ensure inheritance cascades down (e.g., shoulder -> elbow -> wrist)
for _ in range(3):
    for bone, parent in kinematic_parents.items():
        if bone in final_dna["pose"] and parent in final_dna["pose"]:
            if final_dna["pose"][bone]["x"] is None:
                final_dna["pose"][bone]["x"] = final_dna["pose"][parent]["x"]
            if final_dna["pose"][bone]["z"] is None:
                final_dna["pose"][bone]["z"] = final_dna["pose"][parent]["z"]

# Replace any remaining None with 0.0 so the JSON doesn't break
for bone in final_dna["pose"]:
    if final_dna["pose"][bone]["x"] is None: final_dna["pose"][bone]["x"] = 0.0
    if final_dna["pose"][bone]["y"] is None: final_dna["pose"][bone]["y"] = 0.0
    if final_dna["pose"][bone]["z"] is None: final_dna["pose"][bone]["z"] = 0.0

# ========================================================
# 5. UNIVERSAL PROPORTION SNAP (Dynamic Y-Interpolation)
# ========================================================
mesh_min = mesh.get_min_bound()
mesh_max = mesh.get_max_bound()
mesh_height = mesh_max[1] - mesh_min[1]

skel_pts = np.array([[pos['x'], pos['y'], pos['z']] for pos in final_dna["pose"].values()])
skel_min = skel_pts.min(axis=0)
skel_max = skel_pts.max(axis=0)

# --- THE CRANIAL OFFSET FIX ---
# The mediapipe's highest point is the nose. The mesh's highest point is the scalp.
# We must mathematically project the top of the AI's head to prevent vertical stretching.
nose_y = final_dna["pose"].get("nose", {}).get("y", skel_max[1])
sh_L_y = final_dna["pose"].get("shoulder_L", {}).get("y", nose_y)
sh_R_y = final_dna["pose"].get("shoulder_R", {}).get("y", nose_y)
mid_shoulder_y = (sh_L_y + sh_R_y) / 2.0

# Project the crown based on standard cranial anatomy
cranial_offset = abs(nose_y - mid_shoulder_y) * 0.6 
true_skel_max_y = skel_max[1] + cranial_offset
true_skel_height = true_skel_max_y - skel_min[1]

# Dynamic Scale: Pure ratio of widths
scale_factor = mesh_height / true_skel_height if true_skel_height > 0 else 1.0

for name in final_dna["pose"]:
    # Scale X and Z normally
    final_dna["pose"][name]['x'] *= scale_factor
    final_dna["pose"][name]['z'] *= scale_factor
    
    # DYNAMIC Y-SNAP: Linear Interpolation using the TRUE skeleton height
    y_from_ai_floor = final_dna["pose"][name]['y'] - skel_min[1]
    y_ratio = y_from_ai_floor / true_skel_height if true_skel_height > 0 else 0
    final_dna["pose"][name]['y'] = (y_ratio * mesh_height) + mesh_min[1]

if "hip_L" in final_dna["pose"] and "hip_R" in final_dna["pose"]:
    final_dna["pose"]["pelvis"] = {
        "x": (final_dna["pose"]["hip_L"]["x"] + final_dna["pose"]["hip_R"]["x"]) / 2,
        "y": (final_dna["pose"]["hip_L"]["y"] + final_dna["pose"]["hip_R"]["y"]) / 2,
        "z": (final_dna["pose"]["hip_L"]["z"] + final_dna["pose"]["hip_R"]["z"]) / 2
    }
if "shoulder_L" in final_dna["pose"] and "shoulder_R" in final_dna["pose"]:
    final_dna["pose"]["neck"] = {
        "x": (final_dna["pose"]["shoulder_L"]["x"] + final_dna["pose"]["shoulder_R"]["x"]) / 2,
        "y": (final_dna["pose"]["shoulder_L"]["y"] + final_dna["pose"]["shoulder_R"]["y"]) / 2,
        "z": (final_dna["pose"]["shoulder_L"]["z"] + final_dna["pose"]["shoulder_R"]["z"]) / 2
    }

with open('VOIDX_CORTEX_DNA.json', 'w') as f:
    json.dump(final_dna, f, indent=4)

print("SUCCESS: 3D_Autorigger_mediapipe's Cortex Processing Completed with Pure Math.")
