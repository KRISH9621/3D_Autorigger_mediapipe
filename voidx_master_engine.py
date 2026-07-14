import trimesh
import open3d as o3d
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import math

print("VOIDX ENGINE: Igniting True X-Ray Core Architecture...")

# ========================================================
# PHASE 1: DATA PREPARATION & HUMAN TRUST
# ========================================================
try:
    trimesh_mesh = trimesh.load('male_t_pose.glb', force='mesh')
    with open('VOIDX_CORTEX_DNA.json', 'r') as f:
        dna = json.load(f)
except Exception as e:
    print("CRITICAL: Missing .glb or VOIDX_CORTEX_DNA.json.")
    exit()

manual_path = 'VOIDX_MANUAL_PINS.json'
if os.path.exists(manual_path):
    with open(manual_path, 'r') as f:
        full_pins = json.load(f)["manual_overrides"]
    print(f"Loaded {len(full_pins)} Human Overrides.")
else:
    full_pins = {}

for joint_name, coords in full_pins.items():
    if joint_name in dna["pose"]:
        dna["pose"][joint_name]['x'] = coords['x']
        dna["pose"][joint_name]['y'] = coords['y']

final_dna = {"pose": {}}
laser_origin_z = trimesh_mesh.bounds[0][2] - 50.0 
proximity_engine = trimesh.proximity.ProximityQuery(trimesh_mesh)
# ========================================================
# PHASE 2: MACRO CORE INJECTION (The Z-Anchor Fix)
# ========================================================
macro_bones = ["pelvis", "neck", "shoulder_L", "shoulder_R", "elbow_L", "elbow_R", "wrist_L", "wrist_R", "hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"]
print("\nExecuting Phase 2: Manual X/Y with Z-Anchor...")

# DYNAMIC ANCHOR: Calculate true shoulder width before the loop starts
pre_sh_L = dna["pose"].get("shoulder_L", {"x": 1.0})
pre_sh_R = dna["pose"].get("shoulder_R", {"x": -1.0})
true_shoulder_width = abs(pre_sh_L["x"] - pre_sh_R["x"])
if true_shoulder_width < 0.1: true_shoulder_width = 2.0 # Fallback

# ---> NEW: We define the hierarchy OUTSIDE the loop so it only loads once <---
kinematic_parents = {
    "wrist_L": "elbow_L", "elbow_L": "shoulder_L", "shoulder_L": "neck",
    "wrist_R": "elbow_R", "elbow_R": "shoulder_R", "shoulder_R": "neck",
    "ankle_L": "knee_L", "knee_L": "hip_L", "hip_L": "pelvis",
    "ankle_R": "knee_R", "knee_R": "hip_R", "hip_R": "pelvis",
    "neck": "pelvis"
}

for bone_name in macro_bones:
    
    # THE SURGICAL FIX: Left Elbow & Left Wrist
    if bone_name in ["elbow_L", "wrist_L"] and bone_name in full_pins:
        target_x = full_pins[bone_name]['x']
        target_y = full_pins[bone_name]['y']
        if "shoulder_L" in full_pins:
            safe_z = full_pins["shoulder_L"]['z']
        else:
            safe_z = full_pins[bone_name]['z']
        final_dna["pose"][bone_name] = {"x": float(target_x), "y": float(target_y), "z": float(safe_z)}
        continue

    if bone_name in full_pins:
        final_dna["pose"][bone_name] = {
            "x": float(full_pins[bone_name]['x']),
            "y": float(full_pins[bone_name]['y']),
            "z": float(full_pins[bone_name]['z'])
        }
        continue

    # ---> NEW: The upgraded raycast logic replaces the old block here <---
    if bone_name in dna["pose"]:
        target_x = dna["pose"][bone_name]['x']
        target_y = dna["pose"][bone_name]['y']
        fallback_z = dna["pose"][bone_name]['z']
        
        # --- THE SURFACE INITIALIZER FIX ---
        # AI depth (fallback_z) is an illusion. If we raycast using it, we miss the mesh entirely.
        # We use the proximity engine FIRST to find the actual physical skin of the mesh.
        try:
            closest_init, _, _ = proximity_engine.on_surface(np.array([[target_x, target_y, fallback_z]]))
            if len(closest_init) > 0:
                physical_z = closest_init[0][2] # The true physical depth of the skin
            else:
                physical_z = fallback_z
        except Exception:
            physical_z = fallback_z

        # ========================================================
        # THE GRAVITY DROP: Y-Axis Volumetric Centering for Arms
        # ========================================================
        if "elbow" in bone_name or "wrist" in bone_name:
            laser_origin_y = trimesh_mesh.bounds[1][1] + 50.0
            
            # --- FIX: Y-RAYCAST MICRO-GRID ---
            z_pad = 0.05 * true_shoulder_width 
            
            # We now fire down using the PHYSICAL Z, guaranteeing we hit the arm cylinder
            y_ray_origins = np.array([
                [target_x, laser_origin_y, physical_z],
                [target_x, laser_origin_y, physical_z + z_pad],
                [target_x, laser_origin_y, physical_z - z_pad],
                [target_x, laser_origin_y, physical_z + (z_pad * 2)],
                [target_x, laser_origin_y, physical_z - (z_pad * 2)]
            ])
            y_ray_directions = np.array([[0, -1, 0]] * 5) # Fire straight down
            
            y_locs, y_index_ray, _ = trimesh_mesh.ray.intersects_location(
                ray_origins=y_ray_origins, ray_directions=y_ray_directions
            )
            
            if len(y_locs) >= 2:
                valid_y_centers = []
                for i in range(5):
                    ray_hits = y_locs[y_index_ray == i]
                    if len(ray_hits) >= 2:
                        y_sort = ray_hits[:, 1].argsort()
                        bottom_y = ray_hits[y_sort[0]][1]
                        top_y = ray_hits[y_sort[-1]][1]
                        valid_y_centers.append(bottom_y + (abs(top_y - bottom_y) * 0.5))
                
                if valid_y_centers:
                    # Override the AI's Y with the true mathematical center of the mesh's arm height
                    target_y = sum(valid_y_centers) / len(valid_y_centers)

        # --- FIX 3: MICRO-GRID RAYCAST PADDING ---
        pad = 0.02 * true_shoulder_width # 2% padding based on dynamic body scale
        ray_origins = np.array([
            [target_x, target_y, laser_origin_z],            # Center
            [target_x + pad, target_y, laser_origin_z],      # Right
            [target_x - pad, target_y, laser_origin_z],      # Left
            [target_x, target_y + pad, laser_origin_z],      # Up
            [target_x, target_y - pad, laser_origin_z]       # Down
        ])
        ray_directions = np.array([[0, 0, 1]] * 5) 
        
        # Trimesh returns index_ray to tell us which of our 5 rays hit what
        locations, index_ray, _ = trimesh_mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )

        if len(locations) >= 2:
            valid_z_centers = []
            
            # Evaluate all hits grouped by which ray spawned them
            for i in range(5):
                ray_hits = locations[index_ray == i]
                if len(ray_hits) >= 2:
                    sort_idx = ray_hits[:, 2].argsort()
                    front_z = ray_hits[sort_idx[0]][2]
                    back_z = ray_hits[sort_idx[-1]][2] 
                    
                    thickness = abs(back_z - front_z)
                    
                    if "wrist" in bone_name or "ankle" in bone_name: max_t = true_shoulder_width * 0.15
                    elif "knee" in bone_name or "elbow" in bone_name: max_t = true_shoulder_width * 0.25
                    else: max_t = true_shoulder_width * 0.60
                    
                    if thickness > max_t: thickness = max_t
                    valid_z_centers.append(front_z + (thickness * 0.5))
            
            # Average the valid centers from our micro-grid
            if valid_z_centers:
                centered_z = sum(valid_z_centers) / len(valid_z_centers)
                final_dna["pose"][bone_name] = {"x": float(target_x), "y": float(target_y), "z": float(centered_z)}
            else:
                final_dna["pose"][bone_name] = {"x": float(target_x), "y": float(target_y), "z": float(locations[0][2])}
                
        elif len(locations) == 1:
            final_dna["pose"][bone_name] = {"x": float(target_x), "y": float(target_y), "z": float(locations[0][2])}
        else:
            # --- FIX 2: CONSTRAIN THE FALLBACK ---
            # Raycast missed completely. Clamp to parent Z to stop ghosting.
            parent_name = kinematic_parents.get(bone_name)
            if parent_name and parent_name in final_dna["pose"]:
                safe_fallback_z = final_dna["pose"][parent_name]["z"]
            else:
                safe_fallback_z = fallback_z
                
            try:
                closest, _, _ = proximity_engine.on_surface(np.array([[target_x, target_y, safe_fallback_z]]))
                if len(closest) > 0:
                    # THE FIX: We stop closest[0][0] and closest[0][1] from overwriting target_x and target_y.
                    # Force the joint to keep the AI's orthographic width, and ONLY update the Z-depth.
                    final_dna["pose"][bone_name] = {"x": float(target_x), "y": float(target_y), "z": float(closest[0][2])}
                else:
                    final_dna["pose"][bone_name] = {"x": float(target_x), "y": float(target_y), "z": float(safe_fallback_z)}
            except Exception:
                final_dna["pose"][bone_name] = {"x": float(target_x), "y": float(target_y), "z": float(safe_fallback_z)}


# ========================================================
# PHASE 3: TOES & FEET RESTORATION
# ========================================================
print("\nExecuting Phase 3: Restoring Feet & Toes...")
feet_bones = ["heel_L", "heel_R", "foot_index_L", "foot_index_R"]
for foot_bone in feet_bones:
    if foot_bone in full_pins:
        final_dna["pose"][foot_bone] = full_pins[foot_bone] 
    elif foot_bone in dna["pose"]:
        parent_ankle = "ankle_L" if "_L" in foot_bone else "ankle_R"
        anchor_z = final_dna["pose"].get(parent_ankle, {}).get("z", 0)
        final_dna["pose"][foot_bone] = {"x": dna["pose"][foot_bone]['x'], "y": dna["pose"][foot_bone]['y'], "z": anchor_z}
# ========================================================
# PHASE 4: THE INDEPENDENT DUAL-MACHINES (True Proportional Mapping)
# ========================================================
print("\nExecuting Phase 4: Independent Dual-Machine Hand Matrix...")

# We skip the dynamic rendering entirely.
# We pull the perfect, orthographic, distortion-free front view generated by the Cortex.
target_image = "DEBUG_PERFECT_FRONT.png"

if not os.path.exists(target_image):
    print(f"CRITICAL ERROR: {target_image} not found. Run Cortex Vision first.")
    exit()

holistic_ai = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2, min_detection_confidence=0.1)
image_rgb = cv2.cvtColor(cv2.imread(target_image), cv2.COLOR_BGR2RGB)
results = holistic_ai.process(image_rgb) # <--- THIS IS WHERE 'results' IS CREATED

hand_joints = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]


def process_hand(side_prefix, hand_landmarks):
    if not hand_landmarks: return
    if f"wrist_{side_prefix}" not in final_dna["pose"]: return
    if not results.pose_landmarks: return
    
    true_wrist = final_dna["pose"][f"wrist_{side_prefix}"]
    ai_hand_wrist = hand_landmarks.landmark[0]
    
    # ========================================================
    # SHATTERING WALL 1 & 2: LOCAL FOREARM SCALING (2D to 2D)
    # ========================================================
    # We abandon the shoulders. We use the local elbow-to-wrist segment.
    true_elbow = final_dna["pose"].get(f"elbow_{side_prefix}")
    if not true_elbow: return
    
    # --- FIX: THE 2D PROJECTION ---
    # Delete the Z-axis calculation here. We MUST compare 2D to 2D!
    true_forearm_length_2d = math.sqrt(
        (true_elbow["x"] - true_wrist["x"])**2 +
        (true_elbow["y"] - true_wrist["y"])**2
    )
    
    # 2. Calculate the 2D AI Forearm Length
    pose_wrist_idx = 15 if side_prefix == "L" else 16
    pose_elbow_idx = 13 if side_prefix == "L" else 14
    
    ai_pose_wrist = results.pose_landmarks.landmark[pose_wrist_idx]
    ai_pose_elbow = results.pose_landmarks.landmark[pose_elbow_idx]
    
    ai_forearm_length = math.sqrt(
        (ai_pose_elbow.x - ai_pose_wrist.x)**2 + 
        (ai_pose_elbow.y - ai_pose_wrist.y)**2
    )
    
    # 3. The Precision Ratio (Apples to Apples)
    if ai_forearm_length > 0:
        hand_scale = true_forearm_length_2d / ai_forearm_length
    else:
        hand_scale = 0.1
        
    # ========================================================
    # SHATTERING WALL 3: RELATIVE Z-DEPTH EXTRACTION
    # ========================zbfb ================================
    for i, lm in enumerate(hand_landmarks.landmark):
        bone_name = f"{hand_joints[i]}_{side_prefix}"
        if bone_name in full_pins:
            final_dna["pose"][bone_name] = full_pins[bone_name]
        else:
            # 1. Extract the raw offsets directly from the AI's local coordinate system
            offset_x = lm.x - ai_hand_wrist.x
            offset_y = lm.y - ai_hand_wrist.y
            offset_z = lm.z - ai_hand_wrist.z  # <-- RESTORE THE Z-DEPTH OFFSET!
            
            # 2. Scale all axes dynamically using the forearm ratio
            final_x = true_wrist["x"] + (offset_x * hand_scale)
            final_y = true_wrist["y"] - (offset_y * hand_scale) 
            
            # 3. Apply the scale to the relative depth and add it to the true wrist Z.
            # Depending on your specific camera orientation, you may need to flip the sign (+ or -) 
            # if the fingers curve backward into the arm instead of outward.
            final_z = true_wrist["z"] - (offset_z * hand_scale) 
            
            final_dna["pose"][bone_name] = {
                "x": float(final_x),
                "y": float(final_y),
                "z": float(final_z) 
            }
if results.right_hand_landmarks:
    process_hand("R", results.right_hand_landmarks)
if results.left_hand_landmarks:
    process_hand("L", results.left_hand_landmarks)

# Clean up leftover basic joints ONLY if they weren't manually pinned
for old in ["thumb_L", "index_L", "pinky_L", "thumb_R", "index_R", "pinky_R"]:
    if old not in full_pins:
        final_dna["pose"].pop(old, None)

with open('VOIDX_ULTIMATE_DNA.json', 'w') as f:
    json.dump(final_dna, f, indent=4)

print("\nSUCCESS: X-Ray Pipeline Execution Complete. Data saved to VOIDX_ULTIMATE_DNA.json")