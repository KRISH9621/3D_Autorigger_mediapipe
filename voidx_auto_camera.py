from builtins import print

import open3d as o3d
import cv2
import mediapipe as mp
import numpy as np
import json
import math
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print("VOIDX ENGINE: Initializing Autonomous Triangulation Pipeline...")

# --- MEDIAPIPE TWEAKS ---
# Complexity dropped to 1 and confidence to 0.1 to force detection on synthetic AI blobs
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.1)

def render_virtual_cameras(mesh_path):
    print(f"Loading AI Blob: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    # --- VISUAL CONTRAST TWEAKS ---
    # Paint the mesh a pseudo-skin tone so MediaPipe doesn't think it's a gray statue
    mesh.paint_uniform_color([0.9, 0.8, 0.7])
    
    bbox = mesh.get_axis_aligned_bounding_box()
    mesh_center = bbox.get_center()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1024, height=1024)
    vis.add_geometry(mesh)
    
    # Pure black background for maximum silhouette contrast
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0]) 
    
    ctr = vis.get_view_control()
    target_angles = [0, 90, 180, 270, 360]

    for angle in target_angles:
        print(f"\nPositioning Virtual Camera at {angle}°...")
        rad = math.radians(angle)
        
        # Calculate orbital position
        cam_dir_x = math.sin(rad)
        cam_dir_z = math.cos(rad)
        
        # Native camera tracking to prevent clipping plane deletion
        ctr.set_lookat(mesh_center)
        ctr.set_up([0, 1, 0]) 
        ctr.set_front([cam_dir_x, 0, cam_dir_z]) 
        ctr.set_zoom(1.2) 
        
        # Flush the visual buffer to ensure the frame renders before capture
        for _ in range(15):
            vis.poll_events()
            vis.update_renderer()
        
        image_path = f"DEBUG_CARDINAL_{angle}.png"
        vis.capture_screen_image(image_path)
        
        # --- MEDIAPIPE PROCESSING & RIG FORMATION ---
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            print(f"SUCCESS: Skeleton detected at {angle}°. Constructing Rig DNA...")
            
            # --- VISUAL DEBUG: DRAW THE RIG ---
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Save the new image showing the skeleton on top of the mesh
            rig_debug_path = f"DEBUG_RIG_CARDINAL_{angle}.png"
            cv2.imwrite(rig_debug_path, annotated_image)
            print(f"-> Saved visual rig overlay to {rig_debug_path}")
            # ----------------------------------
            
            def calc_3d(idx):
                lm = results.pose_landmarks.landmark[idx]
                return {"x": lm.x, "y": lm.y, "z": lm.z}

            # FULL KINEMATIC CHAIN RESTORED
            pose_data = {
                # Upper Body
                "shoulder_L": calc_3d(11),
                "shoulder_R": calc_3d(12),
                "elbow_L": calc_3d(13),
                "elbow_R": calc_3d(14),
                "wrist_L": calc_3d(15),
                "wrist_R": calc_3d(16),
                
                # Lower Body
                "hip_L": calc_3d(23),
                "hip_R": calc_3d(24),
                "knee_L": calc_3d(25),
                "knee_R": calc_3d(26),
                "ankle_L": calc_3d(27),
                "ankle_R": calc_3d(28),
                
                # Feet (Crucial for grounding the mesh)
                "heel_L": calc_3d(29),
                "heel_R": calc_3d(30),
                "foot_index_L": calc_3d(31),
                "foot_index_R": calc_3d(32)
            }

            # Mathematical centers
            pose_data["pelvis"] = {
                "x": (pose_data["hip_L"]['x'] + pose_data["hip_R"]['x']) / 2,
                "y": (pose_data["hip_L"]['y'] + pose_data["hip_R"]['y']) / 2,
                "z": (pose_data["hip_L"]['z'] + pose_data["hip_R"]['z']) / 2
            }
            
            pose_data["neck"] = {
                "x": (pose_data["shoulder_L"]['x'] + pose_data["shoulder_R"]['x']) / 2,
                "y": (pose_data["shoulder_L"]['y'] + pose_data["shoulder_R"]['y']) / 2,
                "z": (pose_data["shoulder_L"]['z'] + pose_data["shoulder_R"]['z']) / 2
            }

            # Save angle-specific rig
            angle_filename = f'SMPL_Anchors_{angle}.json'
            with open(angle_filename, 'w') as f:
                json.dump({"pose": pose_data}, f, indent=4)
            print(f"-> Exported {angle_filename}")

            # Overwrite the master file on 0 degrees to feed the rest of your pipeline
            if angle == 0:
                with open('SMPL_Anchors.json', 'w') as f:
                    json.dump({"pose": pose_data}, f, indent=4)
                print("-> Updated master SMPL_Anchors.json")

        else:
            print(f"WARNING: MediaPipe failed to see the skeleton at {angle}°")

    vis.destroy_window()
    print("\nVirtual Cameras shut down.")

if __name__ == "__main__":
    # Ensure this matches your actual test file
    target_mesh = "male_t_pose.glb" 
    
    if os.path.exists(target_mesh):
        render_virtual_cameras(target_mesh)
    else:
        print(f"CRITICAL ERROR: Cannot find {target_mesh} in the current folder.")