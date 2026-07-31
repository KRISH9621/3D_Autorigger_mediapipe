# A Mathematical Approach for Auto-Rigging 3D Models Using 2D MediaPipe

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/) <!-- Note: Update this link once your paper is live! -->

Official code repository for the arXiv preprint: **"A Mathematical Approach for Auto-Rigging 3D Model Using 2D MediaPipe"**.

This repository contains the Python implementation of the **VoidX Engine** auto-rigging pipeline. Unlike traditional methods that rely on probabilistic neural networks to guess 3D depth (Z-axis) from 2D images, this pipeline treats depth as a solvable geometric variable. By rotating a 3D mesh in front of an orthographic camera and applying trigonometric inversion, we achieve mathematically exact, volumetrically centered skeletal structures requiring zero training data.

---

## 🔑 Key Features
*   **Semantic Jurisdiction Zones:** A strict filtering system that prevents 2D pose estimators (like MediaPipe) from hallucinating joint positions on self-occluded limbs.
*   **Decoupled Axis Projection:** Extracts true 3D depth (Z) using pure trigonometric inversion from multi-view 2D screen coordinates.
*   **Volumetric Master Engine:** Uses dynamic shoulder-width anchoring and micro-grid raycasting to perfectly center joints inside the mesh geometry.
*   **Anatomical Thickness Capping:** Prevents rays from bleeding through thin limbs (like wrists) into the torso behind them.

---

## 📊 The Math Behind It
Standard 2D pose estimators only provide 2D screen coordinates $(x_{screen}, y_{screen})$. To find the true depth $(Z_{true})$, we rotate the mesh by an angle $\theta$ and use the following geometric relationship:

$$ Z_{true} = \frac{x_{screen}(\theta) - X_{true} \cos(\theta)}{\sin(\theta)} $$

Once the 3D coordinates are extracted, the 14 core joints are anchored to the volumetric center using a Gravity Drop raycast and Anatomical Thickness Capping:

$$ Z_{centered} = Z_{front\_skin} + \frac{\min(\Delta Z, \Delta Z_{max\_cap})}{2} $$

---

## 🛠️ Installation & Usage

### Prerequisites
Make sure you have Python 3.x installed. You will need the following libraries:
```bash
pip install open3d trimesh mediapipe opencv-python numpy plotly

### 🚀 Running the Pipeline

The VoidX pipeline operates in two sequential stages, followed by an optional visualization step. 

Ensure your target 3D mesh (e.g., `target_mesh.glb`) is placed in the root directory. Then, you can execute the entire pipeline by running the following script commands sequentially in your terminal:

```bash
# ==========================================
# STAGE 1: VoidX Cortex Vision
# Extracts 2D-to-3D spatial data via orthographic scanning
# ==========================================
python voidx_cortex_vision.py
# -> Outputs: VOIDX_CORTEX_DNA.json & DEBUG_PERFECT_FRONT.png

# ==========================================
# STAGE 2: VoidX Master Engine
# Performs volumetric raycasting and dynamic anchoring
# ==========================================
python voidx_master_engine.py
# -> Outputs: VOIDX_ULTIMATE_DNA.json

# ==========================================
# VISUALIZATION (Optional but Recommended)
# Renders the final 3D rig inside the transparent mesh
# ==========================================
python voidx_plotly_viewer.py
# -> Outputs: debug_rig_viewer.html (Opens in browser)
