import os

print("--- Starting VoidX Pipeline ---")
print("[1/4] Running Cortex Vision...")
os.system("python voidx_cortex_vision.py")

print("[2/4] Running Master Engine...")
os.system("python voidx_master_engine.py")

print("[3/4] Launching Plotly Viewer...")
os.system("python voidx_plotly_viewer.py")

print("[4/4] Running auto camera...")
os.system("python voidx_auto_camera.py")
print("--- Pipeline Complete ---")
