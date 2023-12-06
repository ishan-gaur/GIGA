import numpy as np
import trimesh

from pathlib import Path
# mesh_path = Path.cwd() / 'data/experiments/23-12-05-05-09-53/grasps/round_000_trial_000.obj'
mesh_path = Path.cwd() / 'data/experiments/23-12-05-05-09-53/meshes/round_000_trial_000_aff.obj'
scene_path = Path.cwd() / 'data/experiments/23-12-05-05-09-53/meshes/round_000_trial_000_scene.obj'
mesh = trimesh.load_mesh(mesh_path)
scene_obj = trimesh.load_mesh(scene_path)

scene = trimesh.Scene()
scene.add_geometry(scene_obj)
png = scene.save_image()
with open('scene.png', 'wb') as f:
    f.write(png)
