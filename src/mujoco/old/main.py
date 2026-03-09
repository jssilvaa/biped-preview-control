from __future__ import annotations
import numpy as np
import mujoco

from sim import run_simulation, MurookaSimConfig

STAND_KEY = 0
BASE_BODY_ID = 1
FLOOR_GEOM_NAME = "floor"
SITE_NAMES = ["left_foot", "right_foot"]
FOOT_VERTS = np.array([
    [ 0.07,  0.035, -0.03],
    [ 0.07, -0.035, -0.03],
    [-0.07,  0.035, -0.03],
    [-0.07, -0.035, -0.03],
], dtype=float)

def _main():
    model = mujoco.MjModel.from_xml_path("models/unitree_g1/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, STAND_KEY)
    site_vertex_offsets={
        "left_foot": FOOT_VERTS,
        "right_foot": FOOT_VERTS,
    }   

    cfg = MurookaSimConfig(
        dt=1e-3,
        N=5_000,
        mu=0.6,
        enable_motion_refs=True,
        floor_geom_name=FLOOR_GEOM_NAME,
        site_names=SITE_NAMES,
        site_vertex_offsets=site_vertex_offsets,
        base_body_id=BASE_BODY_ID,
        I_diag=model.body_inertia[BASE_BODY_ID],
        viz=True,
        display_every=2,
    )

    out = run_simulation(model, data, cfg=cfg)
    print("done", out["q_log"].shape[0])

if __name__ == "__main__":
    _main()