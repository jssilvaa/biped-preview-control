import mujoco
import sys
import os
from contact_measurement import PatchGeomMap, build_patch_geom_map_from_sites

def _main(): 
  model = mujoco.MjModel.from_xml_path("models/unitree_g1/scene.xml")
  data = mujoco.MjData(model)

  # Prints (nq, nv, nu)
  print(model.nq, model.nv, model.nu)

  # Prints (body_id, body_name)
  print("============================")
  print(f"nbody: {model.nbody}")
  for id in range(model.nbody): 
    print(f"{id}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id)}")

  # Prints (site_id, site_name)
  print("============================")
  print(f"nsite: {model.nsite}")
  for id in range(model.nsite): 
    print(f"{id}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, id)}")

  # Prints (geom_id, geom_name)
  print("============================")
  print(f"ngeom: {model.ngeom}")
  for id in range(model.ngeom): 
    print(f"{id}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, id)}")
    break # all the rest has name None (besides 0 which is the floor, i.e. floor_gid = 0)

  # Prints (key_id, key_name)
  print("============================")
  print(f"nkey: {model.nkey}")
  for id in range(model.nkey): 
    print(f"{id}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, id)}")

  # Prints (joint_id, joint_name)
  print("============================")
  print(f"njnt: {model.njnt}")
  for id in range (model.njnt): 
    print(f"{id}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, id)}\n" \
          f"qposadr: {model.jnt_qposadr[id]}\t\t" \
          f"qdofadr: {model.jnt_dofadr[id]}"
    )

  # Prints (actuator_id, actuator_name)
  print("============================")
  print(f"nu: {model.nu}")
  for id in range (model.nu): 
    print(f"{id}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, id)}")
    print(f"transmission id: {model.actuator_trnid[id]}")
    print(f"transmission type: {model.actuator_trntype[id]}")
    print(f"actuator gear: {model.actuator_gear[id]}")
    print(f"actuator ctrlrange (lo): {model.actuator_ctrlrange[id, 0]}")
    print(f"actuator ctrlrange (hi): {model.actuator_ctrlrange[id, 1]}")

  patch_geom_map: PatchGeomMap = build_patch_geom_map_from_sites(model, [2])
  print(patch_geom_map)

if __name__ == "__main__": 
  _main()

# RESULTS 
G1_BASE_BODY = "pelvis"             # bid: 1 
G1_LEFT_FOOT_SITE = "left_foot"     # sid: 1
G1_RIGHT_FOOT_SITE = "right_foot"   # sid: 2
G1_LEFT_FOOT_GEOMS = {14, 15, 16, 17, 18}
G1_RIGHT_FOOT_GEOMS = {29, 30, 31, 32, 33}