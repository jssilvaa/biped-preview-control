from __future__ import annotations 
from dataclasses import dataclass 
import numpy as np 
import mujoco 

from control_types import ContactModel 
from misc import site_attached_geom_ids


@dataclass(frozen=True)
class PatchGeomMap: 
  """
  Takes a contact patch to a set of geom ids belonging to that patch
  """
  patch_geom_ids: list[set[int]]  # size like ContactModel.patches 


def build_patch_geom_map_from_sites(model: mujoco.MjModel, site_ids: list[int]) -> PatchGeomMap: 
  """
  Each patch is anchored to a site. 
  We map patch -> geoms attached to the site's body.
  """
  patch_geom_ids: list[set[int]] = []
  for sid in site_ids: 
    gids = site_attached_geom_ids(model, sid)
    patch_geom_ids.append(set(int(g) for g in gids))
  return PatchGeomMap(patch_geom_ids=patch_geom_ids)


def _contact_R_wc(contact: mujoco.MjContact) -> np.ndarray: 
  return np.asarray(contact.frame, dtype=float).reshape(3,3).copy()


def _contact_normal_world(contact: mujoco.MjContact) -> np.ndarray: 
  n = np.asarray(contact.frame, dtype=float)[:3].copy()
  nn = float(np.linalg.norm(n))
  if nn < 1e-12: 
    return np.array([0.0, 0.0, 1.0], dtype=float)
  return n / nn 


def measure_patch_wrenches_world(
    model: mujoco.MjModel, 
    data: mujoco.MjData, 
    *,
    floor_geom_id: int, 
    contact_model: ContactModel, 
    geom_map: PatchGeomMap, 
    min_normal_force: float = 0.0 
) -> list[np.ndarray]: 
  """
  Returns a list of wrenches per patch, each wrench (6,) in world frame: 
    w = [F_world; Tau_world_about_site_pos]
  Convention is to make the limb_end (p_w in ContactPatch)

  - Sums contributions from all MuJoCo contacts involving a robot geom assigned to that patch. 
  - Uses mj_contactForce which returns wrench in contact frame at contact point. 
  - Converts to world and shifts torque to patch origin using tau_o = tau_c + (p_c-p_o ) x f
  """
  if len(contact_model.patches) != len(geom_map.patch_geom_ids): 
    raise ValueError("contact_model.patches and geom_map.patch_geom_ids must have same length")
  
  w_patch = [np.zeros(6, dtype=float) for _ in contact_model.patches]
  patch_origin = [np.asarray(patch.p_w, dtype=float).reshape(3,) for patch in contact_model.patches]

  wrench_c = np.zeros(6,dtype=float)
  for cid in range(int(data.ncon)): 
    c = data.contact[cid]
    g1, g2 = int(c.geom1), int(c.geom2)

    if (g1 != int(floor_geom_id)) and (g2 != int(floor_geom_id)): 
      continue 

    robot_gid = g2 if g1 == int(floor_geom_id) else g1 

    if int(model.geom_bodyid[robot_gid]) == 0: # world body
      continue 

    R_wc = _contact_R_wc(c)
    n_w = _contact_normal_world(c)

    mujoco.mj_contactForce(model, data, cid, wrench_c)
    f_c = wrench_c[:3].copy()
    tau_c = wrench_c[3:].copy()

    # mj_contactForce returns force on geom2 by geom1 in contact frame, order signal accordingly
    sign = 1.0 if (robot_gid == g2) else -1.0
    f_w = sign * (R_wc.T @ f_c)
    tau_w_at_contact = sign * (R_wc.T @ tau_c)

    fn = float(np.dot(f_w, n_w))
    if fn < float(min_normal_force): 
      continue 

    p_contact = np.asarray(c.pos, dtype=float).reshape(3,).copy()
    
    # assign to patch via geom membership 
    assigned = False 
    for i, gset in enumerate(geom_map.patch_geom_ids): 
      if robot_gid in gset: 
        p0 = patch_origin[i]
        w_patch[i][:3] += f_w
        w_patch[i][3:] += tau_w_at_contact + np.cross(p_contact - p0, f_w)
        assigned = True 
        break 

    # if no patch mathes, drop the contact 
    if not assigned: 
      print(f"{robot_gid} not assigned to any known robot patch in {geom_map.__repr__}")

  return w_patch 