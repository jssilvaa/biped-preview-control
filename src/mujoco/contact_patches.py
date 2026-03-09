from __future__ import annotations 
from dataclasses import dataclass 
from typing import Literal
import numpy as np 
import mujoco 

from control_types import ContactPatch, ContactModel 

ContactFrameMode = Literal["site", "world_up"]


@dataclass(frozen=True)
class PatchSpec: 
  """
  Defines a patch anchored to a MuJoCo site. 
  `vertex_offsets_site`: (Nv,3) offsets in R3 expressed in the site frame.
  If Nv=1 and offset=[0,0,0], patch is a point contact at the site position. 
  """
  name: str 
  site_id: int 
  vertex_offsets_site: np.ndarray   # (Nv,3)


def _site_R_ws(data: mujoco.MjData, site_id: int) -> np.ndarray: 
  return np.asarray(data.site_xmat[site_id], dtype=float).reshape(3,3).copy()

def _site_p_w(data: mujoco.MjData, site_id: int) -> np.ndarray: 
  return np.asarray(data.site_xpos[site_id], dtype=float).reshape(3,).copy()


def _R_world_up_from_site_yaw(R_ws: np.ndarray) -> np.ndarray: 
  """
  Build R_wc with z = world up, x from site x-axis projected onto xy plane 
  """
  z = np.array([0.0, 0.0, 1.0], dtype=float)
  x_raw = np.asarray(R_ws[:, 0], dtype=float).reshape(3,)
  x_raw[2] = 0.0 
  nx = float(np.linalg.norm(x_raw))
  if nx < 1e-9:
      x_raw = np.array([1.0, 0.0, 0.0], dtype=float)
  else:
      x_raw = x_raw / nx
  y = np.cross(z, x_raw)
  ny = float(np.linalg.norm(y))
  if ny < 1e-9:
      y = np.array([0.0, 1.0, 0.0], dtype=float)
  else:
      y = y / ny
  x = np.cross(y, z)
  return np.stack([x, y, z], axis=1)  # columns are contact axes in world


def build_contact_model_from_sites(
    model: mujoco.MjModel, 
    data: mujoco.MjData, 
    *,
    mu: float, 
    patch_specs: list[PatchSpec],
    frame_mode: ContactFrameMode = "world_up",
) -> ContactModel:
  if not (mu > 0 and np.isfinite(mu)): 
    raise ValueError(f"mu must be positive finite. got {mu}")
  
  patches: list[ContactPatch] = []
  for spec in patch_specs: 
    if not (0 <= spec.site_id < model.nsite): 
      raise ValueError(f"Invalid site_id {spec.site_id} for patch {spec.name}")
    
    offs = np.asarray(spec.vertex_offsets_site, dtype=float)
    if offs.ndim != 2 or offs.shape[1] != 3 or not np.all(np.isfinite(offs)): 
      raise ValueError(f"vertex_offsets_size must be finite (Nv,3), got {offs.shape} for {spec.name} instead")

    R_ws = _site_R_ws(data, spec.site_id)
    p_w = _site_p_w(data, spec.site_id)
    verts_w = p_w.reshape(1,3) + (R_ws @ offs.T).T # (Nv,3)

    if frame_mode == "site": 
       R_wc = R_ws 
    elif frame_mode == "world_up": 
       R_wc = _R_world_up_from_site_yaw(R_ws)
    else: 
       raise ValueError(f"Unknown frame_mode: {frame_mode}")

    patches.append(ContactPatch(
      name=spec.name,
      vertices_world=verts_w,
      p_w=p_w,
      R_wc=R_wc,
    ))

  return ContactModel(patches=patches, mu=float(mu))