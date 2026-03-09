from __future__ import annotations

import time 
from dataclasses import dataclass

import numpy as np 
import mujoco 
import mujoco.viewer


@dataclass 
class ContactNormalForce:
  cid: int 
  floor_geom_id: int 
  robot_geom_id: int 
  position_world: np.ndarray
  normal_world: np.ndarray 
  force_world: np.ndarray 
  normal_force: float 


class Viz: 
  def __init__(
    self, 
    m: mujoco.MjModel, 
    d: mujoco.MjData, 
    show_contact_normals: bool=True, 
    min_normal_force: float = 5.0, 
    contact_force_scale: float = 1.0e-2,
    contact_force_width: float = 1.5e-2, 
    floor_geom_name: str="floor"): 
    if mujoco.viewer is None: 
      raise RuntimeError("mujoco.viewer is not available in this environment")
   
    self.m = m 
    self.viewer = mujoco.viewer.launch_passive(m,d)
    self.show_contact_normals = show_contact_normals
    self.min_normal_force = min_normal_force 
    self.contact_force_scale = contact_force_scale
    self.floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name) 
    self.identity_mat = np.eye(3, dtype=float).reshape(-1)
    self.arrow_rgba = np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32)
    self.arrow_width = contact_force_width
    self.realtime: bool = True

  def update(self, data: mujoco.MjData): 
    lock = getattr(self.viewer, "lock", None)
    if callable(lock): 
      with self.viewer.lock(): 
        self._update_contact_force_geoms(data)
    self.viewer.sync()
    if not self.realtime:
      time.sleep(2*self.m.opt.timestep) # slow-mo 
    else: 
      time.sleep(self.m.opt.timestep)

  def _update_contact_force_geoms(self, d: mujoco.MjData): 
    scene = self.viewer.user_scn
    scene.ngeom = 0
    if not self.show_contact_normals or self.floor_geom_id < 0: 
      return 
    
    forces = self._extract_floor_contact_normals(d, self.floor_geom_id, self.min_normal_force)
    for force in forces: 
      if scene.ngeom >= scene.maxgeom: 
        break 
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        geom, 
        int(mujoco.mjtGeom.mjGEOM_ARROW), 
        np.zeros((3,), dtype=float),
        np.zeros((3,), dtype=float), 
        self.identity_mat, 
        self.arrow_rgba,
      )
      p0 = force.position_world
      p1 = force.position_world + self.contact_force_scale * force.force_world
      mujoco.mjv_connector(
        geom, 
        int(mujoco.mjtGeom.mjGEOM_ARROW), 
        self.arrow_width, 
        p0,
        p1
      )

      scene.ngeom += 1

  def _contact_frame_normal_world(self, contact: mujoco.MjContact): 
    """ returns the unit contact normal in world frame coordinates """
    normal = np.asarray(contact.frame, dtype=float).copy()[:3] 
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12: 
      return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / norm 
  
  def _extract_floor_contact_normals(self, data: mujoco.MjData, floor_geom_id: int, min_normal_force: float=5.0) -> list[ContactNormalForce]: 
    """ extracts active robot to floor contacts and returns world frame normal forces on the robot. """
    contacts: list[ContactNormalForce] = []
    wrench = np.zeros((6,), dtype=float)

    for cid in range(int(data.ncon)): 
      contact = data.contact[cid]
      geom1 = int(contact.geom1)
      geom2 = int(contact.geom2)

      if (geom1 != floor_geom_id) and (geom2 != floor_geom_id): 
        continue 

      robot_geom_id = geom2 if geom1 == floor_geom_id else geom1 
      if int(self.m.geom_bodyid[robot_geom_id]) == 0:
        continue 

      mujoco.mj_contactForce(self.m, data, cid, wrench)
      normal_force = float(wrench[0])
      if normal_force < float(min_normal_force): 
        continue 

      normal_world = self._contact_frame_normal_world(contact)
      
      # note mj_contactForce reports force on geom2 by geom1, we want to represent the force on the robot geom regardless of the ordering of geoms 
      sign = 1.0 if robot_geom_id == geom2 else -1.0
      force_world = sign * normal_force * normal_world

      contacts.append(
        ContactNormalForce(
          cid=cid, 
          floor_geom_id=floor_geom_id,
          robot_geom_id=robot_geom_id,
          position_world=np.asarray(contact.pos, dtype=float).copy(),
          normal_world=normal_world,
          force_world=force_world, 
          normal_force=normal_force
        )
      )
    
    return contacts
  
  def close(self):
    if hasattr(self.viewer, "close"): 
      self.viewer.close()
