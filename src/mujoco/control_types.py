from __future__ import annotations 
from dataclasses import dataclass, field 
from typing import Optional, Literal, Any 

import numpy as np 

#TaskKind = Literal["equality", "inequality", "objective"]
#TaskPriority = Literal["hard", "soft"]

#NOTE Using omega_world as phi_dot is an approximation. 
@dataclass
class BaseState:  
  """ Base orientation state in world frame """
  R_world: np.ndarray       # (3,3)
  omega_world: np.ndarray   # (3,)
  phi_world: np.ndarray     # (3,)


@dataclass
class CentroidalState: 
  com: np.ndarray         # (3,)
  com_vel: np.ndarray     # (3,)
  h: np.ndarray           # (6,) [linear; angular]
  hdot_bias: np.ndarray   # (6,) = Hdot_G qdot approx 
  Ag: np.ndarray          # (6, nv)


@dataclass
class CentroidalMeasured: 
  """ Actual centroidal measured state """
  com: np.ndarray         # (3,)
  com_vel: np.ndarray     # (3,) 
  base: BaseState | None  # None if not provided 


@dataclass 
class CentroidalDesired: 
  com: np.ndarray           # (3,)
  com_vel: np.ndarray       # (3,)
  base_R_world: np.ndarray | None 
  base_omega_world: np.ndarray | None 
  com_acc: np.ndarray | None = None   # (3,)
  base_alpha_world: np.ndarray | None = None


@dataclass
class CentroidalReference: 
  com_ref: np.ndarray       # (3,)
  com_vel_ref: np.ndarray   # (3,)
  com_acc_ref: np.ndarray   # (3,)
  h_ref: np.ndarray         # (6,)
  hdot_ff_ref: np.ndarray   # (6,) feedforward momentum rate 
  base_rpy_ref: Optional[np.ndarray] = None 
  base_omega_ref: Optional[np.ndarray] = None 
  meta: dict[str, Any] = field(default_factory=dict)


@dataclass 
class ResultantWrenchCommand: 
  force_world: np.ndarray       # (3,)
  moment_world: np.ndarray      # (3,) about world 
  about: str = "world_origin"   # change the reference here when the above changes


@dataclass 
class ResultantWrenchBar: 
  """ 
  Bar wrench output from the COTG layer 
  [bar_f; bar_n] == [f + m g_world; n - c x f == LDOT i.e. CAM rate about CoM]
  """
  bar_force_world: np.ndarray   # (3,)
  bar_moment_world: np.ndarray  # (3,)


@dataclass 
class WrenchDistributionConfig: 
  mu: float = 0.6 
  fz_max: float | None = None 
  reg: float = 1e-6 
  w_track: float = 1e-6 
  w_reg: float = 1e-4 


@dataclass 
class ContactPatch: 
  """
  Single limb-end contact patch. 
  Vertices-world: contact polygon vertices in world coordinates (Nv,3)
  p_w: limb end origin in world coordinates (3,)
  R_wc: rotation wRc, i.e. from contact frame to world frame (3,3)
    Contact frame z-axis is treated as the normal direction for friction pyramids
    as this is the more general approach to all possible scenarios
    Friction pyramid orientation is based on the local contact yaw 
    i.e. assume flat terrain to start with 
  """
  name: str 
  vertices_world: np.ndarray    # (Nv,3)
  p_w: np.ndarray               # (3,)
  R_wc: np.ndarray              # (3,3)


@dataclass 
class ContactModel: 
  """ 
  Collection patches. One per limb-end. 
  """
  patches: list[ContactPatch]
  mu: float = 0.6              # make this a world default, as in the paper 
