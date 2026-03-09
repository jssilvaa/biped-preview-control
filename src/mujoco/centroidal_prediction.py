from __future__ import annotations 

from control_types import ResultantWrenchBar, CentroidalDesired
from lie_math import Exp

import numpy as np 


def predict_one_step(
    *,
    dt: float, 
    mass: float, 
    I_diag: np.ndarray,
    com: np.ndarray, 
    com_vel: np.ndarray, 
    base_R: np.ndarray | None, 
    base_omega: np.ndarray | None, 
    bar_wp_proj: ResultantWrenchBar,
) -> CentroidalDesired:
  """
  One-step prediction using projected bar wrench: 
    com_ddot = bar_f / m 
    omega_dot approx= I^{-1} bar_n (approximation in paper)
  Integrate one step: 
    c_d = c + cd dt + 0.5 cdd dt^2 
    cd_d = cd + cdd dt 
    omega_d = omega + omegadot dt 
    R_d = Exp(omega_d dt) R (sympletic euler; consistent with small angle approx)
  """
  dt = float(dt)
  if dt <= 0 or not np.isfinite(dt): 
    raise ValueError("dt must be positive finite")
  I_diag = np.asarray(I_diag, dtype=float).reshape(3,)
  if np.any(I_diag <= 0) or not np.all(np.isfinite(I_diag)): 
    raise ValueError("I_diag must be positive finite (3,)")
  
  c = np.asarray(com, dtype=float).reshape(3,)
  cd = np.asarray(com_vel, dtype=float).reshape(3,)
  bf = np.asarray(bar_wp_proj.bar_force_world, dtype=float).reshape(3,)
  bn = np.asarray(bar_wp_proj.bar_moment_world, dtype=float).reshape(3,)

  cdd = bf / float(mass)
  c_d = c + cd * dt + 0.5 * cdd * dt**2
  cd_d = cd + cdd * dt 

  R_d = None 
  omega_d = None 
  if base_R is not None and base_omega is not None: 
    R = np.asarray(base_R, dtype=float).reshape(3,3)
    omega = np.asarray(base_omega, dtype=float).reshape(3,)
    omegadot = bn / I_diag 
    omega_d = omega + omegadot * dt 
    R_d = Exp(omega_d * dt) @ R # approximation (small angle), must prove this later or confirm with the paper

  return CentroidalDesired(
    com=c_d,
    com_vel=cd_d,
    base_R_world=R_d,
    base_omega_world=omega_d,
  )