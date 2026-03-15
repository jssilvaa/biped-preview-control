from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from control_types import ResultantWrenchBar, CentroidalMeasured, CentroidalDesired
from lie_math import logvec


@dataclass(frozen=True)
class StabilizerGains: 
  Kp_lin: np.ndarray  # (3,3)
  Kd_lin: np.ndarray  # (3,3)
  Kp_ang: np.ndarray  # (3,3)
  Kd_ang: np.ndarray  # (3,3)

  @staticmethod 
  def diagonal(
    kp_lin=(2000.0, 2000.0, 2000.0),
    kd_lin=(666.0, 666.0, 666.0),
    kp_ang=(0.0, 0.0, 0.0),
    kd_ang=(0.0, 0.0, 0.0),
  ) -> StabilizerGains: 
    return StabilizerGains(
      Kp_lin=np.diag(np.asarray(kp_lin, dtype=float)),
      Kd_lin=np.diag(np.asarray(kd_lin, dtype=float)),
      Kp_ang=np.diag(np.asarray(kp_ang, dtype=float)),
      Kd_ang=np.diag(np.asarray(kd_ang, dtype=float)),
    )

  @staticmethod
  def murooka_table_iii() -> StabilizerGains:
    return StabilizerGains.diagonal(
      kp_lin=(2000.0, 2000.0, 2000.0),
      kd_lin=(666.0, 666.0, 666.0),
      kp_ang=(0.0, 0.0, 0.0),
      kd_ang=(0.0, 0.0, 0.0),
    )
  

def stabilize_bar_wrench(
    *,
    bar_wp_proj: ResultantWrenchBar,
    desired: CentroidalDesired,
    measured: CentroidalMeasured, 
    gains: StabilizerGains
) -> tuple[ResultantWrenchBar, dict]: 
  """
  Implements paper Eq (11)-(12) in rotation matrix form: 
    Δbar_w = [ KpL(c_d-c_a) + KdL(cd_d - cd_a) 
               KpA log(R_d R_a^T) + KdA(omega_d - omega_a) ]
    bar_wd = bar_wp' + Δbar_w
  """
  c_a = np.asarray(measured.com, dtype=float).reshape(3,)
  cd_a = np.asarray(measured.com_vel, dtype=float).reshape(3,)
  c_d = np.asarray(desired.com, dtype=float).reshape(3,)
  cd_d = np.asarray(desired.com_vel, dtype=float).reshape(3,)

  dbf = gains.Kp_lin @ (c_d - c_a) + gains.Kd_lin @ (cd_d - cd_a)

  dbn = np.zeros(3, dtype=float)
  ang_dbg = {"enabled": False}

  if desired.base_R_world is not None or desired.base_omega_world is not None: 
    if measured.base is None: 
      raise ValueError("Angular stabilization requested but measured.base is None")
    if desired.base_R_world is None or desired.base_omega_world is None: 
      raise ValueError("Angular stabilization requested but desired base_R / omega missing")
    
    R_a = np.asarray(measured.base.R_world, dtype=float).reshape(3,3)
    omega_a = np.asarray(measured.base.omega_world, dtype=float).reshape(3,)
    R_d = np.asarray(desired.base_R_world, dtype=float).reshape(3,3)
    omega_d = np.asarray(desired.base_omega_world, dtype=float).reshape(3,)

    # Rotation Error Vector 
    e_R = logvec(R_d @ R_a.T)
    dbn = gains.Kp_ang @ e_R + gains.Kd_ang @ (omega_d - omega_a)
    ang_dbg = {"enabled": True, "e_R": e_R, "omega_err": (omega_d - omega_a)}

  bar_wd = ResultantWrenchBar(
    bar_force_world=np.asarray(bar_wp_proj.bar_force_world, dtype=float).reshape(3,) + dbf, 
    bar_moment_world=np.asarray(bar_wp_proj.bar_moment_world, dtype=float).reshape(3,) + dbn 
  )

  dbg = {
    "delta_bar_force": dbf,
    "delta_bar_moment": dbn, 
    "ang": ang_dbg, 
    "bar_wp_proj": np.hstack((bar_wp_proj.bar_force_world, bar_wp_proj.bar_moment_world)),
    "bar_wd": np.hstack((bar_wd.bar_force_world, bar_wd.bar_moment_world))
  }
  return bar_wd, dbg 
