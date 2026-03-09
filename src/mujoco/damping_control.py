from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from lie_math import compose_rotvec, Exp, logvec


@dataclass(frozen=True)
class DampingGains: 
  Kd: np.ndarray  # (6,) diagonal entries 
  Ks: np.ndarray  # (6,) diagonal entries 
  Kf: np.ndarray  # (6,) diagonal entries 

  def validate(self): 
    for name, v in [("Kd", self.Kd), ("Ks", self.Ks), ("Kf", self.Kf)]: 
      v = np.asarray(v, dtype=float).reshape(6,)
      if not np.all(np.isfinite(v)): 
        raise ValueError(f"{name} non-finite")
    if np.any(np.asarray(self.Kd) <= 0): 
      raise ValueError("Kd must be strictly positive elementwise")
    

@dataclass
class ComplianceState: 
  """
  Δr = [Δp; Δphi] where Δphi = log(Rc Rd^T)
  Stored as (6,) in world coords for translation and rotation vector for orientation. 
  """
  dr: np.ndarray  # (6,)

  @staticmethod
  def zero() -> ComplianceState: 
    return ComplianceState(dr=np.zeros(6, dtype=float))
  

def damping_step( 
    *, 
    dt: float, 
    gains: DampingGains,
    state: ComplianceState,
    w_meas: np.ndarray,     # (6,)
    w_des: np.ndarray,      # (6,)
) -> ComplianceState: 
  """
  Discrete implementation of paper Eq (16): 
    drdot = -(Ks/Kd) dr + (Kf/Kd)(w_meas - w_des)
    dr_L[k+1] = dr_L[k] + dt drdot_L
    dr_A[k+1] = log( exp(dt drdot_A) exp(dr_A) )
  """
  dt = float(dt)
  if dt <= 0 or not np.isfinite(dt): 
    raise ValueError(f"dt must be positive finite, got {dt}")
  gains.validate()

  dr = np.asarray(state.dr, dtype=float).reshape(6,)
  w_meas = np.asarray(w_meas, dtype=float).reshape(6,)
  w_des = np.asarray(w_des, dtype=float).reshape(6,)

  Kd = np.asarray(gains.Kd, dtype=float).reshape(6,)
  Ks = np.asarray(gains.Ks, dtype=float).reshape(6,)
  Kf = np.asarray(gains.Kf, dtype=float).reshape(6,)

  drdot = -(Ks / Kd) * dr + (Kf / Kd) * (w_meas - w_des)

  dr_next = dr.copy()
  dr_next[0:3] = dr[0:3] + dt * drdot[0:3]
  dr_next[3:6] = compose_rotvec(dt * drdot[3:6], dr[3:6])

  return ComplianceState(dr=dr_next)