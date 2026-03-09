from __future__ import annotations 
from dataclasses import dataclass
import numpy as np 
import mujoco 

from dynamics import actuator_dof_indices 


@dataclass(frozen=True)
class JointServoConfig: 
  kp: float = 200.0 
  kd: float = 20.0 
  ctrl_clip: bool = True 


def compute_motor_ctrl_from_qpos_target(
    model: mujoco.MjModel, 
    data: mujoco.MjData,
    *,
    qpos_des: np.ndarray, 
    qvel_des: np.ndarray | None, 
    cfg: JointServoConfig,
) -> np.ndarray: 
  """
  Torque motor servo 
    tau_dof = kp*(q_des - q) + kd*(qd_des - qd)
  Convert to ctrl via gear: tau = gear * ctrl => ctrl = tau / gear 

  This is actuator-by-actuator using actuator_dof_indices 
  """
  qpos_des = np.asarray(qpos_des, dtype=float).reshape(model.nq,)
  if qvel_des is None: 
    qvel_des = np.zeros(model.nv, dtype=float)
  else: 
    qvel_des = np.asarray(qvel_des, dtype=float).reshape(model.nv)

  dof_of_act = actuator_dof_indices(model)  # (nu,)
  nu = model.nu 
  ctrl = np.zeros(nu, dtype=float)

  for a in range(nu): 
    dof = int(dof_of_act[a]) # note that qpos index is not always dof index. qpos layout is different from nu, nv one 
    jid = int(model.actuator_trnid[a, 0])
    qadr = int(model.jnt_qposadr[jid])
    q = float(data.qpos[qadr])
    qd = float(data.qvel[dof])
    qdes = float(qpos_des[qadr])
    qddes = float(qvel_des[dof])

    tau = cfg.kp * (qdes - q) + cfg.kd * (qddes - qd)
    gear = float(model.actuator_gear[a, 0])
    ctrl[a] = tau / gear if abs(gear) > 1e-12 else 0.0 

  if cfg.ctrl_clip and np.any(model.actuator_ctrllimited): 
    lo = model.actuator_ctrlrange[:, 0].astype(float)
    hi = model.actuator_ctrlrange[:, 1].astype(float)
    ctrl = np.clip(ctrl, lo, hi)
  
  return ctrl 

def compute_position_ctrl_from_qpos_target(model: mujoco.MjModel, qpos_des: np.ndarray) -> np.ndarray:
    qpos_des = np.asarray(qpos_des, dtype=float).reshape(model.nq,)
    ctrl = np.zeros(model.nu, dtype=float)
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        qadr = int(model.jnt_qposadr[jid])
        ctrl[a] = qpos_des[qadr]
    if np.any(model.actuator_ctrllimited):
        lo = model.actuator_ctrlrange[:, 0].astype(float)
        hi = model.actuator_ctrlrange[:, 1].astype(float)
        ctrl = np.clip(ctrl, lo, hi)
    return ctrl