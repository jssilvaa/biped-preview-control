from __future__ import annotations
from typing import Optional 
from control_types import BaseState

import mujoco
import numpy as np

# define gravity 
def set_gravity(model: mujoco.MjModel, g: np.ndarray): 
    """ set world gravity """
    model.opt.gravity[:] = np.asarray(g, dtype=float).reshape(3,) 


# compute total energy 
def total_energy(model: mujoco.MjModel, data: mujoco.MjData) -> float: 
    """ total model energy """
    mujoco.mj_energyPos(model,data)
    mujoco.mj_energyVel(model,data)
    return float(data.energy[0] + data.energy[1])  # [potential, kinetic]


# compute total mass 
def compute_total_mass(model: mujoco.MjModel) -> float: 
    """ total model mass"""
    return float(np.sum(model.body_mass))


# tau limits from xml 
def tau_limits(model: mujoco.MjModel) -> np.ndarray:
    """ Retrieves torque limits from xml with simple joint motors and symmetric ctrlrange with gear gain in transmission """
    assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)  # assume only joints 
    assert np.all(model.actuator_biastype == 0)  # no bias assumption 
    nu = model.nu 
    limits: np.ndarray = np.full(nu, np.inf, dtype=float)
    if np.any(model.actuator_ctrllimited): 
        lo = model.actuator_ctrlrange[:, 0].astype(float)
        hi = model.actuator_ctrlrange[:, 1].astype(float)
        gear = model.actuator_gear[:nu, 0].astype(float)
        # generalized torque limits at actuator output, |tau| <= |gear * ctrl|
        limits = np.maximum(np.abs(gear * lo), np.abs(gear * hi))
    return limits 


# compute actuator momemnt matrix 
def build_actuator_moment_matrix(model: mujoco.MjModel) -> np.ndarray:
    """
    Build M (nv x nu) such that qfrc_actuator = M @ ctrl
    for <motor joint="..."> transmissions.
    """
    nv, nu = model.nv, model.nu
    M = np.zeros((nv, nu))

    # actuator_trnid[a,0] = joint id for joint transmissions
    # jnt_dofadr[jid] = dof index in qvel for that joint (hinge => 1 dof)
    for a in range(nu):
        jid = int(model.actuator_trnid[a, 0])
        assert model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        dof = int(model.jnt_dofadr[jid])
        gear = float(model.actuator_gear[a, 0])  # gear="..." in XML
        M[dof, a] = gear
    return M


# compute centroidal momentum matrix and centroidal momentum
def compute_centroidal_full(m: mujoco.MjModel, d: mujoco.MjData, body: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    whole-body centroidal quantities around system CoM 
    returns:
        `Ag`: (6,nv) The Centroidal Momentum Matrix 
        `h` : (6,) Centroidal Momentum 6D spatial-vector [lin_mom; ang_mom]
    """
    if body is not None and not (0 <= body < m.nbody):
        raise ValueError(f"body index {body} out of bounds for nbody={m.nbody}")
    elif body is None: # assume body is 0 
        body = 0 

    jacp = np.zeros((3, m.nv), dtype=float)
    H = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jacSubtreeCom(m, d, jacp, body)
    mujoco.mj_angmomMat(m, d, H, body)
    mass = float(m.body_subtreemass[body])
    Ag = np.vstack((mass * jacp, H))
    return Ag, Ag @ d.qvel


# build selection matrix S_T from model 
def build_S_T(model: mujoco.MjModel) -> np.ndarray:
    nv = model.nv 
    nu = model.nu 
    if not (0 <= nu <= nv): # we're assuming an underactuated model  
        raise ValueError(f"Invalid dimensions: nv={nv}, nu={nu}")
    S_T = np.zeros((nv, nu), dtype=float)
    for a in range(nu): 
        jid = int(model.actuator_trnid[a, 0])
        dof = int(model.jnt_dofadr[jid])
        S_T[dof, a] = 1.0 
    return S_T


# compute mass matrix M and bias terms h
def compute_M_h(m: mujoco.MjModel, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes:
      - `M(q)`: dense mass-inertia matrix, shape (nv, nv)
      - `h(q, dq)`: nonlinear effects (Coriolis + centrifugal + gravity), shape (nv,)

    MuJoCo stores h(q, dq) in `qfrc_bias`.
    """
    M = np.zeros((m.nv, m.nv), dtype=float)
    mujoco.mj_fullM(m, M, d.qM)
    M = 0.5 * (M + M.T)
    h = d.qfrc_bias.copy()
    return M, h


# return com state variables 
def compute_com_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]: 
    """ 
    returns: 
        `com_pos`: (3,) world-frame CoM of whole model
        `com_vel`: (3,) world-frame CoM velocity of whole model 
    """
    com = np.asarray(data.subtree_com[0], dtype=float).copy() 
    jacp = np.zeros((3, model.nv), dtype=float) 
    mujoco.mj_jacSubtreeCom(model, data, jacp, 0)
    com_vel = jacp @ data.qvel 
    return com, com_vel


def compute_base_state(model: mujoco.MjModel, data: mujoco.MjData, base_bid: int, ) -> BaseState | None: 
    if not (0 <= base_bid < model.nbody): 
        raise ValueError(f"base_body_id out of range: {base_bid}")
    R = np.asarray(data.xmat[base_bid], dtype=float).reshape(3,3).copy()
    res = np.zeros((6,), dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_bid, res, 0)
    omega = res[:3].copy()
    from lie_math import logvec
    phi = logvec(R)
    return BaseState(R_world=R, omega_world=omega, phi_world=phi)


# compute hdot via finite differences from starting configuration q 
def compute_centroidal_hdot_kinematic(model: mujoco.MjModel, data: mujoco.MjData, dt_fd: float = 1e-6) -> np.ndarray: 
    """ 
    Finite-difference estimate of Hdot_G using: 
        hdot = (h(q+qdot*dt, qdot) - h(q,qdot)) / dt
        with qvel held constant 
    """
    Ag0, h0 = compute_centroidal_full(model, data, 0)

    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    act0 = data.act.copy() if model.na > 0 else None 

    qpos_plus = qpos0.copy()
    mujoco.mj_integratePos(model, qpos_plus, qvel0, dt_fd)

    data.qpos[:] = qpos_plus 
    data.qvel[:] = qvel0
    if act0 is not None: 
        data.act[:] = act0     
    mujoco.mj_forward(model, data)

    _, h_plus = compute_centroidal_full(model, data, 0)
    hdot = (h_plus - h0) / dt_fd

    # restore 
    data.qpos[:] = qpos0 
    data.qvel[:] = qvel0 
    if act0 is not None: 
        data.act[:] = act0
    mujoco.mj_forward(model, data)
    
    return hdot


# compute contact wrench map about a point  
def contact_wrench_resultant_map(site_pos_world: np.ndarray, about_point_world: np.ndarray | None = None) -> np.ndarray: 
    """ 
    Build linear map W s.t.: 
        [f_res; tau_res] = W * lambda 
    where lambda stacks contact forces at each site in world frame: [fx1 fy1 fz1 fx2 fy2 ...]^T

    tau_res is moment about `about_point_world` if provided; else about world origin. 
    inputs: 
    `site_pos_world`: (nc,3)
    `about_point_world`: (3,)
    returns: 
    `W`: (6, 3*nc)
    """
    p = np.asarray(site_pos_world, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3: 
        raise ValueError(f"site_pos_world must be (nc,3), got {p.shape}")
    nc = p.shape[0]
    pref = np.zeros(3) if about_point_world is None else np.asarray(about_point_world, dtype=float).reshape(3,)
    
    W = np.zeros((6, 3 * nc), dtype=float)
    I3 = np.eye(3)
    for i in range(nc): 
        r = p[i] - pref
        rx = np.array([
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0]
        ], dtype=float)
        W[0:3, 3*i:3*i+3] = I3 # forces do not change
        W[3:6, 3*i:3*i+3] = rx # apply skew symmetric [pc_i - pref]_x operator to force above 
    return W


def actuator_dof_indices(model: mujoco.MjModel) -> np.ndarray: 
    """ 
    Returns dof index (qvel indexing) driven by each actuator a. 
    Shape: (nu,)
    Assumes joint transmissions (mTRN_JOINT)
    """
    assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
    nu = model.nu 
    dof = np.zeros(nu, dtype=int)
    for a in range(nu): 
        jid = int(model.actuator_trnid[a,0])
        dof[a] = int(model.jnt_dofadr[jid])
    return dof 


def actuated_dof_mask(model: mujoco.MjModel) -> np.ndarray: 
    """
    boolean mask over nv dofs: True if any actuator drives that dof 
    """
    nv = model.nv 
    mask = np.zeros(nv, dtype=bool)
    dof = actuator_dof_indices(model)
    mask[dof] = True 
    return mask 


def actuated_dof_indices_unique(model: mujoco.MjModel) -> np.ndarray: 
    """
    Sorted unique actuated dof indices (nv indexing)
    WARNING: if multiple actuators map to the same dof, uniqueness collapses them. 
    """
    dof = actuator_dof_indices(model)
    return np.sort(np.unique(dof))


def unactuated_dof_indices(model: mujoco.MjModel) -> np.ndarray: 
    nv = model.nv 
    mask = actuated_dof_mask(model)
    return np.flatnonzero(~mask).astype(int)


def tau_limits_per_dof(model: mujoco.MjModel) -> np.ndarray: 
    """
    Returns a length-nv vector tau_lim_dof where: 
        - tau_lim_dof[i] is finite if dof i is actuated by exactly one actuator, 
        - tau_lim_dof[i] = +inf if dof i is unactuated 
        - raises error if more than one actuator maps to any given dof 

    Uses actuator ctrlrange and gear: 
        |tau_actuator| <= |gear * ctrl|
    """
    nv, nu = model.nv, model.nu
    dof = actuator_dof_indices(model)

    # actuator-space torque limits 
    lim_act = tau_limits(model)
    if lim_act.shape != (nu,): 
        raise ValueError("tau_limits(model) must return (nv,)")
    
    # detect multiple actuators on same dof 
    counts = np.zeros(nv, dtype=float)
    for a in range(nu): 
        counts[dof[a]] += 1 
    if np.any(counts > 1): 
        bad = np.flatnonzero(counts > 1).tolist()
        raise ValueError(f"Multiple actuators mapped to the same dof(s) {bad}; "
                         f"Model not supported or define a new policy")
    
    tau_lim_dof = np.full(nv, np.inf, dtype=float)
    for a in range(nu): 
        tau_lim_dof[dof[a]] = float(lim_act[a])
    return tau_lim_dof 