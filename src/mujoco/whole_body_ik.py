from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mujoco

from lie_math import logvec
from misc import check_finite
from wbik_tasks import BaseOrientationTask, CoMTask, PostureTask, SiteTarget, WBIKTaskSet


@dataclass(frozen=True)
class IKConfig:
    max_iters: int = 10
    damping: float = 1e-3        # Levenberg-Marquardt
    step_size: float = 0.5       # integration step on dq
    w_com: float = 200.0
    w_base_rot: float = 5.0
    w_posture: float = 5.0
    w_site_pos: float = 10.0
    w_site_rot: float = 10.0
    base_body_id: int | None = None


def _site_pose_world(data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(data.site_xpos[site_id], dtype=float).reshape(3,).copy()
    R = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()
    return p, R


def _body_pose_world(data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(data.xpos[body_id], dtype=float).reshape(3,).copy()
    R = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    return p, R


def _body_rot_jac(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    Jp = np.zeros((3, model.nv), dtype=float)
    Jr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacBody(model, data, Jp, Jr, body_id)
    return Jr


def _com_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.asarray(data.subtree_com[0], dtype=float).reshape(3,).copy()


def _com_jac(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    J = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSubtreeCom(model, data, J, 0)
    return J


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    com_target: np.ndarray,
    site_targets: list[SiteTarget],
    qpos_nominal: np.ndarray | None,
    cfg: IKConfig,
    base_R_target: np.ndarray | None = None,
) -> np.ndarray:
    com_task = CoMTask(
        p_world=check_finite("com_target", np.asarray(com_target, dtype=float).reshape(3,)),
    )
    base_task = None
    if base_R_target is not None:
        base_task = BaseOrientationTask(
            R_world=check_finite(
                "base_R_target",
                np.asarray(base_R_target, dtype=float).reshape(3, 3),
            ),
        )
    posture_task = None
    if qpos_nominal is not None:
        posture_task = PostureTask(
            qpos_nominal=check_finite(
                "qpos_nominal",
                np.asarray(qpos_nominal, dtype=float).reshape(model.nq,),
            ),
        )
    tasks = WBIKTaskSet(
        com=com_task,
        site_targets=list(site_targets),
        base=base_task,
        posture=posture_task,
    )
    return solve_ik_tasks(model, data, tasks=tasks, cfg=cfg)


def solve_ik_tasks(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    tasks: WBIKTaskSet,
    cfg: IKConfig,
) -> np.ndarray:
    """
    Returns qpos_des computed by damped Gauss-Newton on stacked tasks.

    **Tasks**:
      - CoM position
      - Base/pelvis orientation (optional enforcement)
      - Site position
      - Site orientation
      - Nominal posture tracking in generalized position coordinates

    **Notes**:
      - Works in qvel-space increments dq and integrates with mj_integratePos.
      - Rotation errors use log(R_des R^T).
      - Posture task is implemented via mj_differentiatePos to correctly map
        qpos differences into nq/nv-consistent generalized coordinates.
      - data.qpos is restored before returning.
    """
    q0 = data.qpos.copy()
    com_target = check_finite("tasks.com.p_world", np.asarray(tasks.com.p_world, dtype=float).reshape(3,))
    qpos_nom = q0.copy() if tasks.posture is None else check_finite(
        "tasks.posture.qpos_nominal",
        np.asarray(tasks.posture.qpos_nominal, dtype=float).reshape(model.nq,),
    )
    posture_mask = np.ones(model.nv, dtype=float) if tasks.posture is None or tasks.posture.dof_weight_mask is None else check_finite(
        "tasks.posture.dof_weight_mask",
        np.asarray(tasks.posture.dof_weight_mask, dtype=float).reshape(model.nv,),
    )
    base_R_target = None if tasks.base is None else check_finite(
        "tasks.base.R_world",
        np.asarray(tasks.base.R_world, dtype=float).reshape(3, 3),
    )
    site_targets = list(tasks.site_targets)

    for _ in range(int(cfg.max_iters)):
        mujoco.mj_forward(model, data)

        rows: list[np.ndarray] = []
        rhs: list[np.ndarray] = []
        W: list[np.ndarray] = []

        # CoM position task
        com = _com_world(model, data)
        e_com = com_target - com
        J_com = _com_jac(model, data)
        rows.append(J_com)
        rhs.append(e_com)
        W.append(np.full(3, float(cfg.w_com), dtype=float))

        # Site tasks
        Jp = np.zeros((3, model.nv), dtype=float)
        Jr = np.zeros((3, model.nv), dtype=float)

        for st in site_targets:
            sid = int(st.site_id)
            p, R = _site_pose_world(data, sid)

            # Position
            p_des = check_finite(f"site_target[{sid}].p_world", np.asarray(st.p_world, dtype=float).reshape(3,))
            e_p = p_des - p
            mujoco.mj_jacSite(model, data, Jp, Jr, sid)
            rows.append(Jp.copy())
            rhs.append(e_p)
            W.append(np.full(3, float(cfg.w_site_pos), dtype=float))

            # Orientation
            R_des = check_finite(f"site_target[{sid}].R_world", np.asarray(st.R_world, dtype=float).reshape(3, 3))
            e_R = logvec(R_des @ R.T)
            rows.append(Jr.copy())
            rhs.append(e_R)
            W.append(np.full(3, float(cfg.w_site_rot), dtype=float))

        # Optional base/pelvis orientation task
        if cfg.base_body_id is not None:
            bid = int(cfg.base_body_id)
            if not (0 <= bid < model.nbody):
                raise ValueError(f"cfg.base_body_id={bid} out of range for nbody={model.nbody}")
            if base_R_target is None:
                raise ValueError("cfg.base_body_id is set but base_R_target is None")

            _, R_base = _body_pose_world(data, bid)
            e_R_base = logvec(base_R_target @ R_base.T)
            Jr_base = _body_rot_jac(model, data, bid)

            rows.append(Jr_base)
            rhs.append(e_R_base)
            W.append(np.full(3, float(cfg.w_base_rot), dtype=float))

        # Nominal posture task
        # Use MuJoCo's differentiatePos to correctly map qpos error -> qvel coords
        # such that applying dq would move current qpos toward qpos_nom.
        e_q = np.zeros(model.nv, dtype=float)
        mujoco.mj_differentiatePos(model, e_q, 1.0, data.qpos, qpos_nom)

        rows.append(np.eye(model.nv, dtype=float))
        rhs.append(e_q)
        W.append(float(cfg.w_posture) * posture_mask)

        # Weighted damped least squares:
        #   min || sqrt(W) (A dq - b) ||^2 + λ ||dq||^2
        A = np.vstack(rows)
        b = np.hstack(rhs)
        w = np.hstack(W)

        if np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("IK task weights must be finite and nonnegative")

        ws = np.sqrt(w)
        Aw = ws[:, None] * A
        bw = ws * b

        H = Aw.T @ Aw + float(cfg.damping) * np.eye(model.nv, dtype=float)
        g = Aw.T @ bw
        dq: np.ndarray = np.linalg.solve(H, g)

        mujoco.mj_integratePos(model, data.qpos, dq, float(cfg.step_size))

    q_des = data.qpos.copy()

    # Restore original state
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)

    return q_des
