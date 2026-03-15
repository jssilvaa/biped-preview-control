from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np

from control_types import CentroidalDesired
from dynamics import actuator_dof_indices
from lie_math import logvec
from misc import check_finite
from wbik_qp import ReducedAccelTask, solve_hierarchical_accel_qp
from wbik_tasks import BaseOrientationTask, CoMTask, PostureTask, SiteTarget, WBIKTaskSet


@dataclass(frozen=True)
class AccelIKConfig:
    task_mode: Literal["paper_kinematic", "accel_feedforward_experimental"] = "paper_kinematic"
    solver_mode: Literal["weighted_ls", "staged_nullspace", "hierarchical_qp"] = "weighted_ls"
    damping: float = 1e-6
    nullspace_rcond: float = 1e-8
    w_com: float = 200.0
    w_base_rot: float = 5.0
    w_posture: float = 5.0
    w_site_pos: float = 10.0
    w_site_rot: float = 10.0
    kp_com: float = 100.0
    kd_com: float = 20.0
    kp_base_rot: float = 50.0
    kd_base_rot: float = 10.0
    kp_site_pos: float = 200.0
    kd_site_pos: float = 20.0
    kp_site_rot: float = 50.0
    kd_site_rot: float = 10.0
    kp_posture: float = 5.0
    kd_posture: float = 1.0
    hierarchical_max_com_acc: float = 2.0
    hierarchical_max_base_alpha: float = 6.0
    hierarchical_max_site_pos_acc: float = 1.5
    hierarchical_max_site_rot_acc: float = 10.0
    hierarchical_max_joint_acc: float | None = 200.0
    hierarchical_split_site_stages: bool = False
    hierarchical_preserve_stage1_tol: float = 0.05
    hierarchical_preserve_stage2_tol: float = 0.09
    hierarchical_preserve_site_pos_tol: float | None = None
    hierarchical_preserve_site_rot_tol: float | None = None
    hierarchical_preserve_com_tol: float | None = None
    hierarchical_preserve_base_tol: float | None = None
    hierarchical_preserve_retry_scale: float = 2.0
    hierarchical_preserve_max_retries: int = 2
    jdot_eps: float = 1e-6
    base_body_id: int | None = None
    max_actuated_position_error: float | None = 0.25


@dataclass(frozen=True)
class AccelIKResult:
    qacc_cmd: np.ndarray
    qvel_des: np.ndarray
    qpos_des: np.ndarray
    solver_mode: str = "weighted_ls"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task_residuals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _AccelTask:
    key: str
    idx: int | None
    J: np.ndarray
    jdot_qdot: np.ndarray
    cmd: np.ndarray
    weight: np.ndarray


def _clip_vec_norm(x: np.ndarray, max_norm: float | None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1,)
    if max_norm is None:
        return x.copy()
    max_norm = float(max_norm)
    if not np.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("Task command norm limits must be positive finite when set")
    nrm = float(np.linalg.norm(x))
    if nrm <= max_norm or nrm <= 1e-12:
        return x.copy()
    return (max_norm / nrm) * x


def _task_scale_for_key(task: _AccelTask, cfg: AccelIKConfig) -> np.ndarray:
    if task.key == "com_acc":
        scale = float(cfg.hierarchical_max_com_acc)
    elif task.key == "base_alpha":
        scale = float(cfg.hierarchical_max_base_alpha)
    elif task.key == "site_pos_acc":
        scale = float(cfg.hierarchical_max_site_pos_acc)
    elif task.key == "site_rot_acc":
        scale = float(cfg.hierarchical_max_site_rot_acc)
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid task scale for {task.key}: {scale}")
    return np.full(task.cmd.shape[0], scale, dtype=float)


def _zero3() -> np.ndarray:
    return np.zeros(3, dtype=float)


def _com_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    del model
    return np.asarray(data.subtree_com[0], dtype=float).reshape(3,).copy()


def _com_jac(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    J = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSubtreeCom(model, data, J, 0)
    return J


def _com_vel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return _com_jac(model, data) @ data.qvel


def _site_pose_world(data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(data.site_xpos[site_id], dtype=float).reshape(3,).copy()
    R = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()
    return p, R


def _site_jac(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    Jp = np.zeros((3, model.nv), dtype=float)
    Jr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
    return Jp, Jr


def _site_velocities(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    Jp, Jr = _site_jac(model, data, site_id)
    return Jp @ data.qvel, Jr @ data.qvel


def _body_pose_world(data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(data.xpos[body_id], dtype=float).reshape(3,).copy()
    R = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    return p, R


def _body_rot_jac(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    Jp = np.zeros((3, model.nv), dtype=float)
    Jr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacBody(model, data, Jp, Jr, body_id)
    return Jr


def _body_omega_world(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    return _body_rot_jac(model, data, body_id) @ data.qvel


def _restore_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    act: np.ndarray | None,
) -> None:
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    if act is not None and model.na > 0:
        data.act[:] = act
    mujoco.mj_forward(model, data)


def _actuated_dof_indices(model: mujoco.MjModel) -> np.ndarray:
    if model.nu <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.asarray(actuator_dof_indices(model), dtype=int))


def _actuated_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    if model.nu <= 0:
        return np.zeros(0, dtype=int)
    qpos_ids: list[int] = []
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        qpos_ids.append(int(model.jnt_qposadr[jid]))
    return np.unique(np.asarray(qpos_ids, dtype=int))


def _fd_jdot_qdot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    eps: float,
    velocity_fn,
) -> np.ndarray:
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError(f"jdot finite-difference step must be positive finite, got {eps}")

    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    act0 = data.act.copy() if model.na > 0 else None

    v0 = np.asarray(velocity_fn(model, data), dtype=float).copy()
    qpos_eps = qpos0.copy()
    mujoco.mj_integratePos(model, qpos_eps, qvel0, float(eps))
    data.qpos[:] = qpos_eps
    data.qvel[:] = qvel0
    if act0 is not None:
        data.act[:] = act0
    mujoco.mj_forward(model, data)
    v1 = np.asarray(velocity_fn(model, data), dtype=float).copy()

    _restore_state(model, data, qpos=qpos0, qvel=qvel0, act=act0)
    return (v1 - v0) / float(eps)


def integrate_desired_state(
    model: mujoco.MjModel,
    *,
    qpos_des: np.ndarray,
    qvel_des: np.ndarray,
    qacc_cmd: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(dt)
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"dt must be positive finite, got {dt}")

    qpos_des = check_finite("qpos_des", np.asarray(qpos_des, dtype=float).reshape(model.nq,))
    qvel_des = check_finite("qvel_des", np.asarray(qvel_des, dtype=float).reshape(model.nv,))
    qacc_cmd = check_finite("qacc_cmd", np.asarray(qacc_cmd, dtype=float).reshape(model.nv,))

    qvel_next = qvel_des + qacc_cmd * dt
    qpos_next = qpos_des.copy()
    mujoco.mj_integratePos(model, qpos_next, qvel_next, dt)
    return qpos_next, qvel_next


def _stack_reduced_stage(
    tasks: list[_AccelTask],
    *,
    solve_dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not tasks:
        return None

    A_blocks: list[np.ndarray] = []
    b_blocks: list[np.ndarray] = []
    w_blocks: list[np.ndarray] = []
    for task in tasks:
        J = check_finite(f"{task.key}.J", np.asarray(task.J, dtype=float))
        rhs = check_finite(f"{task.key}.rhs", np.asarray(task.cmd - task.jdot_qdot, dtype=float).reshape(-1,))
        weight = check_finite(f"{task.key}.weight", np.asarray(task.weight, dtype=float).reshape(-1,))
        if J.shape[0] != rhs.shape[0] or rhs.shape[0] != weight.shape[0]:
            raise ValueError(f"Inconsistent task dimensions for {task.key}: J={J.shape}, rhs={rhs.shape}, weight={weight.shape}")
        A_blocks.append(J[:, solve_dofs])
        b_blocks.append(rhs)
        w_blocks.append(weight)

    A = np.vstack(A_blocks)
    b = np.hstack(b_blocks)
    w = np.hstack(w_blocks)
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("Acceleration IK task weights must be finite and nonnegative")
    active = w > 0.0
    if not np.any(active):
        return None
    return A[active], b[active], w[active]


def _solve_weighted_ls_reduced(
    tasks: list[_AccelTask],
    *,
    solve_dofs: np.ndarray,
    damping: float,
    nv: int,
) -> np.ndarray:
    if solve_dofs.size == 0 or not tasks:
        return np.zeros(nv, dtype=float)

    stacked = _stack_reduced_stage(tasks, solve_dofs=solve_dofs)
    if stacked is None:
        return np.zeros(nv, dtype=float)
    A, b, w = stacked
    ws = np.sqrt(w)
    Aw = ws[:, None] * A
    bw = ws * b
    H = Aw.T @ Aw + float(damping) * np.eye(solve_dofs.size, dtype=float)
    g = Aw.T @ bw
    qacc_red = np.linalg.solve(H, g)
    qacc_cmd = np.zeros(nv, dtype=float)
    qacc_cmd[solve_dofs] = qacc_red
    return qacc_cmd


def _solve_staged_nullspace_reduced(
    stages: list[list[_AccelTask]],
    *,
    solve_dofs: np.ndarray,
    damping: float,
    nullspace_rcond: float,
    nv: int,
) -> np.ndarray:
    if solve_dofs.size == 0:
        return np.zeros(nv, dtype=float)

    nd = solve_dofs.size
    x = np.zeros(nd, dtype=float)
    N = np.eye(nd, dtype=float)
    for stage_tasks in stages:
        stacked = _stack_reduced_stage(stage_tasks, solve_dofs=solve_dofs)
        if stacked is None:
            continue
        A, b, w = stacked
        ws = np.sqrt(w)
        Aw = ws[:, None] * A
        bw = ws * b
        AwN = Aw @ N
        rhs = bw - Aw @ x
        H = AwN.T @ AwN + float(damping) * np.eye(nd, dtype=float)
        g = AwN.T @ rhs
        dz = np.linalg.solve(H, g)
        x = x + N @ dz
        if np.linalg.norm(AwN, ord="fro") > 0.0:
            N = N @ (np.eye(nd, dtype=float) - np.linalg.pinv(AwN, rcond=float(nullspace_rcond)) @ AwN)

    qacc_cmd = np.zeros(nv, dtype=float)
    qacc_cmd[solve_dofs] = x
    return qacc_cmd


def _reduced_stage_tasks(
    tasks: list[_AccelTask],
    *,
    solve_dofs: np.ndarray,
    cfg: AccelIKConfig,
) -> list[ReducedAccelTask]:
    out: list[ReducedAccelTask] = []
    for task in tasks:
        out.append(
            ReducedAccelTask(
                key=task.key,
                idx=task.idx,
                A=np.asarray(task.J[:, solve_dofs], dtype=float).copy(),
                b=np.asarray(task.cmd - task.jdot_qdot, dtype=float).reshape(-1,).copy(),
                weight=np.asarray(task.weight, dtype=float).reshape(-1,).copy(),
                scale=_task_scale_for_key(task, cfg),
            )
        )
    return out


def solve_accel_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    desired: CentroidalDesired,
    site_targets: list[SiteTarget],
    qpos_nominal: np.ndarray | None,
    qpos_des_prev: np.ndarray | None,
    qvel_des_prev: np.ndarray | None,
    cfg: AccelIKConfig,
    dt: float,
) -> AccelIKResult:
    com_task = CoMTask(
        p_world=check_finite("desired.com", np.asarray(desired.com, dtype=float).reshape(3,)),
        v_world=check_finite("desired.com_vel", np.asarray(desired.com_vel, dtype=float).reshape(3,)),
        a_world=None if desired.com_acc is None else check_finite(
            "desired.com_acc",
            np.asarray(desired.com_acc, dtype=float).reshape(3,),
        ),
    )
    base_task = None
    if desired.base_R_world is not None:
        base_task = BaseOrientationTask(
            R_world=check_finite(
                "desired.base_R_world",
                np.asarray(desired.base_R_world, dtype=float).reshape(3, 3),
            ),
            omega_world=None if desired.base_omega_world is None else check_finite(
                "desired.base_omega_world",
                np.asarray(desired.base_omega_world, dtype=float).reshape(3,),
            ),
            alpha_world=None if desired.base_alpha_world is None else check_finite(
                "desired.base_alpha_world",
                np.asarray(desired.base_alpha_world, dtype=float).reshape(3,),
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
    return solve_accel_ik_tasks(
        model,
        data,
        tasks=tasks,
        qpos_des_prev=qpos_des_prev,
        qvel_des_prev=qvel_des_prev,
        cfg=cfg,
        dt=dt,
    )


def solve_accel_ik_tasks(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    tasks: WBIKTaskSet,
    qpos_des_prev: np.ndarray | None,
    qvel_des_prev: np.ndarray | None,
    cfg: AccelIKConfig,
    dt: float,
) -> AccelIKResult:
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    act0 = data.act.copy() if model.na > 0 else None
    try:
        actuated_dofs = _actuated_dof_indices(model)
        actuated_qpos = _actuated_qpos_indices(model)
        solve_dofs = actuated_dofs if actuated_dofs.size > 0 else np.arange(model.nv, dtype=int)

        qpos_nom = qpos0.copy() if tasks.posture is None else check_finite(
            "tasks.posture.qpos_nominal",
            np.asarray(tasks.posture.qpos_nominal, dtype=float).reshape(model.nq,),
        )
        posture_mask = np.ones(model.nv, dtype=float) if tasks.posture is None or tasks.posture.dof_weight_mask is None else check_finite(
            "tasks.posture.dof_weight_mask",
            np.asarray(tasks.posture.dof_weight_mask, dtype=float).reshape(model.nv,),
        )
        qpos_des_seed = qpos0.copy()
        if qpos_des_prev is not None:
            qpos_des_prev = check_finite(
                "qpos_des_prev",
                np.asarray(qpos_des_prev, dtype=float).reshape(model.nq,),
            )
            seed_qpos = actuated_qpos if actuated_qpos.size > 0 else np.arange(model.nq, dtype=int)
            qpos_des_seed[seed_qpos] = qpos_des_prev[seed_qpos]
        qvel_des_seed = qvel0.copy()
        if qvel_des_prev is not None:
            qvel_des_prev = check_finite(
                "qvel_des_prev",
                np.asarray(qvel_des_prev, dtype=float).reshape(model.nv,),
            )
            qvel_des_seed[solve_dofs] = qvel_des_prev[solve_dofs]

        site_targets = list(tasks.site_targets)
        residuals: dict[str, Any] = {
            "com_acc": np.full(3, np.nan, dtype=float),
            "base_alpha": np.full(3, np.nan, dtype=float),
            "site_pos_acc": [np.full(3, np.nan, dtype=float) for _ in site_targets],
            "site_rot_acc": [np.full(3, np.nan, dtype=float) for _ in site_targets],
        }
        diagnostics: dict[str, Any] = {
            "stage1_slack_rms": np.nan,
            "stage2_slack_rms": np.nan,
            "stage1_preserved_resid_rms": np.nan,
            "stage2_preserved_resid_rms": np.nan,
            "joint_limit_active_count": 0,
            "preserve_retry_count": 0,
            "stage1_tol_used": np.nan,
            "stage2_tol_used": np.nan,
            "com_cmd_clipped": 0,
            "base_cmd_clipped": 0,
            "site_pos_cmd_clipped": 0,
            "site_rot_cmd_clipped": 0,
        }
        stage_site: list[_AccelTask] = []
        stage_centroid: list[_AccelTask] = []
        stage_posture: list[_AccelTask] = []
        task_cache: list[_AccelTask] = []

        desired_com = check_finite("tasks.com.p_world", np.asarray(tasks.com.p_world, dtype=float).reshape(3,))
        desired_com_vel = None if tasks.com.v_world is None else check_finite(
            "tasks.com.v_world",
            np.asarray(tasks.com.v_world, dtype=float).reshape(3,),
        )
        desired_com_acc = None if tasks.com.a_world is None else check_finite(
            "tasks.com.a_world",
            np.asarray(tasks.com.a_world, dtype=float).reshape(3,),
        )

        if float(cfg.w_com) > 0.0:
            if cfg.task_mode == "accel_feedforward_experimental" and desired_com_acc is None:
                raise ValueError("tasks.com.a_world is required for accel_feedforward_experimental mode")
            com = _com_world(model, data)
            com_vel = _com_vel(model, data)
            J_com = _com_jac(model, data)
            jdot_qdot = _fd_jdot_qdot(model, data, eps=float(cfg.jdot_eps), velocity_fn=_com_vel)
            vel_err = np.zeros(3, dtype=float) if desired_com_vel is None else (desired_com_vel - com_vel)
            if cfg.task_mode == "paper_kinematic":
                com_cmd = float(cfg.kp_com) * (desired_com - com) + float(cfg.kd_com) * vel_err
            elif cfg.task_mode == "accel_feedforward_experimental":
                com_cmd = desired_com_acc + float(cfg.kp_com) * (desired_com - com) + float(cfg.kd_com) * vel_err
            else:
                raise ValueError(f"Unsupported AccelIKConfig.task_mode: {cfg.task_mode}")
            if cfg.solver_mode == "hierarchical_qp":
                com_cmd_clipped = _clip_vec_norm(com_cmd, cfg.hierarchical_max_com_acc)
                if not np.allclose(com_cmd_clipped, com_cmd):
                    diagnostics["com_cmd_clipped"] = 1
                com_cmd = com_cmd_clipped
            task = _AccelTask(
                key="com_acc",
                idx=None,
                J=J_com.copy(),
                jdot_qdot=jdot_qdot.copy(),
                cmd=com_cmd.copy(),
                weight=np.full(3, float(cfg.w_com), dtype=float),
            )
            stage_centroid.append(task)
            task_cache.append(task)

        if float(cfg.w_base_rot) > 0.0 and cfg.base_body_id is not None:
            bid = int(cfg.base_body_id)
            if tasks.base is None:
                raise ValueError("Base rotation task is required when cfg.w_base_rot > 0 and base_body_id is set")
            _, R_base = _body_pose_world(data, bid)
            omega_base = _body_omega_world(model, data, bid)
            J_base = _body_rot_jac(model, data, bid)
            desired_R = check_finite("tasks.base.R_world", np.asarray(tasks.base.R_world, dtype=float).reshape(3, 3))
            desired_omega = _zero3() if tasks.base.omega_world is None else check_finite(
                "tasks.base.omega_world",
                np.asarray(tasks.base.omega_world, dtype=float).reshape(3,),
            )
            desired_alpha = _zero3() if tasks.base.alpha_world is None else check_finite(
                "tasks.base.alpha_world",
                np.asarray(tasks.base.alpha_world, dtype=float).reshape(3,),
            )
            jdot_qdot = _fd_jdot_qdot(
                model,
                data,
                eps=float(cfg.jdot_eps),
                velocity_fn=lambda m, d: _body_omega_world(m, d, bid),
            )
            if cfg.task_mode == "paper_kinematic":
                base_cmd = float(cfg.kp_base_rot) * logvec(desired_R @ R_base.T) + float(cfg.kd_base_rot) * (desired_omega - omega_base)
            elif cfg.task_mode == "accel_feedforward_experimental":
                base_cmd = desired_alpha + float(cfg.kp_base_rot) * logvec(desired_R @ R_base.T) + float(cfg.kd_base_rot) * (desired_omega - omega_base)
            else:
                raise ValueError(f"Unsupported AccelIKConfig.task_mode: {cfg.task_mode}")
            if cfg.solver_mode == "hierarchical_qp":
                base_cmd_clipped = _clip_vec_norm(base_cmd, cfg.hierarchical_max_base_alpha)
                if not np.allclose(base_cmd_clipped, base_cmd):
                    diagnostics["base_cmd_clipped"] = 1
                base_cmd = base_cmd_clipped
            task = _AccelTask(
                key="base_alpha",
                idx=None,
                J=J_base.copy(),
                jdot_qdot=jdot_qdot.copy(),
                cmd=base_cmd.copy(),
                weight=np.full(3, float(cfg.w_base_rot), dtype=float),
            )
            stage_centroid.append(task)
            task_cache.append(task)

        for idx, st in enumerate(site_targets):
            sid = int(st.site_id)
            p, R = _site_pose_world(data, sid)
            Jp, Jr = _site_jac(model, data, sid)
            v_site, omega_site = _site_velocities(model, data, sid)
            p_des = check_finite(f"site_target[{sid}].p_world", np.asarray(st.p_world, dtype=float).reshape(3,))
            R_des = check_finite(f"site_target[{sid}].R_world", np.asarray(st.R_world, dtype=float).reshape(3, 3))
            v_des = np.zeros(3, dtype=float) if st.v_world is None else check_finite(
                f"site_target[{sid}].v_world",
                np.asarray(st.v_world, dtype=float).reshape(3,),
            )
            omega_des = np.zeros(3, dtype=float) if st.omega_world is None else check_finite(
                f"site_target[{sid}].omega_world",
                np.asarray(st.omega_world, dtype=float).reshape(3,),
            )

            if float(cfg.w_site_pos) > 0.0:
                jdot_qdot_p = _fd_jdot_qdot(
                    model,
                    data,
                    eps=float(cfg.jdot_eps),
                    velocity_fn=lambda m, d, sid=sid: _site_velocities(m, d, sid)[0],
                )
                site_p_cmd = float(cfg.kp_site_pos) * (p_des - p) + float(cfg.kd_site_pos) * (v_des - v_site)
                if cfg.solver_mode == "hierarchical_qp":
                    site_p_cmd_clipped = _clip_vec_norm(site_p_cmd, cfg.hierarchical_max_site_pos_acc)
                    if not np.allclose(site_p_cmd_clipped, site_p_cmd):
                        diagnostics["site_pos_cmd_clipped"] = int(diagnostics["site_pos_cmd_clipped"]) + 1
                    site_p_cmd = site_p_cmd_clipped
                task = _AccelTask(
                    key="site_pos_acc",
                    idx=idx,
                    J=Jp.copy(),
                    jdot_qdot=jdot_qdot_p.copy(),
                    cmd=site_p_cmd.copy(),
                    weight=np.full(3, float(cfg.w_site_pos), dtype=float),
                )
                stage_site.append(task)
                task_cache.append(task)

            if float(cfg.w_site_rot) > 0.0:
                jdot_qdot_r = _fd_jdot_qdot(
                    model,
                    data,
                    eps=float(cfg.jdot_eps),
                    velocity_fn=lambda m, d, sid=sid: _site_velocities(m, d, sid)[1],
                )
                site_r_cmd = float(cfg.kp_site_rot) * logvec(R_des @ R.T) + float(cfg.kd_site_rot) * (omega_des - omega_site)
                if cfg.solver_mode == "hierarchical_qp":
                    site_r_cmd_clipped = _clip_vec_norm(site_r_cmd, cfg.hierarchical_max_site_rot_acc)
                    if not np.allclose(site_r_cmd_clipped, site_r_cmd):
                        diagnostics["site_rot_cmd_clipped"] = int(diagnostics["site_rot_cmd_clipped"]) + 1
                    site_r_cmd = site_r_cmd_clipped
                task = _AccelTask(
                    key="site_rot_acc",
                    idx=idx,
                    J=Jr.copy(),
                    jdot_qdot=jdot_qdot_r.copy(),
                    cmd=site_r_cmd.copy(),
                    weight=np.full(3, float(cfg.w_site_rot), dtype=float),
                )
                stage_site.append(task)
                task_cache.append(task)

        if float(cfg.w_posture) > 0.0:
            e_q = np.zeros(model.nv, dtype=float)
            mujoco.mj_differentiatePos(model, e_q, 1.0, data.qpos, qpos_nom)
            ddq_nom = float(cfg.kp_posture) * e_q - float(cfg.kd_posture) * qvel0
            stage_posture.append(
                _AccelTask(
                    key="posture",
                    idx=None,
                    J=np.eye(model.nv, dtype=float),
                    jdot_qdot=np.zeros(model.nv, dtype=float),
                    cmd=ddq_nom.copy(),
                    weight=float(cfg.w_posture) * posture_mask,
                )
            )

        if cfg.solver_mode == "weighted_ls":
            qacc_cmd = _solve_weighted_ls_reduced(
                stage_site + stage_centroid + stage_posture,
                solve_dofs=solve_dofs,
                damping=float(cfg.damping),
                nv=model.nv,
            )
        elif cfg.solver_mode == "staged_nullspace":
            qacc_cmd = _solve_staged_nullspace_reduced(
                [stage_site, stage_centroid, stage_posture],
                solve_dofs=solve_dofs,
                damping=float(cfg.damping),
                nullspace_rcond=float(cfg.nullspace_rcond),
                nv=model.nv,
            )
        elif cfg.solver_mode == "hierarchical_qp":
            reduced_site_tasks = _reduced_stage_tasks(stage_site, solve_dofs=solve_dofs, cfg=cfg)
            reduced_centroid_tasks = _reduced_stage_tasks(stage_centroid, solve_dofs=solve_dofs, cfg=cfg)
            if stage_posture:
                posture_task = stage_posture[0]
                posture_cmd = np.asarray(posture_task.cmd[solve_dofs], dtype=float).reshape(-1,)
                posture_weight = np.asarray(posture_task.weight[solve_dofs], dtype=float).reshape(-1,)
            else:
                posture_cmd = np.zeros(solve_dofs.size, dtype=float)
                posture_weight = np.zeros(solve_dofs.size, dtype=float)
            qp_result = solve_hierarchical_accel_qp(
                model,
                solve_dofs=solve_dofs,
                qpos_seed=qpos_des_seed,
                qvel_seed=qvel_des_seed,
                dt=float(dt),
                stage_site_tasks=reduced_site_tasks,
                stage_centroid_tasks=reduced_centroid_tasks,
                posture_cmd=posture_cmd,
                posture_weight=posture_weight,
                damping=float(cfg.damping),
                joint_acc_limit=cfg.hierarchical_max_joint_acc,
                split_site_stages=bool(cfg.hierarchical_split_site_stages),
                preserve_stage1_tol=float(cfg.hierarchical_preserve_stage1_tol),
                preserve_stage2_tol=float(cfg.hierarchical_preserve_stage2_tol),
                preserve_site_pos_tol=None if cfg.hierarchical_preserve_site_pos_tol is None else float(cfg.hierarchical_preserve_site_pos_tol),
                preserve_site_rot_tol=None if cfg.hierarchical_preserve_site_rot_tol is None else float(cfg.hierarchical_preserve_site_rot_tol),
                preserve_com_tol=None if cfg.hierarchical_preserve_com_tol is None else float(cfg.hierarchical_preserve_com_tol),
                preserve_base_tol=None if cfg.hierarchical_preserve_base_tol is None else float(cfg.hierarchical_preserve_base_tol),
                preserve_retry_scale=float(cfg.hierarchical_preserve_retry_scale),
                preserve_max_retries=int(cfg.hierarchical_preserve_max_retries),
            )
            qacc_cmd = np.zeros(model.nv, dtype=float)
            qacc_cmd[solve_dofs] = qp_result.qacc_red
            diagnostics = {
                **diagnostics,
                "stage1_slack_rms": float(qp_result.stage1_slack_rms),
                "stage2_slack_rms": float(qp_result.stage2_slack_rms),
                "stage1_preserved_resid_rms": float(qp_result.stage1_preserved_resid_rms),
                "stage2_preserved_resid_rms": float(qp_result.stage2_preserved_resid_rms),
                "joint_limit_active_count": int(qp_result.joint_limit_active_count),
                "preserve_retry_count": int(qp_result.preserve_retry_count),
                "stage1_tol_used": float(qp_result.stage1_tol_used),
                "stage2_tol_used": float(qp_result.stage2_tol_used),
            }
        else:
            raise ValueError(f"Unsupported AccelIKConfig.solver_mode: {cfg.solver_mode}")

        qacc_cmd = check_finite("qacc_cmd", np.asarray(qacc_cmd, dtype=float).reshape(model.nv,))
        qpos_des, qvel_des = integrate_desired_state(
            model,
            qpos_des=qpos_des_seed,
            qvel_des=qvel_des_seed,
            qacc_cmd=qacc_cmd,
            dt=dt,
        )
        if cfg.max_actuated_position_error is not None and actuated_qpos.size > 0:
            max_err = float(cfg.max_actuated_position_error)
            if max_err <= 0.0 or not np.isfinite(max_err):
                raise ValueError("cfg.max_actuated_position_error must be positive finite when set")
            dq = qpos_des[actuated_qpos] - qpos0[actuated_qpos]
            qpos_des[actuated_qpos] = qpos0[actuated_qpos] + np.clip(dq, -max_err, max_err)

        for task in task_cache:
            actual = task.J @ qacc_cmd + task.jdot_qdot
            err = actual - task.cmd
            if task.key == "com_acc":
                residuals["com_acc"] = err
            elif task.key == "base_alpha":
                residuals["base_alpha"] = err
            elif task.key == "site_pos_acc":
                residuals["site_pos_acc"][int(task.idx)] = err
            elif task.key == "site_rot_acc":
                residuals["site_rot_acc"][int(task.idx)] = err

        return AccelIKResult(
            qacc_cmd=qacc_cmd,
            qvel_des=qvel_des,
            qpos_des=qpos_des,
            solver_mode=str(cfg.solver_mode),
            diagnostics=diagnostics,
            task_residuals=residuals,
        )
    finally:
        _restore_state(model, data, qpos=qpos0, qvel=qvel0, act=act0)
