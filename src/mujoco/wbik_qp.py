from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import osqp

from misc import to_csc


@dataclass(frozen=True)
class ReducedAccelTask:
    key: str
    idx: int | None
    A: np.ndarray
    b: np.ndarray
    weight: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class ActiveTaskBlock:
    key: str
    idx: int | None
    A: np.ndarray
    b: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class HierarchicalQPResult:
    qacc_red: np.ndarray
    stage1_slack_rms: float
    stage2_slack_rms: float
    stage1_preserved_resid_rms: float
    stage2_preserved_resid_rms: float
    joint_limit_active_count: int
    preserve_retry_count: int
    stage1_tol_used: float
    stage2_tol_used: float


def _active_task_blocks(tasks: list[ReducedAccelTask]) -> list[ActiveTaskBlock]:
    out: list[ActiveTaskBlock] = []
    for task in tasks:
        A = np.asarray(task.A, dtype=float)
        b = np.asarray(task.b, dtype=float).reshape(-1,)
        w = np.asarray(task.weight, dtype=float).reshape(-1,)
        scale = np.asarray(task.scale, dtype=float).reshape(-1,)
        if A.shape[0] != b.shape[0] or b.shape[0] != w.shape[0]:
            raise ValueError(f"Inconsistent ReducedAccelTask dimensions for {task.key}")
        if scale.shape != b.shape:
            raise ValueError(f"Task scale shape mismatch for {task.key}")
        if np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError(f"Task weights for {task.key} must be finite and nonnegative")
        if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
            raise ValueError(f"Task scales for {task.key} must be finite and positive")
        A_scaled = A / scale[:, None]
        b_scaled = b / scale
        active = w > 0.0
        if not np.any(active):
            continue
        out.append(
            ActiveTaskBlock(
                key=task.key,
                idx=task.idx,
                A=A_scaled[active],
                b=b_scaled[active],
                weight=w[active],
            )
        )
    return out


def _stack_active_task_blocks(blocks: list[ActiveTaskBlock]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not blocks:
        return None
    A = np.vstack([blk.A for blk in blocks])
    b = np.hstack([blk.b for blk in blocks])
    w = np.hstack([blk.weight for blk in blocks])
    return A, b, w


def _family_tol_for_key(
    key: str,
    *,
    default_tol: float,
    preserve_site_pos_tol: float | None,
    preserve_site_rot_tol: float | None,
    preserve_com_tol: float | None,
    preserve_base_tol: float | None,
) -> float:
    if key == "site_pos_acc":
        tol = default_tol if preserve_site_pos_tol is None else float(preserve_site_pos_tol)
    elif key == "site_rot_acc":
        tol = default_tol if preserve_site_rot_tol is None else float(preserve_site_rot_tol)
    elif key == "com_acc":
        tol = default_tol if preserve_com_tol is None else float(preserve_com_tol)
    elif key == "base_alpha":
        tol = default_tol if preserve_base_tol is None else float(preserve_base_tol)
    else:
        tol = float(default_tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"Preservation tolerance for {key} must be finite and >= 0, got {tol}")
    return tol


def _preserve_blocks_from_active(
    blocks: list[ActiveTaskBlock],
    *,
    x_ref: np.ndarray,
    default_tol: float,
    preserve_site_pos_tol: float | None,
    preserve_site_rot_tol: float | None,
    preserve_com_tol: float | None,
    preserve_base_tol: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float]:
    preserve_A: list[np.ndarray] = []
    preserve_l: list[np.ndarray] = []
    preserve_u: list[np.ndarray] = []
    used_tols: list[float] = []
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1,)
    for block in blocks:
        tol = _family_tol_for_key(
            block.key,
            default_tol=default_tol,
            preserve_site_pos_tol=preserve_site_pos_tol,
            preserve_site_rot_tol=preserve_site_rot_tol,
            preserve_com_tol=preserve_com_tol,
            preserve_base_tol=preserve_base_tol,
        )
        y = block.A @ x_ref
        preserve_A.append(block.A)
        preserve_l.append(y - tol)
        preserve_u.append(y + tol)
        used_tols.append(tol)
    mean_tol = float(np.mean(used_tols)) if used_tols else 0.0
    return preserve_A, preserve_l, preserve_u, mean_tol


def _joint_position_bounds(
    model: mujoco.MjModel,
    *,
    solve_dofs: np.ndarray,
    qpos_seed: np.ndarray,
    qvel_seed: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    nd = int(solve_dofs.size)
    lb = np.full(nd, -np.inf, dtype=float)
    ub = np.full(nd, np.inf, dtype=float)
    dt = float(dt)
    dt2 = dt * dt
    if dt2 <= 0.0 or not np.isfinite(dt2):
        raise ValueError(f"dt must be positive finite, got {dt}")

    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    for red_idx, dof in enumerate(np.asarray(solve_dofs, dtype=int).reshape(-1,)):
        jid = int(model.dof_jntid[dof])
        if jid < 0:
            continue
        if not bool(model.jnt_limited[jid]):
            continue
        jtype = int(model.jnt_type[jid])
        if jtype not in (hinge, slide):
            continue
        qadr = int(model.jnt_qposadr[jid])
        q = float(qpos_seed[qadr])
        qd = float(qvel_seed[dof])
        qlo = float(model.jnt_range[jid, 0])
        qhi = float(model.jnt_range[jid, 1])
        lb[red_idx] = (qlo - q - qd * dt) / dt2
        ub[red_idx] = (qhi - q - qd * dt) / dt2
    return lb, ub


def _stack_constraint_blocks(
    *,
    nd: int,
    preserve_A: list[np.ndarray],
    preserve_l: list[np.ndarray],
    preserve_u: list[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    slack_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_rows: list[np.ndarray] = []
    l_rows: list[np.ndarray] = []
    u_rows: list[np.ndarray] = []

    for Aeq, leq, ueq in zip(preserve_A, preserve_l, preserve_u):
        Aeq = np.asarray(Aeq, dtype=float)
        leq = np.asarray(leq, dtype=float).reshape(-1,)
        ueq = np.asarray(ueq, dtype=float).reshape(-1,)
        if Aeq.size == 0:
            continue
        if Aeq.shape[0] != leq.shape[0] or Aeq.shape[0] != ueq.shape[0]:
            raise ValueError("Preserved equality dimensions do not match")
        zeros = np.zeros((Aeq.shape[0], slack_dim), dtype=float)
        A_rows.append(np.hstack((Aeq, zeros)))
        l_rows.append(leq)
        u_rows.append(ueq)

    finite = np.isfinite(lb) | np.isfinite(ub)
    if np.any(finite):
        I = np.eye(nd, dtype=float)[finite]
        zeros = np.zeros((I.shape[0], slack_dim), dtype=float)
        A_rows.append(np.hstack((I, zeros)))
        l_rows.append(lb[finite])
        u_rows.append(ub[finite])

    if not A_rows:
        A = np.zeros((0, nd + slack_dim), dtype=float)
        l = np.zeros(0, dtype=float)
        u = np.zeros(0, dtype=float)
    else:
        A = np.vstack(A_rows)
        l = np.hstack(l_rows)
        u = np.hstack(u_rows)
    return A, l, u


def _solve_osqp(
    *,
    P: np.ndarray,
    q: np.ndarray,
    A: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    solver = osqp.OSQP()
    solver.setup(
        P=to_csc(P),
        q=np.asarray(q, dtype=float).reshape(-1,),
        A=to_csc(A),
        l=np.asarray(l, dtype=float).reshape(-1,),
        u=np.asarray(u, dtype=float).reshape(-1,),
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=10_000,
        adaptive_rho=True,
        polishing=False,
    )
    res = solver.solve()
    status = str(res.info.status).lower()
    if res.x is None or "solved" not in status:
        raise RuntimeError(f"OSQP failed: {res.info.status}")
    return np.asarray(res.x, dtype=float).reshape(-1,)


def _solve_slack_stage(
    *,
    nd: int,
    stage: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    preserve_A: list[np.ndarray],
    preserve_l: list[np.ndarray],
    preserve_u: list[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    reg: float,
) -> tuple[np.ndarray, float]:
    if stage is None:
        P = 2.0 * float(reg) * np.eye(nd, dtype=float)
        q = np.zeros(nd, dtype=float)
        A, l, u = _stack_constraint_blocks(
            nd=nd,
            preserve_A=preserve_A,
            preserve_l=preserve_l,
            preserve_u=preserve_u,
            lb=lb,
            ub=ub,
            slack_dim=0,
        )
        x = _solve_osqp(P=P, q=q, A=A, l=l, u=u)
        return x[:nd], 0.0

    A_task, b_task, w_task = stage
    m = int(A_task.shape[0])
    P = np.zeros((nd + m, nd + m), dtype=float)
    P[:nd, :nd] = 2.0 * float(reg) * np.eye(nd, dtype=float)
    P[nd:, nd:] = 2.0 * np.diag(np.square(w_task))
    q = np.zeros(nd + m, dtype=float)

    Aeq_task = np.hstack((A_task, -np.eye(m, dtype=float)))
    A_preserve, l_preserve, u_preserve = _stack_constraint_blocks(
        nd=nd,
        preserve_A=preserve_A,
        preserve_l=preserve_l,
        preserve_u=preserve_u,
        lb=lb,
        ub=ub,
        slack_dim=m,
    )
    A = np.vstack((Aeq_task, A_preserve))
    l = np.hstack((b_task, l_preserve))
    u = np.hstack((b_task, u_preserve))

    z = _solve_osqp(P=P, q=q, A=A, l=l, u=u)
    x = z[:nd]
    slack = z[nd:]
    slack_rms = float(np.sqrt(np.mean(np.square(slack)))) if slack.size > 0 else 0.0
    return x, slack_rms


def _solve_tracking_stage(
    *,
    x_ref: np.ndarray,
    weight: np.ndarray,
    preserve_A: list[np.ndarray],
    preserve_l: list[np.ndarray],
    preserve_u: list[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    reg: float,
) -> np.ndarray:
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1,)
    weight = np.asarray(weight, dtype=float).reshape(-1,)
    nd = int(x_ref.size)
    if weight.shape != (nd,):
        raise ValueError("Tracking-stage weight shape mismatch")
    if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
        raise ValueError("Tracking-stage weights must be finite and nonnegative")

    W2 = np.square(weight)
    P = 2.0 * (np.diag(W2) + float(reg) * np.eye(nd, dtype=float))
    q = -2.0 * W2 * x_ref
    A, l, u = _stack_constraint_blocks(
        nd=nd,
        preserve_A=preserve_A,
        preserve_l=preserve_l,
        preserve_u=preserve_u,
        lb=lb,
        ub=ub,
        slack_dim=0,
    )
    return _solve_osqp(P=P, q=q, A=A, l=l, u=u)


def _bound_activation_count(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, *, tol: float = 1e-6) -> int:
    x = np.asarray(x, dtype=float).reshape(-1,)
    lower_active = np.isfinite(lb) & (np.abs(x - lb) <= tol)
    upper_active = np.isfinite(ub) & (np.abs(x - ub) <= tol)
    return int(np.count_nonzero(lower_active | upper_active))


def solve_hierarchical_accel_qp(
    model: mujoco.MjModel,
    *,
    solve_dofs: np.ndarray,
    qpos_seed: np.ndarray,
    qvel_seed: np.ndarray,
    dt: float,
    stage_site_tasks: list[ReducedAccelTask],
    stage_centroid_tasks: list[ReducedAccelTask],
    posture_cmd: np.ndarray,
    posture_weight: np.ndarray,
    damping: float,
    joint_acc_limit: float | None,
    split_site_stages: bool,
    preserve_stage1_tol: float,
    preserve_stage2_tol: float,
    preserve_site_pos_tol: float | None,
    preserve_site_rot_tol: float | None,
    preserve_com_tol: float | None,
    preserve_base_tol: float | None,
    preserve_retry_scale: float,
    preserve_max_retries: int,
) -> HierarchicalQPResult:
    solve_dofs = np.asarray(solve_dofs, dtype=int).reshape(-1,)
    nd = int(solve_dofs.size)
    if nd == 0:
        return HierarchicalQPResult(
            qacc_red=np.zeros(0, dtype=float),
            stage1_slack_rms=0.0,
            stage2_slack_rms=0.0,
            stage1_preserved_resid_rms=0.0,
            stage2_preserved_resid_rms=0.0,
            joint_limit_active_count=0,
            preserve_retry_count=0,
            stage1_tol_used=0.0,
            stage2_tol_used=0.0,
        )

    lb, ub = _joint_position_bounds(
        model,
        solve_dofs=solve_dofs,
        qpos_seed=np.asarray(qpos_seed, dtype=float).reshape(model.nq,),
        qvel_seed=np.asarray(qvel_seed, dtype=float).reshape(model.nv,),
        dt=float(dt),
    )
    if joint_acc_limit is not None:
        joint_acc_limit = float(joint_acc_limit)
        if not np.isfinite(joint_acc_limit) or joint_acc_limit <= 0.0:
            raise ValueError("joint_acc_limit must be positive finite when set")
        lb = np.maximum(lb, -joint_acc_limit)
        ub = np.minimum(ub, joint_acc_limit)
    site_blocks_all = _active_task_blocks(stage_site_tasks)
    centroid_blocks = _active_task_blocks(stage_centroid_tasks)
    if bool(split_site_stages):
        site_pos_blocks = [blk for blk in site_blocks_all if blk.key == "site_pos_acc"]
        site_rot_blocks = [blk for blk in site_blocks_all if blk.key == "site_rot_acc"]
    else:
        site_pos_blocks = site_blocks_all
        site_rot_blocks = []
    stage_site_pos = _stack_active_task_blocks(site_pos_blocks)
    stage_site_rot = _stack_active_task_blocks(site_rot_blocks)
    stage_centroid = _stack_active_task_blocks(centroid_blocks)

    qacc_site_pos, site_pos_slack_rms = _solve_slack_stage(
        nd=nd,
        stage=stage_site_pos,
        preserve_A=[],
        preserve_l=[],
        preserve_u=[],
        lb=lb,
        ub=ub,
        reg=float(damping),
    )
    tol1_base = float(preserve_stage1_tol)
    tol2_base = float(preserve_stage2_tol)
    retry_scale = float(preserve_retry_scale)
    retry_limit = int(preserve_max_retries)
    if not np.isfinite(tol1_base) or tol1_base < 0.0:
        raise ValueError("preserve_stage1_tol must be finite and >= 0")
    if not np.isfinite(tol2_base) or tol2_base < 0.0:
        raise ValueError("preserve_stage2_tol must be finite and >= 0")
    if not np.isfinite(retry_scale) or retry_scale < 1.0:
        raise ValueError("preserve_retry_scale must be finite and >= 1")
    if retry_limit < 0:
        raise ValueError("preserve_max_retries must be >= 0")

    qacc_site_rot = np.zeros(nd, dtype=float)
    qacc_centroid = np.zeros(nd, dtype=float)
    qacc_final = np.zeros(nd, dtype=float)
    site_rot_slack_rms = 0.0
    stage2_slack_rms = 0.0
    used_tol1 = tol1_base
    used_tol2 = tol2_base
    retry_count = 0
    last_exc: Exception | None = None
    stage1_refs: list[np.ndarray] = []
    stage2_refs: list[np.ndarray] = []
    for retry in range(retry_limit + 1):
        scale = retry_scale ** retry
        used_tol1 = tol1_base * scale
        used_tol2 = tol2_base * scale
        preserve_A_try, preserve_l_try, preserve_u_try, site_pos_used_mean = _preserve_blocks_from_active(
            site_pos_blocks,
            x_ref=qacc_site_pos,
            default_tol=used_tol1,
            preserve_site_pos_tol=None if preserve_site_pos_tol is None else preserve_site_pos_tol * scale,
            preserve_site_rot_tol=None,
            preserve_com_tol=None,
            preserve_base_tol=None,
        )
        try:
            stage1_used_vals: list[float] = []
            if site_pos_blocks:
                stage1_used_vals.append(site_pos_used_mean)

            if stage_site_rot is not None:
                qacc_site_rot, site_rot_slack_rms = _solve_slack_stage(
                    nd=nd,
                    stage=stage_site_rot,
                    preserve_A=preserve_A_try,
                    preserve_l=preserve_l_try,
                    preserve_u=preserve_u_try,
                    lb=lb,
                    ub=ub,
                    reg=float(damping),
                )
                rot_A_try, rot_l_try, rot_u_try, site_rot_used_mean = _preserve_blocks_from_active(
                    site_rot_blocks,
                    x_ref=qacc_site_rot,
                    default_tol=used_tol1,
                    preserve_site_pos_tol=None,
                    preserve_site_rot_tol=None if preserve_site_rot_tol is None else preserve_site_rot_tol * scale,
                    preserve_com_tol=None,
                    preserve_base_tol=None,
                )
                preserve_A_try = [*preserve_A_try, *rot_A_try]
                preserve_l_try = [*preserve_l_try, *rot_l_try]
                preserve_u_try = [*preserve_u_try, *rot_u_try]
                stage1_used_vals.append(site_rot_used_mean)
            else:
                qacc_site_rot = qacc_site_pos.copy()
                site_rot_slack_rms = 0.0

            qacc_centroid, stage2_slack_rms = _solve_slack_stage(
                nd=nd,
                stage=stage_centroid,
                preserve_A=preserve_A_try,
                preserve_l=preserve_l_try,
                preserve_u=preserve_u_try,
                lb=lb,
                ub=ub,
                reg=float(damping),
            )
            stage2_A_try, stage2_l_try, stage2_u_try, stage2_used_mean = _preserve_blocks_from_active(
                centroid_blocks,
                x_ref=qacc_centroid,
                default_tol=used_tol2,
                preserve_site_pos_tol=None,
                preserve_site_rot_tol=None,
                preserve_com_tol=None if preserve_com_tol is None else preserve_com_tol * scale,
                preserve_base_tol=None if preserve_base_tol is None else preserve_base_tol * scale,
            )
            preserve_A_try = [*preserve_A_try, *stage2_A_try]
            preserve_l_try = [*preserve_l_try, *stage2_l_try]
            preserve_u_try = [*preserve_u_try, *stage2_u_try]
            qacc_final = _solve_tracking_stage(
                x_ref=np.asarray(posture_cmd, dtype=float).reshape(nd,),
                weight=np.asarray(posture_weight, dtype=float).reshape(nd,),
                preserve_A=preserve_A_try,
                preserve_l=preserve_l_try,
                preserve_u=preserve_u_try,
                lb=lb,
                ub=ub,
                reg=float(damping),
            )
            stage1_refs = [blk.A @ qacc_site_pos for blk in site_pos_blocks] + [blk.A @ qacc_site_rot for blk in site_rot_blocks]
            stage2_refs = [blk.A @ qacc_centroid for blk in centroid_blocks]
            used_tol1 = float(np.mean(stage1_used_vals)) if stage1_used_vals else 0.0
            used_tol2 = stage2_used_mean
            retry_count = retry
            last_exc = None
            break
        except RuntimeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc

    stage1_preserved_resid_rms = 0.0
    site_blocks_preserved = [*site_pos_blocks, *site_rot_blocks]
    if site_blocks_preserved:
        e1 = [blk.A @ qacc_final - y1 for blk, y1 in zip(site_blocks_preserved, stage1_refs)]
        e1_flat = np.hstack(e1)
        stage1_preserved_resid_rms = float(np.sqrt(np.mean(np.square(e1_flat))))

    stage2_preserved_resid_rms = 0.0
    if centroid_blocks:
        e2 = [blk.A @ qacc_final - y2 for blk, y2 in zip(centroid_blocks, stage2_refs)]
        e2_flat = np.hstack(e2)
        stage2_preserved_resid_rms = float(np.sqrt(np.mean(np.square(e2_flat))))

    site_stage_slacks = []
    if site_pos_blocks:
        site_stage_slacks.append(site_pos_slack_rms)
    if site_rot_blocks:
        site_stage_slacks.append(site_rot_slack_rms)
    stage1_slack_rms = float(np.sqrt(np.mean(np.square(site_stage_slacks)))) if site_stage_slacks else 0.0

    joint_limit_active_count = _bound_activation_count(qacc_final, lb, ub)
    return HierarchicalQPResult(
        qacc_red=qacc_final,
        stage1_slack_rms=stage1_slack_rms,
        stage2_slack_rms=stage2_slack_rms,
        stage1_preserved_resid_rms=stage1_preserved_resid_rms,
        stage2_preserved_resid_rms=stage2_preserved_resid_rms,
        joint_limit_active_count=joint_limit_active_count,
        preserve_retry_count=retry_count,
        stage1_tol_used=used_tol1,
        stage2_tol_used=used_tol2,
    )
