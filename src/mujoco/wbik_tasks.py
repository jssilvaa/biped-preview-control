from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control_types import CentroidalDesired
from misc import check_finite


@dataclass(frozen=True)
class CoMTask:
    p_world: np.ndarray
    v_world: np.ndarray | None = None
    a_world: np.ndarray | None = None


@dataclass(frozen=True)
class BaseOrientationTask:
    R_world: np.ndarray
    omega_world: np.ndarray | None = None
    alpha_world: np.ndarray | None = None


@dataclass(frozen=True)
class SiteTarget:
    site_id: int
    p_world: np.ndarray
    R_world: np.ndarray
    v_world: np.ndarray | None = None
    omega_world: np.ndarray | None = None


@dataclass(frozen=True)
class PostureTask:
    qpos_nominal: np.ndarray
    dof_weight_mask: np.ndarray | None = None


@dataclass(frozen=True)
class WBIKTaskSet:
    com: CoMTask
    site_targets: list[SiteTarget]
    base: BaseOrientationTask | None = None
    posture: PostureTask | None = None


def build_wbik_task_set(
    *,
    desired: CentroidalDesired,
    site_targets: list[SiteTarget],
    qpos_nominal: np.ndarray | None,
    posture_dof_weight_mask: np.ndarray | None = None,
) -> WBIKTaskSet:
    com = CoMTask(
        p_world=check_finite("desired.com", np.asarray(desired.com, dtype=float).reshape(3,)),
        v_world=check_finite("desired.com_vel", np.asarray(desired.com_vel, dtype=float).reshape(3,)),
        a_world=None if desired.com_acc is None else check_finite(
            "desired.com_acc",
            np.asarray(desired.com_acc, dtype=float).reshape(3,),
        ),
    )

    base = None
    if desired.base_R_world is not None:
        base = BaseOrientationTask(
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

    posture = None
    if qpos_nominal is not None:
        posture = PostureTask(
            qpos_nominal=check_finite(
                "qpos_nominal",
                np.asarray(qpos_nominal, dtype=float),
            ).copy(),
            dof_weight_mask=None if posture_dof_weight_mask is None else check_finite(
                "posture_dof_weight_mask",
                np.asarray(posture_dof_weight_mask, dtype=float),
            ).copy(),
        )

    return WBIKTaskSet(
        com=com,
        site_targets=list(site_targets),
        base=base,
        posture=posture,
    )
