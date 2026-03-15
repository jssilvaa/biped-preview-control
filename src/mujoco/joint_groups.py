from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointGroupMap:
    dof_indices: dict[str, np.ndarray]
    qpos_indices: dict[str, np.ndarray]


def _joint_group_name(joint_name: str) -> str | None:
    if joint_name.startswith("left_hip_") or joint_name.startswith("left_knee_") or joint_name.startswith("left_ankle_"):
        return "left_leg"
    if joint_name.startswith("right_hip_") or joint_name.startswith("right_knee_") or joint_name.startswith("right_ankle_"):
        return "right_leg"
    if joint_name.startswith("left_shoulder_") or joint_name.startswith("left_elbow_") or joint_name.startswith("left_wrist_"):
        return "left_arm"
    if joint_name.startswith("right_shoulder_") or joint_name.startswith("right_elbow_") or joint_name.startswith("right_wrist_"):
        return "right_arm"
    if joint_name.startswith("waist_"):
        return "trunk"
    return None


def build_joint_group_map(model: mujoco.MjModel) -> JointGroupMap:
    dof_indices: dict[str, list[int]] = {}
    qpos_indices: dict[str, list[int]] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        group = _joint_group_name(name)
        if group is None:
            continue
        dof_adr = int(model.jnt_dofadr[jid])
        qpos_adr = int(model.jnt_qposadr[jid])
        if dof_adr >= 0:
            dof_indices.setdefault(group, []).append(dof_adr)
        qpos_indices.setdefault(group, []).append(qpos_adr)

    return JointGroupMap(
        dof_indices={k: np.asarray(v, dtype=int) for k, v in dof_indices.items()},
        qpos_indices={k: np.asarray(v, dtype=int) for k, v in qpos_indices.items()},
    )


def contact_group_from_site_name(site_name: str) -> str | None:
    if site_name == "left_foot":
        return "left_leg"
    if site_name == "right_foot":
        return "right_leg"
    if site_name == "left_palm":
        return "left_arm"
    if site_name == "right_palm":
        return "right_arm"
    return None


def posture_dof_weight_mask_from_contacts(
    model: mujoco.MjModel,
    *,
    site_names: list[str] | None,
    patch_active: np.ndarray | None,
    joint_groups: JointGroupMap | None = None,
) -> np.ndarray:
    mask = np.ones(model.nv, dtype=float)
    free_dofs = min(6, model.nv)
    mask[:free_dofs] = 0.0

    if joint_groups is None:
        joint_groups = build_joint_group_map(model)

    if site_names is None or patch_active is None:
        return mask

    active = np.asarray(patch_active, dtype=bool).reshape(-1)
    if len(site_names) != active.shape[0]:
        raise ValueError("site_names and patch_active must have the same length")

    for site_name, is_active in zip(site_names, active):
        if not is_active:
            continue
        group = contact_group_from_site_name(site_name)
        if group is None:
            continue
        dofs = joint_groups.dof_indices.get(group)
        if dofs is not None and dofs.size > 0:
            mask[dofs] = 0.0

    return mask
