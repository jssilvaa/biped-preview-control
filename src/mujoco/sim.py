# sim.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import mujoco

from misc import site_ids
from viz import Viz

from dynamics import compute_com_state, compute_base_state
from contact_patches import PatchSpec
from contact_patches import build_contact_model_from_sites
from contact_measurement import (
    build_patch_geom_map_from_sites,
    measure_patch_wrenches_world,
)

from reference_sequences import sine_com_bar_ref_seq, sine_com_ref_seq, zeros_bar_seq
from stack_controller import (
    StackController,
    StackControllerConfig,
    MurookaReferenceCommand,
)

from damping_control import ComplianceState, damping_step
from contact_phase import (
  PhaseGains, 
  ContactHysteresis, 
  normal_force_in_patch_frame, 
  select_patch_gains
)

from wbik_tasks import SiteTarget, build_wbik_task_set
from whole_body_ik import IKConfig, solve_ik_tasks
from whole_body_accel_ik import AccelIKConfig, solve_accel_ik_tasks
from joint_groups import build_joint_group_map, posture_dof_weight_mask_from_contacts
from joint_servo import (
  JointServoConfig,
  compute_affine_actuator_ctrl_from_joint_pd_targets,
  compute_motor_ctrl_from_qpos_target,
  compute_position_ctrl_from_qpos_target,
)
from lie_math import Exp
from murooka_wrench import contact_wrench_about_origin_to_bar


def _resultant_wrench_from_patch_wrenches(
  patch_wrenches_world: list[np.ndarray],
  contact_model,
) -> np.ndarray:
  w_total = np.zeros(6, dtype=float)
  for i, w_patch in enumerate(patch_wrenches_world):
    w_patch = np.asarray(w_patch, dtype=float).reshape(6,)
    p0 = np.asarray(contact_model.patches[i].p_w, dtype=float).reshape(3,)
    F = w_patch[:3]
    tau0 = w_patch[3:] + np.cross(p0, F)
    w_total[:3] += F
    w_total[3:] += tau0
  return w_total


def _model_uses_motor_actuators(model: mujoco.MjModel) -> bool:
  if model.nu <= 0:
    return False
  bias = np.asarray(model.actuator_biasprm[:, :3], dtype=float)
  return bool(np.allclose(bias, 0.0))


@dataclass(frozen=True)
class MurookaSimConfig:
  dt: float = 1e-3 
  N: int = 50_000 

  floor_geom_name: str = "floor"
  site_names: list[str] | None = None   # required 

  # contact patch geometry, per site 
  # map: site_name -> (Nv,3) vertex offsets expressed in SITE frame 
  site_vertex_offsets: dict[str, np.ndarray] | None = None 
  mu: float = 0.6 

  # centroidal / preview / stabilizer
  base_body_id: int | None = None       # must fill in for angular stabilization 
  I_diag: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=float))
  preview_dt: float = 5e-3 # match dt above
  preview_horizon_steps: int = 400 
  preview_controller_mode: str = "lqt"
  preview_linear_position_scale: float = 0.05
  preview_angular_position_scale: float = 0.10
  preview_nominal_freq_hz: float = 0.5
  preview_q_pos_override: float | None = None
  preview_q_wrench_override: float | None = None
  preview_r_jerk_override: float | None = None
  preview_ki_pos_override: float | None = None
  preview_state_source: str = "measured"
  preview_blend_alpha: float = 0.05
  enable_centroidal_stabilizer: bool = True
  execution_backend: str = "position_ik"

  # preview references 
  enable_motion_refs: bool = True 
  motion_axis: int = 0          # 0=x, 1=y, 2=z
  motion_amp: float = 0.2      # meters
  motion_freq_hz: float = 0.5   # Hz
  motion_match_bar_force_ref: bool = True
  reference_advance_steps: int = 0  # shift reference sequence forward in time to compensate preview lag

  # generator QP regularization terms 
  reg_projection: float = 1e-9
  reg_distribution: float = 1e-9
  w_tan_projection: float = 0.0 
  w_tan_distribution: float = 0.0 

  # contact phase logic
  phase_gains: PhaseGains = PhaseGains.murooka_table_ii()
  fn_on: float = 30.0 
  fn_off: float = 10.0 
  enable_damping_control: bool = True

  # IK 
  ik_cfg: IKConfig = IKConfig(base_body_id=base_body_id)
  accel_ik_cfg: AccelIKConfig = field(default_factory=AccelIKConfig)

  # observer hooks (inactive in Phase A)
  observer_external_bar_wrench_world: np.ndarray | None = None
  disturbance_gate_active: bool = False

  # joint servo 
  servo_mode: str = "auto"
  servo_cfg: JointServoConfig = JointServoConfig()

  # viz 
  viz: bool = False 
  display_every: int = 2 


def run_simulation( 
    model: mujoco.MjModel, 
    data: mujoco.MjData, 
    *, 
    cfg: MurookaSimConfig, 
) -> dict[str, Any]: 
  if cfg.site_names is None or len(cfg.site_names) == 0: 
    raise ValueError("cfg.site_names must be provided and non-empty")
  
  model.opt.timestep = float(cfg.dt)
  mujoco.mj_forward(model, data)

  sids = site_ids(model, cfg.site_names)
  nc = len(sids)
  joint_groups = build_joint_group_map(model)

  floor_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.floor_geom_name))
  if floor_gid < 0: 
    raise ValueError("floor geom not found")
  
  # Build patch specs once (stance patches)
  patch_specs: list[PatchSpec] = []
  for name, sid in zip(cfg.site_names, sids): 
    offs = None 
    if cfg.site_vertex_offsets is not None and name in cfg.site_vertex_offsets:
      offs = np.asarray(cfg.site_vertex_offsets[name], dtype=float)
    if offs is None: 
      offs = np.zeros((1,3), dtype=float)
    if offs.ndim != 2 or offs.shape[1] != 3 or not np.all(np.isfinite(offs)): 
      raise ValueError(f"site_vertex_offsets[{name}] must be finite (Nv,3), got {offs.shape}")
    patch_specs.append(PatchSpec(name=name, site_id=int(sid), vertex_offsets_site=offs))
  
  # Build Stack Controller
  stack = StackController(
    model=model, 
    cfg=StackControllerConfig(
      enable_preview_planner=True,
      enable_wrench_projection=True, 
      enable_wrench_distribution=True,
      enable_centroidal_stabilizer=bool(cfg.enable_centroidal_stabilizer),
      base_body_id=cfg.base_body_id,
      I_diag=np.asarray(cfg.I_diag, dtype=float).reshape(3,),
      mu=float(cfg.mu),
      preview_dt=float(cfg.preview_dt), 
      preview_horizon_steps=int(cfg.preview_horizon_steps),
      preview_controller_mode=str(cfg.preview_controller_mode),
      preview_linear_position_scale=float(cfg.preview_linear_position_scale),
      preview_angular_position_scale=float(cfg.preview_angular_position_scale),
      preview_nominal_freq_hz=float(cfg.preview_nominal_freq_hz),
      preview_q_pos_override=None if cfg.preview_q_pos_override is None else float(cfg.preview_q_pos_override),
      preview_q_wrench_override=None if cfg.preview_q_wrench_override is None else float(cfg.preview_q_wrench_override),
      preview_r_jerk_override=None if cfg.preview_r_jerk_override is None else float(cfg.preview_r_jerk_override),
      preview_ki_pos_override=None if cfg.preview_ki_pos_override is None else float(cfg.preview_ki_pos_override),
      preview_state_source=cfg.preview_state_source,
      preview_blend_alpha=float(cfg.preview_blend_alpha),
      reg_projection=float(cfg.reg_projection),
      reg_distribution=float(cfg.reg_distribution),
      w_tan_projection=float(cfg.w_tan_projection),
      w_tan_distribution=float(cfg.w_tan_distribution),
      contact_frame_mode="world_up",
    ),
  )

  # Patch to geom membership mapping 
  geom_map = build_patch_geom_map_from_sites(model, sids)

  # Compliance states per patch 
  comp = [ComplianceState.zero() for _ in range(nc)]
  qpos_des_state = data.qpos.copy()
  qvel_des_state = data.qvel.copy()

  ik_cfg = cfg.ik_cfg if cfg.ik_cfg.base_body_id is not None else replace(cfg.ik_cfg, base_body_id=cfg.base_body_id)
  accel_ik_cfg = cfg.accel_ik_cfg if cfg.accel_ik_cfg.base_body_id is not None else replace(cfg.accel_ik_cfg, base_body_id=cfg.base_body_id)

  # Nominal (stance) site pose targets (captured at k=0)
  mujoco.mj_forward(model, data)
  stance_pos = [np.asarray(data.site_xpos[s], dtype=float).reshape(3,).copy() for s in sids]
  stance_R = [np.asarray(data.site_xmat[s], dtype=float).reshape(3,3).copy() for s in sids]
  qpos_nominal = data.qpos.copy()

  # contact phase hysteresis (disabled for static + sinusoidal stances in phase 1)
  # hyst = ContactHysteresis(fn_on=float(cfg.fn_on), fn_off=float(cfg.fn_off))
  # hyst.reset(nc)

  viewer = Viz(model, data, floor_geom_name=cfg.floor_geom_name) if cfg.viz else None 
  
  #LOGS
  # Position + State Logs  
  q_log = np.zeros((cfg.N, model.nq), dtype=float)
  com_meas_log = np.zeros((cfg.N, 3), dtype=float)
  com_des_log = np.zeros((cfg.N, 3), dtype=float)
  com_preview_log = np.zeros((cfg.N, 3), dtype=float)
  com_ref_cmd_log = np.zeros((cfg.N, 3), dtype=float)
  patch_active_log = np.zeros((cfg.N, nc), dtype=bool)
  fn_log = np.zeros((cfg.N, nc), dtype=float)

  # Wrench Desired + Measured Logs 
  w_des_log = np.zeros((cfg.N, nc, 6), dtype=float)
  w_meas_log = np.zeros((cfg.N, nc, 6), dtype=float)
  dr_log = np.zeros((cfg.N, nc, 6), dtype=float)
  drdot_log = np.zeros((cfg.N, nc, 6), dtype=float)

  # Per block logs: 
  # Includes: 
  # - Planned, projected, desired 
  # - Commanded, real, and error norm between both 6D wrenches 
  bar_wp_log = np.zeros((cfg.N, 6), dtype=float)
  bar_wp_proj_log = np.zeros((cfg.N, 6), dtype=float)
  bar_wd_log = np.zeros((cfg.N, 6), dtype=float)
  bar_f_ref_cmd_log = np.zeros((cfg.N, 3), dtype=float)
  bar_n_ref_cmd_log = np.zeros((cfg.N, 3), dtype=float)

  
  w_cmd_log = np.zeros((cfg.N, 6), dtype=float)
  w_real_log = np.full((cfg.N, 6), np.nan, dtype=float)
  w_meas_resultant_log = np.full((cfg.N, 6), np.nan, dtype=float)
  w_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  w_force_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  w_moment_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  w_exec_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  w_exec_force_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  w_exec_moment_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  preview_state_pre_sync_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  preview_vel_pre_sync_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  preview_acc_pre_sync_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  preview_alpha_pre_sync_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  preview_acc_post_sync_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  preview_alpha_post_sync_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  stabilizer_overwrite_ratio_log = np.full(cfg.N, np.nan, dtype=float)
  qacc_cmd_log = np.full((cfg.N, model.nv), np.nan, dtype=float)
  qvel_des_log = np.full((cfg.N, model.nv), np.nan, dtype=float)
  qpos_des_log = np.full((cfg.N, model.nq), np.nan, dtype=float)
  ctrl_log = np.full((cfg.N, model.nu), np.nan, dtype=float)
  wbik_com_acc_residual_log = np.full((cfg.N, 3), np.nan, dtype=float)
  wbik_base_alpha_residual_log = np.full((cfg.N, 3), np.nan, dtype=float)
  wbik_site_pos_acc_residual_log = np.full((cfg.N, nc, 3), np.nan, dtype=float)
  wbik_site_rot_acc_residual_log = np.full((cfg.N, nc, 3), np.nan, dtype=float)
  wbik_stage1_slack_rms_log = np.full(cfg.N, np.nan, dtype=float)
  wbik_stage2_slack_rms_log = np.full(cfg.N, np.nan, dtype=float)
  wbik_stage1_preserved_resid_rms_log = np.full(cfg.N, np.nan, dtype=float)
  wbik_stage2_preserved_resid_rms_log = np.full(cfg.N, np.nan, dtype=float)
  wbik_joint_limit_active_count_log = np.full(cfg.N, np.nan, dtype=float)
  wbik_preserve_retry_count_log = np.full(cfg.N, np.nan, dtype=float)
  wbik_stage1_tol_used_log = np.full(cfg.N, np.nan, dtype=float)
  wbik_stage2_tol_used_log = np.full(cfg.N, np.nan, dtype=float)
  accel_ik_failed_log = np.zeros(cfg.N, dtype=bool)
  
  #NOTE: Due to the way the control is made, patches are measured from the previous state, and gains match the patch status from the previous control cycle 
  # patches with 1*cfg.dt seconds of delay
  patch_active = np.ones((nc,), dtype=bool)

  # Measure first CoM 
  com0_ref, _, = compute_com_state(model, data)
  total_mass = float(np.sum(model.body_mass))
  base_ref = compute_base_state(model, data, cfg.base_body_id) if cfg.base_body_id is not None else None 
  
  # initialize reference scalars 
  com_ref_world: np.ndarray | None = None 
  phi_ref_world: np.ndarray | None = None 

  # initialize reference sequences
  com_ref_seq: np.ndarray | None = None 
  phi_ref_seq: np.ndarray | None = None
  bar_f_ref_seq: np.ndarray | None = None
  bar_n_ref_seq: np.ndarray | None = None
  
  try: 
    for k in range(cfg.N): 
      mujoco.mj_forward(model, data)

      # Build contact model for current configuration 
      com_now, _ = compute_com_state(model, data)
      com_meas_log[k] = com_now 
      if base_ref is not None and phi_ref_world is None: 
        phi_ref_world = base_ref.phi_world

      # reference command construction 
      Nh = int(cfg.preview_horizon_steps)
      k_preview = int(np.floor((k * cfg.dt) / cfg.preview_dt)) if cfg.dt != cfg.preview_dt else k

      if cfg.enable_motion_refs:
          k_ref = k_preview + int(cfg.reference_advance_steps)
          if cfg.motion_match_bar_force_ref:
            com_ref_seq, bar_f_ref_seq = sine_com_bar_ref_seq(
              k_ref,
              cfg.preview_dt,
              Nh,
              com0_ref,
              axis=int(cfg.motion_axis),
              amp=float(cfg.motion_amp),
              freq_hz=float(cfg.motion_freq_hz),
              mass=total_mass,
            )
          else:
            com_ref_seq = sine_com_ref_seq(
              k_ref, cfg.preview_dt, Nh, com0_ref,
              axis=int(cfg.motion_axis),
              amp=float(cfg.motion_amp),
              freq_hz=float(cfg.motion_freq_hz),
            )
            bar_f_ref_seq = None
          com_ref_world = np.asarray(com_ref_seq[0], dtype=float).reshape(3,)
          # For metrics: log the actual current-time reference (not the advanced one)
          if int(cfg.reference_advance_steps) > 0:
            com_ref_now = sine_com_ref_seq(
              k_preview, cfg.preview_dt, 1, com0_ref,
              axis=int(cfg.motion_axis), amp=float(cfg.motion_amp), freq_hz=float(cfg.motion_freq_hz),
            )[0]
            com_ref_world_log = np.asarray(com_ref_now, dtype=float).reshape(3,)
          else:
            com_ref_world_log = com_ref_world
      else: 
          com_ref_world = np.asarray(com0_ref, dtype=float).reshape(3,)

      if phi_ref_world is not None: 
        phi_ref_seq = np.tile(phi_ref_world, (Nh,1))
      
      if bar_f_ref_seq is None:
        bar_f_ref_seq, bar_n_ref_seq = zeros_bar_seq(Nh)
      else:
        _, bar_n_ref_seq = zeros_bar_seq(Nh)
      bar_f_ref_world = np.asarray(bar_f_ref_seq[0], dtype=float).reshape(3,)
      bar_n_ref_world = np.asarray(bar_n_ref_seq[0], dtype=float).reshape(3,)

      com_ref_cmd_log[k] = com_ref_world_log if cfg.enable_motion_refs and int(cfg.reference_advance_steps) > 0 else com_ref_world
      bar_f_ref_cmd_log[k] = bar_f_ref_world
      bar_n_ref_cmd_log[k] = bar_n_ref_world

      ref_cmd = MurookaReferenceCommand(
          com_ref_world=com_ref_world,
          phi_ref_world=None if cfg.base_body_id is None else phi_ref_world,
          bar_f_ref_world=bar_f_ref_world,
          bar_n_ref_world=bar_n_ref_world,
          com_ref_seq_world=com_ref_seq,
          bar_f_ref_seq_world=bar_f_ref_seq,
          phi_ref_seq_world=phi_ref_seq,
          bar_n_ref_seq_world=bar_n_ref_seq,
      )

      contact_model_now = build_contact_model_from_sites(
        model,
        data,
        mu=float(cfg.mu),
        patch_specs=patch_specs,
        frame_mode="world_up",
      )
      w_meas_pre = measure_patch_wrenches_world(
        model,
        data,
        floor_geom_id=floor_gid,
        contact_model=contact_model_now,
        geom_map=geom_map,
        min_normal_force=0.0,
      )
      w_meas_resultant = _resultant_wrench_from_patch_wrenches(
        w_meas_pre,
        contact_model_now,
      )
      bar_w_meas = contact_wrench_about_origin_to_bar(
        w_meas_resultant,
        com_world=com_now,
        mass=total_mass,
        gravity_world=model.opt.gravity.copy(),
      )
      com_acc_meas = np.asarray(bar_w_meas.bar_force_world, dtype=float).reshape(3,) / float(total_mass)
      alpha_meas = np.asarray(bar_w_meas.bar_moment_world, dtype=float).reshape(3,) / np.asarray(cfg.I_diag, dtype=float).reshape(3,)

      # run the full pipeline with patch active 
      patch_active_log[k] = patch_active
      out = stack.step(
        data=data,
        patch_specs=patch_specs,
        ref_cmd=ref_cmd,
        patch_active=patch_active,
        measured_com_acc_world=com_acc_meas,
        measured_alpha_world=alpha_meas,
        measured_resultant_wrench_world=w_meas_resultant,
        external_bar_wrench_estimate_world=cfg.observer_external_bar_wrench_world,
        disturbance_gate_active=bool(cfg.disturbance_gate_active),
      )

      # log data 
      bar_wp_log[k] = np.hstack((out.bar_wp.bar_force_world, out.bar_wp.bar_moment_world))
      bar_wp_proj_log[k] = np.hstack((out.bar_wp_proj.bar_force_world, out.bar_wp_proj.bar_moment_world))
      bar_wd_log[k] = np.hstack((out.bar_wd.bar_force_world, out.bar_wd.bar_moment_world))
      w_cmd_log[k] = np.asarray(out.w_cmd_world_origin, dtype=float).reshape(6,)
      if out.w_real_world_origin is not None:
          w_real = np.asarray(out.w_real_world_origin, dtype=float).reshape(6,)
          w_real_log[k] = w_real
          w_force_err_norm_log[k] = np.linalg.norm(w_real[:3] - w_cmd_log[k, :3])
          w_moment_err_norm_log[k] = np.linalg.norm(w_real[3:] - w_cmd_log[k, 3:])
      if out.w_err_norm is not None:
          w_err_norm_log[k] = float(out.w_err_norm)

      # preview target 
      com_preview_log[k] = np.asarray(out.preview_state.com_ref, dtype=float).reshape(3,)

      # get measured and desired wrenches 
      w_meas = w_meas_pre
      if len(w_meas) != nc: 
         raise RuntimeError(f"w_meas length mismatch. expected {nc}, got {len(w_meas)} instead.")

      w_des = out.patch_wrenches_world
      if len(w_des) != nc: 
        raise RuntimeError(f"patch_wrenches_world length mismatch. expected {nc}, got {len(w_des)}") 
      
      w_meas_resultant_log[k] = w_meas_resultant
      w_exec_force_err_norm_log[k] = np.linalg.norm(w_cmd_log[k, :3] - w_meas_resultant[:3])
      w_exec_moment_err_norm_log[k] = np.linalg.norm(w_cmd_log[k, 3:] - w_meas_resultant[3:])
      w_exec_err_norm_log[k] = np.linalg.norm(w_cmd_log[k] - w_meas_resultant)

      dbg = out.debug
      pre_sync = dbg.get("preview_state_error_pre_sync")
      if pre_sync is not None:
        preview_state_pre_sync_err_norm_log[k] = float(pre_sync.get("com_err_norm", np.nan))
        preview_vel_pre_sync_err_norm_log[k] = float(pre_sync.get("com_vel_err_norm", np.nan))
        preview_acc_pre_sync_err_norm_log[k] = float(pre_sync.get("com_acc_err_norm", np.nan))
        preview_alpha_pre_sync_err_norm_log[k] = float(pre_sync.get("alpha_err_norm", np.nan))
      post_sync = dbg.get("preview_state_error_post_sync")
      if post_sync is not None:
        preview_acc_post_sync_err_norm_log[k] = float(post_sync.get("com_acc_err_norm", np.nan))
        preview_alpha_post_sync_err_norm_log[k] = float(post_sync.get("alpha_err_norm", np.nan))
      stabilizer_overwrite_ratio_log[k] = float(dbg.get("stabilizer_overwrite_ratio", np.nan))

      # compute normal forces in patch frame 
      fn = np.zeros(nc, dtype=float)
      for i in range(nc): 
        Fw = np.asarray(w_meas[i], dtype=float).reshape(6,)[:3]
        Rwc = np.asarray(out.contact_model.patches[i].R_wc, dtype=float).reshape(3,3)
        fn[i] = normal_force_in_patch_frame(Rwc, Fw)
      fn_log[k] = fn 
      
      # select damping gains per patch based on phase 
      gains_per_patch = select_patch_gains(patch_active, cfg.phase_gains)
      # patch_active = hyst.update(fn)

      # Damping Control Update --> compliance dr
      if cfg.enable_damping_control:
        for i in range(nc): 
          comp[i] = damping_step(
            dt=float(cfg.dt), 
            gains=gains_per_patch[i], 
            state=comp[i],
            w_meas=np.asarray(w_meas[i], dtype=float).reshape(6,),
            w_des=np.asarray(w_des[i], dtype=float).reshape(6,),
          )
      else:
        for i in range(nc):
          comp[i] = ComplianceState.zero()

      # IK Targets: stance Pose + compliance offsets 
      site_targets: list[SiteTarget] = []
      for i, sid in enumerate(sids): 
        dp = comp[i].dr[:3]
        dphi = comp[i].dr[3:]
        p_t = stance_pos[i] + dp 
        R_t = Exp(dphi) @ stance_R[i]
        site_targets.append(
          SiteTarget(
            site_id=int(sid),
            p_world=p_t,
            R_world=R_t,
            v_world=np.asarray(comp[i].drdot[:3], dtype=float).reshape(3,),
            omega_world=np.asarray(comp[i].drdot[3:], dtype=float).reshape(3,),
          )
        )

      # CoM target 
      com_t = np.asarray(out.desired_state.com, dtype=float).reshape(3,)
      com_des_log[k] = com_t 
      posture_mask = posture_dof_weight_mask_from_contacts(
        model,
        site_names=cfg.site_names,
        patch_active=patch_active,
        joint_groups=joint_groups,
      )
      wbik_tasks = build_wbik_task_set(
        desired=out.desired_state,
        site_targets=site_targets,
        qpos_nominal=qpos_nominal,
        posture_dof_weight_mask=posture_mask,
      )

      qpos_des = qpos_des_state.copy()
      if cfg.execution_backend == "position_ik":
        qpos_des = solve_ik_tasks(
          model,
          data,
          tasks=wbik_tasks,
          cfg=ik_cfg,
        )
        qpos_des_state = qpos_des.copy()
        qpos_des_log[k] = qpos_des
      elif cfg.execution_backend == "accel_wbik":
        try:
          accel_result = solve_accel_ik_tasks(
            model,
            data,
            tasks=wbik_tasks,
            qpos_des_prev=qpos_des_state,
            qvel_des_prev=qvel_des_state,
            cfg=accel_ik_cfg,
            dt=float(cfg.dt),
          )
          qpos_des = accel_result.qpos_des
          qpos_des_state = accel_result.qpos_des.copy()
          qvel_des_state = accel_result.qvel_des.copy()
          qacc_cmd_log[k] = accel_result.qacc_cmd
          qvel_des_log[k] = accel_result.qvel_des
          qpos_des_log[k] = accel_result.qpos_des
          wbik_com_acc_residual_log[k] = np.asarray(accel_result.task_residuals["com_acc"], dtype=float).reshape(3,)
          wbik_base_alpha_residual_log[k] = np.asarray(accel_result.task_residuals["base_alpha"], dtype=float).reshape(3,)
          wbik_stage1_slack_rms_log[k] = float(accel_result.diagnostics.get("stage1_slack_rms", np.nan))
          wbik_stage2_slack_rms_log[k] = float(accel_result.diagnostics.get("stage2_slack_rms", np.nan))
          wbik_stage1_preserved_resid_rms_log[k] = float(accel_result.diagnostics.get("stage1_preserved_resid_rms", np.nan))
          wbik_stage2_preserved_resid_rms_log[k] = float(accel_result.diagnostics.get("stage2_preserved_resid_rms", np.nan))
          wbik_joint_limit_active_count_log[k] = float(accel_result.diagnostics.get("joint_limit_active_count", np.nan))
          wbik_preserve_retry_count_log[k] = float(accel_result.diagnostics.get("preserve_retry_count", np.nan))
          wbik_stage1_tol_used_log[k] = float(accel_result.diagnostics.get("stage1_tol_used", np.nan))
          wbik_stage2_tol_used_log[k] = float(accel_result.diagnostics.get("stage2_tol_used", np.nan))
          for i in range(nc):
            wbik_site_pos_acc_residual_log[k, i] = np.asarray(accel_result.task_residuals["site_pos_acc"][i], dtype=float).reshape(3,)
            wbik_site_rot_acc_residual_log[k, i] = np.asarray(accel_result.task_residuals["site_rot_acc"][i], dtype=float).reshape(3,)
        except Exception:
          accel_ik_failed_log[k] = True
          qpos_des = qpos_des_state.copy()
          qacc_cmd_log[k] = np.zeros(model.nv, dtype=float)
          qvel_des_log[k] = qvel_des_state.copy()
          qpos_des_log[k] = qpos_des_state.copy()
      else:
        raise ValueError(f"Unsupported execution_backend: {cfg.execution_backend}")

      servo_mode = str(cfg.servo_mode)
      if servo_mode == "auto":
        if cfg.execution_backend == "accel_wbik":
          servo_mode = "motor_pd" if _model_uses_motor_actuators(model) else "actuator_pd"
        else:
          servo_mode = "position_target"

      if servo_mode == "position_target":
        ctrl = compute_position_ctrl_from_qpos_target(model, qpos_des)
      elif servo_mode == "actuator_pd":
        ctrl = compute_affine_actuator_ctrl_from_joint_pd_targets(
          model,
          data,
          qpos_des=qpos_des,
          qvel_des=qvel_des_state,
          cfg=cfg.servo_cfg,
        )
      elif servo_mode == "motor_pd":
        ctrl = compute_motor_ctrl_from_qpos_target(
          model,
          data,
          qpos_des=qpos_des,
          qvel_des=qvel_des_state,
          cfg=cfg.servo_cfg,
        )
      else:
        raise ValueError(f"Unsupported servo_mode: {cfg.servo_mode}")
      data.ctrl[:] = ctrl
      ctrl_log[k] = ctrl

      mujoco.mj_step(model, data)

      # logs 
      q_log[k] = data.qpos.copy()
      for i in range(nc): 
        w_des_log[k, i] = np.asarray(w_des[i], dtype=float).reshape(6,)
        w_meas_log[k, i] = np.asarray(w_meas[i], dtype=float).reshape(6,)
        dr_log[k, i] = comp[i].dr.copy()
        drdot_log[k, i] = comp[i].drdot.copy()
      
      if viewer is not None and (k % int(cfg.display_every) == 0): 
        viewer.update(data)

    return dict(
      q_log=q_log, 
      com_meas_log=com_meas_log,
      com_preview_log=com_preview_log,
      com_des_log=com_des_log, 
      com_ref_cmd_log=com_ref_cmd_log,
      patch_active_log=patch_active_log,
      fn_log=fn_log, 
      w_des_log=w_des_log, 
      w_meas_log=w_meas_log,
      dr_log=dr_log,
      drdot_log=drdot_log,
      bar_wp_log=bar_wp_log, 
      bar_wp_proj_log=bar_wp_proj_log,
      bar_wd_log=bar_wd_log,
      bar_f_ref_cmd_log=bar_f_ref_cmd_log,
      bar_n_ref_cmd_log=bar_n_ref_cmd_log,
      w_cmd_log=w_cmd_log,
      w_real_log=w_real_log,
      w_meas_resultant_log=w_meas_resultant_log,
      w_err_norm_log=w_err_norm_log,
      w_force_err_norm_log=w_force_err_norm_log,
      w_moment_err_norm_log=w_moment_err_norm_log,
      w_exec_err_norm_log=w_exec_err_norm_log,
      w_exec_force_err_norm_log=w_exec_force_err_norm_log,
      w_exec_moment_err_norm_log=w_exec_moment_err_norm_log,
      preview_state_pre_sync_err_norm_log=preview_state_pre_sync_err_norm_log,
      preview_vel_pre_sync_err_norm_log=preview_vel_pre_sync_err_norm_log,
      preview_acc_pre_sync_err_norm_log=preview_acc_pre_sync_err_norm_log,
      preview_alpha_pre_sync_err_norm_log=preview_alpha_pre_sync_err_norm_log,
      preview_acc_post_sync_err_norm_log=preview_acc_post_sync_err_norm_log,
      preview_alpha_post_sync_err_norm_log=preview_alpha_post_sync_err_norm_log,
      stabilizer_overwrite_ratio_log=stabilizer_overwrite_ratio_log,
      qacc_cmd_log=qacc_cmd_log,
      qvel_des_log=qvel_des_log,
      qpos_des_log=qpos_des_log,
      ctrl_log=ctrl_log,
      wbik_com_acc_residual_log=wbik_com_acc_residual_log,
      wbik_base_alpha_residual_log=wbik_base_alpha_residual_log,
      wbik_site_pos_acc_residual_log=wbik_site_pos_acc_residual_log,
      wbik_site_rot_acc_residual_log=wbik_site_rot_acc_residual_log,
      wbik_stage1_slack_rms_log=wbik_stage1_slack_rms_log,
      wbik_stage2_slack_rms_log=wbik_stage2_slack_rms_log,
      wbik_stage1_preserved_resid_rms_log=wbik_stage1_preserved_resid_rms_log,
      wbik_stage2_preserved_resid_rms_log=wbik_stage2_preserved_resid_rms_log,
      wbik_joint_limit_active_count_log=wbik_joint_limit_active_count_log,
      wbik_preserve_retry_count_log=wbik_preserve_retry_count_log,
      wbik_stage1_tol_used_log=wbik_stage1_tol_used_log,
      wbik_stage2_tol_used_log=wbik_stage2_tol_used_log,
      accel_ik_failed_log=accel_ik_failed_log,
      wbik_solver_mode=str(accel_ik_cfg.solver_mode),
    )
  
  finally: 
    if viewer is not None: 
      viewer.close()
