# sim.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import mujoco

from misc import site_ids
from viz import Viz

from dynamics import compute_com_state, compute_base_state
from contact_patches import PatchSpec
from contact_measurement import (
    ContactMeasurementDiagnostics,
    build_patch_geom_map_from_sites,
    measure_patch_wrenches_world,
)

from reference_sequences import sine_com_bar_ref_seq, sine_com_ref_seq, zeros_bar_seq
from centroidal_stabilizer import StabilizerGains
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

from whole_body_ik import IKConfig, SiteTarget, solve_ik
from joint_servo import JointServoConfig, compute_motor_ctrl_from_qpos_target, compute_position_ctrl_from_qpos_target
from lie_math import Exp, logvec


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
  preview_state_source: str = "measured"
  preview_blend_alpha: float = 0.05
  preview_max_drift: float = 0.02  # [m] hard clamp for desired_delay mode; 0 = use blend_alpha instead
  preview_q_pos_override: float | None = None
  preview_q_wrench_override: float | None = None
  preview_r_jerk_override: float | None = None
  preview_ki_pos_override: float | None = None

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
  enable_centroidal_stabilizer: bool = True
  enable_damping_control: bool = True

  # IK 
  ik_cfg: IKConfig = IKConfig(base_body_id=base_body_id)

  # joint servo 
  servo_cfg: JointServoConfig = JointServoConfig()

  # viz 
  viz: bool = False 
  display_every: int = 2 


def _ik_task_residual_norm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos_eval: np.ndarray,
    com_target: np.ndarray,
    site_targets: list[SiteTarget],
    base_body_id: int | None,
    base_R_target: np.ndarray | None,
) -> float:
  qpos0 = data.qpos.copy()
  try:
    data.qpos[:] = np.asarray(qpos_eval, dtype=float).reshape(model.nq,)
    mujoco.mj_forward(model, data)

    residuals: list[float] = []
    com_eval, _ = compute_com_state(model, data)
    residuals.append(float(np.linalg.norm(np.asarray(com_target, dtype=float).reshape(3,) - com_eval)))

    for st in site_targets:
      sid = int(st.site_id)
      p = np.asarray(data.site_xpos[sid], dtype=float).reshape(3,)
      R = np.asarray(data.site_xmat[sid], dtype=float).reshape(3, 3)
      residuals.append(float(np.linalg.norm(np.asarray(st.p_world, dtype=float).reshape(3,) - p)))
      residuals.append(float(np.linalg.norm(logvec(np.asarray(st.R_world, dtype=float).reshape(3, 3) @ R.T))))

    if base_body_id is not None and base_R_target is not None:
      R_base = np.asarray(data.xmat[int(base_body_id)], dtype=float).reshape(3, 3)
      residuals.append(float(np.linalg.norm(logvec(np.asarray(base_R_target, dtype=float).reshape(3, 3) @ R_base.T))))

    return float(np.linalg.norm(np.asarray(residuals, dtype=float))) if residuals else 0.0
  finally:
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)


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
  
  # Stabilizer gains from Murooka Appendix-A Eq. (19) — DCM correspondence:
  #   K_PL = m * omega^2 * K_xi
  #   K_DL = m * omega  * K_xi        =>  zeta = 0.5 analytically; numerically ~0.707 for K_xi=2
  #   where omega = sqrt(g / z_c) is the LIPM natural frequency.
  #
  # K_xi=2 matches the paper's Table III for HRP-5P (m=105, omega=3.21 => KP=2163, KD=674).
  # Scaling by G1 mass instead of transplanting the absolute gains avoids CWC saturation:
  #   G1 (m≈35, omega≈3.40): KP≈807, KD≈238, peak wrench ≈46% CWC
  #   Paper values direct (KP=2000): peak wrench ≈102% CWC → inevitable clip → fall
  _com0_init, _ = compute_com_state(model, data)
  _mass_total   = float(np.sum(model.body_mass))
  _z_c          = float(_com0_init[2])                            # standing CoM height [m]
  _g_mag        = float(np.linalg.norm(model.opt.gravity))
  _omega_lipm   = float(np.sqrt(_g_mag / _z_c)) if _z_c > 0.01 else 3.4
  _K_xi         = 2.0                                             # paper's proportional gain factor
  _kp_lin       = _mass_total * _omega_lipm**2 * _K_xi            # K_PL = m * omega^2 * K_xi
  _kd_lin       = _mass_total * _omega_lipm    * _K_xi            # K_DL = m * omega   * K_xi
  _stab_gains = StabilizerGains.diagonal(
    kp_lin=(_kp_lin, _kp_lin, _kp_lin),
    kd_lin=(_kd_lin, _kd_lin, _kd_lin),
  )

  # Build Stack Controller
  stack = StackController(
    model=model,
    cfg=StackControllerConfig(
      enable_preview_planner=True,
      enable_wrench_projection=True,
      enable_centroidal_stabilizer=bool(cfg.enable_centroidal_stabilizer),
      enable_wrench_distribution=True,
      base_body_id=cfg.base_body_id,
      I_diag=np.asarray(cfg.I_diag, dtype=float).reshape(3,),
      mu=float(cfg.mu),
      sim_dt=float(cfg.dt),
      preview_dt=float(cfg.preview_dt),
      preview_horizon_steps=int(cfg.preview_horizon_steps),
      preview_controller_mode=str(cfg.preview_controller_mode),
      preview_linear_position_scale=float(cfg.preview_linear_position_scale),
      preview_angular_position_scale=float(cfg.preview_angular_position_scale),
      preview_nominal_freq_hz=float(cfg.preview_nominal_freq_hz),
      preview_state_source=cfg.preview_state_source,
      preview_blend_alpha=float(cfg.preview_blend_alpha),
      preview_max_drift=float(cfg.preview_max_drift),
      preview_q_pos_override=cfg.preview_q_pos_override,
      preview_q_wrench_override=cfg.preview_q_wrench_override,
      preview_r_jerk_override=cfg.preview_r_jerk_override,
      preview_ki_pos_override=cfg.preview_ki_pos_override,
      stabilizer_gains=_stab_gains,
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
  compliance_norm_log = np.zeros((cfg.N, nc, 2), dtype=float)
  ik_residual_norm_log = np.zeros(cfg.N, dtype=float)
  unassigned_patch_contact_count_log = np.zeros(cfg.N, dtype=int)
  site_target_pos_log = np.zeros((cfg.N, nc, 3), dtype=float)
  site_target_rotvec_log = np.zeros((cfg.N, nc, 3), dtype=float)
  base_target_rotvec_log = np.zeros((cfg.N, 3), dtype=float)

  # Per block logs: 
  # Includes: 
  # - Planned, projected, desired 
  # - Commanded, real, and error norm between both 6D wrenches 
  bar_wp_log = np.zeros((cfg.N, 6), dtype=float)
  bar_wp_proj_log = np.zeros((cfg.N, 6), dtype=float)
  bar_wd_log = np.zeros((cfg.N, 6), dtype=float)
  com_des_from_bar_wd_log = np.zeros((cfg.N, 3), dtype=float)
  bar_f_ref_cmd_log = np.zeros((cfg.N, 3), dtype=float)
  bar_n_ref_cmd_log = np.zeros((cfg.N, 3), dtype=float)
  projection_resultant_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  
  
  w_cmd_log = np.zeros((cfg.N, 6), dtype=float)
  w_real_log = np.full((cfg.N, 6), np.nan, dtype=float)
  w_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  w_force_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  w_moment_err_norm_log = np.full(cfg.N, np.nan, dtype=float)
  
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

      # run the full pipeline with patch active 
      patch_active_log[k] = patch_active
      out = stack.step(data=data, patch_specs=patch_specs, ref_cmd=ref_cmd, patch_active=patch_active)

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
      w_meas = measure_patch_wrenches_world(
         model, data, 
         floor_geom_id=floor_gid, 
         contact_model=out.contact_model, 
         geom_map=geom_map,
          min_normal_force=0.0,
         diagnostics=(meas_diag := ContactMeasurementDiagnostics()),
      )
      if len(w_meas) != nc: 
         raise RuntimeError(f"w_meas length mismatch. expected {nc}, got {len(w_meas)} instead.")
      unassigned_patch_contact_count_log[k] = int(meas_diag.unassigned_robot_contacts)

      w_des = out.patch_wrenches_world
      if len(w_des) != nc: 
        raise RuntimeError(f"patch_wrenches_world length mismatch. expected {nc}, got {len(w_des)}") 
      
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
        site_targets.append(SiteTarget(site_id=int(sid), p_world=p_t, R_world=R_t))

      # CoM target 
      com_t = np.asarray(out.desired_state.com, dtype=float).reshape(3,)
      com_des_log[k] = com_t 
      com_des_from_bar_wd_log[k] = np.asarray(out.desired_from_bar_wd.com, dtype=float).reshape(3,)

      # Base R target 
      base_R_t = np.asarray(out.desired_state.base_R_world, dtype=float).reshape(3,3)
      base_target_rotvec_log[k] = logvec(base_R_t)

      qpos_des = solve_ik(
        model, data,
        com_target=com_t, 
        site_targets=site_targets,
        qpos_nominal=qpos_nominal,
        cfg=cfg.ik_cfg,
        base_R_target=base_R_t,
      )
      ik_residual_norm_log[k] = _ik_task_residual_norm(
        model,
        data,
        qpos_eval=qpos_des,
        com_target=com_t,
        site_targets=site_targets,
        base_body_id=cfg.base_body_id,
        base_R_target=base_R_t,
      )

      ctrl = compute_position_ctrl_from_qpos_target(model, qpos_des)
      data.ctrl[:] = ctrl 

      mujoco.mj_step(model, data)

      # logs 
      q_log[k] = data.qpos.copy()
      for i in range(nc): 
        w_des_log[k, i] = np.asarray(w_des[i], dtype=float).reshape(6,)
        w_meas_log[k, i] = np.asarray(w_meas[i], dtype=float).reshape(6,)
        dr_log[k, i] = comp[i].dr.copy()
        compliance_norm_log[k, i, 0] = float(np.linalg.norm(comp[i].dr[:3]))
        compliance_norm_log[k, i, 1] = float(np.linalg.norm(comp[i].dr[3:]))
        site_target_pos_log[k, i] = np.asarray(site_targets[i].p_world, dtype=float).reshape(3,)
        site_target_rotvec_log[k, i] = logvec(np.asarray(site_targets[i].R_world, dtype=float).reshape(3, 3))
      projection_resultant_err_norm_log[k] = float(out.debug.get("projection_resultant_err_norm", np.nan))
      
      if viewer is not None and (k % int(cfg.display_every) == 0): 
        viewer.update(data)

    return dict(
      q_log=q_log, 
      ref_log=com_ref_cmd_log,
      preview_log=com_preview_log,
      desired_log=com_des_log,
      desired_from_bar_wd_log=com_des_from_bar_wd_log,
      measured_log=com_meas_log,
      com_meas_log=com_meas_log,
      com_preview_log=com_preview_log,
      com_des_log=com_des_log, 
      com_des_from_bar_wd_log=com_des_from_bar_wd_log,
      com_ref_cmd_log=com_ref_cmd_log,
      patch_active_log=patch_active_log,
      fn_log=fn_log, 
      w_des_log=w_des_log, 
      w_meas_log=w_meas_log,
      dr_log=dr_log,
      compliance_norm_log=compliance_norm_log,
      ik_residual_norm_log=ik_residual_norm_log,
      unassigned_patch_contact_count_log=unassigned_patch_contact_count_log,
      site_target_pos_log=site_target_pos_log,
      site_target_rotvec_log=site_target_rotvec_log,
      base_target_rotvec_log=base_target_rotvec_log,
      bar_wp_log=bar_wp_log, 
      bar_wp_proj_log=bar_wp_proj_log,
      bar_wd_log=bar_wd_log,
      bar_f_ref_cmd_log=bar_f_ref_cmd_log,
      bar_n_ref_cmd_log=bar_n_ref_cmd_log,
      projection_resultant_err_norm_log=projection_resultant_err_norm_log,
      w_cmd_log=w_cmd_log,
      w_real_log=w_real_log,
      w_err_norm_log=w_err_norm_log,
      w_exec_err_norm_log=w_err_norm_log,
      w_force_err_norm_log=w_force_err_norm_log,
      w_exec_force_err_norm_log=w_force_err_norm_log,
      w_moment_err_norm_log=w_moment_err_norm_log,
      w_exec_moment_err_norm_log=w_moment_err_norm_log,
    )
  
  finally: 
    if viewer is not None: 
      viewer.close()
