from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import mujoco

from control_types import (
    BaseState,
    CentroidalReference,
    CentroidalMeasured,
    CentroidalDesired,
    ResultantWrenchBar,
    ContactModel,
)
from dynamics import (
    compute_total_mass,
    compute_com_state,
)
from preview_centroidal import CentroidalPreviewPlanner, PreviewConfig
from contact_patches import PatchSpec, build_contact_model_from_sites
from wrench_qp_generators import (
    project_planned_bar_wrench,
    distribute_desired_bar_wrench,
)
from centroidal_prediction import predict_one_step
from centroidal_stabilizer import StabilizerGains, stabilize_bar_wrench
from murooka_wrench import bar_to_contact_wrench_about_origin
from lie_math import logvec


def _body_R_world(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()


def _body_omega_world(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    """
    Uses mj_objectVelocity (no guessing).
    res[:3] is angular velocity in world coordinates.
    """
    res = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, res, 0)
    return res[:3].copy()


@dataclass
class MurookaReferenceCommand: 
  """ 
  command interface for planner 

  com_ref_world: desired CoM position reference for planner 
  phi_ref_world: desired base orientation reference as rotation vector 
  bar_f_ref_world: desired bar force reference, set to 0 
  bar_n_ref_world: desired bar moment reference, set to 0 
  """
  # scalar references 
  com_ref_world: np.ndarray 
  phi_ref_world: np.ndarray | None = None 
  bar_f_ref_world: np.ndarray | None = None 
  bar_n_ref_world: np.ndarray | None = None 

  # horizon sequences (Nh,3)
  com_ref_seq_world: np.ndarray | None = None
  phi_ref_seq_world: np.ndarray | None = None
  bar_f_ref_seq_world: np.ndarray | None = None
  bar_n_ref_seq_world: np.ndarray | None = None


@dataclass 
class StackControllerConfig: 
  enable_preview_planner: bool = True 
  enable_wrench_projection: bool = True 
  enable_wrench_distribution: bool = True 
  enable_centroidal_stabilizer: bool = True

  base_body_id: int | None = None 
  I_diag: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=float))

  mu: float = 0.6 
  site_vertex_offsets: dict[int, np.ndarray] | None = None 

  sim_dt: float = 1e-3 
  preview_dt: float = 1e-3
  preview_horizon_steps: int = 400 
  preview_controller_mode: Literal["lqt", "lqt_normalized", "preview_servo"] = "lqt"
  preview_linear_position_scale: float = 0.05
  preview_angular_position_scale: float = 0.10
  preview_nominal_freq_hz: float = 0.5
  preview_q_pos_override: float | None = None
  preview_q_wrench_override: float | None = None
  preview_r_jerk_override: float | None = None
  preview_ki_pos_override: float | None = None
  preview_state_source: Literal["desired_delay", "measured", "open_loop", "blended"] = "measured"
  preview_blend_alpha: float = 0.05

  stabilizer_gains: StabilizerGains = StabilizerGains.murooka_table_iii()

  # regularization for gernerator QPs 
  reg_projection: float = 1e-9
  reg_distribution: float = 1e-9

  # extra cost + stay away from friction cone bounds 
  w_tan_projection: float = 0.0 
  w_tan_distribution: float = 0.0 

  # contact frame mode to determine generator matrix local reference frame 
  contact_frame_mode: Literal["site", "world_up"] = "world_up"


@dataclass
class StackMurookaOutputs: 
  """ 
  Pure Murooka outputs used by the runtime exectuor (damping + IK)
  """
  contact_model: ContactModel

  # planned ,projected planned, desired bar wrench 
  bar_wp: ResultantWrenchBar
  bar_wp_proj: ResultantWrenchBar
  bar_wd: ResultantWrenchBar

  # desired contact wrench about world origin 
  w_cmd_world_origin: np.ndarray    # (6,)

  # distribution solution 
  lam: np.ndarray | None                  # (nlambda,)
  patch_wrenches_world: list[np.ndarray]  # list of (6,) about each patch origin 

  preview_state: CentroidalReference
  desired_state: CentroidalDesired 
  measured_state: CentroidalMeasured 

  # diagnostic fields
  w_real_world_origin: np.ndarray | None = None 
  w_err_norm: float | None = None 

  # debug dict
  debug: dict = field(default_factory=dict)


@dataclass
class DelayedPreviewState:
  com: np.ndarray
  com_vel: np.ndarray
  com_acc: np.ndarray
  phi_world: np.ndarray | None
  omega_world: np.ndarray | None
  alpha_world: np.ndarray | None


class StackController: 
  """ 
  Explicit Murooka pipeline (preview -> projection -> one-step prediction -> stabilizer -> distribution).
  Torque QP Layer can consume: 
    - wrench_cmd (cwrench about origin)
    - lambda_ref_per_site (net force per site, if point-contact is kept in QP)
  """
  def __init__(self, model: mujoco.MjModel, cfg: StackControllerConfig): 
    self.model = model 
    self.cfg = cfg 
    self.mass = compute_total_mass(model)

    I_diag = np.asarray(cfg.I_diag, dtype=float).reshape(3,)
    if np.any(I_diag <= 0) or not np.all(np.isfinite(I_diag)): 
      raise ValueError("cfg.I_diag must be positive finite (3,)")
    
    self.I_diag = I_diag
    
    self.preview = None 
    if cfg.enable_preview_planner:
      if cfg.preview_controller_mode == "lqt":
        lin_cfg = PreviewConfig.build_linear(dt=cfg.preview_dt, horizon_steps=cfg.preview_horizon_steps)
        ang_cfg = PreviewConfig.build_angular(dt=cfg.preview_dt, horizon_steps=cfg.preview_horizon_steps)
      elif cfg.preview_controller_mode == "lqt_normalized":
        lin_cfg = PreviewConfig.build_linear_normalized(
          dt=cfg.preview_dt,
          horizon_steps=cfg.preview_horizon_steps,
          position_scale=cfg.preview_linear_position_scale,
          nominal_freq_hz=cfg.preview_nominal_freq_hz,
        )
        ang_cfg = PreviewConfig.build_angular_normalized(
          dt=cfg.preview_dt,
          horizon_steps=cfg.preview_horizon_steps,
          position_scale=cfg.preview_angular_position_scale,
          nominal_freq_hz=cfg.preview_nominal_freq_hz,
        )
      elif cfg.preview_controller_mode == "preview_servo":
        lin_cfg = PreviewConfig.build_linear_preview_servo(
          dt=cfg.preview_dt,
          horizon_steps=cfg.preview_horizon_steps,
          position_scale=cfg.preview_linear_position_scale,
          nominal_freq_hz=cfg.preview_nominal_freq_hz,
        )
        ang_cfg = PreviewConfig.build_angular_preview_servo(
          dt=cfg.preview_dt,
          horizon_steps=cfg.preview_horizon_steps,
          position_scale=cfg.preview_angular_position_scale,
          nominal_freq_hz=cfg.preview_nominal_freq_hz,
        )
      else:
        raise ValueError(f"Unsupported preview_controller_mode: {cfg.preview_controller_mode}")
      self._apply_preview_overrides(lin_cfg)
      self._apply_preview_overrides(ang_cfg)
      self.preview = CentroidalPreviewPlanner(
        mass=self.mass, 
        I_diag=self.I_diag,
        lin_cfg=lin_cfg,
        ang_cfg=ang_cfg,
      )

      self._preview_inited = False
      self._delayed_preview_state: DelayedPreviewState | None = None
      self._preview_elapsed_since_update = 0.0
      self._held_preview_ref: CentroidalReference | None = None
      self._held_bar_wp: ResultantWrenchBar | None = None

  def _apply_preview_overrides(self, preview_cfg: PreviewConfig) -> None:
    if self.cfg.preview_q_pos_override is not None:
      preview_cfg.q_pos = float(self.cfg.preview_q_pos_override)
    if self.cfg.preview_q_wrench_override is not None:
      preview_cfg.q_wrench = float(self.cfg.preview_q_wrench_override)
    if self.cfg.preview_r_jerk_override is not None:
      preview_cfg.r_jerk = float(self.cfg.preview_r_jerk_override)
    if self.cfg.preview_ki_pos_override is not None:
      preview_cfg.ki_pos = float(self.cfg.preview_ki_pos_override)

  @staticmethod
  def _copy_preview_delay_state(
    desired: CentroidalDesired,
    com_acc: np.ndarray,
    alpha_world: np.ndarray | None,
  ) -> DelayedPreviewState:
    phi_world = None
    omega_world = None
    if desired.base_R_world is not None and desired.base_omega_world is not None:
      phi_world = logvec(np.asarray(desired.base_R_world, dtype=float).reshape(3, 3))
      omega_world = np.asarray(desired.base_omega_world, dtype=float).reshape(3,).copy()
    alpha_copy = None if alpha_world is None else np.asarray(alpha_world, dtype=float).reshape(3,).copy()
    if desired.com_acc is not None:
      com_acc = np.asarray(desired.com_acc, dtype=float).reshape(3,)
    if desired.base_alpha_world is not None:
      alpha_copy = np.asarray(desired.base_alpha_world, dtype=float).reshape(3,).copy()
    return DelayedPreviewState(
      com=np.asarray(desired.com, dtype=float).reshape(3,).copy(),
      com_vel=np.asarray(desired.com_vel, dtype=float).reshape(3,).copy(),
      com_acc=np.asarray(com_acc, dtype=float).reshape(3,).copy(),
      phi_world=phi_world,
      omega_world=omega_world,
      alpha_world=alpha_copy,
    )

  def _sync_preview_from_desired_delay(self) -> bool:
    if self.preview is None or self._delayed_preview_state is None:
      return False

    delayed = self._delayed_preview_state

    self.preview.sync_state(
      delayed.com,
      delayed.com_vel,
      coma0=delayed.com_acc,
      phi0=delayed.phi_world,
      omega0=delayed.omega_world,
      alpha0=delayed.alpha_world,
    )
    return True

  def _preview_update_due(self) -> bool:
    if not self._preview_inited:
      return True

    if self.cfg.preview_dt <= self.cfg.sim_dt + 1e-15:
      return True

    self._preview_elapsed_since_update += float(self.cfg.sim_dt)
    if self._preview_elapsed_since_update + 1e-15 < float(self.cfg.preview_dt):
      return False

    self._preview_elapsed_since_update = self._preview_elapsed_since_update % float(self.cfg.preview_dt)
    return True

  @staticmethod
  def _preview_state_error(
    preview_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    com: np.ndarray,
    com_vel: np.ndarray,
    com_acc: np.ndarray | None,
    phi_world: np.ndarray | None,
    omega_world: np.ndarray | None,
    alpha_world: np.ndarray | None,
  ) -> dict[str, np.ndarray | float]:
    p_c, p_v, p_a, p_phi, p_omega, p_alpha = preview_state
    err: dict[str, np.ndarray | float] = {
      "com_err": p_c - np.asarray(com, dtype=float).reshape(3,),
      "com_vel_err": p_v - np.asarray(com_vel, dtype=float).reshape(3,),
      "com_acc_err": (
        p_a - np.asarray(com_acc, dtype=float).reshape(3,)
        if com_acc is not None else np.full(3, np.nan, dtype=float)
      ),
      "phi_err": (
        p_phi - np.asarray(phi_world, dtype=float).reshape(3,)
        if phi_world is not None else np.full(3, np.nan, dtype=float)
      ),
      "omega_err": (
        p_omega - np.asarray(omega_world, dtype=float).reshape(3,)
        if omega_world is not None else np.full(3, np.nan, dtype=float)
      ),
      "alpha_err": (
        p_alpha - np.asarray(alpha_world, dtype=float).reshape(3,)
        if alpha_world is not None else np.full(3, np.nan, dtype=float)
      ),
    }
    err["com_err_norm"] = float(np.linalg.norm(err["com_err"]))
    err["com_vel_err_norm"] = float(np.linalg.norm(err["com_vel_err"]))
    err["com_acc_err_norm"] = float(np.linalg.norm(err["com_acc_err"])) if com_acc is not None else float("nan")
    err["phi_err_norm"] = float(np.linalg.norm(err["phi_err"])) if phi_world is not None else float("nan")
    err["omega_err_norm"] = float(np.linalg.norm(err["omega_err"])) if omega_world is not None else float("nan")
    err["alpha_err_norm"] = float(np.linalg.norm(err["alpha_err"])) if alpha_world is not None else float("nan")
    return err
  
  def _estimate_base(self, data: mujoco.MjData) -> BaseState | None: 
    if self.cfg.base_body_id is None: 
      return None 
    bid = int(self.cfg.base_body_id)
    if not (0 <= bid < self.model.nbody): 
      raise ValueError(f"base_body_id out of range: {bid}")
    R = _body_R_world(data, bid)
    omega = _body_omega_world(self.model, data, bid)
    from lie_math import logvec
    phi = logvec(R)
    return BaseState(R_world=R, omega_world=omega, phi_world=phi)
  
  def step(
    self, 
    *,
    data: mujoco.MjData, 
    patch_specs: list[PatchSpec], 
    ref_cmd: MurookaReferenceCommand,
    patch_active: np.ndarray | None = None,
    measured_com_acc_world: np.ndarray | None = None,
    measured_alpha_world: np.ndarray | None = None,
    measured_resultant_wrench_world: np.ndarray | None = None,
    external_bar_wrench_estimate_world: np.ndarray | None = None,
    disturbance_gate_active: bool = False,
  ) -> StackMurookaOutputs: 
    com, com_vel = compute_com_state(self.model, data)
    base = self._estimate_base(data)
    measured = CentroidalMeasured(com=com, com_vel=com_vel, base=base)
    phi_meas = None if base is None else base.phi_world
    omega_meas = None if base is None else base.omega_world

    if self.preview is None: 
      raise ValueError("Preview Planner disabled.")
    
    contact_model = build_contact_model_from_sites(
      self.model, data, 
      mu=self.cfg.mu, 
      patch_specs=patch_specs,
      frame_mode=self.cfg.contact_frame_mode,
    )

    Nh = self.preview.lin_cfg.horizon_steps

    # scalar fallback 
    com_ref = np.asarray(ref_cmd.com_ref_world, dtype=float).reshape(3,)
    if not np.all(np.isfinite(com_ref)): 
      raise ValueError("ref_cmd.com_ref_world must be finite")
    
    bar_f_ref = np.zeros(3, dtype=float) if ref_cmd.bar_f_ref_world is None else np.asarray(ref_cmd.bar_f_ref_world, dtype=float).reshape(3,)
    if not np.all(np.isfinite(bar_f_ref)): 
      raise ValueError("ref_cmd.bar_f_ref_world must be finite")
      
    # angular defaults depend on base being active
    if self.cfg.base_body_id is None: 
      phi_ref = np.zeros(3, dtype=float)
      bar_n_ref = np.zeros(3, dtype=float)
    else: 
      if ref_cmd.phi_ref_world is None: 
        raise ValueError("base_body_id was set, but no ref_cmd.phi_ref_world was provided")
      phi_ref = np.asarray(ref_cmd.phi_ref_world, dtype=float).reshape(3,)
      if not np.all(np.isfinite(phi_ref)):
        raise ValueError("ref_cmd.phi_ref_world must be finite")
      bar_n_ref = np.zeros(3, dtype=float) if ref_cmd.bar_n_ref_world is None else np.asarray(ref_cmd.bar_n_ref_world, dtype=float).reshape(3,)
      if not np.all(np.isfinite(bar_n_ref)): 
        raise ValueError("ref_cmd.bar_n_ref_world must be finite")
      
    # sequences take precedence if they exist 
    com_ref_seq = None
    phi_ref_seq = None 
    bar_f_ref_seq = None 
    bar_n_ref_seq = None

    if ref_cmd.com_ref_seq_world is not None: 
      com_ref_seq = np.asarray(ref_cmd.com_ref_seq_world, dtype=float)
      if com_ref_seq.shape != (Nh, 3) or not np.all(np.isfinite(com_ref_seq)): 
        raise ValueError(f"ref_cmd.com_ref_seq_world must be finite {(Nh,3)}")
      bar_f_ref_seq = np.zeros((Nh,3), dtype=float) if ref_cmd.bar_f_ref_seq_world is None else np.asarray(ref_cmd.bar_f_ref_seq_world, dtype=float)
      if bar_f_ref_seq.shape != (Nh, 3) or not np.all(np.isfinite(bar_f_ref_seq)): 
        raise ValueError(f"bar_f_ref_seq_world must be finite {(Nh,3)}")
      
      if self.cfg.base_body_id is None: 
        phi_ref_seq = np.zeros((Nh,3), dtype=float)
        bar_n_ref_seq = np.zeros((Nh,3), dtype=float)
      else: 
        if ref_cmd.phi_ref_seq_world is None:
          raise ValueError("base_body_id set: phi_ref_seq_world must be provided when using sequence mode")
        phi_ref_seq = np.asarray(ref_cmd.phi_ref_seq_world, dtype=float)
        if phi_ref_seq.shape != (Nh, 3) or not np.all(np.isfinite(phi_ref_seq)):
          raise ValueError(f"phi_ref_seq_world must be finite {(Nh,3)}")
        bar_n_ref_seq = np.zeros((Nh, 3), dtype=float) if ref_cmd.bar_n_ref_seq_world is None else np.asarray(ref_cmd.bar_n_ref_seq_world, dtype=float)
        if bar_n_ref_seq.shape != (Nh, 3) or not np.all(np.isfinite(bar_n_ref_seq)):
          raise ValueError(f"bar_n_ref_seq_world must be finite {(Nh,3)}")
        
    preview_updated = False
    just_initialized = False

    # init / sync preview internal state
    if not self._preview_inited: 
      self._preview_inited = True 
      just_initialized = True
      omega0 = base.omega_world if base is not None else None 
      phi0 = base.phi_world if base is not None else phi_ref
      self.preview.reset(com0=com, comv0=com_vel, phi0=phi0, omega0=omega0)
    preview_state_pre_sync = None
    preview_state_pre_err: dict | None = None
    preview_state_post_err: dict | None = None

    if just_initialized or self._preview_update_due():
      preview_state_pre_sync = self.preview.measured_state()
      preview_state_pre_err = self._preview_state_error(
        preview_state_pre_sync,
        com=com,
        com_vel=com_vel,
        com_acc=measured_com_acc_world,
        phi_world=phi_meas,
        omega_world=omega_meas,
        alpha_world=measured_alpha_world,
      )
      if self.cfg.preview_state_source == "measured":
        self.preview.sync_state(
          com,
          com_vel,
          coma0=measured_com_acc_world,
          phi0=phi_meas,
          omega0=omega_meas,
          alpha0=measured_alpha_world,
        )
      elif self.cfg.preview_state_source == "desired_delay":
        synced = self._sync_preview_from_desired_delay()
        if not synced:
          self.preview.sync_state(
            com,
            com_vel,
            coma0=measured_com_acc_world,
            phi0=phi_meas,
            omega0=omega_meas,
            alpha0=measured_alpha_world,
          )
      elif self.cfg.preview_state_source == "open_loop":
        pass  # LQT evolves from its own predicted state; no measurement sync
      elif self.cfg.preview_state_source == "blended":
        self.preview.blend_with_meas(self.cfg.preview_blend_alpha, com, com_vel, base)
      else:
        raise ValueError(f"Unsupported preview_state_source: {self.cfg.preview_state_source}")

      if self.cfg.preview_state_source in ("measured", "desired_delay"):
        preview_state_post_err = self._preview_state_error(
          self.preview.measured_state(),
          com=com,
          com_vel=com_vel,
          com_acc=measured_com_acc_world,
          phi_world=phi_meas,
          omega_world=omega_meas,
          alpha_world=measured_alpha_world,
        )

      if com_ref_seq is None: 
        ref, bar_wp = self.preview.step_constant(
          com_ref=com_ref, 
          phi_ref=phi_ref, 
          bar_f_ref=bar_f_ref, 
          bar_n_ref=bar_n_ref,
        )
      else: 
        ref, bar_wp = self.preview.step_preview(
          com_ref_seq=com_ref_seq, 
          phi_ref_seq=phi_ref_seq,
          bar_f_ref_seq=bar_f_ref_seq,
          bar_n_ref_seq=bar_n_ref_seq,
        )

      phi_acc = np.asarray(ref.meta.get("phi_acc", [0.0, 0.0, 0.0]), dtype=float).reshape(3,)
      bar_wp = ResultantWrenchBar(
        bar_force_world=np.asarray(bar_wp.bar_force_world, dtype=float).reshape(3,),
        bar_moment_world=self.I_diag * phi_acc,
      )
      self._held_preview_ref = ref
      self._held_bar_wp = bar_wp
      preview_updated = True
    else:
      ref = self._held_preview_ref
      bar_wp = self._held_bar_wp
      if ref is None or bar_wp is None:
        raise RuntimeError("Preview hold requested before any preview state was cached")

    # 2. Wrench Projection, QP --> 'bar_wp_proj' 
    if self.cfg.enable_wrench_projection: 
      bar_wp_proj, dbg_proj = project_planned_bar_wrench(
        bar_wp=bar_wp, 
        com_planned_world=np.asarray(ref.com_ref, dtype=float).reshape(3,),
        mass=self.mass, 
        gravity_world=self.model.opt.gravity.copy(), 
        contact_model=contact_model, 
        reg=float(self.cfg.reg_projection),
        patch_active=patch_active,
        w_tan=self.cfg.w_tan_projection,
      )
    else: 
      bar_wp_proj = bar_wp 
      dbg_proj = {"status": "disabled"}
    

    # 3. One-step prediction --> desired rd 
    from lie_math import Exp
    phi_p = np.asarray(ref.meta["phi"])
    omega_p = np.asarray(ref.meta["omega"])
    com_acc_des = np.asarray(bar_wp_proj.bar_force_world, dtype=float).reshape(3,) / float(self.mass)
    alpha_des = None if base is None else np.asarray(bar_wp_proj.bar_moment_world, dtype=float).reshape(3,) / self.I_diag
    desired = predict_one_step(
      dt=float(self.cfg.sim_dt), 
      mass=self.mass, 
      I_diag=self.I_diag, 
      com=ref.com_ref,
      com_vel=ref.com_vel_ref,
      base_R=(Exp(phi_p)) if base is not None else None, 
      base_omega=omega_p, 
      bar_wp_proj=bar_wp_proj,
    )


    # 4/5. Stabilizer --> bar_wd 
    if self.cfg.enable_centroidal_stabilizer:
      bar_wd, dbg_stab = stabilize_bar_wrench(
        bar_wp_proj=bar_wp_proj, 
        desired=desired,
        measured=measured,
        gains=self.cfg.stabilizer_gains,
      )
    else:
      bar_wd = bar_wp_proj
      dbg_stab = {"status": "disabled"}
    bar_wp_proj_vec = np.hstack((bar_wp_proj.bar_force_world, bar_wp_proj.bar_moment_world))
    bar_wd_vec = np.hstack((bar_wd.bar_force_world, bar_wd.bar_moment_world))
    stab_delta = bar_wd_vec - bar_wp_proj_vec
    stab_delta_norm = float(np.linalg.norm(stab_delta))
    stab_overwrite_ratio = stab_delta_norm / (float(np.linalg.norm(bar_wp_proj_vec)) + 1e-9)


    # Convert bar_wd --> wd (i.e. contact wrench about world origin)
    w_cmd = bar_to_contact_wrench_about_origin(
      bar_wd, 
      com_world=measured.com, 
      mass=self.mass,
      gravity_world=self.model.opt.gravity.copy(),
    )


    # 6. Distribution QP --> lambda and per-patch wrenches  
    lam = None 
    w_real_world_origin = None 
    w_err_norm = None 
    patch_wrenches_world: list[np.ndarray] = [np.zeros(6, dtype=float) for _ in contact_model.patches]
    dbg_dist = {"status": "disabled"}
    
    if self.cfg.enable_wrench_distribution: 
      lam, dbg_dist = distribute_desired_bar_wrench(
        bar_wd=bar_wd, 
        com_actual_world=measured.com, # paper lists best results with measured rather than desired
        mass=self.mass,
        gravity_world=self.model.opt.gravity.copy(),
        contact_model=contact_model, 
        reg=float(self.cfg.reg_distribution),
        patch_active=patch_active,
        w_tan=self.cfg.w_tan_distribution,
      )

      patch_wrenches_world = [ 
        np.asarray(w, dtype=float).reshape(6,) for w in dbg_dist["patch_wrenches_world"]
      ]
      if len(patch_wrenches_world) != len(contact_model.patches): 
        raise RuntimeError("patch_wrenches_world length mismatch with conatct_model.patches")
      
      w_real_world_origin = np.asarray(dbg_dist["w_real"], dtype=float).reshape(6,)
      w_d = np.asarray(dbg_dist["w_d"], dtype=float).reshape(6,)
      w_err_norm = float(np.linalg.norm(w_real_world_origin - w_d))

    dbg = {
      "ref": ref,
      "projection": dbg_proj,
      "stabilizer": dbg_stab, 
      "distribution": dbg_dist, 
      "preview_state_source": self.cfg.preview_state_source,
      "preview_updated": preview_updated,
      "preview_state_pre_sync": preview_state_pre_sync,
      "preview_state_error_pre_sync": preview_state_pre_err,
      "preview_state_error_post_sync": preview_state_post_err,
      "stabilizer_delta_bar": stab_delta,
      "stabilizer_delta_bar_norm": stab_delta_norm,
      "stabilizer_overwrite_ratio": stab_overwrite_ratio,
      "observer_hooks": {
        "measured_resultant_wrench_world": None if measured_resultant_wrench_world is None else np.asarray(measured_resultant_wrench_world, dtype=float).reshape(6,),
        "measured_com_acc_world": None if measured_com_acc_world is None else np.asarray(measured_com_acc_world, dtype=float).reshape(3,),
        "external_bar_wrench_estimate_world": None if external_bar_wrench_estimate_world is None else np.asarray(external_bar_wrench_estimate_world, dtype=float).reshape(6,),
        "disturbance_gate_active": bool(disturbance_gate_active),
      },
    }

    self._delayed_preview_state = self._copy_preview_delay_state(
      desired,
      com_acc=com_acc_des,
      alpha_world=alpha_des,
    )

    return StackMurookaOutputs(
      contact_model=contact_model,
      bar_wp=bar_wp,
      bar_wp_proj=bar_wp_proj,
      bar_wd=bar_wd, 
      w_cmd_world_origin=w_cmd, 
      lam=lam,
      patch_wrenches_world=patch_wrenches_world,
      preview_state=ref,
      desired_state=desired,
      measured_state=measured,
      w_real_world_origin=w_real_world_origin,
      w_err_norm=w_err_norm,
      debug=dbg,
    )
    
