from __future__ import annotations 
from dataclasses import dataclass 
import numpy as np 

from control_types import CentroidalReference, ResultantWrenchBar, BaseState
from preview_lqt import LQTModel, LQTWeights, FiniteHorizonPreviewLQT


@dataclass
class PreviewConfig:
  dt: float # table i points 0.005s as default
  horizon_steps: int # table i points 400 as default (i.e. to reach a 2s horizon)
  q_pos: float
  q_wrench: float 
  r_jerk: float
  ki_pos: float = 2.0  # integral gain on position error to eliminate finite-horizon SS offset
  ki_max: float = 0.01  # anti-windup clamp for integral (meters)

  @staticmethod
  def build_linear(dt: float, horizon_steps: int):
    q_pos: float = 2e2
    # q_wrench penalises bar force (= m*a).  With motion_match_bar_force_ref=True
    # (dynamically consistent f̄_ref = m·a_ref), the force and position objectives
    # are aligned and q_wrench HELPS tracking:
    #   q_wrench=5e-4 + consistent refs → LQT ratio=0.998 at 0.5 Hz
    #   q_wrench=2e-5 + consistent refs → LQT ratio=1.015 (slight overshoot)
    # With f̄_ref=0 (inconsistent), 5e-4 gives 0.769 (conflict between objectives).
    q_wrench: float = 5e-4
    r_jerk: float = 1e-8
    ki_pos: float = 0.0  # DARE terminal cost eliminates finite-horizon DC offset
    return PreviewConfig(dt=dt, horizon_steps=horizon_steps, q_pos=q_pos, q_wrench=q_wrench, r_jerk=r_jerk, ki_pos=ki_pos)

  @staticmethod
  def build_angular(dt: float, horizon_steps: int):
     q_pos: float = 1e2
     q_wrench: float = 5e-3
     r_jerk: float = 1e-8
     ki_pos: float = 0.0  # DARE terminal cost eliminates finite-horizon DC offset
     return PreviewConfig(dt=dt, horizon_steps=horizon_steps, q_pos=q_pos, q_wrench=q_wrench, r_jerk=r_jerk, ki_pos=ki_pos)

  @staticmethod
  def build_linear_normalized(dt: float, horizon_steps: int):
    # Baseline inferred from the prior normalized-LQT tuning sweep:
    # q_pos=q_wrench=1, low jerk regularization, and nonzero integral action.
    return PreviewConfig(dt=dt, horizon_steps=horizon_steps, q_pos=1.0, q_wrench=1.0, r_jerk=1e-8, ki_pos=2.0)

  @staticmethod
  def build_angular_normalized(dt: float, horizon_steps: int):
    return PreviewConfig(dt=dt, horizon_steps=horizon_steps, q_pos=1.0, q_wrench=1.0, r_jerk=1e-8, ki_pos=2.0)


@dataclass 
class TripleIntegratorAxis: 
  """
  x = [p, v, a], u = jerk 
  x_{k+1} = A x_k + B u_k 
  y_k = [p, bar_w]^T where bar_w = gain * a (gain=m for linear, gain=I for angular)
  """
  A: np.ndarray 
  B: np.ndarray 
  C: np.ndarray 
  state_scale: np.ndarray
  output_scale: np.ndarray

  @staticmethod 
  def build(dt: float, output_gain: float) -> TripleIntegratorAxis: 
    A = np.array([
      [1.0, dt, 0.5 * dt**2], 
      [0.0, 1.0, dt], 
      [0.0, 0.0, 1.0],
    ], dtype=float)
    B = np.array([
      [dt**3 / 6.0], 
      [0.5 * dt**2],
      [dt]
    ], dtype=float)
    C = np.array([
      [1.0, 0.0, 0.0],
      [0.0, 0.0, output_gain]
    ], dtype=float)
    return TripleIntegratorAxis(
      A=A,
      B=B,
      C=C,
      state_scale=np.ones(3, dtype=float),
      output_scale=np.ones(2, dtype=float),
    )

  @staticmethod
  def build_normalized(
    dt: float,
    output_gain: float,
    *,
    position_scale: float,
    nominal_freq_hz: float,
  ) -> TripleIntegratorAxis:
    if position_scale <= 0.0:
      raise ValueError("position_scale must be positive")
    if nominal_freq_hz <= 0.0:
      raise ValueError("nominal_freq_hz must be positive")

    raw_axis = TripleIntegratorAxis.build(dt, output_gain)
    omega_n = 2.0 * np.pi * float(nominal_freq_hz)
    pos_scale = float(position_scale)
    vel_scale = omega_n * pos_scale
    acc_scale = omega_n**2 * pos_scale
    jerk_scale = omega_n**3 * pos_scale
    state_scale = np.array([pos_scale, vel_scale, acc_scale], dtype=float)
    output_scale = np.array([pos_scale, float(output_gain) * acc_scale], dtype=float)
    Sx = np.diag(state_scale)
    Sx_inv = np.diag(1.0 / state_scale)
    A = Sx_inv @ raw_axis.A @ Sx
    B = Sx_inv @ raw_axis.B * jerk_scale
    C = np.array([
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0],
    ], dtype=float)
    return TripleIntegratorAxis(
      A=A,
      B=B,
      C=C,
      state_scale=state_scale,
      output_scale=output_scale,
    )


class AxisPreviewController: 
  def __init__(self, axis: TripleIntegratorAxis, cfg: PreviewConfig):
    Qy = np.diag([cfg.q_pos, cfg.q_wrench]).astype(float)
    R = np.array([[cfg.r_jerk]], dtype=float)
    self.dt = float(cfg.dt)
    self.x = np.zeros(3, dtype=float)
    self._state_scale = np.asarray(axis.state_scale, dtype=float).reshape(3,)
    self._output_scale = np.asarray(axis.output_scale, dtype=float).reshape(2,)
    self._ki = float(cfg.ki_pos)
    self._ki_max = float(cfg.ki_max)
    self._integral = 0.0

    self.ctrl = FiniteHorizonPreviewLQT(
      model=LQTModel(A=axis.A, B=axis.B, C=axis.C),
      w=LQTWeights(Qy=Qy, R=R),
      horizon=cfg.horizon_steps,
    )

  def reset(self, p0: float, v0: float, a0: float = 0.0):
    self.x[:] = np.array([p0, v0, a0], dtype=float)
    self._integral = 0.0

  def step(self, yref_seq: np.ndarray) -> tuple[float, float, float]:
    """
    yref_seq: (N,2) of [p_ref; bar_w_ref]
    Returns updated (p, v, a)
    """
    yref_seq = np.asarray(yref_seq, dtype=float)
    yref_scaled = yref_seq / self._output_scale.reshape(1, 2)
    x_scaled = self.x / self._state_scale

    # Integral correction: accumulate position error to eliminate finite-horizon SS offset
    if self._ki > 0.0:
      pos_err = yref_seq[0, 0] - self.x[0]
      self._integral += self._ki * pos_err * self.dt
      self._integral = np.clip(self._integral, -self._ki_max, self._ki_max)
      yref_corrected = yref_seq.copy()
      yref_corrected[:, 0] += self._integral
      yref_scaled = yref_corrected / self._output_scale.reshape(1, 2)
      u0, x1 = self.ctrl.step(x_scaled, yref_seq=yref_scaled)
    else:
      u0, x1 = self.ctrl.step(x_scaled, yref_seq=yref_scaled)
    self.x[:] = self._state_scale * x1
    return self.x[0], self.x[1], self.x[2]


class CentroidalPreviewPlanner: 
  """
  Outputs planned centroidal state rp and planned bar wrench bar_wp 
  Uses rotation-vector components for angular part to avoid euler angles 
  """
  def __init__(
    self,
    mass: float,
    I_diag: np.ndarray,
    lin_cfg: PreviewConfig,
    ang_cfg: PreviewConfig,
    *,
    controller_mode: str = "lqt",
    linear_position_scale: float = 0.05,
    angular_position_scale: float = 0.10,
    nominal_freq_hz: float = 0.5,
  ): 
    self.mass = float(mass)
    I_diag = np.asarray(I_diag, dtype=float).reshape(3,)
    if not np.all(np.isfinite(I_diag)) or np.any(I_diag <= 0.0): 
      raise ValueError("I_diag must be positive definite (3,)")
    
    self.lin_cfg = lin_cfg 
    self.ang_cfg = ang_cfg 
    self.controller_mode = str(controller_mode)
    if self.controller_mode == "lqt":
      lin_axis = [TripleIntegratorAxis.build(lin_cfg.dt, self.mass) for _ in range(3)]
      ang_axis = [TripleIntegratorAxis.build(ang_cfg.dt, float(I_diag[i])) for i in range(3)]
    elif self.controller_mode == "lqt_normalized":
      lin_axis = [
        TripleIntegratorAxis.build_normalized(
          lin_cfg.dt,
          self.mass,
          position_scale=float(linear_position_scale),
          nominal_freq_hz=float(nominal_freq_hz),
        )
        for _ in range(3)
      ]
      ang_axis = [
        TripleIntegratorAxis.build_normalized(
          ang_cfg.dt,
          float(I_diag[i]),
          position_scale=float(angular_position_scale),
          nominal_freq_hz=float(nominal_freq_hz),
        )
        for i in range(3)
      ]
    else:
      raise ValueError(f"Unsupported preview controller mode: {self.controller_mode}")

    self.lin = [AxisPreviewController(axis, lin_cfg) for axis in lin_axis]
    self.ang = [AxisPreviewController(axis, ang_cfg) for axis in ang_axis]

  def reset(self, com0: np.ndarray, comv0: np.ndarray, phi0: np.ndarray | None = None, omega0: np.ndarray | None = None): 
    com0 = np.asarray(com0, dtype=float).reshape(3,)
    comv0 = np.asarray(comv0, dtype=float).reshape(3,)
    phi0 = np.zeros(3) if phi0 is None else np.asarray(phi0, dtype=float).reshape(3,)
    omega0 = np.zeros(3) if omega0 is None else np.asarray(omega0, dtype=float).reshape(3,)
    for i in range(3): 
      self.lin[i].reset(com0[i], comv0[i], 0.0)
      self.ang[i].reset(phi0[i], omega0[i], 0.0)

  def sync_state(
    self,
    com0: np.ndarray,
    comv0: np.ndarray,
    coma0: np.ndarray | None = None,
    phi0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
    alpha0: np.ndarray | None = None,
  ): 
    com0 = np.asarray(com0, dtype=float).reshape(3,)
    comv0 = np.asarray(comv0, dtype=float).reshape(3,)
    for i in range(3): 
       self.lin[i].x[0] = com0[i]
       self.lin[i].x[1] = comv0[i]
    if coma0 is not None:
      coma0 = np.asarray(coma0, dtype=float).reshape(3,)
      for i in range(3):
        self.lin[i].x[2] = coma0[i]

    if phi0 is not None and omega0 is not None:
      phi0 = np.asarray(phi0, dtype=float).reshape(3,)
      omega0 = np.asarray(omega0, dtype=float).reshape(3,)
      for i in range(3):
        self.ang[i].x[0] = phi0[i]
        self.ang[i].x[1] = omega0[i]
    if alpha0 is not None:
      alpha0 = np.asarray(alpha0, dtype=float).reshape(3,)
      for i in range(3):
        self.ang[i].x[2] = alpha0[i]

  def update_from_meas(
    self,
    com0: np.ndarray,
    comv0: np.ndarray,
    base0: BaseState | None = None,
    *,
    coma0: np.ndarray | None = None,
    alpha0: np.ndarray | None = None,
  ): 
    phi0 = None if base0 is None else base0.phi_world
    omega0 = None if base0 is None else base0.omega_world
    self.sync_state(com0, comv0, coma0=coma0, phi0=phi0, omega0=omega0, alpha0=alpha0)

  def blend_position_only(self, alpha: float, com0: np.ndarray, base0: BaseState | None = None):
    """Blend only position toward measurement, preserving model-derived velocity.
    WARNING: causes systematic amplitude attenuation proportional to
    alpha * (1 - robot_tracking_ratio).  At alpha=0.05 with 91% tracking
    this is ~15%.  Prefer clamp_position_drift for desired_delay mode."""
    com0 = np.asarray(com0, dtype=float).reshape(3,)
    for i in range(3):
      self.lin[i].x[0] += alpha * (com0[i] - self.lin[i].x[0])
    if base0 is not None and base0.phi_world is not None:
      phi0 = np.asarray(base0.phi_world, dtype=float).reshape(3,)
      for i in range(3):
        self.ang[i].x[0] += alpha * (phi0[i] - self.ang[i].x[0])

  def clamp_position_drift(self, max_drift: float, com0: np.ndarray, base0: BaseState | None = None):
    """Hard-clamp preview position to within max_drift of measurement.
    Unlike blend_position_only, introduces ZERO attenuation when the natural
    drift stays below the threshold — the LQT runs completely unconstrained.
    Only activates as a safety net during transients when preview diverges.
    Velocity and acceleration are NOT modified (preserves LQT feedforward).
    Typical max_drift: 0.015-0.020 m (natural steady-state drift at
    0.5 Hz / 50 mm amplitude is ~13 mm)."""
    com0 = np.asarray(com0, dtype=float).reshape(3,)
    for i in range(3):
      drift = self.lin[i].x[0] - com0[i]
      if abs(drift) > max_drift:
        self.lin[i].x[0] = com0[i] + np.sign(drift) * max_drift
    if base0 is not None and base0.phi_world is not None:
      phi0 = np.asarray(base0.phi_world, dtype=float).reshape(3,)
      max_drift_ang = max_drift * 10.0  # generous angular threshold
      for i in range(3):
        drift = self.ang[i].x[0] - phi0[i]
        if abs(drift) > max_drift_ang:
          self.ang[i].x[0] = phi0[i] + np.sign(drift) * max_drift_ang

  def blend_with_meas(self, alpha: float, com0: np.ndarray, comv0: np.ndarray, base0: BaseState | None = None):
    """Blend internal predicted state toward measured state: x = (1-α)*x_pred + α*x_meas."""
    com0 = np.asarray(com0, dtype=float).reshape(3,)
    comv0 = np.asarray(comv0, dtype=float).reshape(3,)
    for i in range(3):
      self.lin[i].x[0] += alpha * (com0[i] - self.lin[i].x[0])
      self.lin[i].x[1] += alpha * (comv0[i] - self.lin[i].x[1])
    if base0 is not None and base0.phi_world is not None and base0.omega_world is not None:
      phi0 = np.asarray(base0.phi_world, dtype=float).reshape(3,)
      omega0 = np.asarray(base0.omega_world, dtype=float).reshape(3,)
      for i in range(3):
        self.ang[i].x[0] += alpha * (phi0[i] - self.ang[i].x[0])
        self.ang[i].x[1] += alpha * (omega0[i] - self.ang[i].x[1])

  @staticmethod 
  def _constant_seq(y: np.ndarray, N: int) -> np.ndarray: 
    y = np.asarray(y, dtype=float).reshape(2,)
    return np.tile(y.reshape(1,2), (N,1))
  
  def step_constant(
    self,
    com_ref: np.ndarray,
    bar_f_ref: np.ndarray,
    phi_ref: np.ndarray,
    bar_n_ref: np.ndarray,
  ) -> tuple[CentroidalReference, ResultantWrenchBar]:
    """ 
    Convenience Wrapper. Provides a constant preview reference over the horizon. 
    NOT A PLANNER SUBSTITUTE. 
    """
    N = int(self.lin_cfg.horizon_steps)
    com_ref = np.asarray(com_ref, dtype=float).reshape(3,)
    bar_f_ref = np.asarray(bar_f_ref, dtype=float).reshape(3,)
    phi_ref = np.asarray(phi_ref, dtype=float).reshape(3,)
    bar_n_ref = np.asarray(bar_n_ref, dtype=float).reshape(3,)

    com = np.zeros(3); com_vel = np.zeros(3); com_acc = np.zeros(3)
    phi = np.zeros(3); omega = np.zeros(3); alpha = np.zeros(3)

    for i in range(3): 
      yL = self._constant_seq(np.array([com_ref[i], bar_f_ref[i]]), N)
      p, v, a = self.lin[i].step(yL)
      com[i], com_vel[i], com_acc[i] = p, v, a
    
      yA = self._constant_seq(np.array([phi_ref[i], bar_n_ref[i]]), N)
      pr, vr, ar = self.ang[i].step(yA)
      phi[i], omega[i], alpha[i] = pr, vr, ar
    

    # CentroidalReference for logging and downstraem; stabilizer uses bar wrench separately 
    ref = CentroidalReference(
      com_ref=com,
      com_vel_ref=com_vel,
      com_acc_ref=com_acc,
      h_ref=np.hstack((self.mass * com_vel, np.zeros(3))),
      hdot_ff_ref=np.hstack((self.mass * com_acc, np.zeros(3))),
      meta={"planner": "finite-horizon-preview-lqt", "phi_ref": phi_ref.tolist()},
    )

    ref.meta["phi"] = phi.tolist()
    ref.meta["omega"] = omega.tolist()
    ref.meta["phi_acc"] = alpha.tolist()

    return ref, ResultantWrenchBar(
      bar_force_world=self.mass * com_acc, 
      bar_moment_world=np.zeros(3, dtype=float), # will be overwritten downstream from I_diag*phi_acc, given I_diag is not exposed here 
    )
    # bar moment world is I alpha (simplification). We set that downstream, since I is not exposed here. bar_n_ref is a reference signal; bar output is embedded in the CentroidalReference for now 

  def step_preview(
    self,
    *,
    com_ref_seq: np.ndarray,     # (Nh,3)
    bar_f_ref_seq: np.ndarray | None = None,   # (Nh,3)
    phi_ref_seq: np.ndarray | None = None,      # (Nh,3) or None
    bar_n_ref_seq: np.ndarray | None = None,    # (Nh,3) or None
  ) -> tuple[CentroidalReference, ResultantWrenchBar]:
    Nh = int(self.lin_cfg.horizon_steps)

    com_ref_seq = np.asarray(com_ref_seq, dtype=float)
    if com_ref_seq.shape != (Nh, 3):
        raise ValueError(f"com_ref_seq must be ({Nh},3), got {com_ref_seq.shape}")
    
    bar_f_ref_seq = np.zeros((Nh, 3), dtype=float) if bar_f_ref_seq is None else np.asarray(bar_f_ref_seq, dtype=float)
    if bar_f_ref_seq.shape != (Nh, 3):
        raise ValueError(f"bar_f_ref_seq must be ({Nh},3), got {bar_f_ref_seq.shape}")

    if phi_ref_seq is None:
        phi_ref_seq = np.zeros((Nh, 3), dtype=float)
    else:
        phi_ref_seq = np.asarray(phi_ref_seq, dtype=float)
        if phi_ref_seq.shape != (Nh, 3):
            raise ValueError(f"phi_ref_seq must be ({Nh},3), got {phi_ref_seq.shape}")

    if bar_n_ref_seq is None:
        bar_n_ref_seq = np.zeros((Nh, 3), dtype=float)
    else:
        bar_n_ref_seq = np.asarray(bar_n_ref_seq, dtype=float)
        if bar_n_ref_seq.shape != (Nh, 3):
            raise ValueError(f"bar_n_ref_seq must be ({Nh},3), got {bar_n_ref_seq.shape}")

    com = np.zeros(3); com_vel = np.zeros(3); com_acc = np.zeros(3)
    phi = np.zeros(3); omega = np.zeros(3); alpha = np.zeros(3)

    for i in range(3):
        yL = np.column_stack([com_ref_seq[:, i], bar_f_ref_seq[:, i]])  # (Nh,2)
        p, v, a = self.lin[i].step(yL)
        com[i], com_vel[i], com_acc[i] = p, v, a

        yA = np.column_stack([phi_ref_seq[:, i], bar_n_ref_seq[:, i]])  # (Nh,2)
        pr, vr, ar = self.ang[i].step(yA)
        phi[i], omega[i], alpha[i] = pr, vr, ar

    ref = CentroidalReference(
        com_ref=com,
        com_vel_ref=com_vel,
        com_acc_ref=com_acc,
        h_ref=np.hstack((self.mass * com_vel, np.zeros(3))),
        hdot_ff_ref=np.hstack((self.mass * com_acc, np.zeros(3))),
        meta={"planner": "finite-horizon-preview-lqt"},
    )
    ref.meta["phi"] = phi.tolist()
    ref.meta["omega"] = omega.tolist()
    ref.meta["phi_acc"] = alpha.tolist()

    return ref, ResultantWrenchBar(
        bar_force_world=self.mass * com_acc,
        bar_moment_world=np.zeros(3, dtype=float),  # overwritten downstream using I_diag * phi_acc
    )
