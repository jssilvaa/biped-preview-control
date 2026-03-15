from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from control_types import BaseState, CentroidalReference, ResultantWrenchBar
from preview_lqt import FiniteHorizonPreviewLQT, InfiniteHorizonPreviewLQT, LQTModel, LQTWeights


PreviewControllerMode = Literal["lqt", "lqt_normalized", "preview_servo"]


@dataclass
class PreviewConfig:
  dt: float
  horizon_steps: int
  q_pos: float
  q_wrench: float
  r_jerk: float
  ki_pos: float = 0.0
  ki_max: float = 0.01
  controller_mode: PreviewControllerMode = "lqt"
  position_scale: float = 1.0
  nominal_freq_hz: float = 0.5
  servo_q_integral: float = 10.0
  servo_q_pos: float = 1.0
  servo_q_vel: float = 0.1
  servo_q_acc: float = 0.1

  @staticmethod
  def build_linear(dt: float, horizon_steps: int):
    return PreviewConfig(
      dt=dt,
      horizon_steps=horizon_steps,
      q_pos=2e2,
      q_wrench=5e-4,
      r_jerk=1e-8,
      ki_pos=0.0,
      controller_mode="lqt",
    )

  @staticmethod
  def build_angular(dt: float, horizon_steps: int):
    return PreviewConfig(
      dt=dt,
      horizon_steps=horizon_steps,
      q_pos=1e2,
      q_wrench=5e-3,
      r_jerk=1e-8,
      ki_pos=0.0,
      controller_mode="lqt",
    )

  @staticmethod
  def build_linear_normalized(
    dt: float,
    horizon_steps: int,
    *,
    position_scale: float = 0.05,
    nominal_freq_hz: float = 0.5,
  ):
    return PreviewConfig(
      dt=dt,
      horizon_steps=horizon_steps,
      q_pos=1.0,
      q_wrench=1.0,
      r_jerk=5e-2,
      ki_pos=0.5,
      ki_max=0.2 * position_scale,
      controller_mode="lqt_normalized",
      position_scale=position_scale,
      nominal_freq_hz=nominal_freq_hz,
    )

  @staticmethod
  def build_angular_normalized(
    dt: float,
    horizon_steps: int,
    *,
    position_scale: float = 0.10,
    nominal_freq_hz: float = 0.5,
  ):
    return PreviewConfig(
      dt=dt,
      horizon_steps=horizon_steps,
      q_pos=1.0,
      q_wrench=1.0,
      r_jerk=5e-2,
      ki_pos=0.5,
      ki_max=0.2 * position_scale,
      controller_mode="lqt_normalized",
      position_scale=position_scale,
      nominal_freq_hz=nominal_freq_hz,
    )

  @staticmethod
  def build_linear_preview_servo(
    dt: float,
    horizon_steps: int,
    *,
    position_scale: float = 0.05,
    nominal_freq_hz: float = 0.5,
  ):
    return PreviewConfig(
      dt=dt,
      horizon_steps=horizon_steps,
      q_pos=1.0,
      q_wrench=1.0,
      r_jerk=1e-2,
      controller_mode="preview_servo",
      position_scale=position_scale,
      nominal_freq_hz=nominal_freq_hz,
      servo_q_integral=25.0,
      servo_q_pos=1.0,
      servo_q_vel=0.15,
      servo_q_acc=0.10,
    )

  @staticmethod
  def build_angular_preview_servo(
    dt: float,
    horizon_steps: int,
    *,
    position_scale: float = 0.10,
    nominal_freq_hz: float = 0.5,
  ):
    return PreviewConfig(
      dt=dt,
      horizon_steps=horizon_steps,
      q_pos=1.0,
      q_wrench=1.0,
      r_jerk=1e-2,
      controller_mode="preview_servo",
      position_scale=position_scale,
      nominal_freq_hz=nominal_freq_hz,
      servo_q_integral=10.0,
      servo_q_pos=1.0,
      servo_q_vel=0.10,
      servo_q_acc=0.08,
    )


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

  @staticmethod
  def build(dt: float, output_gain: float) -> TripleIntegratorAxis:
    A = np.array(
      [
        [1.0, dt, 0.5 * dt**2],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
      ],
      dtype=float,
    )
    B = np.array(
      [
        [dt**3 / 6.0],
        [0.5 * dt**2],
        [dt],
      ],
      dtype=float,
    )
    C = np.array(
      [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, output_gain],
      ],
      dtype=float,
    )
    return TripleIntegratorAxis(A=A, B=B, C=C)


@dataclass(frozen=True)
class AxisNormalization:
  position_scale: float
  velocity_scale: float
  acceleration_scale: float
  jerk_scale: float
  wrench_scale: float

  @staticmethod
  def from_nominal(position_scale: float, nominal_freq_hz: float, output_gain: float) -> AxisNormalization:
    pos = max(float(position_scale), 1e-6)
    omega = 2.0 * np.pi * max(float(nominal_freq_hz), 1e-3)
    vel = max(pos * omega, 1e-6)
    acc = max(pos * omega**2, 1e-6)
    jerk = max(pos * omega**3, 1e-6)
    wrench = max(abs(float(output_gain)) * acc, 1e-6)
    return AxisNormalization(
      position_scale=pos,
      velocity_scale=vel,
      acceleration_scale=acc,
      jerk_scale=jerk,
      wrench_scale=wrench,
    )

  def state_scale_matrix(self) -> np.ndarray:
    return np.diag([self.position_scale, self.velocity_scale, self.acceleration_scale]).astype(float)

  def state_inv_scale_matrix(self) -> np.ndarray:
    return np.diag([1.0 / self.position_scale, 1.0 / self.velocity_scale, 1.0 / self.acceleration_scale]).astype(float)


class FiniteHorizonPreviewServo:
  """
  Finite-horizon preview servo on an augmented state:
    eta_{k+1} = eta_k + C x_k - r_k
    x_{k+1} = A x_k + B u_k

  Cost:
    J = sum X_k^T Q X_k + u_k^T R u_k

  where X = [eta; x].
  """

  def __init__(self, A: np.ndarray, B: np.ndarray, Cz: np.ndarray, Q: np.ndarray, R: np.ndarray, horizon: int):
    nx = int(A.shape[0])
    if A.shape != (nx, nx):
      raise ValueError("A must be square")
    if B.shape != (nx, 1):
      raise ValueError("B must be (nx,1)")
    if Cz.shape != (1, nx):
      raise ValueError("Cz must be (1,nx)")
    if Q.shape != (nx + 1, nx + 1):
      raise ValueError("Q must be (nx+1,nx+1)")
    if R.shape != (1, 1):
      raise ValueError("R must be (1,1)")
    if horizon <= 0:
      raise ValueError("horizon must be positive")

    self.nx = nx
    self.N = int(horizon)
    self.A_bar = np.block(
      [
        [np.array([[1.0]], dtype=float), Cz.astype(float)],
        [np.zeros((nx, 1), dtype=float), A.astype(float)],
      ]
    )
    self.B_bar = np.vstack([np.zeros((1, 1), dtype=float), B.astype(float)])
    self.E_bar = np.vstack([np.array([[-1.0]], dtype=float), np.zeros((nx, 1), dtype=float)])
    self.Q = Q.astype(float)
    self.R = R.astype(float)
    self._P: list[np.ndarray] = [np.zeros((nx + 1, nx + 1), dtype=float) for _ in range(self.N + 1)]
    self._K: list[np.ndarray] = [np.zeros((1, nx + 1), dtype=float) for _ in range(self.N)]
    self._S: list[np.ndarray] = [np.zeros((1, 1), dtype=float) for _ in range(self.N)]

    for t in range(self.N - 1, -1, -1):
      Pt1 = self._P[t + 1]
      S = self.R + self.B_bar.T @ Pt1 @ self.B_bar
      S = 0.5 * (S + S.T)
      K = np.linalg.solve(S, self.B_bar.T @ Pt1 @ self.A_bar)
      self._S[t] = S
      self._K[t] = K
      Acl = self.A_bar - self.B_bar @ K
      self._P[t] = Acl.T @ Pt1 @ Acl + self.Q
      self._P[t] = 0.5 * (self._P[t] + self._P[t].T)

  def step(self, x0: np.ndarray, ref_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x0, dtype=float).reshape(self.nx + 1,)
    ref_seq = np.asarray(ref_seq, dtype=float).reshape(self.N,)

    p = [np.zeros((self.nx + 1,), dtype=float) for _ in range(self.N + 1)]
    for t in range(self.N - 1, -1, -1):
      Kt = self._K[t]
      Pt1 = self._P[t + 1]
      dt_vec = (self.E_bar[:, 0] * ref_seq[t]).reshape(self.nx + 1,)
      AclT = (self.A_bar - self.B_bar @ Kt).T
      p[t] = AclT @ (p[t + 1] - Pt1 @ dt_vec)

    S0 = self._S[0]
    K0 = self._K[0]
    d0 = (self.E_bar[:, 0] * ref_seq[0]).reshape(self.nx + 1,)
    u_ff = np.linalg.solve(S0, self.B_bar.T @ (p[1] - self._P[1] @ d0))
    u0 = -K0 @ x + u_ff
    x1 = self.A_bar @ x + self.B_bar[:, 0] * float(u0[0]) + d0
    return u0.reshape(1,), x1


class AxisPreviewController:
  def __init__(self, axis: TripleIntegratorAxis, cfg: PreviewConfig):
    Qy = np.diag([cfg.q_pos, cfg.q_wrench]).astype(float)
    R = np.array([[cfg.r_jerk]], dtype=float)
    self.dt = float(cfg.dt)
    self.x = np.zeros(3, dtype=float)
    self.ctrl = InfiniteHorizonPreviewLQT(
      model=LQTModel(A=axis.A, B=axis.B, C=axis.C),
      w=LQTWeights(Qy=Qy, R=R),
      horizon=cfg.horizon_steps,
    )

  def reset(self, p0: float, v0: float, a0: float = 0.0):
    self.x[:] = np.array([p0, v0, a0], dtype=float)

  def step(self, yref_seq: np.ndarray) -> tuple[float, float, float]:
    _, x1 = self.ctrl.step(self.x, yref_seq=np.asarray(yref_seq, dtype=float))
    self.x[:] = x1
    return self.x[0], self.x[1], self.x[2]


class AxisNormalizedLQTController:
  def __init__(self, axis: TripleIntegratorAxis, cfg: PreviewConfig, normalization: AxisNormalization):
    self.dt = float(cfg.dt)
    self.x = np.zeros(3, dtype=float)
    self._ki = float(cfg.ki_pos)
    self._ki_max = float(cfg.ki_max) / normalization.position_scale
    self._integral = 0.0
    self.norm = normalization

    Sx = normalization.state_scale_matrix()
    Sx_inv = normalization.state_inv_scale_matrix()
    Su = float(normalization.jerk_scale)
    A_hat = Sx_inv @ axis.A @ Sx
    B_hat = Sx_inv @ axis.B * Su
    C_hat = np.array(
      [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
      ],
      dtype=float,
    )
    Qy = np.diag([cfg.q_pos, cfg.q_wrench]).astype(float)
    R = np.array([[cfg.r_jerk]], dtype=float)
    self.ctrl = FiniteHorizonPreviewLQT(
      model=LQTModel(A=A_hat, B=B_hat, C=C_hat),
      w=LQTWeights(Qy=Qy, R=R),
      horizon=cfg.horizon_steps,
    )

  def reset(self, p0: float, v0: float, a0: float = 0.0):
    self.x[:] = np.array([p0, v0, a0], dtype=float)
    self._integral = 0.0

  def _state_to_norm(self) -> np.ndarray:
    return np.array(
      [
        self.x[0] / self.norm.position_scale,
        self.x[1] / self.norm.velocity_scale,
        self.x[2] / self.norm.acceleration_scale,
      ],
      dtype=float,
    )

  def _state_from_norm(self, x_hat: np.ndarray) -> None:
    self.x[:] = np.array(
      [
        x_hat[0] * self.norm.position_scale,
        x_hat[1] * self.norm.velocity_scale,
        x_hat[2] * self.norm.acceleration_scale,
      ],
      dtype=float,
    )

  def _refs_to_norm(self, yref_seq: np.ndarray) -> np.ndarray:
    yref_seq = np.asarray(yref_seq, dtype=float)
    out = yref_seq.copy()
    out[:, 0] /= self.norm.position_scale
    out[:, 1] /= self.norm.wrench_scale
    return out

  def step(self, yref_seq: np.ndarray) -> tuple[float, float, float]:
    x_hat = self._state_to_norm()
    yref_hat = self._refs_to_norm(yref_seq)
    if self._ki > 0.0:
      pos_err = float(yref_hat[0, 0]) - x_hat[0]
      self._integral += self._ki * pos_err * self.dt
      self._integral = np.clip(self._integral, -self._ki_max, self._ki_max)
      yref_hat = yref_hat.copy()
      yref_hat[:, 0] += self._integral
    _, x1_hat = self.ctrl.step(x_hat, yref_seq=yref_hat)
    self._state_from_norm(x1_hat)
    return self.x[0], self.x[1], self.x[2]


class AxisPreviewServoController:
  def __init__(self, axis: TripleIntegratorAxis, cfg: PreviewConfig, normalization: AxisNormalization):
    self.dt = float(cfg.dt)
    self.x = np.zeros(3, dtype=float)
    self.eta = 0.0
    self.norm = normalization
    Sx = normalization.state_scale_matrix()
    Sx_inv = normalization.state_inv_scale_matrix()
    Su = float(normalization.jerk_scale)
    A_hat = Sx_inv @ axis.A @ Sx
    B_hat = Sx_inv @ axis.B * Su
    Cz = np.array([[1.0, 0.0, 0.0]], dtype=float)
    Q = np.diag(
      [
        cfg.servo_q_integral,
        cfg.servo_q_pos,
        cfg.servo_q_vel,
        cfg.servo_q_acc,
      ]
    ).astype(float)
    R = np.array([[cfg.r_jerk]], dtype=float)
    self.ctrl = FiniteHorizonPreviewServo(A=A_hat, B=B_hat, Cz=Cz, Q=Q, R=R, horizon=cfg.horizon_steps)

  def reset(self, p0: float, v0: float, a0: float = 0.0):
    self.x[:] = np.array([p0, v0, a0], dtype=float)
    self.eta = 0.0

  def _state_to_norm(self) -> np.ndarray:
    return np.array(
      [
        self.x[0] / self.norm.position_scale,
        self.x[1] / self.norm.velocity_scale,
        self.x[2] / self.norm.acceleration_scale,
      ],
      dtype=float,
    )

  def _state_from_norm(self, x_hat: np.ndarray) -> None:
    self.x[:] = np.array(
      [
        x_hat[0] * self.norm.position_scale,
        x_hat[1] * self.norm.velocity_scale,
        x_hat[2] * self.norm.acceleration_scale,
      ],
      dtype=float,
    )

  def step(self, yref_seq: np.ndarray) -> tuple[float, float, float]:
    yref_seq = np.asarray(yref_seq, dtype=float)
    pref_seq = yref_seq[:, 0] / self.norm.position_scale
    x_hat = self._state_to_norm()
    X0 = np.hstack((np.array([self.eta], dtype=float), x_hat))
    _, X1 = self.ctrl.step(X0, pref_seq)
    self.eta = float(X1[0])
    self._state_from_norm(np.asarray(X1[1:], dtype=float).reshape(3,))
    return self.x[0], self.x[1], self.x[2]


def _make_axis_controller(axis: TripleIntegratorAxis, cfg: PreviewConfig, output_gain: float):
  if cfg.controller_mode == "lqt":
    return AxisPreviewController(axis, cfg)
  normalization = AxisNormalization.from_nominal(
    position_scale=cfg.position_scale,
    nominal_freq_hz=cfg.nominal_freq_hz,
    output_gain=output_gain,
  )
  if cfg.controller_mode == "lqt_normalized":
    return AxisNormalizedLQTController(axis, cfg, normalization)
  if cfg.controller_mode == "preview_servo":
    return AxisPreviewServoController(axis, cfg, normalization)
  raise ValueError(f"Unsupported preview controller mode: {cfg.controller_mode}")


class CentroidalPreviewPlanner:
  """
  Outputs planned centroidal state rp and planned bar wrench bar_wp.
  Uses rotation-vector components for angular part to avoid Euler angles.
  """

  def __init__(self, mass: float, I_diag: np.ndarray, lin_cfg: PreviewConfig, ang_cfg: PreviewConfig):
    self.mass = float(mass)
    I_diag = np.asarray(I_diag, dtype=float).reshape(3,)
    if not np.all(np.isfinite(I_diag)) or np.any(I_diag <= 0.0):
      raise ValueError("I_diag must be positive definite (3,)")

    self.lin_cfg = lin_cfg
    self.ang_cfg = ang_cfg
    self.lin = [_make_axis_controller(TripleIntegratorAxis.build(lin_cfg.dt, self.mass), lin_cfg, self.mass) for _ in range(3)]
    self.ang = [
      _make_axis_controller(TripleIntegratorAxis.build(ang_cfg.dt, float(I_diag[i])), ang_cfg, float(I_diag[i]))
      for i in range(3)
    ]

  def reset(
    self,
    com0: np.ndarray,
    comv0: np.ndarray,
    phi0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
  ):
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

  def update_from_meas(self, com0: np.ndarray, comv0: np.ndarray, base0: BaseState | None = None):
    phi0 = None if base0 is None else base0.phi_world
    omega0 = None if base0 is None else base0.omega_world
    self.sync_state(com0, comv0, phi0=phi0, omega0=omega0)

  def measured_state(
    self,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    com = np.array([self.lin[i].x[0] for i in range(3)], dtype=float)
    com_vel = np.array([self.lin[i].x[1] for i in range(3)], dtype=float)
    com_acc = np.array([self.lin[i].x[2] for i in range(3)], dtype=float)
    phi = np.array([self.ang[i].x[0] for i in range(3)], dtype=float)
    omega = np.array([self.ang[i].x[1] for i in range(3)], dtype=float)
    alpha = np.array([self.ang[i].x[2] for i in range(3)], dtype=float)
    return com, com_vel, com_acc, phi, omega, alpha

  def blend_with_meas(self, alpha: float, com0: np.ndarray, comv0: np.ndarray, base0: BaseState | None = None):
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
    return np.tile(y.reshape(1, 2), (N, 1))

  def _planner_name(self) -> str:
    mode = self.lin_cfg.controller_mode
    if mode == "lqt":
      return "finite-horizon-preview-lqt"
    if mode == "lqt_normalized":
      return "finite-horizon-preview-lqt-normalized"
    if mode == "preview_servo":
      return "finite-horizon-preview-servo"
    return str(mode)

  def step_constant(
    self,
    com_ref: np.ndarray,
    bar_f_ref: np.ndarray,
    phi_ref: np.ndarray,
    bar_n_ref: np.ndarray,
  ) -> tuple[CentroidalReference, ResultantWrenchBar]:
    N = int(self.lin_cfg.horizon_steps)
    com_ref = np.asarray(com_ref, dtype=float).reshape(3,)
    bar_f_ref = np.asarray(bar_f_ref, dtype=float).reshape(3,)
    phi_ref = np.asarray(phi_ref, dtype=float).reshape(3,)
    bar_n_ref = np.asarray(bar_n_ref, dtype=float).reshape(3,)

    com = np.zeros(3, dtype=float)
    com_vel = np.zeros(3, dtype=float)
    com_acc = np.zeros(3, dtype=float)
    phi = np.zeros(3, dtype=float)
    omega = np.zeros(3, dtype=float)
    alpha = np.zeros(3, dtype=float)

    for i in range(3):
      yL = self._constant_seq(np.array([com_ref[i], bar_f_ref[i]], dtype=float), N)
      p, v, a = self.lin[i].step(yL)
      com[i], com_vel[i], com_acc[i] = p, v, a

      yA = self._constant_seq(np.array([phi_ref[i], bar_n_ref[i]], dtype=float), N)
      pr, vr, ar = self.ang[i].step(yA)
      phi[i], omega[i], alpha[i] = pr, vr, ar

    ref = CentroidalReference(
      com_ref=com,
      com_vel_ref=com_vel,
      com_acc_ref=com_acc,
      h_ref=np.hstack((self.mass * com_vel, np.zeros(3))),
      hdot_ff_ref=np.hstack((self.mass * com_acc, np.zeros(3))),
      meta={"planner": self._planner_name(), "phi_ref": phi_ref.tolist()},
    )
    ref.meta["phi"] = phi.tolist()
    ref.meta["omega"] = omega.tolist()
    ref.meta["phi_acc"] = alpha.tolist()

    return ref, ResultantWrenchBar(
      bar_force_world=self.mass * com_acc,
      bar_moment_world=np.zeros(3, dtype=float),
    )

  def step_preview(
    self,
    *,
    com_ref_seq: np.ndarray,
    bar_f_ref_seq: np.ndarray | None = None,
    phi_ref_seq: np.ndarray | None = None,
    bar_n_ref_seq: np.ndarray | None = None,
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

    com = np.zeros(3, dtype=float)
    com_vel = np.zeros(3, dtype=float)
    com_acc = np.zeros(3, dtype=float)
    phi = np.zeros(3, dtype=float)
    omega = np.zeros(3, dtype=float)
    alpha = np.zeros(3, dtype=float)

    for i in range(3):
      yL = np.column_stack([com_ref_seq[:, i], bar_f_ref_seq[:, i]])
      p, v, a = self.lin[i].step(yL)
      com[i], com_vel[i], com_acc[i] = p, v, a

      yA = np.column_stack([phi_ref_seq[:, i], bar_n_ref_seq[:, i]])
      pr, vr, ar = self.ang[i].step(yA)
      phi[i], omega[i], alpha[i] = pr, vr, ar

    ref = CentroidalReference(
      com_ref=com,
      com_vel_ref=com_vel,
      com_acc_ref=com_acc,
      h_ref=np.hstack((self.mass * com_vel, np.zeros(3))),
      hdot_ff_ref=np.hstack((self.mass * com_acc, np.zeros(3))),
      meta={"planner": self._planner_name()},
    )
    ref.meta["phi"] = phi.tolist()
    ref.meta["omega"] = omega.tolist()
    ref.meta["phi_acc"] = alpha.tolist()

    return ref, ResultantWrenchBar(
      bar_force_world=self.mass * com_acc,
      bar_moment_world=np.zeros(3, dtype=float),
    )
