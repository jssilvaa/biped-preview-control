from __future__ import annotations 
from dataclasses import dataclass 

from misc import assert_shape, check_finite

import numpy as np 

@dataclass(frozen=True)
class LQTModel: 
  A: np.ndarray   # (nx,nx)
  B: np.ndarray   # (nx,nu)
  C: np.ndarray   # (ny,nx)

@dataclass(frozen=True)
class LQTWeights: 
  Qy: np.ndarray  # (ny,ny)
  R: np.ndarray   # (nu,nu)


class FiniteHorizonPreviewLQT:
  """
  Finite-Horizon LQT for output tracking:
    x_{t+1} = A x_t + B u_t
    y_t = C x_t
  Minimize sum_{t=0..N-1} (y_t - yref_t)^T Qy (y_t - yref_t) + u_t^T R u_t
  Applies only u_0 each call (receding horizon), but uses future yref sequence (preview control).
  correct DP solution for the defined cost.
  """
  def __init__(self, model: LQTModel, w: LQTWeights, horizon: int):
    A = check_finite("A", model.A)
    B = check_finite("B", model.B)
    C = check_finite("C", model.C)
    Qy = check_finite("Qy", w.Qy)
    R = check_finite("R", w.R)

    nx = A.shape[0]
    if A.shape != (nx, nx):
      raise ValueError("A must be square")
    if B.shape[0] != nx:
      raise ValueError("B rows must match A")
    nu = B.shape[1]
    ny = C.shape[0]
    if C.shape[1] != nx:
      raise ValueError("C cols must match A")
    if Qy.shape != (ny, ny):
      raise ValueError("Qy must match output dimension")
    if R.shape != (nu, nu):
      raise ValueError("R must match input dimension")
    if horizon <= 0:
      raise ValueError("horizon must be positive")

    self.A, self.B, self.C = A, B, C
    self.Qy, self.R = Qy, R
    self.nx, self.nu, self.ny = nx, nu, ny
    self.N = int(horizon)

    # Precompute Riccati recursion P_t and gains K_t (time-varying)
    Qx = C.T @ Qy @ C
    P = [np.zeros((nx, nx), dtype=float) for _ in range(self.N + 1)]
    K = [np.zeros((nu, nx), dtype=float) for _ in range(self.N)]
    S = [np.zeros((nu, nu), dtype=float) for _ in range(self.N)]

    # terminal cost P_N = 0 (paper uses no terminal term)
    P[self.N][:] = 0.0

    for t in range(self.N - 1, -1, -1):
      Pt1 = P[t + 1]
      St = R + B.T @ Pt1 @ B
      # Stabilize inversion explicitly
      St = 0.5 * (St + St.T)
      S[t] = St
      Kt = np.linalg.solve(St, B.T @ Pt1 @ A)  # (nu,nx)
      K[t] = Kt
      Acl = A - B @ Kt
      P[t] = Acl.T @ Pt1 @ Acl + Qx
      P[t] = 0.5 * (P[t] + P[t].T)

    self._P = P
    self._K = K
    self._S = S

  def step(self, x0: np.ndarray, yref_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    x0: (nx,)
    yref_seq: (N, ny) giving yref[t] for t=0..N-1 (preview)
    Returns:
      u0: (nu,)
      x1: (nx,) next state after applying u0 once
    """
    x = assert_shape("x0", np.asarray(x0, dtype=float).reshape(self.nx,), (self.nx,))
    yref_seq = check_finite("yref_seq", yref_seq)
    if yref_seq.shape != (self.N, self.ny):
      raise ValueError(f"yref_seq must be (N,ny)=({self.N},{self.ny}), got {yref_seq.shape}")

    # DP linear term recursion: J_t(x) = x^T P_t x - 2 x^T p_t + const
    p = [np.zeros((self.nx,), dtype=float) for _ in range(self.N + 1)]
    p[self.N][:] = 0.0
    CTQ = self.C.T @ self.Qy

    for t in range(self.N - 1, -1, -1):
      Kt = self._K[t]
      AclT = (self.A - self.B @ Kt).T
      p[t] = AclT @ p[t + 1] + CTQ @ yref_seq[t]

    # u0 = -K0 x0 + S0^{-1} B^T p1
    S0 = self._S[0]
    K0 = self._K[0]
    u_ff = np.linalg.solve(S0, self.B.T @ p[1])
    u0 = -K0 @ x + u_ff
    x1 = self.A @ x + self.B @ u0
    return u0, x1


class InfiniteHorizonPreviewLQT:
  """
  Infinite-horizon preview tracker for:
    x_{k+1} = A x_k + B u_k
    y_k = C x_k

  Cost:
    sum_{i=k..inf} (y_i - yref_i)^T Qy (y_i - yref_i) + u_i^T R u_i

  The control law matches the fixed-gain preview structure used in Murooka Eq. (5):
    u_k = -Kfb x_k + sum_{i=1..Nh} Kff[i] yref_{k+i}
  """

  def __init__(
    self,
    model: LQTModel,
    w: LQTWeights,
    horizon: int,
    *,
    tol: float = 1e-12,
    max_iter: int = 100000,
  ):
    A = check_finite("A", model.A)
    B = check_finite("B", model.B)
    C = check_finite("C", model.C)
    Qy = check_finite("Qy", w.Qy)
    R = check_finite("R", w.R)

    nx = A.shape[0]
    if A.shape != (nx, nx):
      raise ValueError("A must be square")
    if B.shape[0] != nx:
      raise ValueError("B rows must match A")
    nu = B.shape[1]
    ny = C.shape[0]
    if C.shape[1] != nx:
      raise ValueError("C cols must match A")
    if Qy.shape != (ny, ny):
      raise ValueError("Qy must match output dimension")
    if R.shape != (nu, nu):
      raise ValueError("R must match input dimension")
    if horizon <= 0:
      raise ValueError("horizon must be positive")

    self.A, self.B, self.C = A, B, C
    self.Qy, self.R = Qy, R
    self.nx, self.nu, self.ny = nx, nu, ny
    self.N = int(horizon)

    Qx = C.T @ Qy @ C
    P = Qx.copy()
    for _ in range(max_iter):
      S = R + B.T @ P @ B
      S = 0.5 * (S + S.T)
      K = np.linalg.solve(S, B.T @ P @ A)
      P_next = Qx + A.T @ P @ A - A.T @ P @ B @ K
      P_next = 0.5 * (P_next + P_next.T)
      if np.linalg.norm(P_next - P, ord="fro") <= tol * max(1.0, np.linalg.norm(P_next, ord="fro")):
        P = P_next
        break
      P = P_next
    else:
      raise RuntimeError("Infinite-horizon preview Riccati iteration did not converge")

    self.P = P
    self.S = 0.5 * ((R + B.T @ P @ B) + (R + B.T @ P @ B).T)
    self.Kfb = np.linalg.solve(self.S, B.T @ P @ A)
    self.Acl = A - B @ self.Kfb

    M = C.T @ Qy
    preview_gains = []
    AclT_power = np.eye(nx, dtype=float)
    for _ in range(self.N):
      gain = np.linalg.solve(self.S, B.T @ (AclT_power @ M))
      preview_gains.append(gain)
      AclT_power = self.Acl.T @ AclT_power
    self.Kff = preview_gains

  def step(self, x0: np.ndarray, yref_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = assert_shape("x0", np.asarray(x0, dtype=float).reshape(self.nx,), (self.nx,))
    yref_seq = check_finite("yref_seq", yref_seq)
    if yref_seq.shape != (self.N, self.ny):
      raise ValueError(f"yref_seq must be (N,ny)=({self.N},{self.ny}), got {yref_seq.shape}")

    future_refs = np.vstack((yref_seq[1:], yref_seq[-1:]))
    u_ff = np.zeros((self.nu,), dtype=float)
    for i, gain in enumerate(self.Kff):
      u_ff += gain @ future_refs[i]
    u0 = -self.Kfb @ x + u_ff
    x1 = self.A @ x + self.B @ u0
    return u0, x1
