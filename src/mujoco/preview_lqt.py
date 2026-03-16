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


def _solve_dare_terminal(
    A: np.ndarray,
    B: np.ndarray,
    Qx: np.ndarray,
    R: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> np.ndarray:
  """
  Solve the Discrete Algebraic Riccati Equation (DARE) by value iteration:
    P_{k+1} = A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A + Qx
  Iterates from P_0 = Qx until ||P_{k+1} - P_k||_inf < tol.

  scipy.linalg.solve_discrete_are uses a Schur-based algorithm that is
  numerically unreliable when R is very small (R^{-1} large), producing
  DARE residuals O(0.1) instead of O(1e-12).  The iterative method
  converges to the true P_inf with residual < 1e-12 on this problem.

  Returns P_inf >= 0 (symmetrised).
  """
  P = Qx.astype(float, copy=True)
  for i in range(max_iter):
    S = R + B.T @ P @ B
    S = 0.5 * (S + S.T)
    K = np.linalg.solve(S, B.T @ P @ A)   # (nu, nx)
    Acl = A - B @ K
    P_new = Acl.T @ P @ Acl + Qx
    P_new = 0.5 * (P_new + P_new.T)
    delta = float(np.max(np.abs(P_new - P)))
    P = P_new
    if delta < tol:
      return P
  raise RuntimeError(
      f"_solve_dare_terminal: did not converge in {max_iter} iterations "
      f"(last delta={delta:.2e}, tol={tol:.2e})"
  )


class FiniteHorizonPreviewLQT:
  """
  DARE-initialized Preview LQT for output tracking (Katayama 1985 / Murooka Eq. 5):
    x_{t+1} = A x_t + B u_t
    y_t = C x_t
  Minimize sum_{t=0..N-1} (y_t - yref_t)^T Qy (y_t - yref_t) + u_t^T R u_t
                           + x_N^T P_inf x_N

  P_inf is the infinite-horizon DARE solution used as terminal cost.
  This ensures the Riccati gains K_t converge to the steady-state K_fb
  within the first few steps of the backward recursion, matching the
  infinite-horizon preview control structure of the paper.

  Applies only u_0 each call (receding horizon), using the full future
  yref sequence (preview feedforward).
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

    # Precompute Riccati recursion P_t and gains K_t
    Qx = C.T @ Qy @ C

    # Solve DARE for infinite-horizon terminal cost
    P_inf = _solve_dare_terminal(A, B, Qx, R)
    P_inf = 0.5 * (P_inf + P_inf.T)
    self._P_inf = P_inf

    P = [np.zeros((nx, nx), dtype=float) for _ in range(self.N + 1)]
    K = [np.zeros((nu, nx), dtype=float) for _ in range(self.N)]
    S = [np.zeros((nu, nu), dtype=float) for _ in range(self.N)]

    # Terminal cost P_N = P_inf (DARE solution)
    P[self.N][:] = P_inf

    for t in range(self.N - 1, -1, -1):
      Pt1 = P[t + 1]
      St = R + B.T @ Pt1 @ B
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

    # Store steady-state closed-loop matrix for the linear term recursion
    S_ss = R + B.T @ P_inf @ B
    S_ss = 0.5 * (S_ss + S_ss.T)
    K_ss = np.linalg.solve(S_ss, B.T @ P_inf @ A)
    self._Acl_ss = A - B @ K_ss
    self._S_ss = S_ss

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
    # Terminal: p_N = P_inf @ 0 = 0 (no fixed terminal reference)
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