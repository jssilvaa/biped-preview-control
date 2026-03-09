import numpy as np


def hat(w: np.ndarray) -> np.ndarray:
  """Hat operator: maps a 3-vector to a skew-symmetric matrix in so(3)."""
  w = np.asarray(w, dtype=float).reshape(-1)
  if w.size != 3:
    raise ValueError(f"Expected a 3-vector, got shape={w.shape}")
  wx, wy, wz = w
  return np.array([[0.0, -wz, wy],
                   [wz, 0.0, -wx],
                   [-wy, wx, 0.0]], dtype=float)


def vee(S: np.ndarray, skew_tol: float = 1e-3) -> np.ndarray:
  """Vee operator: maps an so(3) skew-symmetric matrix to a 3-vector."""
  S = np.asarray(S, dtype=float)
  if S.shape != (3, 3):
    raise ValueError(f"Expected a 3x3 matrix, got shape={S.shape}")
  if np.linalg.norm(S + S.T, ord="fro") > skew_tol:
    raise ValueError("Input matrix is not skew-symmetric within tolerance.")
  return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def Log(R: np.ndarray, eps: float = 1e-8, eps_pi: float = 1e-5, clip_tol: float = 1e-5) -> np.ndarray:
  """
  Principal SO(3) logarithm.
  Returns S in so(3) such that exp(S) = R (principal branch).

  Runtime-robust:
  - accepts tiny floating-point drift in trace-based acos argument
  - still raises for genuinely invalid matrices
  """
  R = np.asarray(R, dtype=float)
  if R.shape != (3, 3):
    raise ValueError(f"Expected a 3x3 rotation matrix, got shape={R.shape}")

  arg_raw = float(0.5 * (np.trace(R) - 1.0))

  # tolerate small floating-point drift only
  if arg_raw > 1.0 + clip_tol or arg_raw < -1.0 - clip_tol:
    raise ValueError(f"non-real value in Log theta: {arg_raw}")

  arg = float(np.clip(arg_raw, -1.0, 1.0))
  th = float(np.arccos(arg))

  if th < eps:
    return 0.5 * (R - R.T)

  if np.abs(th - np.pi) < eps_pi:
    diag = np.array([R[0,0], R[1,1], R[2,2]])
    n = np.sqrt(np.maximum((diag + 1.0) / 2.0, 0.0))

    # resolve signs from symmetric off-diagonal structure
    if R[0,1] + R[1,0] < 0: n[1] = -n[1]
    if R[0,2] + R[2,0] < 0: n[2] = -n[2]

    nn = np.linalg.norm(n)
    if nn < 1e-12:
      raise ValueError("Failed to recover axis in Log at pi")
    n = n / nn
    return hat(np.pi * n)

  return 0.5 * th / np.sin(th) * (R - R.T)


def Exp(phi: np.ndarray, eps: float = 1e-12) -> np.ndarray: 
  """ 
  SO(3) exponential map from rotation vector (axis-angle) phi to R. 
  Uses Rodrigues formula. phi is a 3-vector. 
  """
  phi = np.asarray(phi, dtype=float).reshape(-1)
  if phi.size != 3: 
    raise ValueError(f"Expected 3-vector, got {phi.shape}")
  th = np.linalg.norm(phi)
  if th < eps: 
    return np.eye(3) + hat(phi)
  a = phi / th 
  A = hat(a)
  return np.eye(3) + np.sin(th) * A + (1.0 - np.cos(th)) * (A @ A)


def logvec(R: np.ndarray) -> np.ndarray: 
  """
  SO(3) logarithm map returning 3-vector (rotation vector) on principal branch. 
  """
  return vee(Log(R))


def compose_rotvec(delta_phi: np.ndarray, phi: np.ndarray) -> np.ndarray: 
  """ 
  Group composition in rotation-vector coordinates: 
    phi_next = log ( exp( delta_phi^ ) exp( phi^ ))
  """ 
  delta_phi = np.asarray(delta_phi, dtype=float).reshape(3,)
  phi = np.asarray(phi, dtype=float).reshape(3,)
  R = Exp(delta_phi) @ Exp(phi)
  return logvec(R)