from __future__ import annotations 
from dataclasses import dataclass 

import numpy as np 
import osqp 
from scipy.sparse import csc_matrix 

from control_types import ContactModel,ResultantWrenchBar
from murooka_wrench import bar_to_contact_wrench_about_origin, contact_wrench_about_origin_to_bar
from misc import to_csc 
from lie_math import hat 


@dataclass(frozen=True)
class GeneratorMap: 
  """ 
  G maps lambda >= 0 to contact wrench about world origin: 
    w = G lambda 
  """
  G: np.ndarray 
  patch_slices: list[slice]
  force_cols: list[np.ndarray]  # list of (ncols_i,3) ridge forces in world, per patch
  vert_ids: list[np.ndarray]    # list of (ncols_i,) vertex indices within patch


def build_generator_map(cmodel: ContactModel) -> GeneratorMap: 
  """
  For each patch vertex, create the generator map spanning the 4-sided friction pyramid in the contact frame.   (Here pm stands for plus or minus)
    rho_c in { [pm mu, pm mu, 1] }
  Transform to world with R_wc. 
  Column: [rho_w; p_w x rho_w]
  """
  mu = float(cmodel.mu)
  if not (mu > 0 and np.isfinite(mu)): 
    raise ValueError("friction must be positive finite")
  
  # 4 ridge generators 
  ridges_c = np.array([
    [ +mu, +mu, 1.0],
    [ +mu, -mu, 1.0],
    [ -mu, +mu, 1.0],
    [ -mu, -mu, 1.0],
  ], dtype=float)   # (4,3)
  for i, v in enumerate(ridges_c): ridges_c[i,:] = (v / np.linalg.norm(v))

  cols = []
  patch_slices: list[slice] = []
  force_cols: list[np.ndarray] = []
  vert_ids: list[np.ndarray] = []

  col0 = 0
  for patch in cmodel.patches: 
    V = np.asarray(patch.vertices_world, dtype=float)
    if V.ndim != 2 or V.shape[1] != 3: 
      raise ValueError(f"patch {patch.name} vertices_world must be (Nv,3), got {V.shape} instead")
    R = np.asarray(patch.R_wc, dtype=float).reshape(3,3)
    if not np.all(np.isfinite(R)): 
      raise ValueError(f"patch {patch.name} R_wc is invalid")
    
    patch_forces = []
    patch_vert_ids = []
    for j in range(V.shape[0]): 
      p = V[j]
      for k in range(4): 
        rho_w = R @ ridges_c[k]
        cols.append(np.hstack((rho_w, np.cross(p, rho_w))))
        patch_forces.append(rho_w)
        patch_vert_ids.append(j)

    col1 = col0 + 4 * V.shape[0]
    patch_slices.append(slice(col0, col1))
    force_cols.append(np.asarray(patch_forces, dtype=float).reshape(-1,3))
    vert_ids.append(np.asarray(patch_vert_ids, dtype=int).reshape(-1,))
    col0 = col1 
  
  if not cols: 
    raise ValueError("No generator columns produced (Empty contact model)")
  
  G = np.stack(cols, axis=1)  # (6, nlambda)
  return GeneratorMap(G=G, patch_slices=patch_slices, force_cols=force_cols, vert_ids=vert_ids)


def _tangential_penalty_P(gen: GeneratorMap, contact_model: ContactModel, w_tan: float | np.ndarray) -> np.ndarray:
    """
    Adds a quadratic penalty on tangential force components in *local contact frame* per patch:
      sum_i w_tan_i || [1 0 0; 0 1 0] R_wc^T F_i ||^2
    where F_i = sum_j lambda_ij * rho_w_j.

    This biases solutions away from friction pyramid edges.
    """
    n = gen.G.shape[1]
    Padd = np.zeros((n, n), dtype=float)

    if np.isscalar(w_tan):
        w_tan_vec = np.full(len(contact_model.patches), float(w_tan), dtype=float) # pyright: ignore[reportArgumentType]
    else:
        w_tan_vec = np.asarray(w_tan, dtype=float).reshape(len(contact_model.patches),)
    if np.any(w_tan_vec < 0) or not np.all(np.isfinite(w_tan_vec)):
        raise ValueError("w_tan must be finite and >= 0")

    Pt = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]], dtype=float)  # pick tangential

    for i, (patch, ps, fcols) in enumerate(zip(contact_model.patches, gen.patch_slices, gen.force_cols)):
        wi = float(w_tan_vec[i])
        if wi == 0.0:
            continue
        R = np.asarray(patch.R_wc, dtype=float).reshape(3, 3)
        # F_world = fcols.T @ lam_i, so tangential = Pt R^T fcols.T lam_i
        A = Pt @ (R.T @ fcols.T)  # (2, ncols_i)
        Padd[ps, ps] += wi * (A.T @ A)

    return Padd


def solve_lambda_qp(
    G: np.ndarray, 
    w_target: np.ndarray,
    *, 
    reg: float = 1e-9,
    u_ub: np.ndarray | None = None, # per-variable upper bound (inactive patch => 0)
    P_extra: np.ndarray | None = None,  # extra quadratic cost 
  ) -> np.ndarray: 
  """
  Solve: 
    min ||Glam - w||^2 + reg ||lam||^2 
    s.t. lam >= 0 
  """
  G = np.asarray(G, dtype=float)
  if G.shape[0] != 6: 
    raise ValueError(f"G must be (6,n), got {G.shape} instead")
  w = np.asarray(w_target, dtype=float).reshape(6,)
  if not np.all(np.isfinite(w)): 
    raise ValueError("w_target contains non-finite values")
  if not (reg >= 0 and np.isfinite(reg)): 
    raise ValueError("reg must be finite and >= 0")
  
  n = G.shape[1]
  P = (G.T @ G) + float(reg) * np.eye(n)
  if P_extra is not None: 
    P_extra = np.asarray(P_extra, dtype=float)
    if P_extra.shape != (n, n): 
      raise ValueError(f"P_extra shape mismatch. Expected {G.shape}, got {P.shape}")
    P += P_extra
  P = 0.5 * (P + P.T)
  q = -(G.T @ w)

  # lam >= 0 as A=I, l=0, u=+inf
  A = np.eye(n)
  l = np.zeros(n)
  if u_ub is None:
      u = np.full(n, np.inf, dtype=float)
  else:
      u = np.asarray(u_ub, dtype=float).reshape(n,)
      if np.any(np.isnan(u)):
          raise ValueError("u_ub contains NaN values")
      if np.any(u < 0):
          raise ValueError("u_ub must be >= 0 elementwise")
      if np.any(np.isneginf(u)):
          raise ValueError("u_ub must not contain -inf")

  solver = osqp.OSQP()
  solver.setup(
    P=to_csc(P),
    q=q,
    A=to_csc(A),
    l=l, u=u,
    verbose=False,
    eps_abs=1e-8, eps_rel=1e-8,
    max_iter=50_000, 
    adaptive_rho=True,
    polishing=False,
  )
  res = solver.solve()
  if res.x is None: 
    raise RuntimeError(f"OSQP failed: {res.info.status}")
  return res.x.astype(float)


def _lambda_upper_bounds_from_patch_active(gen: GeneratorMap, patch_active: list[bool] | np.ndarray | None) -> np.ndarray | None:
    if patch_active is None:
        return None
    a = np.asarray(patch_active, dtype=bool).reshape(len(gen.patch_slices),)
    u = np.full(gen.G.shape[1], np.inf, dtype=float)
    for is_on, ps in zip(a, gen.patch_slices):
        if not bool(is_on):
            u[ps] = 0.0
    return u


def project_planned_bar_wrench(
    *,
    bar_wp: ResultantWrenchBar,
    com_planned_world: np.ndarray,
    mass: float, 
    gravity_world: np.ndarray,
    contact_model: ContactModel,
    reg: float = 1e-9,
    patch_active: list[bool] | np.ndarray | None = None, 
    w_tan: float = 0.0 
) -> tuple[ResultantWrenchBar, dict]: 
  """
  Paper Eq (9)-(10) but implemented without ambiguous symbols: 
    1) Convert planned bar wrench to contact wrench about origin using c_p 
    2) Solve lam_opt to match that contact wrench under lam >= 0 
    3) Convert projected contact wrench to projected bar wrench using c_p 
  """
  gen = build_generator_map(contact_model)
  w_p = bar_to_contact_wrench_about_origin(bar_wp, com_planned_world, mass, gravity_world)

  u_ub = _lambda_upper_bounds_from_patch_active(gen, patch_active=patch_active)
  P_extra = _tangential_penalty_P(gen, contact_model=contact_model, w_tan=w_tan) if w_tan > 0 else None 

  lam = solve_lambda_qp(gen.G, w_p, reg=reg, u_ub=u_ub, P_extra=P_extra)
  w_proj = gen.G @ lam 
  bar_proj = contact_wrench_about_origin_to_bar(w_proj, com_planned_world, mass, gravity_world)

  dbg = {
    "w_p": w_p, 
    "w_proj": w_proj, 
    "bar_wp": np.hstack((bar_wp.bar_force_world, bar_wp.bar_moment_world)), 
    "bar_wp_proj": np.hstack((bar_proj.bar_force_world, bar_proj.bar_moment_world)),
    "lambda": lam, 
    "G": gen.G
  }
  return bar_proj, dbg 


def patch_wrenches_from_lambda_world(
    gen: GeneratorMap, 
    contact_model: ContactModel, 
    lam: np.ndarray
) -> list[np.ndarray]: 
  """ 
  Returns per-patch wrench (6,) in world frame, about each site origin for each patch. 
  Uses the same generator columns used to build G 

  For patch i: 
    w_i_origin = sum_j lam_j [rho_w; p_w x rho_w] then shift moment to site origin: 
      tau_about_p0 = tau_about_world_origin - p0 x F 
  """
  lam = np.asarray(lam, dtype=float).reshape(-1)
  if lam.shape[0] != gen.G.shape[1]: 
    raise ValueError("lam length must match generator columns")
  
  out: list[np.ndarray] = []
  for patch, ps in zip(contact_model.patches, gen.patch_slices): 
    w0 = gen.G[:, ps] @ lam[ps] # wrench about world origin 
    F = w0[:3]
    tau0 = w0[3:]
    p0 = np.asarray(patch.p_w, dtype=float).reshape(3,)
    tau_p0 = tau0 - np.cross(p0, F)
    out.append(np.hstack((F, tau_p0)))
  return out 


def distribute_desired_bar_wrench(
    *,
    bar_wd: ResultantWrenchBar, 
    com_actual_world: np.ndarray,
    mass: float, 
    gravity_world: np.ndarray,
    contact_model: ContactModel,
    reg: float = 1e-9,
    patch_active: list[bool] | np.ndarray | None = None, 
    w_tan: float = 0.0, 
) -> tuple[np.ndarray, dict]: 
  """
  Paper Eq (13)-(14): 
    wd = bar_wd + [m g_up; c_a x f_d]
  Here we implement: 
    wd (contact wrench about origin) computed using c_a, then solve lambda 
  """
  gen = build_generator_map(contact_model)
  w_d = bar_to_contact_wrench_about_origin(bar_wd, com_actual_world, mass, gravity_world)

  u_ub = _lambda_upper_bounds_from_patch_active(gen, patch_active)
  P_extra = _tangential_penalty_P(gen, contact_model, w_tan) if w_tan > 0 else None 

  lam = solve_lambda_qp(gen.G, w_d, reg=reg, u_ub=u_ub, P_extra=P_extra)
  w_real = gen.G @ lam 

  patch_forces = []
  for ps, fcols in zip(gen.patch_slices, gen.force_cols): 
    lam_i = lam[ps]
    f_i = (lam_i.reshape(-1,1) * fcols).sum(axis=0)
    patch_forces.append(f_i)
  patch_wrenches = patch_wrenches_from_lambda_world(gen, contact_model, lam)

  dbg = {
    "w_d": w_d, 
    "w_real": w_real, 
    "lambda": lam,
    "patch_forces_world": patch_forces, # list of (3,)
    "patch_wrenches_world": patch_wrenches, # list of (6,) about p0, at each patch 
    "G": gen.G,
  }
  return lam, dbg 