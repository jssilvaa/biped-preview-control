from __future__ import annotations 
import numpy as np 
from control_types import ResultantWrenchBar

def require_shape(x: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray: 
  x = np.asarray(x, dtype=float)
  if x.shape != shape: 
    raise ValueError(f"{name} must be shape {shape}, got {x.shape}")
  if not np.all(np.isfinite(x)): 
    raise ValueError(f"{name} contains non-finite values")
  return x 


def bar_to_contact_wrench_about_origin(
    bar: ResultantWrenchBar,
    com_world: np.ndarray, # paper says stability is better in eq. (13) if c_a is used instead of c_d
    mass: float, 
    gravity_world: np.ndarray, 
) -> np.ndarray: 
  """ 
  Convert bar wrench (bar_f, bar_n) to contact wrench w = [f; n0], where n0 is moment about world origin.

  f = bar_f - m g_world 
  n0 = bar_n + c x f
  """
  c = require_shape(com_world, (3,), "com_world")
  g = require_shape(gravity_world, (3,), "gravity_world")
  bf = require_shape(bar.bar_force_world, (3,), "bar_force_world")
  bn = require_shape(bar.bar_moment_world, (3,), "bar_moment_world")

  f = bf - float(mass) * g 
  n0 = bn + np.cross(c, f)
  return np.hstack((f, n0))
  

def contact_wrench_about_origin_to_bar(
    w: np.ndarray,
    com_world: np.ndarray,
    mass: float,
    gravity_world: np.ndarray,
) -> ResultantWrenchBar:
  """
  Convert contact wrench w=[f; n0] (moment about origin) into bar wrench.

  bar_f = f + m g_world
  bar_n = n0 - c x f
  """
  w = require_shape(np.asarray(w, dtype=float).reshape(6,), (6,), "w")
  c = require_shape(com_world, (3,), "com_world")
  g = require_shape(gravity_world, (3,), "gravity_world")

  f = w[:3]
  n0 = w[3:]
  bf = f + float(mass) * g
  bn = n0 - np.cross(c, f)
  return ResultantWrenchBar(bar_force_world=bf, bar_moment_world=bn)