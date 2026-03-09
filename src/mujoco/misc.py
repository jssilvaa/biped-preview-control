import mujoco 
import numpy as np 

from typing import Any
from scipy.sparse import csc_matrix 


def site_ids(model: mujoco.MjModel, names: list[str]) -> list[int]: 
	ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)) for n in names]
	if any(id < 0 for id in ids): 
		missing = [name for (name, id) in zip(names, ids) if id < 0]
		raise ValueError(f"Unknown site(s): {missing}")
	return ids 


def geom_ids(model: mujoco.MjModel, names: list[str]) -> list[int]: 
	ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)) for n in names] 
	if any(id < 0 for id in ids): 
		missing = [name for (name, id) in zip(names, ids) if id < 0]
		raise ValueError(f"Unknown geom(s): {missing}")
	return ids 


def site_attached_geom_ids(model: mujoco.MjModel, site_id: int) -> list[int]: 
	bid = int(model.site_bodyid[site_id])
	gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid]
	return gids 


def safe_array(x, shape=None) -> np.ndarray | None : 
    if x is None: 
        return None 
    arr = np.asarray(x, dtype=float)
    if shape is not None and arr.shape != shape:
        return None 
    if not np.all(np.isfinite(arr)): 
        return None 
    return arr 


def check_finite(name: str, x: np.ndarray) -> np.ndarray: 
  x = np.asarray(x, dtype=float)
  if not np.all(np.isfinite(x)): 
    raise ValueError(f"{name} contains non-finite values")
  return x 

def assert_shape(name: str, x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray: 
  x = check_finite(name, x)
  if x.shape != shape: 
    raise ValueError(f"{name} must be {shape}, got {x.shape} instead")
  return x 

def to_csc(A: np.ndarray) -> csc_matrix: 
  return csc_matrix(A.astype(float))