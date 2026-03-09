from __future__ import annotations
import numpy as np 


def sine_com_ref_seq( 
    k_preview: int, 
    dt_preview: float, 
    Nh: int, 
    com0: np.ndarray,
    *,
    axis: int, 
    amp: float, 
    freq_hz: float, 
) -> np.ndarray:
  """ 
  Build a Horizon sequence for CoM position reference: 
    com_reference(t) = com0 + amp * sin(2πft) * e_axis 
  
    Inputs: 
      `k_preview`: current preview index
      `dt_preview`: preview discretization step (from PreviewConfig.dt, must match MurookaSimConfig.dt)
      `Nh`: preview horizon steps 
      `com0`: base offset (3,)
      `axis`: 0/1/2 for x/y/z
      `amp`: amplitude (m)
      `freq_hz`: frequency (hz)
    Outputs: 
      `com_ref_seq`: (Nh,3)
  """
  if Nh <= 0:
      raise ValueError("Nh must be positive")
  dt_preview = float(dt_preview)
  if not np.isfinite(dt_preview) or dt_preview <= 0:
      raise ValueError("dt_preview must be positive finite")
  if axis not in (0, 1, 2):
      raise ValueError("axis must be 0,1,2")
  amp = float(amp)
  freq_hz = float(freq_hz)
  if not np.isfinite(amp) or not np.isfinite(freq_hz):
      raise ValueError("amp and freq_hz must be finite")
  
  com0 = np.asarray(com0, dtype=float).reshape(3,)
  if not np.all(np.isfinite(com0)): 
      raise ValueError("com0 must be finite (3,)")
  
  t = (k_preview + np.arange(Nh)) * dt_preview
  sig = amp * np.sin(2.0 * np.pi * freq_hz * t)

  out = np.tile(com0.reshape(1, 3), (Nh, 1)) # (Nh, 3) repeated com0 
  out[:, axis] = out[:, axis] + sig # given axis `x`, add the sinusoidal term to the current com 
  return out 


def sine_com_bar_ref_seq(
        k_preview: int,
        dt_preview: float,
        Nh: int,
        com0: np.ndarray,
        *,
        axis: int,
        amp: float,
        freq_hz: float,
        mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a dynamically consistent sinusoidal CoM reference together with the
    matching bar-force reference under the simplified centroidal model used by the
    preview controller:

        c_ref(t) = c0 + A sin(omega t) e_axis
        bar_f_ref(t) = m c_ddot_ref(t) = -m A omega^2 sin(omega t) e_axis
    """
    if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("mass must be positive finite")

    com_ref_seq = sine_com_ref_seq(
            k_preview,
            dt_preview,
            Nh,
            com0,
            axis=axis,
            amp=amp,
            freq_hz=freq_hz,
    )

    omega = 2.0 * np.pi * float(freq_hz)
    t = (k_preview + np.arange(Nh)) * float(dt_preview)
    cdd = -float(amp) * (omega ** 2) * np.sin(omega * t)

    bar_f_ref_seq = np.zeros((Nh, 3), dtype=float)
    bar_f_ref_seq[:, axis] = float(mass) * cdd
    return com_ref_seq, bar_f_ref_seq


def zeros_bar_seq(Nh: int) -> tuple[np.ndarray, np.ndarray]: 
    """ 
    Returns (bar_f_ref_seq, bar_n_ref_seq) each (Nh,3) = zeros
    """
    if Nh <= 0: 
        raise ValueError("Nh must be positive")
    z = np.zeros((Nh, 3), dtype=float)
    return z.copy(), z.copy()


def zeros_phi_seq(Nh: int) -> np.ndarray: 
    """
    Convenience: returns phi_ref_seq (Nh,3) = zeros. 
    """
    if Nh <= 0: 
        raise ValueError("Nh must be positive")
    return np.zeros((Nh,3), dtype=float)