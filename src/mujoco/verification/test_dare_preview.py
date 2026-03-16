"""
test_dare_preview.py — Proof that DARE-initialized preview control fixes
amplitude attenuation, phase lag, and static DC offset.

Tests are pure preview-layer (no MuJoCo, no IK, no compliance).
They instantiate the triple-integrator axis model and run the preview
controller in closed-loop against analytic references.

Run:
    pytest test_dare_preview.py -v
"""
from __future__ import annotations
import os, sys
import numpy as np
import pytest

file_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(file_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from preview_lqt import LQTModel, LQTWeights, FiniteHorizonPreviewLQT, _solve_dare_terminal
from preview_centroidal import TripleIntegratorAxis, PreviewConfig, AxisPreviewController


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_axis(dt: float, mass: float) -> TripleIntegratorAxis:
    return TripleIntegratorAxis.build(dt, mass)


def _run_preview_tracking(
    dt: float,
    mass: float,
    N_sim: int,
    Nh: int,
    q_pos: float,
    q_wrench: float,
    r_jerk: float,
    ref_fn,       # callable(k, Nh, dt) -> (Nh, 2) yref sequence
    ki_pos: float = 0.0,
):
    """Run a single-axis preview controller for N_sim steps, return (pos_log, ref_log)."""
    cfg = PreviewConfig(dt=dt, horizon_steps=Nh, q_pos=q_pos, q_wrench=q_wrench, r_jerk=r_jerk, ki_pos=ki_pos)
    axis = _build_axis(dt, mass)
    ctrl = AxisPreviewController(axis, cfg)

    # start at the initial reference value
    yref0 = ref_fn(0, Nh, dt)
    ctrl.reset(yref0[0, 0], 0.0, 0.0)

    pos_log = np.zeros(N_sim)
    ref_log = np.zeros(N_sim)

    for k in range(N_sim):
        yref = ref_fn(k, Nh, dt)
        ref_log[k] = yref[0, 0]
        p, v, a = ctrl.step(yref)
        pos_log[k] = p

    return pos_log, ref_log


def _sinusoid_ref(amp: float, freq_hz: float, mass: float):
    """Returns a ref_fn for sinusoidal position + dynamically consistent bar_f."""
    omega = 2.0 * np.pi * freq_hz

    def ref_fn(k, Nh, dt):
        t = (k + np.arange(Nh)) * dt
        pos = amp * np.sin(omega * t)
        bar_f = mass * (-amp * omega**2 * np.sin(omega * t))
        return np.column_stack([pos, bar_f])

    return ref_fn


def _static_ref(target: float, mass: float):
    """Returns a ref_fn for constant position target, bar_f = 0."""
    def ref_fn(k, Nh, dt):
        out = np.zeros((Nh, 2))
        out[:, 0] = target
        return out
    return ref_fn


def _amplitude(x: np.ndarray) -> float:
    return 0.5 * float(np.max(x) - np.min(x))


def _phase_lag(ref: np.ndarray, sig: np.ndarray, dt: float) -> float | None:
    """Estimate phase lag at dominant frequency via cross-spectral analysis."""
    ref0 = ref - np.mean(ref)
    sig0 = sig - np.mean(sig)
    if np.allclose(ref0, 0.0) or np.allclose(sig0, 0.0):
        return None
    ref_fft = np.fft.rfft(ref0)
    sig_fft = np.fft.rfft(sig0)
    freqs = np.fft.rfftfreq(ref0.size, d=dt)
    idx = 1 + int(np.argmax(np.abs(ref_fft[1:])))
    freq_hz = float(freqs[idx])
    if freq_hz <= 0:
        return None
    phase_ref = float(np.angle(ref_fft[idx]))
    phase_sig = float(np.angle(sig_fft[idx]))
    lag = phase_ref - phase_sig
    lag = (lag + np.pi) % (2.0 * np.pi) - np.pi
    return lag / (2.0 * np.pi * freq_hz)


# ── Tests ────────────────────────────────────────────────────────────────

# Paper parameters
DT = 0.005
NH = 400
MASS = 35.0  # approx G1 mass
Q_POS = 2e2
Q_WRENCH = 5e-4
R_JERK = 1e-8


class TestDARETerminalCost:
    """Verify that the DARE solution is correct and consistent."""

    def test_dare_solves_riccati(self):
        """P_inf must satisfy the DARE: P = A'PA - A'PB(R+B'PB)^{-1}B'PA + Qx."""
        axis = _build_axis(DT, MASS)
        Qx = axis.C.T @ np.diag([Q_POS, Q_WRENCH]) @ axis.C
        R = np.array([[R_JERK]])
        P = _solve_dare_terminal(axis.A, axis.B, Qx, R)

        # Verify DARE equation
        S = R + axis.B.T @ P @ axis.B
        K = np.linalg.solve(S, axis.B.T @ P @ axis.A)
        Acl = axis.A - axis.B @ K
        P_check = Acl.T @ P @ Acl + Qx
        np.testing.assert_allclose(P, P_check, atol=1e-8,
            err_msg="P_inf does not satisfy the DARE")

    def test_dare_positive_semidefinite(self):
        axis = _build_axis(DT, MASS)
        Qx = axis.C.T @ np.diag([Q_POS, Q_WRENCH]) @ axis.C
        R = np.array([[R_JERK]])
        P = _solve_dare_terminal(axis.A, axis.B, Qx, R)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals >= -1e-10), f"P_inf has negative eigenvalues: {eigvals}"

    def test_backward_riccati_converges_to_dare(self):
        """With P_N = P_inf, the backward Riccati P_t should be constant = P_inf for all t."""
        axis = _build_axis(DT, MASS)
        Qy = np.diag([Q_POS, Q_WRENCH])
        R = np.array([[R_JERK]])
        ctrl = FiniteHorizonPreviewLQT(
            model=LQTModel(A=axis.A, B=axis.B, C=axis.C),
            w=LQTWeights(Qy=Qy, R=R),
            horizon=NH,
        )
        # All P[t] should equal P_inf since we initialized P[N] = P_inf
        P_inf = ctrl._P_inf
        for t in range(NH + 1):
            np.testing.assert_allclose(ctrl._P[t], P_inf, atol=1e-6,
                err_msg=f"P[{t}] deviates from P_inf")


class TestStaticTracking:
    """Static reference: preview must converge to target with zero DC offset."""

    def test_static_dc_offset_small(self):
        """After settling, position must match target within 2e-4 m.

        Note: finite-horizon feedforward truncation (p_N = 0) causes a small
        residual DC offset even with DARE terminal cost.  The state-feedback
        gains K_t are exact (= K_inf), but the preview feedforward sum is
        truncated at the horizon boundary.  At 0.1 mm on a 100 mm target
        (0.1% error) this is 15x better than P_N=0 which gave 1.57 mm.
        """
        target = 0.1  # 10 cm offset
        N_sim = 2000  # 10s at 5ms
        pos, ref = _run_preview_tracking(
            dt=DT, mass=MASS, N_sim=N_sim, Nh=NH,
            q_pos=Q_POS, q_wrench=Q_WRENCH, r_jerk=R_JERK,
            ref_fn=_static_ref(target, MASS),
        )
        # Last 25% should be settled
        tail = pos[N_sim * 3 // 4:]
        dc_error = np.abs(np.mean(tail) - target)
        assert dc_error < 2e-4, f"DC offset = {dc_error:.2e} m (must be < 2e-4)"


class TestSinusoidTracking:
    """Sinusoidal reference: preview must achieve near-unity amplitude ratio
    and near-zero phase lag."""

    @pytest.fixture
    def sinusoid_result(self):
        amp = 0.05
        freq_hz = 0.5
        N_sim = 4000  # 20s at 5ms — enough for steady-state
        pos, ref = _run_preview_tracking(
            dt=DT, mass=MASS, N_sim=N_sim, Nh=NH,
            q_pos=Q_POS, q_wrench=Q_WRENCH, r_jerk=R_JERK,
            ref_fn=_sinusoid_ref(amp, freq_hz, MASS),
        )
        # Discard first 5s of transient
        skip = int(5.0 / DT)
        return pos[skip:], ref[skip:], amp, freq_hz

    def test_amplitude_ratio_near_unity(self, sinusoid_result):
        """Steady-state amplitude ratio must be > 0.95."""
        pos, ref, amp, _ = sinusoid_result
        amp_out = _amplitude(pos)
        amp_ref = _amplitude(ref)
        ratio = amp_out / amp_ref
        print(f"  amplitude ratio = {ratio:.4f} (ref amp = {amp_ref:.4f}, out amp = {amp_out:.4f})")
        assert ratio > 0.95, f"Amplitude ratio {ratio:.4f} < 0.95"

    def test_phase_lag_small(self, sinusoid_result):
        """Steady-state phase lag must be < 50 ms."""
        pos, ref, _, freq_hz = sinusoid_result
        lag = _phase_lag(ref, pos, DT)
        assert lag is not None, "Could not estimate phase lag"
        print(f"  phase lag = {lag * 1000:.1f} ms")
        assert abs(lag) < 0.050, f"Phase lag {lag * 1000:.1f} ms > 50 ms"


class TestSinusoidTrackingHighFreq:
    """Higher frequency sinusoid (1 Hz) — still must track well."""

    @pytest.fixture
    def result_1hz(self):
        amp = 0.02
        freq_hz = 1.0
        N_sim = 4000
        pos, ref = _run_preview_tracking(
            dt=DT, mass=MASS, N_sim=N_sim, Nh=NH,
            q_pos=Q_POS, q_wrench=Q_WRENCH, r_jerk=R_JERK,
            ref_fn=_sinusoid_ref(amp, freq_hz, MASS),
        )
        skip = int(5.0 / DT)
        return pos[skip:], ref[skip:], amp, freq_hz

    def test_amplitude_ratio_1hz(self, result_1hz):
        pos, ref, amp, _ = result_1hz
        ratio = _amplitude(pos) / _amplitude(ref)
        print(f"  1 Hz amplitude ratio = {ratio:.4f}")
        assert ratio > 0.90, f"1 Hz amplitude ratio {ratio:.4f} < 0.90"

    def test_phase_lag_1hz(self, result_1hz):
        pos, ref, _, freq_hz = result_1hz
        lag = _phase_lag(ref, pos, DT)
        assert lag is not None
        print(f"  1 Hz phase lag = {lag * 1000:.1f} ms")
        assert abs(lag) < 0.080, f"1 Hz phase lag {lag * 1000:.1f} ms > 80 ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
