"""
test_dare_preview_standalone.py — Pure-numpy version of DARE proof tests.
No scipy required. Solves DARE by iterating the Riccati backward recursion
until convergence (which IS convergence to P_inf by definition).

Run:  python3 test_dare_preview_standalone.py
"""
from __future__ import annotations
import numpy as np
import sys, os

# ── DARE solver (pure numpy, no scipy) ──────────────────────────────────

def solve_dare_iterative(A, B, Qx, R, tol=1e-12, max_iter=5000):
    """Iterate backward Riccati from P=Qx until convergence."""
    P = Qx.copy()
    for i in range(max_iter):
        S = R + B.T @ P @ B
        S = 0.5 * (S + S.T)
        K = np.linalg.solve(S, B.T @ P @ A)
        Acl = A - B @ K
        P_new = Acl.T @ P @ Acl + Qx
        P_new = 0.5 * (P_new + P_new.T)
        if np.max(np.abs(P_new - P)) < tol:
            return P_new
        P = P_new
    raise RuntimeError(f"DARE did not converge in {max_iter} iterations")

# ── Triple integrator model ─────────────────────────────────────────────

def build_axis(dt, mass):
    A = np.array([
        [1.0, dt, 0.5*dt**2],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ])
    B = np.array([
        [dt**3/6.0],
        [0.5*dt**2],
        [dt]
    ])
    C = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, mass]
    ])
    return A, B, C

# ── Preview LQT (standalone, mirrors preview_lqt.py logic) ─────────────

class PreviewLQT:
    def __init__(self, A, B, C, Qy, R, N):
        self.A, self.B, self.C = A, B, C
        self.Qy, self.R = Qy, R
        self.N = N
        nx = A.shape[0]
        nu = B.shape[1]
        
        Qx = C.T @ Qy @ C
        P_inf = solve_dare_iterative(A, B, Qx, R)
        self.P_inf = P_inf
        
        P = [np.zeros((nx,nx)) for _ in range(N+1)]
        K = [np.zeros((nu,nx)) for _ in range(N)]
        S = [np.zeros((nu,nu)) for _ in range(N)]
        
        P[N][:] = P_inf  # DARE terminal cost
        
        for t in range(N-1, -1, -1):
            Pt1 = P[t+1]
            St = R + B.T @ Pt1 @ B
            St = 0.5*(St + St.T)
            S[t] = St
            Kt = np.linalg.solve(St, B.T @ Pt1 @ A)
            K[t] = Kt
            Acl = A - B @ Kt
            P[t] = Acl.T @ Pt1 @ Acl + Qx
            P[t] = 0.5*(P[t] + P[t].T)
        
        self.P = P
        self.K = K
        self.S = S
    
    def step(self, x0, yref_seq):
        nx = self.A.shape[0]
        N = self.N
        ny = self.C.shape[0]
        CTQ = self.C.T @ self.Qy
        
        p = [np.zeros(nx) for _ in range(N+1)]
        for t in range(N-1, -1, -1):
            Kt = self.K[t]
            AclT = (self.A - self.B @ Kt).T
            p[t] = AclT @ p[t+1] + CTQ @ yref_seq[t]
        
        u_ff = np.linalg.solve(self.S[0], self.B.T @ p[1])
        u0 = -self.K[0] @ x0 + u_ff
        x1 = self.A @ x0 + self.B @ u0
        return u0, x1

# ── Axis preview controller ─────────────────────────────────────────────

class AxisController:
    def __init__(self, A, B, C, Qy, R, N):
        self.lqt = PreviewLQT(A, B, C, Qy, R, N)
        self.A = A
        self.x = np.zeros(3)
    
    def reset(self, p0, v0, a0=0.0):
        self.x[:] = [p0, v0, a0]
    
    def step(self, yref_seq):
        u0, x1 = self.lqt.step(self.x, yref_seq)
        self.x[:] = x1
        return self.x[0], self.x[1], self.x[2]

# ── Parameters (paper Table I) ──────────────────────────────────────────

DT = 0.005
NH = 400
MASS = 35.0
Q_POS = 2e2
Q_WRENCH = 5e-4
R_JERK = 1e-8

# ── Helpers ──────────────────────────────────────────────────────────────

def amplitude(x):
    return 0.5 * float(np.max(x) - np.min(x))

def phase_lag(ref, sig, dt):
    ref0 = ref - np.mean(ref)
    sig0 = sig - np.mean(sig)
    if np.allclose(ref0, 0) or np.allclose(sig0, 0):
        return None
    ref_fft = np.fft.rfft(ref0)
    sig_fft = np.fft.rfft(sig0)
    freqs = np.fft.rfftfreq(ref0.size, d=dt)
    idx = 1 + int(np.argmax(np.abs(ref_fft[1:])))
    freq_hz = float(freqs[idx])
    if freq_hz <= 0:
        return None
    lag = float(np.angle(ref_fft[idx])) - float(np.angle(sig_fft[idx]))
    lag = (lag + np.pi) % (2*np.pi) - np.pi
    return lag / (2*np.pi*freq_hz)

def run_tracking(dt, mass, N_sim, Nh, q_pos, q_wrench, r_jerk, ref_fn):
    A, B, C = build_axis(dt, mass)
    Qy = np.diag([q_pos, q_wrench])
    R = np.array([[r_jerk]])
    ctrl = AxisController(A, B, C, Qy, R, Nh)
    
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

def sinusoid_ref(amp, freq_hz, mass):
    omega = 2*np.pi*freq_hz
    def ref_fn(k, Nh, dt):
        t = (k + np.arange(Nh)) * dt
        pos = amp * np.sin(omega * t)
        bar_f = mass * (-amp * omega**2 * np.sin(omega * t))
        return np.column_stack([pos, bar_f])
    return ref_fn

def static_ref(target, mass):
    def ref_fn(k, Nh, dt):
        out = np.zeros((Nh, 2))
        out[:, 0] = target
        return out
    return ref_fn

# ── Tests ────────────────────────────────────────────────────────────────

def test_dare_solves_riccati():
    A, B, C = build_axis(DT, MASS)
    Qx = C.T @ np.diag([Q_POS, Q_WRENCH]) @ C
    R = np.array([[R_JERK]])
    P = solve_dare_iterative(A, B, Qx, R)
    
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    Acl = A - B @ K
    P_check = Acl.T @ P @ Acl + Qx
    err = np.max(np.abs(P - P_check))
    assert err < 1e-8, f"DARE residual = {err:.2e}"
    print(f"  DARE residual = {err:.2e} ✓")

def test_dare_psd():
    A, B, C = build_axis(DT, MASS)
    Qx = C.T @ np.diag([Q_POS, Q_WRENCH]) @ C
    R = np.array([[R_JERK]])
    P = solve_dare_iterative(A, B, Qx, R)
    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals}"
    print(f"  eigenvalues = {eigvals} ✓")

def test_backward_riccati_constant():
    A, B, C = build_axis(DT, MASS)
    Qy = np.diag([Q_POS, Q_WRENCH])
    R = np.array([[R_JERK]])
    lqt = PreviewLQT(A, B, C, Qy, R, NH)
    
    max_dev = 0.0
    for t in range(NH + 1):
        dev = np.max(np.abs(lqt.P[t] - lqt.P_inf))
        max_dev = max(max_dev, dev)
    assert max_dev < 1e-6, f"Max P deviation from P_inf = {max_dev:.2e}"
    print(f"  max P[t] deviation from P_inf = {max_dev:.2e} ✓")

def test_static_dc_offset():
    target = 0.1
    N_sim = 2000
    pos, ref = run_tracking(DT, MASS, N_sim, NH, Q_POS, Q_WRENCH, R_JERK,
                            static_ref(target, MASS))
    tail = pos[N_sim*3//4:]
    dc_error = abs(np.mean(tail) - target)
    assert dc_error < 2e-4, f"DC offset = {dc_error:.2e} m (must be < 2e-4)"
    print(f"  DC offset = {dc_error:.2e} m ✓")

def test_sinusoid_amplitude_0_5hz():
    amp, freq = 0.05, 0.5
    N_sim = 4000
    pos, ref = run_tracking(DT, MASS, N_sim, NH, Q_POS, Q_WRENCH, R_JERK,
                            sinusoid_ref(amp, freq, MASS))
    skip = int(5.0 / DT)
    pos_ss, ref_ss = pos[skip:], ref[skip:]
    ratio = amplitude(pos_ss) / amplitude(ref_ss)
    assert ratio > 0.95, f"Amplitude ratio {ratio:.4f} < 0.95"
    print(f"  0.5 Hz amplitude ratio = {ratio:.4f} ✓")

def test_sinusoid_phase_lag_0_5hz():
    amp, freq = 0.05, 0.5
    N_sim = 4000
    pos, ref = run_tracking(DT, MASS, N_sim, NH, Q_POS, Q_WRENCH, R_JERK,
                            sinusoid_ref(amp, freq, MASS))
    skip = int(5.0 / DT)
    lag = phase_lag(ref[skip:], pos[skip:], DT)
    assert lag is not None
    assert abs(lag) < 0.050, f"Phase lag {lag*1000:.1f} ms > 50 ms"
    print(f"  0.5 Hz phase lag = {lag*1000:.1f} ms ✓")

def test_sinusoid_amplitude_1hz():
    amp, freq = 0.02, 1.0
    N_sim = 4000
    pos, ref = run_tracking(DT, MASS, N_sim, NH, Q_POS, Q_WRENCH, R_JERK,
                            sinusoid_ref(amp, freq, MASS))
    skip = int(5.0 / DT)
    ratio = amplitude(pos[skip:]) / amplitude(ref[skip:])
    assert ratio > 0.90, f"1 Hz amplitude ratio {ratio:.4f} < 0.90"
    print(f"  1 Hz amplitude ratio = {ratio:.4f} ✓")

def test_sinusoid_phase_lag_1hz():
    amp, freq = 0.02, 1.0
    N_sim = 4000
    pos, ref = run_tracking(DT, MASS, N_sim, NH, Q_POS, Q_WRENCH, R_JERK,
                            sinusoid_ref(amp, freq, MASS))
    skip = int(5.0 / DT)
    lag = phase_lag(ref[skip:], pos[skip:], DT)
    assert lag is not None
    assert abs(lag) < 0.080, f"1 Hz phase lag {lag*1000:.1f} ms > 80 ms"
    print(f"  1 Hz phase lag = {lag*1000:.1f} ms ✓")

# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("DARE solves Riccati", test_dare_solves_riccati),
        ("DARE positive semidefinite", test_dare_psd),
        ("Backward Riccati = P_inf everywhere", test_backward_riccati_constant),
        ("Static DC offset eliminated", test_static_dc_offset),
        ("0.5 Hz amplitude ratio > 0.95", test_sinusoid_amplitude_0_5hz),
        ("0.5 Hz phase lag < 50 ms", test_sinusoid_phase_lag_0_5hz),
        ("1 Hz amplitude ratio > 0.90", test_sinusoid_amplitude_1hz),
        ("1 Hz phase lag < 80 ms", test_sinusoid_phase_lag_1hz),
    ]
    
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("ALL TESTS PASSED")
    sys.exit(1 if failed > 0 else 0)
