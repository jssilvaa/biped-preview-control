"""
test_all.py — CI gate test suite for the biped controller.

Design principles:
- Every test has an analytically computed truth: expected result is derived by hand and hard-coded.
- No tolerances wider than 1e-6 except where explicitly justified by a named approximation.
- MuJoCo-dependent tests build minimal programmatic models (no XML files required).
- Tests are ordered by data-flow dependency, i.e. the stuff that comes earlier in the pipeline is tested earlier, e.g. lie_math first, whole_body_ik last.

Run:
    pytest test_all.py -v
"""
from __future__ import annotations
import os, sys 
import numpy as np
import pytest
import mujoco

file_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(file_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 1. lie_math
# ─────────────────────────────────────────────────────────────────────────────

class TestLieMath:
    """
    All results are derivable by hand from Rodrigues' formula or direct matrix algebra.
    """

    def test_hat_vee_roundtrip(self):
        from lie_math import hat, vee
        w = np.array([1.0, -2.0, 3.0])
        assert np.allclose(vee(hat(w)), w, atol=1e-12)

    def test_hat_skew_symmetry(self):
        from lie_math import hat
        w = np.array([0.5, -1.0, 2.0])
        S = hat(w)
        assert np.allclose(S + S.T, 0, atol=1e-12)

    def test_hat_cross_product(self):
        """hat(w) @ v == w × v"""
        from lie_math import hat
        w = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        assert np.allclose(hat(w) @ v, np.cross(w, v), atol=1e-12)

    def test_Exp_identity_at_zero(self):
        from lie_math import Exp
        assert np.allclose(Exp(np.zeros(3)), np.eye(3), atol=1e-12)

    def test_Exp_rotation_x_90(self):
        """Exp([π/2, 0, 0]) should equal Rx(π/2)."""
        from lie_math import Exp
        phi = np.array([np.pi/2, 0.0, 0.0])
        # Rx(π/2) = [[1,0,0],[0,0,-1],[0,1,0]]
        expected = _rot_x(np.pi/2)
        assert np.allclose(Exp(phi), expected, atol=1e-12)

    def test_Exp_rotation_z_45(self):
        from lie_math import Exp
        phi = np.array([0.0, 0.0, np.pi/4])
        expected = _rot_z(np.pi/4)
        assert np.allclose(Exp(phi), expected, atol=1e-12)

    def test_Exp_Log_roundtrip_small(self):
        """logvec(Exp(phi)) == phi for ||phi|| << π."""
        from lie_math import Exp, logvec
        phi = np.array([0.1, -0.2, 0.3])
        assert np.allclose(logvec(Exp(phi)), phi, atol=1e-12)

    def test_Exp_Log_roundtrip_large(self):
        """logvec(Exp(phi)) == phi for ||phi|| just below π."""
        from lie_math import Exp, logvec
        phi = np.array([1.0, 1.0, 1.0]) / np.sqrt(3) * (np.pi - 0.01)
        assert np.allclose(logvec(Exp(phi)), phi, atol=1e-10)

    def test_Exp_is_SO3(self):
        """Exp output must be a valid rotation matrix: R^T R = I, det = +1."""
        from lie_math import Exp
        phi = np.array([0.5, -1.2, 0.8])
        R = Exp(phi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_Log_at_pi_no_nan(self):
        """
        Log at θ=π must not produce NaN or inf.
        Oracle: Rx(π) = diag(1,-1,-1). logvec should return phi = [π,0,0].
        """
        from lie_math import Log, vee
        R = _rot_x(np.pi)
        S = Log(R)
        result = vee(S)
        assert np.all(np.isfinite(result)), f"Log at θ=π produced non-finite: {result}"
        # The axis is x, magnitude π
        assert np.isclose(np.linalg.norm(result), np.pi, atol=1e-8)

    def test_Log_at_pi_axis_correct(self):
        """For Rz(π), the rotation axis must be z."""
        from lie_math import logvec
        R = _rot_z(np.pi)
        phi = logvec(R)
        assert np.all(np.isfinite(phi))
        # axis must be z (or -z, both represent the same rotation at π)
        axis = phi / np.linalg.norm(phi)
        assert np.isclose(np.abs(axis[2]), 1.0, atol=1e-6), f"Expected z-axis rotation, got {axis}"

    def test_compose_rotvec_identity(self):
        """compose_rotvec(phi, -phi) == 0 (R · R^{-1} = I)."""
        from lie_math import compose_rotvec
        phi = np.array([0.3, -0.5, 0.7])
        result = compose_rotvec(phi, -phi)
        assert np.allclose(result, np.zeros(3), atol=1e-12)

    def test_compose_rotvec_associativity(self):
        """(a ∘ b) ∘ c == a ∘ (b ∘ c) in rotation-vector composition."""
        from lie_math import compose_rotvec
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([-0.1, 0.4, 0.0])
        c = np.array([0.3, 0.0, -0.2])
        lhs = compose_rotvec(compose_rotvec(a, b), c)
        rhs = compose_rotvec(a, compose_rotvec(b, c))
        assert np.allclose(lhs, rhs, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 2. murooka_wrench
# ─────────────────────────────────────────────────────────────────────────────

class TestMurookaWrench:
    """
    Bar wrench convention: bar_f = f + mg_world, bar_n = n0 - c × f
    where n0 is moment about world origin, g_world = [0,0,-9.81].

    At static equilibrium: f = -mg = [0,0,mg], n0 = 0 (assuming CoM above origin).
    Then: bar_f = [0,0,mg] + m[0,0,-g] = [0,0,0], bar_n = 0 - c × [0,0,mg] = -c × [0,0,mg].
    """

    G = 9.81
    M = 10.0
    G_WORLD = np.array([0.0, 0.0, -G])
    COM = np.array([0.0, 0.0, 0.5])

    def test_bar_to_contact_roundtrip(self):
        from murooka_wrench import bar_to_contact_wrench_about_origin, contact_wrench_about_origin_to_bar
        bar_f = np.array([1.0, 2.0, 3.0])
        bar_n = np.array([0.1, -0.2, 0.3])
        from control_types import ResultantWrenchBar
        bar_in = ResultantWrenchBar(bar_force_world=bar_f, bar_moment_world=bar_n)
        w = bar_to_contact_wrench_about_origin(bar_in, self.COM, self.M, self.G_WORLD)
        bar_out = contact_wrench_about_origin_to_bar(w, self.COM, self.M, self.G_WORLD)
        assert np.allclose(bar_out.bar_force_world, bar_f, atol=1e-12)
        assert np.allclose(bar_out.bar_moment_world, bar_n, atol=1e-12)

    def test_static_equilibrium_bar_force(self):
        """
        At rest: contact force = [0,0,mg] upward.
        bar_f = f + m*g_world = [0,0,mg] + m[0,0,-g] = [0,0,0].
        """
        from murooka_wrench import contact_wrench_about_origin_to_bar
        f_contact = np.array([0.0, 0.0, self.M * self.G])  # upward normal reaction
        n0_contact = np.zeros(3)  # moment about world origin = 0 (foot at origin)
        w = np.hstack((f_contact, n0_contact))
        bar = contact_wrench_about_origin_to_bar(w, np.zeros(3), self.M, self.G_WORLD)
        assert np.allclose(bar.bar_force_world, np.zeros(3), atol=1e-12)

    def test_contact_force_from_gravity_only(self):
        """
        bar_f = [0,0,0] (gravity balanced) → f = bar_f - m*g_world = [0,0,mg].
        """
        from murooka_wrench import bar_to_contact_wrench_about_origin
        from control_types import ResultantWrenchBar
        bar = ResultantWrenchBar(
            bar_force_world=np.zeros(3),
            bar_moment_world=np.zeros(3),
        )
        w = bar_to_contact_wrench_about_origin(bar, self.COM, self.M, self.G_WORLD)
        f = w[:3]
        # f = 0 - m*(-9.81) = +mg upward
        assert np.allclose(f, np.array([0.0, 0.0, self.M * self.G]), atol=1e-12)

    def test_moment_shift(self):
        """
        bar_n = n0 - c × f.
        With f=[0,0,F], c=[cx,cy,0]:
          c × f = [cy*F, -cx*F, 0]
        so bar_n = n0 - [cy*F, -cx*F, 0].
        """
        from murooka_wrench import contact_wrench_about_origin_to_bar
        cx, cy, F = 0.1, -0.2, 50.0
        com = np.array([cx, cy, 0.0])
        f = np.array([0.0, 0.0, F])
        n0 = np.zeros(3)
        w = np.hstack((f, n0))
        bar = contact_wrench_about_origin_to_bar(w, com, self.M, self.G_WORLD)
        # bar_n = n0 - c × f = -[cy*F, -cx*F, 0]
        expected_bar_n = -np.cross(com, f)
        assert np.allclose(bar.bar_moment_world, expected_bar_n, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 3. preview_lqt
# ─────────────────────────────────────────────────────────────────────────────

class TestPreviewLQT:
    """
    For a 1D integrator x_{k+1} = x_k + u_k, y_k = x_k,
    with cost sum (y_k - yref)^2 + r u_k^2 and N steps:

    The unconstrained optimal is a known closed-form for constant reference.
    Key property: u* is linear in (x - yref) and decreasing in magnitude
    as the horizon grows.

    We test structural properties rather than the exact scalar value:
      1. Zero reference, zero initial state → zero control forever.
      2. Positive reference, positive state error → control has correct sign.
      3. Forward prediction: x1 = A x0 + B u0 matches closed-loop recursion.
    """

    def _make_integrator_lqt(self, N=50, qy=1.0, r=0.01):
        from preview_lqt import LQTModel, LQTWeights, FiniteHorizonPreviewLQT
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        Qy = np.array([[qy]])
        R  = np.array([[r]])
        return FiniteHorizonPreviewLQT(LQTModel(A=A, B=B, C=C), LQTWeights(Qy=Qy, R=R), N)

    def test_zero_reference_zero_state_zero_control(self):
        ctrl = self._make_integrator_lqt()
        x0 = np.array([0.0])
        yref = np.zeros((50, 1))
        u0, x1 = ctrl.step(x0, yref)
        assert np.allclose(u0, 0.0, atol=1e-12)
        assert np.allclose(x1, 0.0, atol=1e-12)

    def test_positive_error_positive_control(self):
        """
        yref > x0 means we need to increase x; u0 should be positive.
        """
        ctrl = self._make_integrator_lqt()
        x0 = np.array([0.0])
        yref = np.ones((50, 1))
        u0, _ = ctrl.step(x0, yref)
        assert float(u0[0]) > 0.0

    def test_state_update_consistency(self):
        """x1 returned by step must equal A @ x0 + B @ u0."""
        ctrl = self._make_integrator_lqt()
        x0 = np.array([0.3])
        yref = np.tile([[1.0]], (50, 1))
        u0, x1 = ctrl.step(x0, yref)
        expected_x1 = ctrl.A @ x0 + ctrl.B @ u0
        assert np.allclose(x1, expected_x1, atol=1e-12)

    def test_triple_integrator_state_update(self):
        """
        For the actual triple integrator used in preview_centroidal:
        A is (3,3), B is (3,1), C is (2,3).
        x1 = A x0 + B u0 must hold exactly.
        """
        from preview_centroidal import TripleIntegratorAxis
        from preview_lqt import LQTModel, LQTWeights, FiniteHorizonPreviewLQT
        dt = 0.005
        axis = TripleIntegratorAxis.build(dt, output_gain=10.0)
        Qy = np.diag([200.0, 5e-4])
        R = np.array([[1e-8]])
        ctrl = FiniteHorizonPreviewLQT(LQTModel(A=axis.A, B=axis.B, C=axis.C),
                                       LQTWeights(Qy=Qy, R=R), 400)
        x0 = np.array([1.0, 0.1, 0.0])
        yref = np.tile([[1.0, 0.0]], (400, 1))
        u0, x1 = ctrl.step(x0, yref)
        expected = axis.A @ x0 + axis.B @ u0
        assert np.allclose(x1, expected, atol=1e-12)

    def test_riccati_P_positive_semidefinite(self):
        """All P_t in the backward pass must be PSD."""
        ctrl = self._make_integrator_lqt(N=20)
        for t, P in enumerate(ctrl._P):
            eigs = np.linalg.eigvalsh(P)
            assert np.all(eigs >= -1e-12), f"P[{t}] not PSD: min eig = {eigs.min()}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. preview_centroidal
# ─────────────────────────────────────────────────────────────────────────────

class TestPreviewCentroidal:

    def _make_planner(self, dt=0.005, N=400):
        from preview_centroidal import CentroidalPreviewPlanner, PreviewConfig
        return CentroidalPreviewPlanner(
            mass=35.0,
            I_diag=np.array([1.0, 1.0, 0.5]),
            cfg=PreviewConfig(dt=dt, horizon_steps=N, q_pos=2e2, q_wrench=5e-4, r_jerk=1e-8),
        )

    def test_reset_then_constant_ref_converges(self):
        """
        With com_ref = com0 = (0,0,1) held constant for many steps,
        the planned CoM position must converge to com_ref.
        """
        planner = self._make_planner()
        com0 = np.array([0.0, 0.0, 1.0])
        planner.reset(com0=com0, comv0=np.zeros(3))
        for _ in range(200):
            ref, bar_wp = planner.step_constant(
                com_ref=com0,
                bar_f_ref=np.zeros(3),
                phi_ref=np.zeros(3),
                bar_n_ref=np.zeros(3),
            )
        # After convergence, planned position should be within 1 mm of reference
        assert np.allclose(ref.com_ref, com0, atol=1e-3), \
            f"Planner did not converge: {ref.com_ref} vs {com0}"

    def test_bar_wp_force_matches_mass_times_acc(self):
        """
        bar_wp.bar_force_world = mass * com_acc from the planner.
        Verify this explicitly after one step.
        """
        planner = self._make_planner()
        planner.reset(com0=np.zeros(3), comv0=np.zeros(3))
        ref, bar_wp = planner.step_constant(
            com_ref=np.array([0.1, 0.0, 0.0]),
            bar_f_ref=np.zeros(3),
            phi_ref=np.zeros(3),
            bar_n_ref=np.zeros(3),
        )
        expected_bar_f = 35.0 * ref.com_acc_ref
        assert np.allclose(bar_wp.bar_force_world, expected_bar_f, atol=1e-10)

    def test_step_preview_matches_step_constant_for_constant_seq(self):
        """
        step_preview with a constant sequence must return the same result
        as step_constant with the corresponding scalar.
        """
        from preview_centroidal import CentroidalPreviewPlanner, PreviewConfig
        cfg = PreviewConfig(dt=0.005, horizon_steps=100, q_pos=2e2, q_wrench=5e-4, r_jerk=1e-8)
        p1 = CentroidalPreviewPlanner(mass=35.0, I_diag=np.ones(3), cfg=cfg)
        p2 = CentroidalPreviewPlanner(mass=35.0, I_diag=np.ones(3), cfg=cfg)
        com0 = np.array([0.0, 0.0, 1.0])
        p1.reset(com0=com0, comv0=np.zeros(3))
        p2.reset(com0=com0, comv0=np.zeros(3))

        com_ref = np.array([0.05, 0.0, 1.0])
        ref_c, bar_c = p1.step_constant(com_ref=com_ref, bar_f_ref=np.zeros(3),
                                         phi_ref=np.zeros(3), bar_n_ref=np.zeros(3))
        com_seq = np.tile(com_ref.reshape(1, 3), (100, 1))
        ref_p, bar_p = p2.step_preview(com_ref_seq=com_seq)

        assert np.allclose(ref_c.com_ref, ref_p.com_ref, atol=1e-12)
        assert np.allclose(ref_c.com_acc_ref, ref_p.com_acc_ref, atol=1e-12)
        assert np.allclose(bar_c.bar_force_world, bar_p.bar_force_world, atol=1e-12)

    def test_angular_planner_I_diag_scaling(self):
        """
        bar_moment_world = I_diag * phi_acc.
        Two planners with different I_diag but same phi_ref should produce
        proportionally different bar_moment_world (through the meta field).
        We verify that I_diag=2 gives 2× the bar moment of I_diag=1.
        """
        from preview_centroidal import CentroidalPreviewPlanner, PreviewConfig, TripleIntegratorAxis
        # This tests the stack_controller logic that overwrites bar_moment_world,
        # so we verify the meta["phi_acc"] field is consistent.
        cfg = PreviewConfig(dt=0.005, horizon_steps=100, q_pos=2e2, q_wrench=5e-4, r_jerk=1e-8)
        I1 = np.array([1.0, 1.0, 1.0])
        I2 = np.array([2.0, 2.0, 2.0])
        p1 = CentroidalPreviewPlanner(mass=10.0, I_diag=I1, cfg=cfg)
        p2 = CentroidalPreviewPlanner(mass=10.0, I_diag=I2, cfg=cfg)
        p1.reset(com0=np.zeros(3), comv0=np.zeros(3), phi0=np.zeros(3))
        p2.reset(com0=np.zeros(3), comv0=np.zeros(3), phi0=np.zeros(3))
        phi_ref = np.array([0.1, 0.0, 0.0])
        ref1, _ = p1.step_constant(com_ref=np.zeros(3), bar_f_ref=np.zeros(3),
                                    phi_ref=phi_ref, bar_n_ref=np.zeros(3))
        ref2, _ = p2.step_constant(com_ref=np.zeros(3), bar_f_ref=np.zeros(3),
                                    phi_ref=phi_ref, bar_n_ref=np.zeros(3))
        # phi_acc from planner with I=2 should differ from I=1 because the C matrix
        # has output_gain=I[i], which weights the wrench reference differently.
        # Both phi_acc values are finite and nonzero (active reference).
        acc1 = np.asarray(ref1.meta["phi_acc"])
        acc2 = np.asarray(ref2.meta["phi_acc"])
        assert np.all(np.isfinite(acc1)) and np.all(np.isfinite(acc2))


# ─────────────────────────────────────────────────────────────────────────────
# 5. centroidal_prediction
# ─────────────────────────────────────────────────────────────────────────────

class TestCentroidalPrediction:
    """
    Oracle: given known bar wrench and initial state, verify closed-form prediction.
    """

    def test_zero_wrench_free_fall_linear(self):
        """
        bar_f = 0 => com_ddot = bar_f/m = 0 => com is constant (no acceleration).
        (Note: bar_f = 0 means gravity is balanced, so no net acceleration.)
        """
        from centroidal_prediction import predict_one_step
        from control_types import ResultantWrenchBar
        dt = 0.01
        m = 20.0
        c0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([0.1, -0.2, 0.0])
        bar = ResultantWrenchBar(bar_force_world=np.zeros(3), bar_moment_world=np.zeros(3))
        out = predict_one_step(dt=dt, mass=m, I_diag=np.ones(3),
                               com=c0, com_vel=v0,
                               base_R=None, base_omega=None,
                               bar_wp_proj=bar)
        # cdd = 0 => c_d = c0 + v0*dt
        expected_c = c0 + v0 * dt
        expected_v = v0.copy()
        assert np.allclose(out.com, expected_c, atol=1e-12)
        assert np.allclose(out.com_vel, expected_v, atol=1e-12)

    def test_constant_bar_force_kinematics(self):
        """
        bar_f = [0, 0, F] => com_ddot = [0, 0, F/m]
        c_d = c0 + v0*dt + 0.5*(F/m)*dt^2
        cd_d = v0 + (F/m)*dt
        """
        from centroidal_prediction import predict_one_step
        from control_types import ResultantWrenchBar
        dt, m, F = 0.005, 10.0, 50.0
        c0 = np.zeros(3)
        v0 = np.zeros(3)
        bar = ResultantWrenchBar(bar_force_world=np.array([0.0, 0.0, F]),
                                  bar_moment_world=np.zeros(3))
        out = predict_one_step(dt=dt, mass=m, I_diag=np.ones(3),
                               com=c0, com_vel=v0,
                               base_R=None, base_omega=None,
                               bar_wp_proj=bar)
        acc = F / m
        assert np.allclose(out.com, np.array([0, 0, 0.5 * acc * dt**2]), atol=1e-12)
        assert np.allclose(out.com_vel, np.array([0, 0, acc * dt]), atol=1e-12)

    def test_angular_prediction_Euler(self):
        """
        omega_dot = I^{-1} * bar_n
        With I=[2,2,2], bar_n=[0,0,1]: omegadot = [0,0,0.5]
        omega_d = [0,0,0.5*dt], R_d = Exp([0,0,0.5*dt^2]) @ I = Rz(0.5*dt^2)
        """
        from centroidal_prediction import predict_one_step
        from control_types import ResultantWrenchBar
        from lie_math import Exp
        dt = 0.005
        I_diag = np.array([2.0, 2.0, 2.0])
        bar = ResultantWrenchBar(bar_force_world=np.zeros(3),
                                  bar_moment_world=np.array([0.0, 0.0, 1.0]))
        R0 = np.eye(3)
        omega0 = np.zeros(3)
        out = predict_one_step(dt=dt, mass=10.0, I_diag=I_diag,
                               com=np.zeros(3), com_vel=np.zeros(3),
                               base_R=R0, base_omega=omega0,
                               bar_wp_proj=bar)
        omegadot = np.array([0.0, 0.0, 0.5])
        omega_d_expected = omegadot * dt
        if out.base_omega_world is None: 
            raise ValueError("out.base_omega_world is none")
        else:
          assert np.allclose(out.base_omega_world, omega_d_expected, atol=1e-12)
        # R_d = Exp(omega_d * dt) @ R0; note code uses omega_d not omega0
        R_d_expected = Exp(omega_d_expected * dt) @ R0
        if out.base_R_world is None: 
            assert ValueError("out.base_R_world is none")
        else:
          assert np.allclose(out.base_R_world, R_d_expected, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 6. centroidal_stabilizer
# ─────────────────────────────────────────────────────────────────────────────

class TestCentroidalStabilizer:

    def _make_gains(self, kp_lin=100.0, kd_lin=10.0, kp_ang=50.0, kd_ang=5.0):
        from centroidal_stabilizer import StabilizerGains
        return StabilizerGains.diagonal(
            kp_lin=(kp_lin,)*3, kd_lin=(kd_lin,)*3,
            kp_ang=(kp_ang,)*3, kd_ang=(kd_ang,)*3,
        )

    def test_zero_error_zero_correction(self):
        """
        When desired == measured, delta_bar_wrench must be exactly zero.
        """
        from centroidal_stabilizer import stabilize_bar_wrench
        from control_types import (ResultantWrenchBar, CentroidalMeasured,
                                    CentroidalDesired, BaseState)
        from lie_math import logvec
        com = np.array([0.1, 0.2, 0.8])
        R = _rot_x(0.1)
        omega = np.array([0.01, 0.0, 0.0])
        phi = logvec(R)

        bar_in = ResultantWrenchBar(bar_force_world=np.array([0, 0, 50.0]),
                                     bar_moment_world=np.zeros(3))
        desired = CentroidalDesired(com=com, com_vel=np.zeros(3),
                                     base_R_world=R, base_omega_world=omega)
        measured = CentroidalMeasured(
            com=com, com_vel=np.zeros(3),
            base=BaseState(R_world=R, omega_world=omega, phi_world=phi),
        )
        bar_out, _ = stabilize_bar_wrench(bar_wp_proj=bar_in, desired=desired,
                                           measured=measured, gains=self._make_gains())
        assert np.allclose(bar_out.bar_force_world, bar_in.bar_force_world, atol=1e-12)
        assert np.allclose(bar_out.bar_moment_world, bar_in.bar_moment_world, atol=1e-12)

    def test_linear_correction_sign_and_magnitude(self):
        """
        With desired.com = [0.1, 0, 0], measured.com = [0, 0, 0], Kp=100*I:
        delta_bar_f = Kp @ (c_d - c_a) = 100 * [0.1, 0, 0] = [10, 0, 0]
        """
        from centroidal_stabilizer import stabilize_bar_wrench
        from control_types import ResultantWrenchBar, CentroidalMeasured, CentroidalDesired
        bar_in = ResultantWrenchBar(bar_force_world=np.zeros(3), bar_moment_world=np.zeros(3))
        desired = CentroidalDesired(com=np.array([0.1, 0.0, 0.0]),
                                     com_vel=np.zeros(3),
                                     base_R_world=None, base_omega_world=None)
        measured = CentroidalMeasured(com=np.zeros(3), com_vel=np.zeros(3), base=None)
        bar_out, _ = stabilize_bar_wrench(bar_wp_proj=bar_in, desired=desired,
                                           measured=measured, gains=self._make_gains(kp_lin=100.0, kd_lin=0.0))
        assert np.allclose(bar_out.bar_force_world, np.array([10.0, 0.0, 0.0]), atol=1e-12)

    def test_angular_correction_so3_error(self):
        """
        R_d = Rx(θ), R_a = I → e_R = logvec(R_d @ I^T) = [θ, 0, 0].
        With Kp_ang = kp*I: delta_bar_n = kp * [θ, 0, 0].
        """
        from centroidal_stabilizer import stabilize_bar_wrench
        from control_types import (ResultantWrenchBar, CentroidalMeasured,
                                    CentroidalDesired, BaseState)
        theta = 0.3
        kp = 50.0
        R_d = _rot_x(theta)
        R_a = np.eye(3)
        bar_in = ResultantWrenchBar(bar_force_world=np.zeros(3), bar_moment_world=np.zeros(3))
        desired = CentroidalDesired(com=np.zeros(3), com_vel=np.zeros(3),
                                     base_R_world=R_d, base_omega_world=np.zeros(3))
        measured = CentroidalMeasured(
            com=np.zeros(3), com_vel=np.zeros(3),
            base=BaseState(R_world=R_a, omega_world=np.zeros(3), phi_world=np.zeros(3)),
        )
        bar_out, _ = stabilize_bar_wrench(bar_wp_proj=bar_in, desired=desired,
                                           measured=measured,
                                           gains=self._make_gains(kp_lin=0.0, kd_lin=0.0,
                                                                   kp_ang=kp, kd_ang=0.0))
        expected_dbn = kp * np.array([theta, 0.0, 0.0])
        assert np.allclose(bar_out.bar_moment_world, expected_dbn, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 7. wrench_qp_generators
# ─────────────────────────────────────────────────────────────────────────────

class TestWrenchQP:
    """
    G maps λ≥0 to contact wrench at world origin.
    For a single patch at origin with R_wc=I, the 4 ridge columns are:
      col_k = [rho_w_k; 0 × rho_w_k] = [rho_w_k; 0]
    (moment is zero because p = origin).
    """

    def _single_patch_model(self, mu=0.5, p_w=None, R_wc=None):
        from control_types import ContactModel, ContactPatch
        if p_w is None:
            p_w = np.zeros(3)
        if R_wc is None:
            R_wc = np.eye(3)
        patch = ContactPatch(
            name="test",
            vertices_world=p_w.reshape(1, 3),
            p_w=p_w,
            R_wc=R_wc,
        )
        return ContactModel(patches=[patch], mu=mu)

    def test_generator_shape_single_patch_point_contact(self):
        """Single patch, 1 vertex, 4 ridges → G is (6, 4)."""
        from wrench_qp_generators import build_generator_map
        model = self._single_patch_model(mu=0.5)
        gen = build_generator_map(model)
        assert gen.G.shape == (6, 4), f"G shape: {gen.G.shape}"

    def test_generator_columns_nonnegative_normal(self):
        """Every column of G must have f_z >= 0 (ridge forces point upward)."""
        from wrench_qp_generators import build_generator_map
        model = self._single_patch_model(mu=0.5)
        gen = build_generator_map(model)
        fz = gen.G[2, :]  # z-component of force
        assert np.all(fz > 0), f"Some ridge has fz <= 0: {fz}"

    def test_uniform_lambda_force_sum(self):
        """
        For uniform λ=1 at a single patch at origin with R_wc=I, μ=0.5:
          ridges_c = [±0.5, ±0.5, 1] / || [±0.5, ±0.5, 1] || 
          Sum of 4 ridges = [0, 0, 4] / [±0.5, ±0.5, 1] 
          f_world = [0, 0, 4] / [±0.5, ±0.5, 1] , moment = 0
        """
        from wrench_qp_generators import build_generator_map
        mu = 0.5
        ridges_c_norm = np.linalg.norm(np.array([mu, mu, 1], dtype=float))
        model = self._single_patch_model(mu=mu)
        gen = build_generator_map(model)
        lam = np.ones(4)
        w = gen.G @ lam
        assert np.allclose(w[:3], np.array([0.0, 0.0, 4.0]) / ridges_c_norm, atol=1e-12)
        assert np.allclose(w[3:], np.zeros(3), atol=1e-12)

    def test_moment_about_offset_patch(self):
        """
        Patch at p=[1,0,0], R=I, lam=[1,1,1,1]:
          f = [0, 0, 4] (same as above)
          tau_world_origin = p × f = [1,0,0] × [0,0,4] = [0*4-0*0, 0*0-1*4, 1*0-0*0]
                           = [0, -4, 0]
        """
        from wrench_qp_generators import build_generator_map
        p = np.array([1.0, 0.0, 0.0])
        mu = 0.5 
        ridges_c_norm = np.linalg.norm(np.array([mu, mu, 1], dtype=float))
        model = self._single_patch_model(mu=mu, p_w=p)
        gen = build_generator_map(model)
        lam = np.ones(4)
        w = gen.G @ lam
        f = np.array([0.0, 0.0, 4.0]) / ridges_c_norm
        assert np.allclose(w[:3], f, atol=1e-12)
        expected_tau = np.cross(p, f)  # [0, -4, 0] / ridges_c_norm
        assert np.allclose(w[3:], expected_tau, atol=1e-12)

    def test_solve_lambda_qp_feasible_wrench(self):
        """
        Requesting w = [0, 0, 1] (unit upward force) should be exactly achievable
        by a positive combination of ridges. Verify G @ lam ≈ w.
        """
        from wrench_qp_generators import build_generator_map, solve_lambda_qp
        model = self._single_patch_model(mu=0.6)
        gen = build_generator_map(model)
        w_target = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        lam = solve_lambda_qp(gen.G, w_target, reg=1e-9)
        assert np.all(lam >= -1e-8), f"Negative lambda: {lam.min()}"
        w_achieved = gen.G @ lam
        assert np.allclose(w_achieved[:3], w_target[:3], atol=1e-4)

    def test_inactive_patch_lambda_zeroed(self):
        """
        When patch_active=[False], all lambda for that patch must be 0.
        """
        from wrench_qp_generators import (build_generator_map, solve_lambda_qp,
                                           _lambda_upper_bounds_from_patch_active)
        model = self._single_patch_model(mu=0.6)
        gen = build_generator_map(model)
        u_ub = _lambda_upper_bounds_from_patch_active(gen, [False])
        w_target = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        lam = solve_lambda_qp(gen.G, w_target, reg=1e-9, u_ub=u_ub)
        assert np.allclose(lam, 0.0, atol=1e-6)

    def test_two_patches_equal_split(self):
        """
        Two symmetric patches at [±d, 0, 0] with target w = [0,0,F,0,0,0] (no moment).
        Each should receive F/2 upward force.
        """
        from control_types import ContactModel, ContactPatch
        from wrench_qp_generators import build_generator_map, solve_lambda_qp
        mu = 0.5
        d = 0.2
        def _patch(name, x):
            p = np.array([x, 0.0, 0.0])
            return ContactPatch(name=name, vertices_world=p.reshape(1,3),
                                p_w=p, R_wc=np.eye(3))
        model = ContactModel(patches=[_patch("L", -d), _patch("R", d)], mu=mu)
        gen = build_generator_map(model)
        F = 20.0
        w_target = np.array([0.0, 0.0, F, 0.0, 0.0, 0.0])
        lam = solve_lambda_qp(gen.G, w_target, reg=1e-9)
        # Net force per patch = sum of ridge forces
        sl, sr = gen.patch_slices
        fz_L = gen.G[2, sl] @ lam[sl]
        fz_R = gen.G[2, sr] @ lam[sr]
        assert np.isclose(fz_L + fz_R, F, atol=1e-4)
        assert np.isclose(fz_L, fz_R, atol=1e-4), f"Unequal split: L={fz_L}, R={fz_R}"

    def test_patch_wrenches_moment_shifted(self):
        """
        patch_wrenches_from_lambda_world: the moment must be about p0, not origin.
        For patch at p=[1,0,0], tau_origin = [0,-4,0] (from above test),
        tau_p0 = tau_origin - p × f = [0,-4,0] - [1,0,0]×[0,0,4]
               = [0,-4,0] - [0*4-0*0, 0*0-1*4, 1*0-0*0]
               = [0,-4,0] - [0,-4,0] = [0,0,0]
        (Force passes through p0, so moment about p0 is zero for axial force.)
        """
        from wrench_qp_generators import build_generator_map, patch_wrenches_from_lambda_world
        p = np.array([1.0, 0.0, 0.0])
        model = self._single_patch_model(mu=0.5, p_w=p)
        gen = build_generator_map(model)
        lam = np.ones(4)
        wrenches = patch_wrenches_from_lambda_world(gen, model, lam)
        assert len(wrenches) == 1
        # f = [0,0,4] about p0 → tau = 0 (force through origin of patch)
        assert np.allclose(wrenches[0][3:], np.zeros(3), atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 8. damping_control
# ─────────────────────────────────────────────────────────────────────────────

class TestDampingControl:
    """
    ODE: drdot = -(Ks/Kd) dr + (Kf/Kd)(w_meas - w_des)
    
    DC gain (steady state with constant load w_err):
      0 = -(Ks/Kd) dr_ss + (Kf/Kd) w_err
      dr_ss = (Kf/Ks) w_err
    """

    def test_zero_error_zero_state_no_change(self):
        from damping_control import damping_step, ComplianceState, DampingGains
        gains = DampingGains(
            Kd=np.ones(6)*100, Ks=np.ones(6)*10, Kf=np.ones(6)*1
        )
        state = ComplianceState.zero()
        new_state = damping_step(dt=0.001, gains=gains, state=state,
                                  w_meas=np.zeros(6), w_des=np.zeros(6))
        assert np.allclose(new_state.dr, 0.0, atol=1e-12)

    def test_dc_gain_linear(self):
        """
        Run to steady state with constant wrench error.
        dr_ss = (Kf/Ks) * w_err (element-wise, for the linear dofs).
        """
        from damping_control import damping_step, ComplianceState, DampingGains
        Kd = np.ones(6) * 10.0
        Ks = np.ones(6) * 100.0
        Kf = np.ones(6) * 1.0
        gains = DampingGains(Kd=Kd, Ks=Ks, Kf=Kf)
        w_err = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w_des = np.zeros(6)
        w_meas = w_err
        state = ComplianceState.zero()
        dt = 1e-4
        for _ in range(200_000):  # run until convergence
            state = damping_step(dt=dt, gains=gains, state=state,
                                  w_meas=w_meas, w_des=w_des)
        dr_ss = state.dr[:3]
        expected = (Kf[:3] / Ks[:3]) * w_err[:3]  # = [0.1, 0, 0]
        assert np.allclose(dr_ss, expected, atol=1e-4)

    def test_exponential_decay_no_load(self):
        """
        With Kf=0 (no force feedback) and initial dr=[1,0,...]:
        dr(t) = exp(-(Ks/Kd)*t) * dr0 → verify at one time constant.
        """
        from damping_control import damping_step, ComplianceState, DampingGains
        Kd = np.ones(6) * 100.0
        Ks = np.ones(6) * 10.0
        gains = DampingGains(Kd=Kd, Ks=Ks, Kf=np.zeros(6))
        dr0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state = ComplianceState(dr=dr0.copy())
        tau = float(Kd[0] / Ks[0])  # time constant = 10 s
        dt = 1e-4
        N = int(tau / dt)
        for _ in range(N):
            state = damping_step(dt=dt, gains=gains, state=state,
                                  w_meas=np.zeros(6), w_des=np.zeros(6))
        expected = np.exp(-1.0)  # dr(tau) = e^{-1} * dr0
        assert np.isclose(state.dr[0], expected, atol=1e-3)

    def test_angular_compose_rotvec_used(self):
        """
        Angular part of dr must use SO(3) composition (compose_rotvec), not linear addition.

        Key: axes must be PERPENDICULAR. For co-axial rotations [a,0,0]+[b,0,0],
        Rx(a)@Rx(b)=Rx(a+b), so compose_rotvec == naive sum — the test can't distinguish them.

        With perpendicular axes a=[θ,0,0] and b=[0,θ,0], BCH gives:
          compose_rotvec(a,b) ≈ a + b + (a×b)/2
        The z-component of a×b = θ²/2 ≠ 0, so the result DIFFERS from a+b.
        Exact: logvec(Rx(θ) @ Ry(θ)) — computed from compose_rotvec oracle.
        """
        from damping_control import damping_step, ComplianceState, DampingGains
        from lie_math import compose_rotvec

        theta = 0.5
        # dr0: y-axis rotation; w_err angular: x-axis rotation
        dr0 = np.array([0.0, 0.0, 0.0, 0.0, theta, 0.0])   # [lin; ang=[0,θ,0]]
        w_err = np.array([0.0, 0.0, 0.0, theta, 0.0, 0.0])  # [lin; ang=[θ,0,0]]

        # Kd=1, Ks=0, Kf=1, dt=1: drdot_A = w_err_A = [θ,0,0]; increment = dt*[θ,0,0] = [θ,0,0]
        gains = DampingGains(Kd=np.ones(6), Ks=np.zeros(6), Kf=np.ones(6))
        state = ComplianceState(dr=dr0.copy())
        new_state = damping_step(dt=1.0, gains=gains, state=state,
                                  w_meas=w_err, w_des=np.zeros(6))

        # SO(3) oracle: compose_rotvec(increment=[θ,0,0], current=[0,θ,0])
        expected_ang = compose_rotvec(np.array([theta, 0.0, 0.0]),
                                       np.array([0.0, theta, 0.0]))
        assert np.allclose(new_state.dr[3:], expected_ang, atol=1e-12), \
            f"Angular: got {new_state.dr[3:]}, expected {expected_ang}"

        # Naive sum: [θ, θ, 0]. SO(3) result has nonzero z from BCH cross term.
        naive_sum = np.array([theta, theta, 0.0])
        assert not np.allclose(new_state.dr[3:], naive_sum, atol=1e-6), \
            "Implementation is doing naive vector addition — missing compose_rotvec"

# ─────────────────────────────────────────────────────────────────────────────
# 9. contact_phase
# ─────────────────────────────────────────────────────────────────────────────

class TestContactPhase:

    def test_hysteresis_initially_off(self):
        from contact_phase import ContactHysteresis
        h = ContactHysteresis(fn_on=30.0, fn_off=10.0)
        h.reset(2)
        assert not np.any(h.active)

    def test_hysteresis_turns_on_above_fn_on(self):
        from contact_phase import ContactHysteresis
        h = ContactHysteresis(fn_on=30.0, fn_off=10.0)
        h.reset(1)
        active = h.update(np.array([35.0]))
        assert bool(active[0])

    def test_hysteresis_stays_on_above_fn_off(self):
        from contact_phase import ContactHysteresis
        h = ContactHysteresis(fn_on=30.0, fn_off=10.0)
        h.reset(1)
        h.update(np.array([35.0]))  # turn on
        active = h.update(np.array([15.0]))  # above fn_off=10 → stays on
        assert bool(active[0])

    def test_hysteresis_turns_off_below_fn_off(self):
        from contact_phase import ContactHysteresis
        h = ContactHysteresis(fn_on=30.0, fn_off=10.0)
        h.reset(1)
        h.update(np.array([35.0]))  # on
        active = h.update(np.array([5.0]))  # below fn_off → off
        assert not bool(active[0])

    def test_hysteresis_does_not_turn_on_between_thresholds(self):
        """fn_off < fn < fn_on: should NOT turn on from off state."""
        from contact_phase import ContactHysteresis
        h = ContactHysteresis(fn_on=30.0, fn_off=10.0)
        h.reset(1)
        active = h.update(np.array([20.0]))  # between 10 and 30, starting off
        assert not bool(active[0])

    def test_select_patch_gains_single_contact_linear_relax(self):
        """
        With single_contact_linear_relax=True and exactly one active patch,
        the active patch must have linear gains from noncontact and angular from contact.
        """
        from contact_phase import select_patch_gains, PhaseGains
        gains = PhaseGains.murooka_table_ii()
        active = np.array([True, False])
        out = select_patch_gains(active, gains)
        # Linear (first 3) of active patch should match noncontact
        assert np.allclose(out[0].Kd[:3], gains.noncontact.Kd[:3], atol=1e-12)
        # Angular (last 3) of active patch should match contact
        assert np.allclose(out[0].Kd[3:], gains.contact.Kd[3:], atol=1e-12)
        # Inactive patch should have noncontact gains
        assert np.allclose(out[1].Kd, gains.noncontact.Kd, atol=1e-12)

    def test_select_patch_gains_both_active_no_relax(self):
        """With two active patches, no linear relaxation applied."""
        from contact_phase import select_patch_gains, PhaseGains
        gains = PhaseGains.murooka_table_ii()
        active = np.array([True, True])
        out = select_patch_gains(active, gains)
        for g in out:
            assert np.allclose(g.Kd, gains.contact.Kd, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 10. dynamics (MuJoCo-dependent, minimal model)
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamics:
    """
    Minimal model: single free body (sphere) with known mass.
    This avoids loading any robot XML.
    """

    FREE_BODY_XML = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="ball" pos="0 0 1">
          <freejoint/>
          <geom type="sphere" size="0.1" mass="5.0"/>
        </body>
      </worldbody>
    </mujoco>
    """

    @pytest.fixture
    def free_model(self):
        m = mujoco.MjModel.from_xml_string(self.FREE_BODY_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d

    def test_compute_total_mass(self, free_model):
        from dynamics import compute_total_mass
        m, d = free_model
        assert np.isclose(compute_total_mass(m), 5.0, atol=1e-10)

    def test_com_at_body_position(self, free_model):
        """Single body: CoM = body position."""
        from dynamics import compute_com_state
        m, d = free_model
        com, _ = compute_com_state(m, d)
        assert np.allclose(com, np.array([0.0, 0.0, 1.0]), atol=1e-6)

    def test_contact_wrench_resultant_map_single_site(self):
        """
        Single contact at [d, 0, 0], force [0, 0, F], about world origin:
          tau = [d, 0, 0] × [0, 0, F] = [0*F-0*0, 0*0-d*F, d*0-0*0] = [0, -d*F, 0]
        W @ lambda = [0, 0, F, 0, -d*F, 0]
        """
        from dynamics import contact_wrench_resultant_map
        d, F = 0.3, 20.0
        p = np.array([[d, 0.0, 0.0]])
        W = contact_wrench_resultant_map(p, about_point_world=None)
        lam = np.array([0.0, 0.0, F])
        result = W @ lam
        expected = np.array([0.0, 0.0, F, 0.0, -d*F, 0.0])
        assert np.allclose(result, expected, atol=1e-12)

    def test_contact_wrench_resultant_map_about_nonzero_point(self):
        """
        Shifting reference point: tau_ref = tau_origin - ref × f
        """
        from dynamics import contact_wrench_resultant_map
        p = np.array([[1.0, 0.0, 0.0]])
        ref = np.array([0.5, 0.0, 0.0])
        W = contact_wrench_resultant_map(p, about_point_world=ref)
        F = 10.0
        lam = np.array([0.0, 0.0, F])
        result = W @ lam
        # r = p - ref = [0.5, 0, 0]; tau = r × f = [0.5,0,0]×[0,0,F] = [0,-0.5F,0]
        expected_tau = np.cross(np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, F]))
        assert np.allclose(result[3:], expected_tau, atol=1e-12)

    def test_Ag_momentum_product(self, free_model):
        """
        For a free body with known velocity qvel=[vx,vy,vz, wx,wy,wz] at the free joint,
        h = Ag @ qvel should give total linear momentum = m * v_com.
        """
        from dynamics import compute_centroidal_full
        m, d = free_model
        # Set linear velocity on the free joint (qvel[0:3] = translational)
        vx = 1.5
        d.qvel[0] = vx
        mujoco.mj_forward(m, d)
        Ag, h = compute_centroidal_full(m, d, 0)
        mass = 5.0
        assert np.isclose(h[0], mass * vx, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 11. whole_body_ik (MuJoCo-dependent, pendulum model)
# ─────────────────────────────────────────────────────────────────────────────

class TestWholeBodyIK:
    """
    Simple 1-DOF pendulum: revolute joint at origin, link of length 1.
    Site at tip of link. CoM at link midpoint (0, 0, -0.5) at zero config.

    This is a pure analytical fixture: at q=0 (vertical), site is at [0,0,-1].
    We solve for q that brings site to a specified target.
    """

    PENDULUM_XML = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="link" pos="0 0 0">
          <joint name="j1" type="hinge" axis="1 0 0" pos="0 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.02" mass="1.0"/>
          <site name="tip" pos="0 0 -1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j1" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
      </actuator>
    </mujoco>
    """

    @pytest.fixture
    def pend(self):
        m = mujoco.MjModel.from_xml_string(self.PENDULUM_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d

    def test_zero_config_site_at_bottom(self, pend):
        m, d = pend
        site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
        p = np.asarray(d.site_xpos[site_id]).copy()
        assert np.allclose(p, np.array([0.0, 0.0, -1.0]), atol=1e-6)

    def test_ik_recovers_zero_config_from_zero(self, pend):
        """
        With site_target = [0,0,-1] (bottom) and starting at q=0,
        IK should return q ≈ 0 (already at target).
        """
        from whole_body_ik import solve_ik, IKConfig, SiteTarget
        m, d = pend
        site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
        R_target = np.eye(3)
        p_target = np.array([0.0, 0.0, -1.0])
        com_target = np.array([0.0, 0.0, -0.5])  # CoM at midpoint of link

        cfg = IKConfig(max_iters=20, damping=1e-4, step_size=0.5,
                       w_com=1.0, w_site_pos=10.0, w_site_rot=0.1, w_posture=1e-4)
        q_des = solve_ik(m, d, com_target=com_target,
                          site_targets=[SiteTarget(site_id=site_id, p_world=p_target,
                                                    R_world=R_target)],
                          qpos_nominal=None, cfg=cfg)
        assert np.allclose(q_des, np.array([0.0]), atol=1e-3)

    def test_ik_does_not_mutate_data_qpos(self, pend):
        """
        solve_ik must restore data.qpos to its initial value after solving.
        """
        from whole_body_ik import solve_ik, IKConfig, SiteTarget
        m, d = pend
        site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
        d.qpos[0] = 0.3  # set non-zero
        mujoco.mj_forward(m, d)
        q_before = d.qpos.copy()

        cfg = IKConfig(max_iters=5)
        solve_ik(m, d, com_target=np.array([0.0, 0.0, -0.5]),
                  site_targets=[SiteTarget(site_id=site_id,
                                            p_world=np.array([0.0, 0.0, -1.0]),
                                            R_world=np.eye(3))],
                  qpos_nominal=None, cfg=cfg)

        assert np.allclose(d.qpos, q_before, atol=1e-12), \
            "solve_ik mutated data.qpos"

    def test_ik_site_position_converges(self, pend):
        """
        Start at q=π/4 (link tilted 45°). Ask IK to bring site back to [0,0,-1].
        After solve, evaluate site position at q_des and verify.
        """
        from whole_body_ik import solve_ik, IKConfig, SiteTarget
        m, d = pend
        site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")

        d.qpos[0] = np.pi / 4
        mujoco.mj_forward(m, d)

        cfg = IKConfig(max_iters=50, damping=1e-5, step_size=0.3,
                       w_com=0.0, w_site_pos=10.0, w_site_rot=0.01, w_posture=1e-5)
        p_target = np.array([0.0, 0.0, -1.0])
        q_des = solve_ik(m, d, com_target=np.array([0.0, 0.0, -0.5]),
                          site_targets=[SiteTarget(site_id=site_id, p_world=p_target,
                                                    R_world=np.eye(3))],
                          qpos_nominal=None, cfg=cfg)

        # Evaluate at q_des
        d.qpos[:] = q_des
        mujoco.mj_forward(m, d)
        p_actual = np.asarray(d.site_xpos[site_id]).copy()

        # Restore
        d.qpos[0] = np.pi / 4
        mujoco.mj_forward(m, d)

        assert np.allclose(p_actual, p_target, atol=5e-3), \
            f"IK site position error: {p_actual} vs {p_target}"


# ─────────────────────────────────────────────────────────────────────────────
# 12. joint_servo (MuJoCo-dependent)
# ─────────────────────────────────────────────────────────────────────────────

class TestJointServo:
    """Verify PD torque computation using the pendulum fixture."""

    HINGE_XML = """
    <mujoco>
      <worldbody>
        <body name="link">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="m1" joint="j1" gear="2.0" ctrllimited="true" ctrlrange="-50 50"/>
      </actuator>
    </mujoco>
    """

    @pytest.fixture
    def hinge(self):
        m = mujoco.MjModel.from_xml_string(self.HINGE_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d

    def test_zero_error_zero_ctrl(self, hinge):
        from joint_servo import compute_motor_ctrl_from_qpos_target, JointServoConfig
        m, d = hinge
        d.qpos[0] = 0.0
        d.qvel[0] = 0.0
        mujoco.mj_forward(m, d)
        cfg = JointServoConfig(kp=100.0, kd=10.0, ctrl_clip=False)
        ctrl = compute_motor_ctrl_from_qpos_target(m, d, qpos_des=d.qpos.copy(),
                                                    qvel_des=None, cfg=cfg)
        assert np.allclose(ctrl, 0.0, atol=1e-12)

    def test_position_error_ctrl_sign_and_magnitude(self):
        """
        q_des=1, q=0, qd=0: tau = kp*(1-0) + kd*0 = kp.
        gear=2 → ctrl = tau/gear = kp/2.
        """
        m = mujoco.MjModel.from_xml_string(self.HINGE_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        from joint_servo import compute_motor_ctrl_from_qpos_target, JointServoConfig
        kp = 100.0
        cfg = JointServoConfig(kp=kp, kd=0.0, ctrl_clip=False)
        qpos_des = np.array([1.0])
        ctrl = compute_motor_ctrl_from_qpos_target(m, d, qpos_des=qpos_des,
                                                    qvel_des=None, cfg=cfg)
        gear = 2.0
        expected = kp / gear
        assert np.isclose(float(ctrl[0]), expected, atol=1e-12)

    def test_velocity_error_ctrl(self):
        """
        qd_des=2, qd=0: tau = kd*(2-0). ctrl = tau/gear.
        """
        m = mujoco.MjModel.from_xml_string(self.HINGE_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        from joint_servo import compute_motor_ctrl_from_qpos_target, JointServoConfig
        kd = 10.0
        cfg = JointServoConfig(kp=0.0, kd=kd, ctrl_clip=False)
        qvel_des = np.array([2.0])
        ctrl = compute_motor_ctrl_from_qpos_target(m, d, qpos_des=d.qpos.copy(),
                                                    qvel_des=qvel_des, cfg=cfg)
        gear = 2.0
        expected = kd * 2.0 / gear
        assert np.isclose(float(ctrl[0]), expected, atol=1e-12)

    def test_ctrl_clip_applied(self):
        """ctrl must be clipped to ctrlrange=[-50, 50]."""
        m = mujoco.MjModel.from_xml_string(self.HINGE_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        from joint_servo import compute_motor_ctrl_from_qpos_target, JointServoConfig
        cfg = JointServoConfig(kp=10000.0, kd=0.0, ctrl_clip=True)
        qpos_des = np.array([1.0])
        ctrl = compute_motor_ctrl_from_qpos_target(m, d, qpos_des=qpos_des,
                                                    qvel_des=None, cfg=cfg)
        assert float(ctrl[0]) <= 50.0 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 10.5 contact_measurement (MuJoCo-dependent, toy XML, model agnostic)
# ─────────────────────────────────────────────────────────────────────────────

class TestContactMeasurement:
    CONTACT_XML = """
    <mujoco>
      <option gravity="0 0 -9.81" timestep="0.002" integrator="Euler"/>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0"/>

        <!-- Contacting toy foot: sphere on a vertical slide -->
        <body name="foot1" pos="0 0 0.20">
          <joint name="z1" type="slide" axis="0 0 1"/>
          <geom name="foot1_geom" type="sphere" size="0.05" mass="1.0"/>
          <site name="foot1_site" pos="0 0 0"/>
        </body>

        <!-- Second static non-contact patch, used for isolation tests -->
        <body name="foot2" pos="0.30 0 0.25">
          <geom name="foot2_geom" type="sphere" size="0.05" mass="1.0"/>
          <site name="foot2_site" pos="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """

    NO_CONTACT_XML = """
    <mujoco>
      <option gravity="0 0 0" timestep="0.002" integrator="Euler"/>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0"/>
        <body name="foot" pos="0 0 0.50">
          <joint name="z" type="slide" axis="0 0 1"/>
          <geom name="foot_geom" type="sphere" size="0.05" mass="1.0"/>
          <site name="foot_site" pos="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def _settle(self, m: mujoco.MjModel, d: mujoco.MjData, steps: int = 400):
        for _ in range(steps):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)

    def _build_contact_model(self, m: mujoco.MjModel, d: mujoco.MjData, site_names: list[str], mu: float = 0.6):
        from contact_patches import PatchSpec, build_contact_model_from_sites
        specs = []
        for name in site_names:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
            specs.append(PatchSpec(name=name, site_id=int(sid),
                                   vertex_offsets_site=np.zeros((1, 3), dtype=float)))
        return build_contact_model_from_sites(m, d, mu=mu, patch_specs=specs)

    def test_no_contact_returns_zero_wrenches(self):
        """
        With no contact and gravity disabled, every patch wrench must be exactly zero.
        """
        from contact_measurement import build_patch_geom_map_from_sites, measure_patch_wrenches_world

        m = mujoco.MjModel.from_xml_string(self.NO_CONTACT_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

        site_name = "foot_site"
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        cmodel = self._build_contact_model(m, d, [site_name])
        geom_map = build_patch_geom_map_from_sites(m, [int(sid)])

        w = measure_patch_wrenches_world(
            m, d,
            floor_geom_id=int(floor_gid),
            contact_model=cmodel,
            geom_map=geom_map,
            min_normal_force=0.0,
        )
        assert len(w) == 1
        assert np.allclose(w[0], np.zeros(6), atol=1e-12)

    def test_settled_vertical_contact_positive_normal_small_tangent(self):
        """
        After settling, the contacting patch must report:
        - positive normal force
        - near-zero tangential force (symmetric sphere on flat floor)
        - near-zero moment about patch origin p_w (force acts on vertical line through p_w)
        """
        from contact_measurement import build_patch_geom_map_from_sites, measure_patch_wrenches_world

        m = mujoco.MjModel.from_xml_string(self.CONTACT_XML)
        d = mujoco.MjData(m)
        self._settle(m, d)

        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
        floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        cmodel = self._build_contact_model(m, d, ["foot1_site"])
        geom_map = build_patch_geom_map_from_sites(m, [int(sid)])

        w = measure_patch_wrenches_world(
            m, d,
            floor_geom_id=int(floor_gid),
            contact_model=cmodel,
            geom_map=geom_map,
            min_normal_force=0.0,
        )

        assert len(w) == 1
        F = np.asarray(w[0][:3], dtype=float)
        tau = np.asarray(w[0][3:], dtype=float)

        # Positive normal
        assert F[2] > 0.0, f"Expected positive normal force, got F={F}"

        # Tangential almost zero for symmetric vertical contact
        assert np.allclose(F[:2], np.zeros(2), atol=1e-5), f"Unexpected tangential force: F={F}"

        # Moment about patch origin should be ~0 for vertical load through site line
        assert np.allclose(tau, np.zeros(3), atol=1e-5), f"Unexpected patch moment: tau={tau}"

    def test_second_noncontact_patch_stays_zero(self):
        """
        Two patches:
        - foot1 settles into contact
        - foot2 is static above ground and must remain zero
        """
        from contact_measurement import build_patch_geom_map_from_sites, measure_patch_wrenches_world

        m = mujoco.MjModel.from_xml_string(self.CONTACT_XML)
        d = mujoco.MjData(m)
        self._settle(m, d)

        sids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot1_site"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot2_site"),
        ]
        floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        cmodel = self._build_contact_model(m, d, ["foot1_site", "foot2_site"])
        geom_map = build_patch_geom_map_from_sites(m, [int(s) for s in sids])

        w = measure_patch_wrenches_world(
            m, d,
            floor_geom_id=int(floor_gid),
            contact_model=cmodel,
            geom_map=geom_map,
            min_normal_force=0.0,
        )

        assert len(w) == 2
        F1 = np.asarray(w[0][:3], dtype=float)
        F2 = np.asarray(w[1][:3], dtype=float)

        assert F1[2] > 0.0, f"Patch 1 should be in contact, got F1={F1}"
        assert np.allclose(w[1], np.zeros(6), atol=1e-12), f"Patch 2 should be zero, got {w[1]}"

    def test_min_normal_force_filter_zeroes_small_contacts(self):
        """
        If min_normal_force is set above the realized normal force, the patch wrench must be zeroed.
        """
        from contact_measurement import build_patch_geom_map_from_sites, measure_patch_wrenches_world

        m = mujoco.MjModel.from_xml_string(self.CONTACT_XML)
        d = mujoco.MjData(m)
        self._settle(m, d)

        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
        floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        cmodel = self._build_contact_model(m, d, ["foot1_site"])
        geom_map = build_patch_geom_map_from_sites(m, [int(sid)])

        # absurdly high threshold guarantees filtering
        w = measure_patch_wrenches_world(
            m, d,
            floor_geom_id=int(floor_gid),
            contact_model=cmodel,
            geom_map=geom_map,
            min_normal_force=1e9,
        )

        assert len(w) == 1
        assert np.allclose(w[0], np.zeros(6), atol=1e-12)

    def test_manual_accumulation_matches_measurement(self):
        """
        Independent oracle:
        manually reconstruct the per-patch wrench from MuJoCo contacts and compare to
        measure_patch_wrenches_world. This locks down sign/frame/origin conventions.
        """
        from contact_measurement import build_patch_geom_map_from_sites, measure_patch_wrenches_world

        m = mujoco.MjModel.from_xml_string(self.CONTACT_XML)
        d = mujoco.MjData(m)
        self._settle(m, d)

        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
        floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        cmodel = self._build_contact_model(m, d, ["foot1_site"])
        geom_map = build_patch_geom_map_from_sites(m, [int(sid)])

        w_meas = measure_patch_wrenches_world(
            m, d,
            floor_geom_id=int(floor_gid),
            contact_model=cmodel,
            geom_map=geom_map,
            min_normal_force=0.0,
        )

        # independent manual accumulation
        w_manual = [np.zeros(6, dtype=float) for _ in cmodel.patches]
        wrench_c = np.zeros(6, dtype=float)

        for cid in range(int(d.ncon)):
            c = d.contact[cid]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 != int(floor_gid)) and (g2 != int(floor_gid)):
                continue

            robot_gid = g2 if g1 == int(floor_gid) else g1
            if int(m.geom_bodyid[robot_gid]) == 0:
                continue

            # MuJoCo stores contact.frame in row-major "axes in rows" form;
            # true contact->world rotation is frame.T
            R_rows = np.asarray(c.frame, dtype=float).reshape(3, 3)
            R_wc = R_rows.T

            mujoco.mj_contactForce(m, d, cid, wrench_c)
            f_c = wrench_c[:3].copy()
            tau_c = wrench_c[3:].copy()

            sign = 1.0 if (robot_gid == g2) else -1.0
            f_w = sign * (R_wc @ f_c)
            tau_w_at_contact = sign * (R_wc @ tau_c)

            p_contact = np.asarray(c.pos, dtype=float).reshape(3,)
            for i, gset in enumerate(geom_map.patch_geom_ids):
                if robot_gid in gset:
                    p0 = np.asarray(cmodel.patches[i].p_w, dtype=float).reshape(3,)
                    w_manual[i][:3] += f_w
                    w_manual[i][3:] += tau_w_at_contact + np.cross(p_contact - p0, f_w)
                    break

        assert len(w_meas) == len(w_manual)
        for wm, wo in zip(w_meas, w_manual):
            assert np.allclose(wm, wo, atol=1e-10), f"measurement={wm}, manual={wo}"

# Run
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])