from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from sim import run_simulation, MurookaSimConfig


STAND_KEY = 0
BASE_BODY_ID = 1
FLOOR_GEOM_NAME = "floor"
SITE_NAMES = ["left_foot", "right_foot"]

FOOT_VERTS = np.array([
    [ 0.07,  0.035, -0.03],
    [ 0.07, -0.035, -0.03],
    [-0.07,  0.035, -0.03],
    [-0.07, -0.035, -0.03],
], dtype=float)


def _time_axis(out: dict, dt: float) -> np.ndarray:
    n = out["q_log"].shape[0]
    return np.arange(n, dtype=float) * dt


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _savefig(save_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(save_dir / f"{name}.png", dpi=180, bbox_inches="tight")


def _rms(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=axis))


def _tail_slice(n: int, fraction: float = 0.25) -> slice:
    keep = max(1, int(np.ceil(float(fraction) * n)))
    return slice(n - keep, n)


def _skip_prefix_slice(n: int, skip_seconds: float, dt: float) -> slice:
    start = int(np.clip(np.ceil(skip_seconds / dt), 0, max(0, n - 1)))
    return slice(start, n)


def _axis_amplitude(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return 0.5 * float(np.max(x) - np.min(x))


def _dominant_phase_lag(ref: np.ndarray, sig: np.ndarray, dt: float) -> tuple[float, float] | None:
    ref = np.asarray(ref, dtype=float).reshape(-1)
    sig = np.asarray(sig, dtype=float).reshape(-1)
    if ref.size != sig.size or ref.size < 8:
        return None

    ref0 = ref - np.mean(ref)
    sig0 = sig - np.mean(sig)
    if np.allclose(ref0, 0.0) or np.allclose(sig0, 0.0):
        return None

    ref_fft = np.fft.rfft(ref0)
    sig_fft = np.fft.rfft(sig0)
    freqs = np.fft.rfftfreq(ref0.size, d=dt)
    if freqs.size <= 1:
        return None

    idx = 1 + int(np.argmax(np.abs(ref_fft[1:])))
    freq_hz = float(freqs[idx])
    if freq_hz <= 0.0:
        return None

    phase_ref = float(np.angle(ref_fft[idx]))
    phase_sig = float(np.angle(sig_fft[idx]))
    phase_lag = phase_ref - phase_sig
    phase_lag = (phase_lag + np.pi) % (2.0 * np.pi) - np.pi
    time_lag = phase_lag / (2.0 * np.pi * freq_hz)
    return freq_hz, time_lag


def _print_error_block(name: str, err: np.ndarray) -> None:
    rms = _rms(err, axis=0)
    max_abs = np.max(np.abs(err), axis=0)
    mean = np.mean(err, axis=0)
    print(f"{name} RMS [x y z] [m]: {rms}")
    print(f"{name} max abs [x y z] [m]: {max_abs}")
    print(f"{name} mean [x y z] [m]: {mean}")


def _print_windowed_com_metrics(com_ref: np.ndarray, com_prev: np.ndarray, com_des: np.ndarray, com_meas: np.ndarray, dt: float) -> None:
    n = com_meas.shape[0]
    tail = _tail_slice(n, fraction=0.25)
    settled = _skip_prefix_slice(n, skip_seconds=min(1.0, max(0.0, (n - 1) * dt)), dt=dt)

    print("\n--- CoM Stage Errors: Full Run ---")
    _print_error_block("CoM reference-vs-preview", com_ref - com_prev)
    _print_error_block("CoM preview-vs-desired", com_prev - com_des)
    _print_error_block("CoM desired-vs-measured", com_des - com_meas)
    _print_error_block("CoM reference-vs-measured", com_ref - com_meas)

    print("\n--- CoM Stage Errors: Last 25% of Run ---")
    _print_error_block("CoM reference-vs-preview", com_ref[tail] - com_prev[tail])
    _print_error_block("CoM preview-vs-desired", com_prev[tail] - com_des[tail])
    _print_error_block("CoM desired-vs-measured", com_des[tail] - com_meas[tail])
    _print_error_block("CoM reference-vs-measured", com_ref[tail] - com_meas[tail])

    print(f"\n--- CoM Stage Errors: After {settled.start * dt:.3f}s ---")
    _print_error_block("CoM reference-vs-preview", com_ref[settled] - com_prev[settled])
    _print_error_block("CoM preview-vs-desired", com_prev[settled] - com_des[settled])
    _print_error_block("CoM desired-vs-measured", com_des[settled] - com_meas[settled])
    _print_error_block("CoM reference-vs-measured", com_ref[settled] - com_meas[settled])


def _print_static_tail_metrics(com_ref: np.ndarray, com_prev: np.ndarray, com_des: np.ndarray, com_meas: np.ndarray, dt: float) -> None:
    n = com_meas.shape[0]
    tail = _tail_slice(n, fraction=0.25)
    print("\n--- Static / Near-Steady Diagnostics: Last 25% of Run ---")
    _print_error_block("CoM reference-vs-measured", com_ref[tail] - com_meas[tail])
    _print_error_block("CoM preview-vs-measured", com_prev[tail] - com_meas[tail])
    _print_error_block("CoM desired-vs-measured", com_des[tail] - com_meas[tail])


def _print_sinusoid_metrics(com_ref: np.ndarray, com_prev: np.ndarray, com_des: np.ndarray, com_meas: np.ndarray, dt: float) -> None:
    print("\n--- Sinusoid Diagnostics ---")
    labels = ["x", "y", "z"]
    for i, lab in enumerate(labels):
        ref_axis = com_ref[:, i]
        prev_axis = com_prev[:, i]
        des_axis = com_des[:, i]
        meas_axis = com_meas[:, i]

        amp_ref = _axis_amplitude(ref_axis)
        if amp_ref < 1e-6:
            continue

        amp_prev = _axis_amplitude(prev_axis)
        amp_des = _axis_amplitude(des_axis)
        amp_meas = _axis_amplitude(meas_axis)

        print(f"axis {lab}: ref amplitude [m] = {amp_ref:.6f}")
        print(f"axis {lab}: preview/reference amplitude ratio = {amp_prev / amp_ref:.6f}")
        print(f"axis {lab}: desired/reference amplitude ratio = {amp_des / amp_ref:.6f}")
        print(f"axis {lab}: measured/reference amplitude ratio = {amp_meas / amp_ref:.6f}")
        print(f"axis {lab}: reference-vs-measured mean offset [m] = {np.mean(ref_axis - meas_axis):.6e}")

        for pair_name, lhs, rhs in (
            ("reference -> preview", ref_axis, prev_axis),
            ("reference -> desired", ref_axis, des_axis),
            ("reference -> measured", ref_axis, meas_axis),
            ("preview -> measured", prev_axis, meas_axis),
        ):
            lag = _dominant_phase_lag(lhs, rhs, dt)
            if lag is None:
                continue
            freq_hz, time_lag = lag
            print(f"axis {lab}: {pair_name} dominant freq [Hz] = {freq_hz:.6f}, lag [s] = {time_lag:.6f}")


def _plot_com_tracking(out: dict, dt: float, save_dir: Path) -> None:
    t = _time_axis(out, dt)
    com_preview = out["com_preview_log"]
    com_meas = out["com_meas_log"]
    com_des = out["com_des_log"]
    com_ref = out["com_ref_cmd_log"]

    labels = ["x", "y", "z"]
    for i, lab in enumerate(labels):
        plt.figure(figsize=(10, 5))
        plt.plot(t, com_ref[:, i], label=f"reference {lab}")
        plt.plot(t, com_preview[:, i], label=f"preview {lab}")
        plt.plot(t, com_des[:, i], label=f"desired {lab}")
        plt.plot(t, com_meas[:, i], label=f"measured {lab}")
        plt.xlabel("time [s]")
        plt.ylabel("CoM position [m]")
        plt.title(f"CoM tracking ({lab})")
        plt.legend()
        _savefig(save_dir, f"com_tracking_{lab}")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(t, com_ref[:, i] - com_preview[:, i], label=f"reference - preview ({lab})")
        plt.plot(t, com_ref[:, i] - com_meas[:, i], label=f"reference - measured ({lab})")
        plt.plot(t, com_preview[:, i] - com_des[:, i], label=f"preview - desired ({lab})")
        plt.plot(t, com_des[:, i] - com_meas[:, i], label=f"desired - measured ({lab})")
        plt.xlabel("time [s]")
        plt.ylabel("error [m]")
        plt.title(f"CoM tracking errors ({lab})")
        plt.legend()
        _savefig(save_dir, f"com_error_{lab}")
        plt.close()


def _plot_patch_activity(out: dict, dt: float, save_dir: Path) -> None:
    t = _time_axis(out, dt)
    active = out["patch_active_log"].astype(float)
    fn = out["fn_log"]
    nc = active.shape[1]

    plt.figure(figsize=(10, 5))
    for i in range(nc):
        plt.plot(t, active[:, i], label=f"patch {i}")
    plt.xlabel("time [s]")
    plt.ylabel("active")
    plt.title("Patch activity")
    plt.legend()
    _savefig(save_dir, "patch_activity")
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(nc):
        plt.plot(t, fn[:, i], label=f"patch {i}")
    plt.xlabel("time [s]")
    plt.ylabel("normal force [N]")
    plt.title("Patch normal force")
    plt.legend()
    _savefig(save_dir, "patch_normal_force")
    plt.close()


def _plot_patch_wrenches(out: dict, dt: float, save_dir: Path) -> None:
    t = _time_axis(out, dt)
    w_des = out["w_des_log"]
    w_meas = out["w_meas_log"]
    nc = w_des.shape[1]

    for i in range(nc):
        plt.figure(figsize=(10, 5))
        plt.plot(t, w_des[:, i, 2], label="desired fz")
        plt.plot(t, w_meas[:, i, 2], label="measured fz")
        plt.xlabel("time [s]")
        plt.ylabel("force [N]")
        plt.title(f"Patch {i} vertical force")
        plt.legend()
        _savefig(save_dir, f"patch_{i}_fz")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(t, w_des[:, i, 3], label="desired mx")
        plt.plot(t, w_meas[:, i, 3], label="measured mx")
        plt.plot(t, w_des[:, i, 4], label="desired my")
        plt.plot(t, w_meas[:, i, 4], label="measured my")
        plt.plot(t, w_des[:, i, 5], label="desired mz")
        plt.plot(t, w_meas[:, i, 5], label="measured mz")
        plt.xlabel("time [s]")
        plt.ylabel("moment [N·m]")
        plt.title(f"Patch {i} moments about patch origin")
        plt.legend(ncol=2)
        _savefig(save_dir, f"patch_{i}_moments")
        plt.close()

        ferr = np.linalg.norm(w_meas[:, i, :3] - w_des[:, i, :3], axis=1)
        terr = np.linalg.norm(w_meas[:, i, 3:] - w_des[:, i, 3:], axis=1)

        plt.figure(figsize=(10, 5))
        plt.plot(t, ferr, label="||force error||")
        plt.plot(t, terr, label="||moment error||")
        plt.xlabel("time [s]")
        plt.ylabel("error norm")
        plt.title(f"Patch {i} wrench tracking error")
        plt.legend()
        _savefig(save_dir, f"patch_{i}_wrench_error")
        plt.close()


def _plot_compliance(out: dict, dt: float, save_dir: Path) -> None:
    t = _time_axis(out, dt)
    dr = out["dr_log"]
    nc = dr.shape[1]

    for i in range(nc):
        dpos = np.linalg.norm(dr[:, i, :3], axis=1)
        dang = np.linalg.norm(dr[:, i, 3:], axis=1)

        plt.figure(figsize=(10, 5))
        plt.plot(t, dpos, label="||Δp||")
        plt.plot(t, dang, label="||Δφ||")
        plt.xlabel("time [s]")
        plt.ylabel("norm")
        plt.title(f"Patch {i} compliance state")
        plt.legend()
        _savefig(save_dir, f"patch_{i}_compliance_norm")
        plt.close()


def _plot_centroidal_pipeline(out: dict, dt: float, save_dir: Path) -> None:
    t = _time_axis(out, dt)

    bar_wp = out["bar_wp_log"]
    bar_wp_proj = out["bar_wp_proj_log"]
    bar_wd = out["bar_wd_log"]
    bar_f_ref = out["bar_f_ref_cmd_log"]
    bar_n_ref = out["bar_n_ref_cmd_log"]

    w_cmd = out["w_cmd_log"]
    w_real = out["w_real_log"]
    w_err = out["w_err_norm_log"]
    w_f_err = out["w_force_err_norm_log"]
    w_m_err = out["w_moment_err_norm_log"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, bar_f_ref[:, 0], label="bar_f_ref x")
    plt.plot(t, bar_wp[:, 0], label="bar_wp x")
    plt.plot(t, bar_wp_proj[:, 0], label="bar_wp_proj x")
    plt.plot(t, bar_wd[:, 0], label="bar_wd x")
    plt.xlabel("time [s]")
    plt.ylabel("force [N]")
    plt.title("Centroidal pipeline: x bar force")
    plt.legend()
    _savefig(save_dir, "bar_force_x")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, bar_n_ref[:, 1], label="bar_n_ref y")
    plt.plot(t, bar_wp[:, 4], label="bar_wp my")
    plt.plot(t, bar_wp_proj[:, 4], label="bar_wp_proj my")
    plt.plot(t, bar_wd[:, 4], label="bar_wd my")
    plt.xlabel("time [s]")
    plt.ylabel("moment [N·m]")
    plt.title("Centroidal pipeline: y bar moment")
    plt.legend()
    _savefig(save_dir, "bar_moment_y")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, np.linalg.norm(w_cmd[:, :3], axis=1), label="||cmd force||")
    plt.plot(t, np.linalg.norm(w_real[:, :3], axis=1), label="||real force||")
    plt.xlabel("time [s]")
    plt.ylabel("norm")
    plt.title("Resultant force norm")
    plt.legend()
    _savefig(save_dir, "wrench_force_norm")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, np.linalg.norm(w_cmd[:, 3:], axis=1), label="||cmd moment||")
    plt.plot(t, np.linalg.norm(w_real[:, 3:], axis=1), label="||real moment||")
    plt.xlabel("time [s]")
    plt.ylabel("norm")
    plt.title("Resultant moment norm")
    plt.legend()
    _savefig(save_dir, "wrench_moment_norm")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, w_f_err, label="||force error||")
    plt.plot(t, w_m_err, label="||moment error||")
    plt.xlabel("time [s]")
    plt.ylabel("error norm")
    plt.title("Resultant wrench error split")
    plt.legend()
    _savefig(save_dir, "wrench_error_split")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, w_err, label="||w_real - w_cmd||")
    plt.xlabel("time [s]")
    plt.ylabel("error norm")
    plt.title("Resultant wrench total error")
    plt.legend()
    _savefig(save_dir, "wrench_error_norm")
    plt.close()


def _print_summary(out: dict, dt: float) -> None:
    com_ref = out["com_ref_cmd_log"]
    com_prev = out["com_preview_log"]
    com_des = out["com_des_log"]
    com_meas = out["com_meas_log"]

    fn = out["fn_log"]
    dr = out["dr_log"]
    w_err = out["w_err_norm_log"]
    w_f_err = out["w_force_err_norm_log"]
    w_m_err = out["w_moment_err_norm_log"]

    err_ref = com_ref - com_meas
    err_prev = com_prev - com_meas
    err_des = com_des - com_meas

    com_ref_rms = np.sqrt(np.mean(err_ref**2, axis=0))
    com_ref_max = np.max(np.abs(err_ref), axis=0)

    com_prev_rms = np.sqrt(np.mean(err_prev**2, axis=0))
    com_prev_max = np.max(np.abs(err_prev), axis=0)

    com_des_rms = np.sqrt(np.mean(err_des**2, axis=0))
    com_des_max = np.max(np.abs(err_des), axis=0)

    dpos_max = np.max(np.linalg.norm(dr[:, :, :3], axis=2), axis=0)
    dang_max = np.max(np.linalg.norm(dr[:, :, 3:], axis=2), axis=0)

    print("\n=== RUN SUMMARY ===")
    print(f"duration [s]: {out['q_log'].shape[0] * dt:.3f}")
    print(f"CoM reference-vs-measured RMS [x y z] [m]: {com_ref_rms}")
    print(f"CoM reference-vs-measured max abs [x y z] [m]: {com_ref_max}")
    print(f"CoM preview-vs-measured RMS [x y z] [m]: {com_prev_rms}")
    print(f"CoM preview-vs-measured max abs [x y z] [m]: {com_prev_max}")
    print(f"CoM desired-vs-measured RMS [x y z] [m]: {com_des_rms}")
    print(f"CoM desired-vs-measured max abs [x y z] [m]: {com_des_max}")
    print(f"Mean normal force per patch [N]: {np.mean(fn, axis=0)}")
    print(f"Min normal force per patch [N]:  {np.min(fn, axis=0)}")
    print(f"Max compliance ||Δp|| per patch [m]: {dpos_max}")
    print(f"Max compliance ||Δφ|| per patch [rad]: {dang_max}")
    print(f"Mean resultant force error:  {np.nanmean(w_f_err):.6f}")
    print(f"Max  resultant force error:  {np.nanmax(w_f_err):.6f}")
    print(f"Mean resultant moment error: {np.nanmean(w_m_err):.6f}")
    print(f"Max  resultant moment error: {np.nanmax(w_m_err):.6f}")
    print(f"Mean resultant total error:  {np.nanmean(w_err):.6f}")
    print(f"Max  resultant total error:  {np.nanmax(w_err):.6f}")

    _print_windowed_com_metrics(com_ref, com_prev, com_des, com_meas, dt)

    if _axis_amplitude(com_ref[:, 0]) < 1e-6 and _axis_amplitude(com_ref[:, 1]) < 1e-6 and _axis_amplitude(com_ref[:, 2]) < 1e-6:
        _print_static_tail_metrics(com_ref, com_prev, com_des, com_meas, dt)
    else:
        _print_sinusoid_metrics(com_ref, com_prev, com_des, com_meas, dt)


def make_plots(out: dict, dt: float, save_dir: str | Path = "plots_g1") -> None:
    save_dir = _ensure_dir(save_dir)
    _plot_com_tracking(out, dt, save_dir)
    _plot_patch_activity(out, dt, save_dir)
    _plot_patch_wrenches(out, dt, save_dir)
    _plot_compliance(out, dt, save_dir)
    _plot_centroidal_pipeline(out, dt, save_dir)
    _print_summary(out, dt)
    print(f"\nSaved plots to: {save_dir.resolve()}")


def _main():
    model = mujoco.MjModel.from_xml_path("models/unitree_g1/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, STAND_KEY)

    site_vertex_offsets = {
        "left_foot": FOOT_VERTS,
        "right_foot": FOOT_VERTS,
    }

    cfg = MurookaSimConfig(
        dt=1e-3,
        N=4_000,
        mu=0.6,
        enable_motion_refs=True,
        motion_axis=0,
        motion_amp=.05,
        motion_freq_hz=0.5,
        preview_state_source="measured",
        preview_blend_alpha=0.05,
        reference_advance_steps=60,
        floor_geom_name=FLOOR_GEOM_NAME,
        site_names=SITE_NAMES,
        site_vertex_offsets=site_vertex_offsets,
        base_body_id=BASE_BODY_ID,
        I_diag=model.body_inertia[BASE_BODY_ID],
        viz=False,
        display_every=4,
    )

    out = run_simulation(model, data, cfg=cfg)
    make_plots(out, dt=cfg.dt, save_dir="plots_g1")
    print("done", out["q_log"].shape[0])


if __name__ == "__main__":
    _main()