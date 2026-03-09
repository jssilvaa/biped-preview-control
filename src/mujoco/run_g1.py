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


def _plot_com_tracking(out: dict, dt: float, save_dir: Path) -> None:
    t = _time_axis(out, dt)
    com_preview = out["com_preview_log"]
    com_meas = out["com_meas_log"]
    com_des = out["com_des_log"]
    com_ref = out["com_ref_cmd_log"]

    labels = ["x", "y", "z"]
    for i, lab in enumerate(labels):
        plt.figure(figsize=(10, 5))
        plt.plot(t, com_preview[:, i], label=f"preview {lab}")
        plt.plot(t, com_ref[:, i], label=f"commanded {lab}")
        plt.plot(t, com_des[:, i], label=f"internal desired {lab}")
        plt.plot(t, com_meas[:, i], label=f"measured {lab}")
        plt.xlabel("time [s]")
        plt.ylabel("CoM position [m]")
        plt.title(f"CoM tracking ({lab})")
        plt.legend()
        _savefig(save_dir, f"com_tracking_{lab}")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(t, com_ref[:, i] - com_preview[:, i], label=f"commanded - preview ({lab})")
        plt.plot(t, com_ref[:, i] - com_meas[:, i], label=f"commanded - measured ({lab})")
        plt.plot(t, com_preview[:, i] - com_des[:, i], label=f"preview - internal_desired ({lab})")
        plt.plot(t, com_des[:, i] - com_meas[:, i], label=f"internal desired - measured ({lab})")
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
    com_meas = out["com_meas_log"]
    com_des = out["com_des_log"]
    com_ref = out["com_ref_cmd_log"]

    fn = out["fn_log"]
    dr = out["dr_log"]
    w_err = out["w_err_norm_log"]
    w_f_err = out["w_force_err_norm_log"]
    w_m_err = out["w_moment_err_norm_log"]

    err_cmd = com_ref - com_meas
    err_des = com_des - com_meas

    com_cmd_rms = np.sqrt(np.mean(err_cmd**2, axis=0))
    com_cmd_max = np.max(np.abs(err_cmd), axis=0)

    com_des_rms = np.sqrt(np.mean(err_des**2, axis=0))
    com_des_max = np.max(np.abs(err_des), axis=0)

    dpos_max = np.max(np.linalg.norm(dr[:, :, :3], axis=2), axis=0)
    dang_max = np.max(np.linalg.norm(dr[:, :, 3:], axis=2), axis=0)

    print("\n=== RUN SUMMARY ===")
    print(f"duration [s]: {out['q_log'].shape[0] * dt:.3f}")
    print(f"CoM commanded-vs-measured RMS [x y z] [m]: {com_cmd_rms}")
    print(f"CoM commanded-vs-measured max abs [x y z] [m]: {com_cmd_max}")
    print(f"CoM internal-desired-vs-measured RMS [x y z] [m]: {com_des_rms}")
    print(f"CoM internal-desired-vs-measured max abs [x y z] [m]: {com_des_max}")
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
        floor_geom_name=FLOOR_GEOM_NAME,
        site_names=SITE_NAMES,
        site_vertex_offsets=site_vertex_offsets,
        base_body_id=BASE_BODY_ID,
        I_diag=model.body_inertia[BASE_BODY_ID],
        viz=True,
        display_every=4,
    )

    out = run_simulation(model, data, cfg=cfg)
    make_plots(out, dt=cfg.dt, save_dir="plots_g1")
    print("done", out["q_log"].shape[0])


if __name__ == "__main__":
    _main()