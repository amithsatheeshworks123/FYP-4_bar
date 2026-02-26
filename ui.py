"""
Minimal Matplotlib UI for four-bar path generation with PPO, CMA-ES, and CEM.
Run:
    python -m v2.ui
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons
from matplotlib.animation import FuncAnimation

from v2.path_kinematics import THETA2, solve_rocker_angle, path_reward
from v2.optimizer import optimize  # CMA-ES style
from v2.cem import cem
from v2.ppo import ppo_train, load_checkpoint

# Bounds and state
MIN_LEN, MAX_LEN = 0.05, 0.4
BOUNDS = [(MIN_LEN, MAX_LEN)] * 4 + [(-0.1, 0.2), (-0.1, 0.2)]
params = np.array([0.1, 0.07, 0.12, 0.09, 0.0, 0.0], dtype=float)  # [L1, L2, L3, L4, xO2, yO2]
lock_mask = np.zeros(6, dtype=bool)
target_line = (0.0, 0.0)  # initial y=0
seed = None
PPO_CKPT = "v2/ppo_checkpoint.pt"

# UI
fig = plt.figure(figsize=(12, 7))
ax_link = fig.add_axes([0.05, 0.35, 0.45, 0.6])
ax_link.set_aspect("equal")
ax_link.set_xlim(-0.5, 0.8)
ax_link.set_ylim(-0.5, 0.5)
ax_link.set_title("Four-Bar Path")
link_line, = ax_link.plot([], [], "o-", lw=2)
coupler_plot, = ax_link.plot([], [], "r.", alpha=0.6, markersize=2)
target_plot, = ax_link.plot([], [], "k--", lw=1)

ax_ang = fig.add_axes([0.55, 0.55, 0.4, 0.35])
ax_ang.set_title("Rocker vs Crank")
ax_ang.set_xlabel("Crank (rad)")
ax_ang.set_ylabel("Rocker (deg)")
ang_line, = ax_ang.plot([], [], lw=1.5)
ang_text = ax_ang.text(0.02, 0.95, "", transform=ax_ang.transAxes, va="top", fontsize=8)

ax_locks = fig.add_axes([0.55, 0.35, 0.2, 0.15], facecolor="#eef1f7")
lock_labels = ["Lock L1", "Lock L2", "Lock L3", "Lock L4", "Lock xO2", "Lock yO2"]
lock_checks = CheckButtons(ax_locks, lock_labels, lock_mask)

text_boxes = []
# Order matches params: L1, L2, L3, L4, xO2, yO2
labels = ["L1", "L2", "L3", "L4", "xO2", "yO2"]
starts = [
    (0.05, 0.25),  # L1
    (0.16, 0.25),  # L2
    (0.27, 0.25),  # L3
    (0.38, 0.25),  # L4
    (0.05, 0.18),  # xO2
    (0.16, 0.18),  # yO2
]
for i, (x, y) in enumerate(starts):
    tb_ax = fig.add_axes([x, y, 0.08, 0.05])
    tb = TextBox(tb_ax, labels[i], initial=f"{params[i]:.3f}")
    text_boxes.append(tb)

# If a PPO checkpoint exists, note its best params (applied after helpers are defined)
best_params_tmp = None
try:
    _, _, best_params_tmp = load_checkpoint(PPO_CKPT, BOUNDS, device="cpu")
except Exception:
    best_params_tmp = None

ax_target = fig.add_axes([0.55, 0.25, 0.15, 0.05])
tb_target = TextBox(ax_target, "Target line y=c", initial="0.0")
ax_seed = fig.add_axes([0.72, 0.25, 0.1, 0.05])
tb_seed = TextBox(ax_seed, "Seed", initial="random")

ax_run_cma = fig.add_axes([0.55, 0.15, 0.12, 0.06])
btn_cma = Button(ax_run_cma, "Run CMA-ES", color="#6bbf92", hovercolor="#5aa17c")
ax_run_cem = fig.add_axes([0.69, 0.15, 0.1, 0.06])
btn_cem = Button(ax_run_cem, "Run CEM", color="#6bbf92", hovercolor="#5aa17c")
ax_run_ppo = fig.add_axes([0.81, 0.15, 0.1, 0.06])
btn_ppo = Button(ax_run_ppo, "Run PPO", color="#6bbf92", hovercolor="#5aa17c")
ax_reset = fig.add_axes([0.55, 0.07, 0.15, 0.05])
btn_reset = Button(ax_reset, "Reset", color="#b0b0b0", hovercolor="#9a9a9a")

status_text = fig.text(0.05, 0.05, "", fontsize=9)


def project(p):
    arr = np.array(p, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, ...]
    low = np.array([b[0] for b in BOUNDS])
    high = np.array([b[1] for b in BOUNDS])
    out = []
    for row in arr:
        row = row.copy()
        for i, locked in enumerate(lock_mask):
            if locked:
                row[i] = params[i]
        row = np.clip(row, low, high)
        Ls = row[:4]
        s_idx = Ls.argmin()
        l_idx = Ls.argmax()
        if s_idx != 1:
            Ls[[1, s_idx]] = Ls[[s_idx, 1]]
        if l_idx != 0:
            Ls[[0, l_idx]] = Ls[[l_idx, 0]]
        row[:4] = Ls
        out.append(row)
    out = np.vstack(out)
    return out[0] if single else out


def update_from_boxes():
    vals = []
    for tb in text_boxes:
        try:
            vals.append(float(tb.text))
        except ValueError:
            vals.append(MIN_LEN)
    return project(vals)

# Apply checkpoint params after project is defined
if best_params_tmp is not None:
    params[:] = project(best_params_tmp)
    for i, tb in enumerate(text_boxes):
        tb.set_val(f"{params[i]:.3f}")


def update_target_seed():
    global target_line, seed
    try:
        c = float(tb_target.text)
    except ValueError:
        c = 0.0
    target_line = (0.0, c)
    txt = tb_seed.text.strip().lower()
    seed = None if txt in ("", "none", "random") else int(float(txt))


def draw_linkage(frame_idx=0):
    t2 = THETA2[frame_idx % len(THETA2)]
    th4 = solve_rocker_angle(t2, params)
    if th4 is None:
        link_line.set_data([], [])
        return
    O1 = np.array([0.0, 0.0])
    O2 = np.array([params[4], params[5]])
    O4 = np.array([params[0], 0.0])
    B = O2 + np.array([params[1] * np.cos(t2), params[1] * np.sin(t2)])
    C = O4 + np.array([params[3] * np.cos(th4), params[3] * np.sin(th4)])
    xs = [O1[0], O2[0], B[0], C[0], O4[0]]
    ys = [O1[1], O2[1], B[1], C[1], O4[1]]
    link_line.set_data(xs, ys)


def draw_path():
    # Ensure params respect locks/constraints before evaluating
    projected = project(params)
    r, _, pts, mse, max_dev = path_reward(projected, target_line=target_line, enforce_grashof="crank_rocker")
    if len(pts) == 0:
        coupler_plot.set_data([], [])
    else:
        coupler_plot.set_data(pts[:, 0], pts[:, 1])
    x_ref = np.linspace(-0.2, 0.5, 50)
    y_ref = target_line[0] * x_ref + target_line[1]
    target_plot.set_data(x_ref, y_ref)
    th4 = []
    for t2 in THETA2:
        val = solve_rocker_angle(t2, projected)
        th4.append(np.nan if val is None else val * 180.0 / np.pi)
    ang_line.set_data(THETA2, th4)
    if len(th4) and not np.isnan(th4).all():
        rom = np.nanmax(th4) - np.nanmin(th4)
        ang_text.set_text(f"ROM: {rom:.2f} deg")
    else:
        ang_text.set_text("ROM: --")
    ax_ang.relim()
    ax_ang.autoscale_view()
    status_text.set_text(f"Reward: {r:.3f} | MSE: {mse:.4f} | max_dev: {max_dev:.4f}")


def anim_update(i):
    draw_linkage(i)
    return link_line,


def on_lock(_):
    for i, chk in enumerate(lock_checks.get_status()):
        lock_mask[i] = chk


def run_cma(event):
    del event
    global params
    params[:] = update_from_boxes()
    update_target_seed()
    status_text.set_text("Running CMA-ES...")
    fig.canvas.draw_idle()
    eval_log = []
    def obj(x):
        px = project(x)
        r_list = []
        if px.ndim == 1:
            r, _, _, mse, max_dev = path_reward(px, target_line=target_line, enforce_grashof="crank_rocker")
            r_list = [r]
            eval_log.append((r, mse, max_dev, px))
        else:
            for row in px:
                r, _, _, mse, max_dev = path_reward(row, target_line=target_line, enforce_grashof="crank_rocker")
                r_list.append(r)
                eval_log.append((r, mse, max_dev, row))
        return np.array(r_list)
    best_r, best_p, hist, _ = optimize(obj, params, BOUNDS, iterations=60, population=120, restarts=3, seed=seed)
    # write log
    with open("v2_attempts_cma.csv", "w") as f:
        for r, mse, max_dev, p in eval_log:
            f.write(f"{r},{mse},{max_dev}," + ",".join(f"{v:.6f}" for v in p) + "\n")
    params[:] = project(best_p)
    for i, tb in enumerate(text_boxes):
        tb.set_val(f"{params[i]:.3f}")
    draw_path()
    status_text.set_text(f"CMA-ES best: {best_r:.3f}")
    r_chk, _, _, mse_chk, dev_chk = path_reward(params, target_line=target_line, enforce_grashof="crank_rocker")
    print("CMA-ES best reward:", best_r, "params:", np.round(params, 4), "mse:", mse_chk, "max_dev:", dev_chk)


def run_cem(event):
    del event
    global params
    params[:] = update_from_boxes()
    update_target_seed()
    status_text.set_text("Running CEM...")
    fig.canvas.draw_idle()
    eval_log = []
    def obj(x):
        px = project(x)
        # CEM calls objective per-sample, so return a scalar
        if px.ndim == 1:
            r, _, _, mse, max_dev = path_reward(px, target_line=target_line, enforce_grashof="crank_rocker")
            eval_log.append((r, mse, max_dev, px))
            return float(r)
        # Fallback: batch, return vector
        r_list = []
        for row in px:
            r, _, _, mse, max_dev = path_reward(row, target_line=target_line, enforce_grashof="crank_rocker")
            r_list.append(r)
            eval_log.append((r, mse, max_dev, row))
        return np.array(r_list)
    best_r, best_p, hist = cem(obj, params, BOUNDS, iterations=80, pop=200, elite_frac=0.2, sigma=0.1, seed=seed)
    with open("v2_attempts_cem.csv", "w") as f:
        for r, mse, max_dev, p in eval_log:
            f.write(f"{r},{mse},{max_dev}," + ",".join(f"{v:.6f}" for v in p) + "\n")
    params[:] = project(best_p)
    for i, tb in enumerate(text_boxes):
        tb.set_val(f"{params[i]:.3f}")
    draw_path()
    status_text.set_text(f"CEM best: {best_r:.3f}")
    r_chk, _, _, mse_chk, dev_chk = path_reward(params, target_line=target_line, enforce_grashof="crank_rocker")
    print("CEM best reward:", best_r, "params:", np.round(params, 4), "mse:", mse_chk, "max_dev:", dev_chk)


def run_ppo(event):
    del event
    global params
    params[:] = update_from_boxes()
    update_target_seed()
    status_text.set_text("Running PPO...")
    fig.canvas.draw_idle()
    def obj(x):
        r, _, _, mse, max_dev = path_reward(project(x), target_line=target_line, enforce_grashof="crank_rocker")
        return r
    best_r, best_p, hist, samples_log = ppo_train(
        obj,
        BOUNDS,
        steps=5000,
        batch_size=128,
        lr=5e-4,
        seed=seed,
        log_samples=True,
        checkpoint_path=PPO_CKPT,
    )
    if samples_log is not None:
        with open("v2_attempts_ppo.csv", "w") as f:
            for step, r, p in samples_log:
                f.write(f"{step},{r}," + ",".join(f"{v:.6f}" for v in p) + "\n")
    params[:] = project(best_p)
    for i, tb in enumerate(text_boxes):
        tb.set_val(f"{params[i]:.3f}")
    draw_path()
    status_text.set_text(f"PPO best: {best_r:.3f}")
    r_chk, _, _, mse_chk, dev_chk = path_reward(params, target_line=target_line, enforce_grashof="crank_rocker")
    print("PPO best reward:", best_r, "params:", np.round(params, 4), "mse:", mse_chk, "max_dev:", dev_chk)


def on_reset(event):
    del event
    global params, target_line, seed
    params[:] = np.array([0.1, 0.07, 0.12, 0.09, 0.0, 0.0])
    target_line = (0.0, 0.0)
    seed = None
    for i, tb in enumerate(text_boxes):
        tb.set_val(f"{params[i]:.3f}")
    tb_target.set_val("0.0")
    tb_seed.set_val("random")
    for i, chk in enumerate(lock_checks.get_status()):
        if chk:
            lock_checks.set_active(i)
    draw_path()
    status_text.set_text("Reset.")


for tb in text_boxes:
    tb.on_submit(lambda _: (params.__setitem__(slice(None), update_from_boxes()), draw_path()))
tb_target.on_submit(lambda _: update_target_seed())
tb_seed.on_submit(lambda _: update_target_seed())
lock_checks.on_clicked(on_lock)
btn_cma.on_clicked(run_cma)
btn_cem.on_clicked(run_cem)
btn_ppo.on_clicked(run_ppo)
btn_reset.on_clicked(on_reset)

draw_path()
ani = FuncAnimation(fig, anim_update, interval=30, blit=True)

if __name__ == "__main__":
    plt.show()
