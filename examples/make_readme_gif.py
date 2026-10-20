"""Render assets/arcod_demo.gif: a streamed flutter-approach signal and the
arCod precursor raising an alarm before flutter onset.

Usage (from the repo root):
    python examples/make_readme_gif.py
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "examples")]

from demo_signal import flutter_approach_signal  # noqa: E402
from flutter_precursor import flutterFeatureExtractor  # noqa: E402

SEED = 0
FS = 100
WINDOW = 128
CALIBRATION_SECONDS = 15
ALARM_HOLD = 1.0  # arCod must stay above threshold this long [s]
VIEW = 12.0  # visible time span of the scrolling panels [s]
FRAME_STEP = 0.08  # signal time per GIF frame [s] (2x real time)
FPS = 25  # GIF delays are whole centiseconds; 25 fps keeps the file small
HOLD_FRAMES = 3 * FPS  # pause on the last frame before looping
DPI = 150

# palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
GRID = "#e6e5e1"
SIGNAL = "#2a78d6"
ARCOD = "#6250d6"
CALIB = "#cde2fb"
GOOD = "#0ca30c"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"


def run_precursor(x):
    # calibration on the nominal (cruise) section
    cal = flutterFeatureExtractor("cal", WINDOW, 0, calibrationFlag=True)
    cal.onlineSignal = list(x[: CALIBRATION_SECONDS * FS])
    with contextlib.redirect_stdout(io.StringIO()):
        cal.calibration(0, CALIBRATION_SECONDS * FS)

    # stream the measurements one by one
    ext = flutterFeatureExtractor(
        "demo", WINDOW, cal.calibrationFactor, calibrationFlag=False
    )
    for v in x:
        ext.get_measurement(v)
    return np.array(ext.arCod), ext.threshold_arcod, cal.calibrationFactor


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_3, labelsize=9, length=0, pad=4)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    np.random.seed(SEED)
    t, x, zeta = flutter_approach_signal(seed=SEED)
    arcod, threshold, gamma = run_precursor(x)
    t_arcod = t[len(t) - len(arcod) :]

    hold = int(ALARM_HOLD * FS)
    above = arcod > threshold
    alarm = np.zeros_like(above)
    for i in range(hold, len(above)):
        alarm[i] = above[i - hold : i].all()
    t_alarm = t_arcod[np.argmax(alarm)] if alarm.any() else None
    t_onset = t[np.argmax(zeta <= 0)]
    print(f"gamma={gamma:.2f}, alarm at {t_alarm:.1f}s, onset at {t_onset:.1f}s")

    # figure
    # ------
    plt.rcParams.update({"font.family": "Lato", "font.size": 10})
    fig = plt.figure(figsize=(9.6, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(
        3, 1, height_ratios=[3, 3, 1.1], hspace=0.38,
        left=0.085, right=0.975, top=0.83, bottom=0.08,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])
    for ax in (ax1, ax2, ax3):
        style_axis(ax)
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.text(0.085, 0.945, "arCod flutter precursor", fontsize=16,
             fontweight="bold", color=INK)
    fig.text(0.085, 0.9, "Wing-tip sensor streamed sample by sample  ·  "
             "128-sample sliding window  ·  synthetic data",
             fontsize=10, color=INK_2)

    # status chip (top right)
    chip = FancyBboxPatch((0, 0), 0.01, 0.01, transform=fig.transFigure,
                          boxstyle="round,pad=0.008,rounding_size=0.012",
                          lw=0, zorder=5)
    fig.patches.append(chip)
    chip_text = fig.text(0.968, 0.932, "", ha="right", va="center", fontsize=11,
                         fontweight="bold", color="white", zorder=6)
    clock = fig.text(0.968, 0.885, "", ha="right", va="center", fontsize=9.5,
                     color=INK_2)

    # panel 1: signal
    ymax = 1.1 * np.abs(x).max()
    ax1.set_ylim(-ymax, ymax)
    ax1.set_title("Sensor signal", loc="left", fontsize=10.5, color=INK, pad=4)
    (l_sig,) = ax1.plot([], [], color=SIGNAL, lw=1.0)

    # panel 2: arCod
    ax2.set_ylim(0, 1)
    ax2.set_title("arCod  (autoregressive R²)", loc="left", fontsize=10.5,
                  color=INK, pad=4)
    ax2.axhline(threshold, color=CRITICAL, ls=(0, (4, 3)), lw=1.2)
    ax2.text(0.995, threshold, f"alarm threshold {threshold}", color=INK_2,
             fontsize=9, ha="right", va="center", zorder=4,
             transform=ax2.get_yaxis_transform(),
             bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID, lw=0.8))
    (l_arcod,) = ax2.plot([], [], color=ARCOD, lw=2.0, solid_capstyle="round")
    (d_arcod,) = ax2.plot([], [], "o", ms=7, color=ARCOD, mec=SURFACE, mew=2)
    ax2.set_xlabel("Time [s]", color=INK_2, fontsize=9.5)

    for ax in (ax1, ax2):
        ax.axvspan(0, CALIBRATION_SECONDS, color=CALIB, alpha=0.45, lw=0)
    ax1.text(0.3, 0.86 * ymax, "calibration", color=INK_2, fontsize=9)

    # panel 3: overview of the whole run (ground truth, not used by the method)
    ax3.plot(t, zeta, color=INK_3, lw=1.5)
    ax3.axhline(0, color=INK_3, lw=0.8)
    ax3.set_xlim(0, t[-1])
    ax3.set_ylim(-0.05, 0.11)
    ax3.set_yticks([0, 0.08])
    ax3.set_title("Modal damping — ground truth, not seen by the algorithm",
                  loc="left", fontsize=9.5, color=INK_2, pad=3)
    ax3.axvline(t_onset, color=CRITICAL, lw=1.2)
    ax3.text(t_onset + 0.4, 0.075, "flutter onset", color=INK_2, fontsize=8.5)
    view_box = Rectangle((0, 0), VIEW, 1, transform=ax3.get_xaxis_transform(),
                         color=SIGNAL, alpha=0.12, lw=0)
    ax3.add_patch(view_box)
    (d_zeta,) = ax3.plot([], [], "o", ms=6, color=INK_2, mec=SURFACE, mew=1.5)

    dynamic = []

    def set_chip(text, color):
        chip_text.set_text(text)
        chip.set_facecolor(color)
        fig.canvas.draw()  # needed to measure the text
        bb = chip_text.get_window_extent().transformed(fig.transFigure.inverted())
        chip.set_bounds(bb.x0, bb.y0, bb.width, bb.height)

    def update(frame_time):
        for artist in dynamic:
            artist.remove()
        dynamic.clear()

        # scrolling window: the last VIEW seconds
        x0 = max(0.0, frame_time - VIEW)
        ax2.set_xlim(x0, x0 + VIEW)
        view_box.set_x(x0)

        m = t <= frame_time
        l_sig.set_data(t[m], x[m])
        ma = t_arcod <= frame_time
        l_arcod.set_data(t_arcod[ma], arcod[ma])
        if ma.any():
            d_arcod.set_data([t_arcod[ma][-1]], [arcod[ma][-1]])
        k = np.flatnonzero(m)[-1]
        d_zeta.set_data([t[k]], [zeta[k]])

        # alarm intervals seen so far
        on = alarm & ma
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            for ax in (ax1, ax2):
                dynamic.append(ax.axvspan(t_arcod[s], t_arcod[e - 1],
                                          color=SERIOUS, alpha=0.18, lw=0))
            dynamic.append(ax3.axvspan(t_arcod[s], t_arcod[e - 1],
                                       color=SERIOUS, alpha=0.35, lw=0))

        if frame_time >= t_onset:
            for ax in (ax1, ax2):
                dynamic.append(ax.axvline(t_onset, color=CRITICAL, lw=1.4))
            dynamic.append(ax1.text(t_onset + 0.15, -0.88 * ymax, "flutter onset",
                                    color=CRITICAL, fontsize=9, fontweight="bold"))

        warned = t_alarm is not None and frame_time >= t_alarm
        if warned and frame_time >= t_onset:
            set_chip(f"▲  WARNED {t_onset - t_alarm:.1f} s EARLY", CRITICAL)
        elif warned:
            set_chip("▲  FLUTTER WARNING", CRITICAL)
        else:
            set_chip("●  MONITORING", GOOD)
        clock.set_text(f"t = {frame_time:4.1f} s")
        return []

    frames = list(np.arange(FRAME_STEP, t[-1] + 1e-9, FRAME_STEP))
    frames += [t[-1]] * HOLD_FRAMES
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = os.path.join(ROOT, "assets", "arcod_demo.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # render to lossless video, then encode the GIF with one global palette
    # (no per-frame color shifts) and loop forever
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "frames.mkv")
        anim.save(video, writer=FFMpegWriter(fps=FPS, codec="ffv1"), dpi=DPI,
                  savefig_kwargs={"facecolor": SURFACE})
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-filter_complex",
             "[0:v]split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];"
             "[b][p]paletteuse=dither=none:diff_mode=rectangle",
             "-loop", "0", out],
            check=True,
        )
    print("saved", out)


if __name__ == "__main__":
    main()
