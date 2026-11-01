"""Part 1 experiments to generate the tables of Tasks 1 to 3 shown in the report.

Each ``table_*`` function returns a markdown string ready to be rendered
in a Jupyter cell with ``show(...)``, and is cached to ``cache/``.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import tp1
from part3_exp import _INPUT_DIR, _load_cache, _md_table, _save_cache

DEV = ["dev_01", "dev_02"]
FINAL_SIZE = (320, 180)
SIZES = [(80, 45), (160, 90), FINAL_SIZE, (640, 360)]
RANGES = [np.linspace(5, 80, 60), np.linspace(0.02, 1.5, 60), np.linspace(0.05, 1.0, 60)]


def table_resolution(force_recompute=False):
    """Task 1: read time, signal time and best threshold / F1 of each signal, per frame size."""
    c = None if force_recompute else _load_cache("resolution")
    if c:
        return c["md"]

    gts = {n: json.loads((_INPUT_DIR / f"{n}.json").read_text()) for n in DEV}
    rows = []
    for size in SIZES:
        t0 = time.perf_counter()
        frames = {n: tp1.read_frames(str(_INPUT_DIR / f"{n}.mp4"), size) for n in DEV}
        t1 = time.perf_counter()
        sig = {n: tp1.compute_signals(frames[n]) for n in DEV}
        t2 = time.perf_counter()
        del frames

        best = []
        for i in range(3):
            points = tp1.sweep_threshold({n: sig[n][i] for n in DEV}, gts, RANGES[i])
            T, _, _, f1 = tp1.best_threshold(points)
            best.append(f"{T:.3f} / {f1:.3f}")
        label = f"{size[0]}x{size[1]}"
        rows.append([f"**{label}**" if size == FINAL_SIZE else label,
                     f"{t1 - t0:.1f}", f"{t2 - t1:.1f}", *best])

    md = _md_table(["Size", "Read time (s)", "Signals time (s)",
                    "Pixel: T / F1", "Histogram: T / F1", "ECR: T / F1"], rows)
    _save_cache("resolution", {"md": md})
    return md


# ═══════════════════════════════════════════════════════════════════
# Signals, thresholds and figures (Tasks 2 and 3)
# ═══════════════════════════════════════════════════════════════════
import cv2
import matplotlib.pyplot as plt

import metrics
import plotting

IMAGES = _HERE.parent / "images"
IMAGES.mkdir(exist_ok=True)

SIGNALS = ["pixel", "histogram", "ecr"]
FINAL_T = (47, 0.22, 0.6)             # thresholds used by process_video (tp1.py)

_gt_cache, _sig_cache, _curve_cache = {}, {}, {}


def _gt(name):
    if name not in _gt_cache:
        _gt_cache[name] = json.loads((_INPUT_DIR / f"{name}.json").read_text())
    return _gt_cache[name]


def _signals(name):
    """(d_pixel, d_hist, d_ecr) of one video, computed once per session."""
    if name not in _sig_cache:
        frames = tp1.read_frames(str(_INPUT_DIR / f"{name}.mp4"))
        _sig_cache[name] = tuple(tp1.compute_signals(frames))
    return _sig_cache[name]


def _curves():
    """{signal: [(T, precision, recall), ...]} over the dev videos."""
    if not _curve_cache:
        gts = {n: _gt(n) for n in DEV}
        for i, name in enumerate(SIGNALS):
            _curve_cache[name] = tp1.sweep_threshold({n: _signals(n)[i] for n in DEV}, gts, RANGES[i])
    return _curve_cache


def _show_img(ax, bgr, title):
    """Display a BGR image in a matplotlib axis (lab 2 helper: BGR -> RGB, title, no axis)."""
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def table_thresholds(force_recompute=False):
    """Task 3: best threshold (best F1 on the dev videos) of each signal, and the 2-of-3 vote."""
    c = None if force_recompute else _load_cache("thresholds")
    if c:
        return c["md"]

    rows = []
    for name in SIGNALS:
        T, p, r, f1 = tp1.best_threshold(_curves()[name])
        rows.append([name, f"{T:.3f}", f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])

    gts = {n: _gt(n) for n in DEV}
    vote = {n: tp1.combine_cuts([tp1.detect_cuts(s, T, 2) for s, T in zip(_signals(n), FINAL_T)], 2)
            for n in DEV}
    p, r, f1 = tp1.evaluate_cuts(vote, gts)
    rows.append(["vote 2 of 3", " / ".join(str(t) for t in FINAL_T), f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])

    md = _md_table(["Signal", "T", "cut-P", "cut-R", "cut-F1"], rows)
    _save_cache("thresholds", {"md": md})
    return md


def fig_pr_curves():
    """Task 3 deliverable: precision-recall curve of each signal, sweeping its threshold (dev videos)."""
    fig = plotting.plot_pr_curve(_curves(), title="Cut detection, dev videos",
                                 path=str(IMAGES / "t3_pr_curve_dev.png"))
    plt.close(fig)
    return fig


def fig_signals(name="dev_01", start=500, end=1250):
    """Task 3 deliverable: the three signals over ~30 s with ground-truth cuts and flashes."""
    d_pixel, d_hist, d_ecr = _signals(name)
    fig = plotting.plot_signals({"pixel": d_pixel, "histogram": d_hist, "ecr": d_ecr}, _gt(name),
                                start=start, end=end, path=str(IMAGES / f"t3_signals_{name}.png"))
    plt.close(fig)
    return fig


def fig_ecr_example(name="dev_01", offset=30):
    """Task 2: the ECR on a pair of frames inside a shot and on a pair across the first cut.

    Right column: dilated edges of frame t-1 in grey; edges of frame t in green when they fall
    inside them (they moved little) and in red when they do not (entering edges).
    """
    gt = _gt(name)
    t_cut = gt["transitions"][0]["start"]
    frames = tp1.read_frames(str(_INPUT_DIR / f"{name}.mp4"))
    kernel = np.ones((5, 5), np.uint8)

    fig, axs = plt.subplots(2, 3, figsize=(12, 5))
    for row, (t, label) in enumerate([(t_cut + offset, "inside a shot"), (t_cut, "across a cut")]):
        prev, cur = frames[t - 1], frames[t]
        e1 = cv2.Canny(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), 100, 200)
        e2 = cv2.Canny(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), 100, 200)
        d1 = cv2.dilate(e1, kernel) > 0
        overlay = np.zeros(e2.shape + (3,), np.uint8)
        overlay[d1] = (90, 90, 90)
        overlay[(e2 > 0) & d1] = (0, 200, 0)
        overlay[(e2 > 0) & ~d1] = (220, 0, 0)
        ecr = tp1.edge_change_ratio(e1, e2)
        _show_img(axs[row][0], prev, f"frame {t - 1} ({label})")
        _show_img(axs[row][1], cur, f"frame {t}")
        axs[row][2].imshow(overlay)
        axs[row][2].set_title(f"edges of t: green = near, red = entering   ECR = {ecr:.2f}", fontsize=9)
        axs[row][2].axis("off")
    fig.tight_layout()
    fig.savefig(IMAGES / f"t2_ecr_example_{name}.png", dpi=150)
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Colour space and bins of the histogram signal (Task 2)
# ═══════════════════════════════════════════════════════════════════
# name -> (conversion, channels, bins, ranges) for cv2.calcHist
_HIST_CONFIGS = [
    ("RGB, 8x8x8",         None,               [0, 1, 2], [8, 8, 8],    [0, 256, 0, 256, 0, 256]),
    ("HSV (H,S,V), 8x8x8", cv2.COLOR_BGR2HSV,  [0, 1, 2], [8, 8, 8],    [0, 180, 0, 256, 0, 256]),
    ("HSV (H,S), 4x4",     cv2.COLOR_BGR2HSV,  [0, 1],    [4, 4],       [0, 180, 0, 256]),
    ("HSV (H,S), 8x8",     cv2.COLOR_BGR2HSV,  [0, 1],    [8, 8],       [0, 180, 0, 256]),
    ("HSV (H,S), 16x16",   cv2.COLOR_BGR2HSV,  [0, 1],    [16, 16],     [0, 180, 0, 256]),
]
_HIST_RANGE = RANGES[1]                              # same thresholds as the sweep of the histogram signal in Task 3
FINAL_HIST = "RGB, 8x8x8"                        # what compute_signals (tp1.py) uses


def _hist_signal(frames, conv, channels, bins, ranges):
    hists = []
    for f in frames:
        img = f if conv is None else cv2.cvtColor(f, conv)
        h = cv2.calcHist([img], channels, None, bins, ranges)
        hists.append(h / h.sum())
    return [0] + [tp1.histogram_distance(hists[t - 1], hists[t]) for t in range(1, len(hists))]


def table_colour_space(force_recompute=False):
    """Task 2: colour space and number of bins of the histogram signal (dev videos).

    For each configuration the threshold is swept and the best F1 is shown, together with the flashes
    that trigger a false alarm at that threshold (no verification yet).
    """
    c = None if force_recompute else _load_cache("colour_space")
    if c:
        return c["md"]

    gts = {n: _gt(n) for n in DEV}
    sigs = {cfg[0]: {} for cfg in _HIST_CONFIGS}
    for n in DEV:
        frames = tp1.read_frames(str(_INPUT_DIR / f"{n}.mp4"))
        for name, conv, ch, bins, rng in _HIST_CONFIGS:
            sigs[name][n] = _hist_signal(frames, conv, ch, bins, rng)
        del frames

    rows = []
    for name, *_ in _HIST_CONFIGS:
        T, p, r, f1 = tp1.best_threshold(tp1.sweep_threshold(sigs[name], gts, _HIST_RANGE))
        res = []
        for n in DEV:
            pred = {"transitions": [{"type": "cut", "start": t, "end": t}
                                    for t in tp1.detect_cuts(sigs[name][n], T, 2)], "scenes": {}}
            res.append(metrics.evaluate_video(gts[n], pred))
        tot = metrics.sum_counts(res)
        label = f"**{name}**" if name == FINAL_HIST else name
        rows.append([label, f"{T:.3f}", f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}",
                     f"{tot['flash_fa']}/{tot['num_flashes']}"])

    md = _md_table(["Histogram", "best T", "cut-P", "cut-R", "cut-F1", "flash-FA"], rows)
    _save_cache("colour_space", {"md": md})
    return md
