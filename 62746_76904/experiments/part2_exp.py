"""Part 2 experiments to generate the tables of Task 4 (verification) shown in the report.

Each ``table_*`` function returns a markdown string ready to be rendered
in a Jupyter cell with ``show(...)``, and is cached to ``cache/``.
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import metrics
import tp1
from part3_exp import _INPUT_DIR, _load_cache, _md_table, _save_cache

DEV = ["dev_01", "dev_02"]
TEST = ["test_01", "test_02"]

THRESHOLDS = (47, 0.22, 0.6)
W = 2
G = 4
MIN_MATCHES = 20


def _counts(name):
    """evaluate_video counts of one video, without and with verification."""
    gt = json.loads((_INPUT_DIR / f"{name}.json").read_text())
    frames = tp1.read_frames(str(_INPUT_DIR / f"{name}.mp4"))
    signals = tp1.compute_signals(frames)
    cuts = tp1.combine_cuts([tp1.detect_cuts(s, T, W) for s, T in zip(signals, THRESHOLDS)], 2)
    verified = tp1.verify_cuts(cuts, frames, G, MIN_MATCHES)
    out = []
    for lst in (cuts, verified):
        pred = {"transitions": [{"type": "cut", "start": t, "end": t} for t in lst], "scenes": {}}
        out.append(metrics.evaluate_video(gt, pred))
    return out


def table_verification(force_recompute=False):
    """Task 4: cut precision / recall / F1 and flash false alarms, without and with verification."""
    c = None if force_recompute else _load_cache("verification")
    if c:
        return c["md"]

    rows = []
    for tag, names in (("DEV", DEV), ("TEST", TEST)):
        per_video = [_counts(n) for n in names]
        for k, label in enumerate(("without", "with")):
            tot = metrics.sum_counts([v[k] for v in per_video])
            p, r, f1 = metrics.precision_recall_f1(*tot["cut"])
            rows.append([tag, label, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}",
                         f"{tot['flash_fa']}/{tot['num_flashes']}"])

    md = _md_table(["Set", "Verification", "cut-P", "cut-R", "cut-F1", "flash-FA"], rows)
    _save_cache("verification", {"md": md})
    return md


# ═══════════════════════════════════════════════════════════════════
# Choice of min_matches and example (Task 4)
# ═══════════════════════════════════════════════════════════════════
import cv2
import matplotlib.pyplot as plt

import scene_grouping
from part1_exp import _show_img

IMAGES = _HERE.parent / "images"
IMAGES.mkdir(exist_ok=True)

SWEEP = [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 1000]


def _candidates(frames, name):
    d = tp1.compute_signals(frames)
    return tp1.combine_cuts([tp1.detect_cuts(s, T, W) for s, T in zip(d, THRESHOLDS)], 2)


def _match_count(frames, t):
    a, b = max(0, t - G), min(len(frames) - 1, t + G)
    return scene_grouping.sift_similarity(frames[a], frames[b])


def table_min_matches(force_recompute=False):
    """Task 4: effect of min_matches on the dev videos (the value is chosen here, never on test)."""
    c = None if force_recompute else _load_cache("min_matches")
    if c:
        return c["md"]

    data = {}
    for n in DEV:
        frames = tp1.read_frames(str(_INPUT_DIR / f"{n}.mp4"))
        cuts = _candidates(frames, n)
        data[n] = (json.loads((_INPUT_DIR / f"{n}.json").read_text()), cuts,
                   {t: _match_count(frames, t) for t in cuts})
        del frames

    rows = []
    for M in SWEEP:
        res = []
        for n in DEV:
            gt, cuts, counts = data[n]
            kept = [t for t in cuts if counts[t] < M]
            pred = {"transitions": [{"type": "cut", "start": t, "end": t} for t in kept], "scenes": {}}
            res.append(metrics.evaluate_video(gt, pred))
        tot = metrics.sum_counts(res)
        p, r, f1 = metrics.precision_recall_f1(*tot["cut"])
        label = "no verification" if M == SWEEP[-1] else (f"**{M}**" if M == MIN_MATCHES else str(M))
        rows.append([label, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}", f"{tot['flash_fa']}/{tot['num_flashes']}"])

    md = _md_table(["min_matches", "cut-P", "cut-R", "cut-F1", "flash-FA"], rows)
    _save_cache("min_matches", {"md": md})
    return md


def fig_false_alarm_removed(name="dev_01"):
    """Task 4 deliverable: a flash proposed as a cut and rejected, next to a real cut that is kept.

    Each row shows frames t-g and t+g with the SIFT matches that pass Lowe's ratio test.
    """
    gt = json.loads((_INPUT_DIR / f"{name}.json").read_text())
    frames = tp1.read_frames(str(_INPUT_DIR / f"{name}.mp4"))
    cuts = _candidates(frames, name)
    sift = scene_grouping.SIFTMatcher.get(500)

    def draw(t):
        a, b = max(0, t - G), min(len(frames) - 1, t + G)
        kp1, d1 = sift.detect(frames[a])
        kp2, d2 = sift.detect(frames[b])
        good = sift.matches(kp1, d1, kp2, d2)
        img = cv2.drawMatches(frames[a], kp1, frames[b], kp2, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img, a, b, len(good)

    # first flash whose candidate was removed, and first real cut whose candidate was kept
    flash_t = next(t for f in gt["distractors"] for t in cuts
                   if f["start"] - 2 <= t <= f["end"] + 2 and _match_count(frames, t) >= MIN_MATCHES)
    gt_cuts = [c["start"] for c in gt["transitions"]]
    cut_t = next(t for t in cuts if t in gt_cuts and _match_count(frames, t) < MIN_MATCHES)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6.5))
    for ax, t, kind, verdict in [(axs[0], flash_t, "flash proposed as a cut", "rejected"),
                                 (axs[1], cut_t, "real cut", "kept")]:
        img, a, b, n_matches = draw(t)
        _show_img(ax, img, f"candidate t = {t} ({kind}): frames {a} and {b}, "
                           f"{n_matches} matches (min_matches = {MIN_MATCHES}) -> {verdict}")
    fig.tight_layout()
    fig.savefig(IMAGES / f"t4_false_alarm_removed_{name}.png", dpi=150)
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Failure cases of the cut detector (Section 6.1 of the report)
# ═══════════════════════════════════════════════════════════════════
# (video, frame t, short name): cases found by comparing the final detector with the ground truth
FAILURE_CASES = [
    ("dev_01",  2947, "missed_cut_pixel_only"),
    ("test_01", 870,  "missed_cut_hist_only"),
    ("test_01", 3073, "cut_rejected_by_verification"),
    ("dev_02",  1867, "flash_not_rejected"),
    ("test_02", 2394, "flash_not_rejected_one_frame"),
]


def fig_failure_cases(cases=FAILURE_CASES):
    """One figure per case: frames t-G, t-1, t and t+G, with the three signals and the SIFT matches in the title."""
    paths = {}
    by_video = {}
    for name, t, tag in cases:
        by_video.setdefault(name, []).append((t, tag))
    for name, items in by_video.items():
        frames = tp1.read_frames(str(_INPUT_DIR / f"{name}.mp4"))
        d = tp1.compute_signals(frames)
        for t, tag in items:
            idx = [t - G, t - 1, t, t + G]
            fig, axs = plt.subplots(1, 4, figsize=(14, 2.8))
            for ax, i in zip(axs, idx):
                _show_img(ax, frames[i], f"frame {i}")
            m = _match_count(frames, t)
            fig.suptitle(f"{name}, t = {t}:  pixel = {d[0][t]:.1f}  histogram = {d[1][t]:.2f}  ECR = {d[2][t]:.2f}  "
                         f"|  SIFT matches between {t - G} and {t + G}: {m}", fontsize=10)
            fig.tight_layout()
            path = IMAGES / f"t6_failure_{name}_{t}_{tag}.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            paths[(name, t)] = path
    return paths


# ═══════════════════════════════════════════════════════════════════
# Final cut detector, per video (Sections 5.3 and 5.5 of the report)
# ═══════════════════════════════════════════════════════════════════
def table_cuts(set_name, force_recompute=False):
    """Cut metrics of the final detector (vote + verification), one row per video and the total."""
    names = DEV if set_name == "DEV" else TEST
    key = f"cuts_{set_name.lower()}"
    c = None if force_recompute else _load_cache(key)
    if c:
        return c["md"]

    def row(label, counts):
        p, r, f1 = metrics.precision_recall_f1(*counts["cut"])
        tp, fp, fn = counts["cut"]
        return [label, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}", f"{tp} / {fp} / {fn}",
                f"{counts['flash_fa']}/{counts['num_flashes']}"]

    rows, res = [], []
    for n in names:
        counts = _counts(n)[1]                  # [1] = with verification
        res.append(counts)
        rows.append(row(n, counts))
    rows.append(row(set_name, metrics.sum_counts(res)))

    md = _md_table(["Video", "cut-P", "cut-R", "cut-F1", "TP / FP / FN", "flash-FA"], rows)
    _save_cache(key, {"md": md})
    return md
