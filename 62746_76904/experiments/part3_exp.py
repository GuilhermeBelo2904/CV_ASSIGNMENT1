"""Part 3 experiments to generate every table shown in the report.

Each ``table_*`` function returns a markdown string ready to be rendered
in a Jupyter cell with ``display(Markdown(...))``.  Matrix-level sweeps
are cached to ``cache/`` so repeated notebook runs are fast.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import scene_grouping as sg
import grouping
import metrics

from tp1 import read_frames

# ═══════════════════════════════════════════════════════════════════
# Caching
# ═══════════════════════════════════════════════════════════════════
CACHE_DIR = _HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
_CACHE_VERSION = "v5"


def _cache_path(name):
    return CACHE_DIR / f"{name}_{_CACHE_VERSION}.json"


def _load_cache(name):
    p = _cache_path(name)
    return json.loads(p.read_text()) if p.exists() else None


def _save_cache(name, data):
    _cache_path(name).write_text(json.dumps(data, default=str, indent=2))


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════
_INPUT_DIR = _HERE.parent.parent / "input"
_frames_cache, _gt_cache = {}, {}


def _video(name):
    """Load (gt, frames) once."""
    if name not in _frames_cache:
        _frames_cache[name] = read_frames(
            str(_INPUT_DIR / f"{name}.mp4"), size=(320, 180))
        _gt_cache[name] = json.loads((_INPUT_DIR / f"{name}.json").read_text())
    return _gt_cache[name], _frames_cache[name]


def _gt_pack(gt):
    return {
        "_map":   {str(s["id"]): s["scene"] for s in gt["shots"]},
        "_shots": gt["shots"],
    }


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _plateau(sim, gt_pack, n_points=100, tol=1e-3):
    """Return (T_center, P, R, F1, width) for the best-F1 plateau."""
    n = len(sim)
    upper = sim[np.triu_indices(n, k=1)]
    lo, hi = float(upper.min()), float(upper.max())
    if hi - lo < 1e-6:
        return lo, 0.0, 0.0, 0.0, 0.0
    rows = []
    for t in np.linspace(lo, hi, n_points):
        labs = grouping.group_shots(sim, t)
        pred = {str(s["id"]): int(l) for s, l in zip(gt_pack["_shots"], labs)}
        tp, fp, fn = metrics.pairwise_scene_counts(gt_pack["_map"], pred)
        p, r, f1 = metrics.precision_recall_f1(tp, fp, fn)
        rows.append((float(t), p, r, f1))
    best = max(rr[3] for rr in rows)
    plateau = [rr for rr in rows if rr[3] >= best - tol]
    T_center = float(np.median([rr[0] for rr in plateau]))
    width = plateau[-1][0] - plateau[0][0] if len(plateau) > 1 else 0.0
    closest = min(plateau, key=lambda r: abs(r[0] - T_center))
    return T_center, closest[1], closest[2], closest[3], width


def _three_kfs(video):
    """(gt, [kf_same_a, kf_same_b, kf_diff], scene) for single-pair tests."""
    gt, frames = _video(video)
    groups = defaultdict(list)
    for s in gt["shots"]:
        groups[s["scene"]].append(s)
    scene = next(k for k, v in groups.items() if len(v) >= 2)
    a, b = groups[scene][0], groups[scene][-1]
    c = next(s for s in gt["shots"] if s["scene"] != scene)
    kfs = [sg.extract_keyframes(frames, s, "middle")[0] for s in (a, b, c)]
    return gt, kfs, scene


def _fmt_time(ms):
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms/1000:.1f} s"


# ═══════════════════════════════════════════════════════════════════
# Single-pair tables (4.2.1, 8.1, 8.2, 8.3, 8.4)
# ═══════════════════════════════════════════════════════════════════
def table_bins_single(force_recompute=False):
    """4.2.1 bin resolution on a single keyframe pair."""
    c = None if force_recompute else _load_cache("bins_single")
    if c: return c["md"]
    rows = []
    for video in ("dev_01", "dev_02"):
        _, (a, b, diff), _ = _three_kfs(video)
        for bins in [(8, 8, 8), (16, 16, 16), (30, 32, 32), (60, 64, 64)]:
            s = sg.histogram_similarity(a, b, bins=bins)
            d = sg.histogram_similarity(a, diff, bins=bins)
            px = 57600 / (bins[0] * bins[1] * bins[2])
            px_str = f"{px:.2f}" if px < 1 else f"{px:.1f}"
            rows.append([video, "×".join(map(str, bins)),
                         f"{s:.4f}", f"{d:.4f}", f"{s - d:.4f}", px_str])
    md = _md_table(["Video", "bins", "same", "diff", "gap", "pixels/bin"], rows)
    _save_cache("bins_single", {"md": md, "rows": rows})
    return md


def table_colour_space_single(include_lab=True, force_recompute=False):
    """4.2.1 (HSV vs RGB) and 8.2 (HSV vs Lab vs RGB + means)."""
    key = "colour_single_lab" if include_lab else "colour_single"
    c = None if force_recompute else _load_cache(key)
    if c: return c["md"]

    spaces = [("HSV", sg.histogram_similarity)]
    if include_lab:
        spaces.append(("Lab", sg.lab_similarity))
    spaces.append(("RGB", sg.rgb_similarity))

    raw = []
    for video in ("dev_01", "dev_02"):
        _, (a, b, diff), _ = _three_kfs(video)
        for name, fn in spaces:
            s = fn(a, b)
            d = fn(a, diff)
            raw.append((video, name, s, d, s - d))

    rows = [[v, name, f"{s:.4f}", f"{d:.4f}", f"{g:.4f}"]
            for v, name, s, d, g in raw]

    if include_lab:
        for name, _ in spaces:
            vals = [g for _, nm, _, _, g in raw if nm == name]
            rows.append(["mean", name, "—", "—",
                         f"{sum(vals) / len(vals):.4f}"])

    md = _md_table(["Video", "Space", "same", "diff", "gap"], rows)
    _save_cache(key, {"md": md})
    return md


def table_nfeatures_single(force_recompute=False):
    """8.4 nfeatures sweep on a single keyframe pair."""
    c = None if force_recompute else _load_cache("nfeatures_single")
    if c: return c["md"]
    rows = []
    for video in ("dev_01", "dev_02"):
        _, (a, b, diff), _ = _three_kfs(video)
        for nf in (100, 200, 500, 0):          # 0 = unlimited
            m = sg.SIFTMatcher.get(nfeatures=nf)
            t0 = time.perf_counter()
            kp_a, des_a = m.detect(a)
            kp_b, des_b = m.detect(b)
            kp_c, des_c = m.detect(diff)
            s = m.match_count(kp_a, des_a, kp_b, des_b)
            d = m.match_count(kp_a, des_a, kp_c, des_c)
            dt = (time.perf_counter() - t0) * 1000
            rows.append([video, "unlimited" if nf == 0 else nf,
                         len(kp_a), len(kp_b), s, d, s - d, f"{dt:.1f} ms"])
    md = _md_table(
        ["Video", "nfeatures", "kp_a", "kp_b", "same", "diff", "gap", "time"],
        rows)
    _save_cache("nfeatures_single", {"md": md})
    return md


def table_lowe_single(force_recompute=False):
    """8.4 Lowe's ratio sweep (same/diff/gap + same/diff ratio)."""
    c = None if force_recompute else _load_cache("lowe_single")
    if c: return c["md"]
    ratios = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    data = {"dev_01": [], "dev_02": []}
    for video in ("dev_01", "dev_02"):
        _, (a, b, diff), _ = _three_kfs(video)
        m = sg.SIFTMatcher.get(nfeatures=500)
        kp_a, des_a = m.detect(a)
        kp_b, des_b = m.detect(b)
        kp_c, des_c = m.detect(diff)
        for r in ratios:
            s = m.match_count(kp_a, des_a, kp_b, des_b, ratio=r)
            d = m.match_count(kp_a, des_a, kp_c, des_c, ratio=r)
            data[video].append((r, s, d))

    rows1 = [[f"{r:.2f}",
              f"{next(x[1] for x in data['dev_01'] if x[0]==r)} / "
              f"{next(x[2] for x in data['dev_01'] if x[0]==r)} / "
              f"{next(x[1]-x[2] for x in data['dev_01'] if x[0]==r)}",
              f"{next(x[1] for x in data['dev_02'] if x[0]==r)} / "
              f"{next(x[2] for x in data['dev_02'] if x[0]==r)} / "
              f"{next(x[1]-x[2] for x in data['dev_02'] if x[0]==r)}"]
             for r in ratios]
    t1 = _md_table(
        ["ratio", "dev_01 (same / diff / gap)", "dev_02 (same / diff / gap)"],
        rows1)

    def _ratio(s, d):
        return "∞ (0 false)" if d == 0 else f"{s/d:.2f}"

    rows2 = [[f"{r:.2f}",
              _ratio(next(x[1] for x in data['dev_01'] if x[0]==r),
                     next(x[2] for x in data['dev_01'] if x[0]==r)),
              _ratio(next(x[1] for x in data['dev_02'] if x[0]==r),
                     next(x[2] for x in data['dev_02'] if x[0]==r))]
             for r in ratios]
    t2 = _md_table(["ratio", "dev_01 (same/diff)", "dev_02 (same/diff)"], rows2)

    md = t1 + "\n\n" + t2
    _save_cache("lowe_single", {"md": md})
    return md


def table_ransac_single(force_recompute=False):
    """8.3 geometric filtering on a single keyframe pair."""
    c = None if force_recompute else _load_cache("ransac_single")
    if c: return c["md"]
    transforms = ["none", "translation", "similarity", "affine", "homography"]
    rows = []
    for video in ("dev_01", "dev_02"):
        _, (a, b, diff), _ = _three_kfs(video)
        m = sg.SIFTMatcher.get(nfeatures=500)
        kp_a, des_a = m.detect(a)
        kp_b, des_b = m.detect(b)
        kp_c, des_c = m.detect(diff)
        for tr in transforms:
            t0 = time.perf_counter()
            if tr == "none":
                s = m.match_count(kp_a, des_a, kp_b, des_b, use_ransac=False)
                d = m.match_count(kp_a, des_a, kp_c, des_c, use_ransac=False)
            else:
                s = m.match_count(kp_a, des_a, kp_b, des_b,
                                   use_ransac=True, transform=tr)
                d = m.match_count(kp_a, des_a, kp_c, des_c,
                                   use_ransac=True, transform=tr)
            dt = (time.perf_counter() - t0) * 1000
            rows.append([video, tr, s, d, s - d, f"{dt:.1f} ms"])
    md = _md_table(
        ["Video", "transform", "same", "diff", "gap", "time"], rows)
    _save_cache("ransac_single", {"md": md})
    return md


def table_alpha_single(force_recompute=False):
    """8.1 alpha gap on a single keyframe pair."""
    c = None if force_recompute else _load_cache("alpha_single")
    if c: return c["md"]
    alphas = [0.0, 0.25, 0.50, 0.75, 1.00]
    rows = []
    for video in ("dev_01", "dev_02"):
        _, (a, b, diff), _ = _three_kfs(video)
        cells = []
        for al in alphas:
            s = sg.combined_similarity(a, b, alpha=al)
            d = sg.combined_similarity(a, diff, alpha=al)
            cells.append(f"{s - d:.4f}")
        rows.append([f"{video} gap"] + cells)
    md = _md_table(["Video"] + [f"α={a:.2f}" for a in alphas], rows)
    _save_cache("alpha_single", {"md": md})
    return md


# ═══════════════════════════════════════════════════════════════════
# Matrix tables (5.1.1 – 5.1.6, §5.2)
# ═══════════════════════════════════════════════════════════════════
def _prod_cfg(**overrides):
    return {**sg.DEFAULT_CONFIG, **overrides}


def _run_matrix_sweep(configs, cache_name, force=False):
    """Generic matrix sweep runner. `configs` is a list of (label, cfg, extra)."""
    c = None if force else _load_cache(cache_name)
    if c: return c
    results = []
    for label, cfg, extra in configs:
        f1s, widths, times = [], [], []
        for video in ("dev_01", "dev_02"):
            gt, frames = _video(video)
            t0 = time.perf_counter()
            sim = sg.build_similarity_matrix(frames, gt["shots"], cfg)
            times.append((time.perf_counter() - t0) * 1000)
            _, _, _, f1, w = _plateau(sim, _gt_pack(gt))
            f1s.append(f1)
            widths.append(w)
        results.append({
            "label": label, "extra": extra,
            "f1": f1s, "width": widths,
            "mean_f1": (f1s[0] + f1s[1]) / 2,
            "mean_time": (times[0] + times[1]) / 2,
        })
    _save_cache(cache_name, {"results": results})
    return {"results": results}


def table_k_sweep(force_recompute=False):
    """5.1.1 K-sweep (middle vs uniform, K ∈ {1,3,5,8})."""
    configs = [
        ("middle",  _prod_cfg(kf_strategy="middle",  K=1), "1 centered"),
        ("uniform", _prod_cfg(kf_strategy="uniform", K=1), "1 first"),
        ("uniform", _prod_cfg(kf_strategy="uniform", K=3), "3 evenly"),
        ("uniform", _prod_cfg(kf_strategy="uniform", K=5), "5 evenly"),
        ("uniform", _prod_cfg(kf_strategy="uniform", K=8), "8 evenly"),
    ]
    data = _run_matrix_sweep(configs, "k_sweep", force_recompute)
    rows = []
    for r in data["results"]:
        rows.append([r["label"],
                     "1" if r["extra"].startswith("1") else r["extra"][0],
                     f"{r['f1'][0]:.3f}", f"{r['f1'][1]:.3f}",
                     f"{r['mean_f1']:.3f}",
                     f"{r['width'][0]:.2f} / {r['width'][1]:.2f}",
                     _fmt_time(r["mean_time"])])
    return _md_table(
        ["Strategy", "K", "d01_F1", "d02_F1", "mean F1",
         "width (d01/d02)", "time"], rows)


def table_aggregator_sweep(force_recompute=False):
    """5.1.2 aggregator sweep (max / mean / vote)."""
    configs = [("max",  _prod_cfg(aggregator="max"),  ""),
               ("mean", _prod_cfg(aggregator="mean"), ""),
               ("vote", _prod_cfg(aggregator="vote"), "")]
    data = _run_matrix_sweep(configs, "aggregator_sweep", force_recompute)
    rows = [[r["label"], f"{r['f1'][0]:.3f}", f"{r['f1'][1]:.3f}",
             f"{r['mean_f1']:.3f}",
             f"{r['width'][0]:.2f} / {r['width'][1]:.2f}"]
            for r in data["results"]]
    return _md_table(
        ["Aggregator", "d01 F1", "d02 F1", "mean F1", "width (d01/d02)"], rows)


def table_alpha_sweep(force_recompute=False):
    """5.1.3 alpha sweep on the matrix."""
    configs = [(f"{a:.2f}", _prod_cfg(method="combined", combined_alpha=a), "")
               for a in (0.00, 0.25, 0.50, 0.75, 1.00)]
    data = _run_matrix_sweep(configs, "alpha_sweep", force_recompute)
    rows = [[r["label"], f"{r['f1'][0]:.3f}", f"{r['f1'][1]:.3f}",
             f"{r['mean_f1']:.3f}",
             f"{r['width'][0]:.2f}", f"{r['width'][1]:.2f}",
             _fmt_time(r["mean_time"])]
            for r in data["results"]]
    return _md_table(
        ["α", "d01 F1", "d02 F1", "mean F1", "d01 width", "d02 width", "time"],
        rows)


def table_colour_space_matrix(force_recompute=False):
    """5.1.4 HSV / Lab / RGB on the matrix."""
    configs = [("hist (HSV)", _prod_cfg(method="hist"), ""),
               ("lab",       _prod_cfg(method="lab"),  ""),
               ("rgb",       _prod_cfg(method="rgb"),  "")]
    data = _run_matrix_sweep(configs, "colour_matrix", force_recompute)
    rows = [[r["label"], f"{r['f1'][0]:.3f}", f"{r['f1'][1]:.3f}",
             f"{r['mean_f1']:.3f}",
             f"{r['width'][0]:.2f}", f"{r['width'][1]:.2f}",
             _fmt_time(r["mean_time"])]
            for r in data["results"]]
    return _md_table(
        ["Method", "d01 F1", "d02 F1", "mean F1", "d01 width", "d02 width", "time"],
        rows)


def table_bins_matrix(force_recompute=False):
    """5.1.5 bin resolution on the matrix (two sub-tables)."""
    def _label(b):
        if len(set(b)) == 1:
            return f"{b[0]}³"
        return f"{b[0]}·{b[1]}²"

    configs = [(_label(b), _prod_cfg(bins=b), "")
               for b in [(4, 4, 4), (8, 8, 8), (16, 16, 16), (30, 32, 32)]]
    data = _run_matrix_sweep(configs, "bins_matrix", force_recompute)

    rows1 = [[r["label"], f"{r['f1'][0]:.3f}", f"{r['f1'][1]:.3f}",
              f"{r['mean_f1']:.3f}",
              f"{r['width'][0]:.3f}", f"{r['width'][1]:.3f}",
              f"{(r['width'][0]+r['width'][1])/2:.3f}",
              _fmt_time(r["mean_time"])]
             for r in data["results"]]
    t1 = _md_table(
        ["Bins", "d01 F1", "d02 F1", "mean F1",
         "d01 width", "d02 width", "mean width", "time"], rows1)

    base = next(r for r in data["results"] if r["label"] == "8³")
    base_mean = (base["width"][0] + base["width"][1]) / 2
    rows2 = []
    for r in data["results"]:
        mean_w = (r["width"][0] + r["width"][1]) / 2
        pct = (mean_w / base_mean - 1) * 100
        if r["label"] == "8³":
            delta = "baseline"
        else:
            sign = "−" if pct < 0 else "+"
            delta = f"{sign}{abs(pct):.0f} %"
        rows2.append([r["label"], f"{r['width'][0]:.3f}",
                      f"{r['width'][1]:.3f}", f"{mean_w:.3f}", delta])
    t2 = _md_table(
        ["Bins", "d01 width", "d02 width", "Mean width", "vs 8³"], rows2)
    return t1 + "\n\n" + t2


def table_ransac_matrix(force_recompute=False):
    """5.1.6 RANSAC on vs off, for sift and combined."""
    configs = [
        ("sift",     _prod_cfg(method="sift",     use_ransac=False), {"m": "sift",     "r": False}),
        ("sift",     _prod_cfg(method="sift",     use_ransac=True),  {"m": "sift",     "r": True}),
        ("combined", _prod_cfg(method="combined", use_ransac=False), {"m": "combined", "r": False}),
        ("combined", _prod_cfg(method="combined", use_ransac=True),  {"m": "combined", "r": True}),
    ]
    data = _run_matrix_sweep(configs, "ransac_matrix", force_recompute)

    rows = []
    for r in data["results"]:
        ex = r["extra"]

        d = 2 if ex["m"] == "sift" else 3
        rows.append([ex["m"], "✓" if ex["r"] else "✗",
                     f"{r['f1'][0]:.3f}", f"{r['f1'][1]:.3f}",
                     f"{r['mean_f1']:.3f}",
                     f"{r['width'][0]:.{d}f}", f"{r['width'][1]:.{d}f}",
                     _fmt_time(r["mean_time"])])
    return _md_table(
        ["Method", "RANSAC", "d01 F1", "d02 F1", "mean F1",
         "d01 width", "d02 width", "time"], rows)


def table_2x2(force_recompute=False):
    """5.2 the 2×2 (hist / sift / combined × middle / uniform) + extras."""
    c = None if force_recompute else _load_cache("two_by_two_md")
    if c:
        return c["md"]

    configs = [
        ("hist",     _prod_cfg(method="hist",     kf_strategy="middle",  K=1), {"kf": "middle",  "K": 1}),
        ("hist",     _prod_cfg(method="hist",     kf_strategy="uniform", K=5), {"kf": "uniform", "K": 5}),
        ("sift",     _prod_cfg(method="sift",     kf_strategy="middle",  K=1), {"kf": "middle",  "K": 1}),
        ("sift",     _prod_cfg(method="sift",     kf_strategy="uniform", K=5), {"kf": "uniform", "K": 5}),
        ("combined", _prod_cfg(method="combined", kf_strategy="middle",  K=1), {"kf": "middle",  "K": 1}),
        ("combined", _prod_cfg(method="combined", kf_strategy="uniform", K=5), {"kf": "uniform", "K": 5}),
    ]
    data = _run_matrix_sweep(configs, "two_by_two", force_recompute)

    rows = []
    for r in data["results"]:
        method = r["label"]
        kf = r["extra"]["kf"]
        K = r["extra"]["K"]
        p_r = []
        for video in ("dev_01", "dev_02"):
            gt, frames = _video(video)
            sim = sg.build_similarity_matrix(frames, gt["shots"],
                                             _prod_cfg(method=method,
                                                       kf_strategy=kf, K=K))
            _, p, rec, f1, _ = _plateau(sim, _gt_pack(gt))
            p_r.append((p, rec, f1))
        rows.append([
            method, f"`{kf}`", K,
            f"{p_r[0][0]:.3f}", f"{p_r[0][1]:.3f}", f"{p_r[0][2]:.3f}",
            f"{p_r[1][0]:.3f}", f"{p_r[1][1]:.3f}", f"{p_r[1][2]:.3f}",
            f"{r['mean_f1']:.3f}",
            _fmt_time(r["mean_time"]),
        ])
    md = _md_table(
        ["Method", "KF", "K", "d01 P", "d01 R", "d01 F1",
         "d02 P", "d02 R", "d02 F1", "mean F1", "time"], rows)
    _save_cache("two_by_two_md", {"md": md})
    return md

def table_threshold_plateau(force_recompute=False):
    """5.1.7 optimal-T plateau per video (production config)."""
    c = None if force_recompute else _load_cache("threshold_plateau")
    if c: return c["md"]
    rows = []
    for video in ("dev_01", "dev_02"):
        gt, frames = _video(video)
        gp = _gt_pack(gt)
        sim = sg.build_similarity_matrix(frames, gt["shots"], sg.DEFAULT_CONFIG)
        n = len(sim)
        upper = sim[np.triu_indices(n, k=1)]
        lo, hi = float(upper.min()), float(upper.max())
        sweep = []
        for t in np.linspace(lo, hi, 100):
            labs = grouping.group_shots(sim, t)
            pred = {str(s["id"]): int(l) for s, l in zip(gp["_shots"], labs)}
            tp, fp, fn = metrics.pairwise_scene_counts(gp["_map"], pred)
            _, _, f1 = metrics.precision_recall_f1(tp, fp, fn)
            sweep.append((float(t), f1))
        best = max(r[1] for r in sweep)
        plateau = [r for r in sweep if r[1] >= best - 1e-3]
        T_opt = float(np.median([r[0] for r in plateau]))
        width = plateau[-1][0] - plateau[0][0]
        rows.append([video, f"{T_opt:.4f}", f"{width:.4f}",
                     f"[{plateau[0][0]:.3f}, {plateau[-1][0]:.3f}]"])
    md = _md_table(["Video", "Optimal T", "Plateau width", "Plateau range"], rows)
    _save_cache("threshold_plateau", {"md": md})
    return md

def table_dev_scene_metrics(force_recompute=False):
    """5.4 scene grouping metrics on dev_01, dev_02 and DEV total."""
    c = None if force_recompute else _load_cache("dev_scene_metrics")
    if c: return c["md"]

    per_video = []
    for video in ("dev_01", "dev_02"):
        gt, frames = _video(video)
        gt_shots = gt["shots"]
        sim = sg.build_similarity_matrix(frames, gt_shots, sg.DEFAULT_CONFIG)
        labels = grouping.group_shots(sim, sg.DEFAULT_CONFIG["threshold"])
        gt_map = {str(s["id"]): s["scene"] for s in gt_shots}
        pred_map = {str(s["id"]): int(l) for s, l in zip(gt_shots, labels)}
        tp, fp, fn = metrics.pairwise_scene_counts(gt_map, pred_map)
        n_pred = len(set(labels))
        n_gt = len(set(gt_map.values()))
        per_video.append((video, n_pred, n_gt, tp, fp, fn))

    tp_t = sum(x[3] for x in per_video)
    fp_t = sum(x[4] for x in per_video)
    fn_t = sum(x[5] for x in per_video)

    def _row(name, n_pred, n_gt, tp, fp, fn):
        p, r, f1 = metrics.precision_recall_f1(tp, fp, fn)
        return [name, f"{n_pred} / {n_gt}",
                f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"]

    rows = [_row(v, np_, ng_, tp, fp, fn)
            for v, np_, ng_, tp, fp, fn in per_video]
    n_pred_dev = sum(x[1] for x in per_video)
    n_gt_dev = sum(x[2] for x in per_video)
    rows.append(_row("DEV", n_pred_dev, n_gt_dev, tp_t, fp_t, fn_t))

    md = _md_table(
        ["Video", "Scenes (pred / GT)", "Scene-P", "Scene-R", "Scene-F1"], rows)
    _save_cache("dev_scene_metrics", {"md": md})
    return md

# ═══════════════════════════════════════════════════════════════════
# Convenience: regenerate everything
# ═══════════════════════════════════════════════════════════════════
def run_all(force_recompute=False):
    """Regenerate every table. Returns dict name → markdown."""
    return {
        "bins_single":         table_bins_single(force_recompute),
        "colour_single":       table_colour_space_single(include_lab=True,
                                                 force_recompute=force_recompute),
        "nfeatures_single":    table_nfeatures_single(force_recompute),
        "lowe_single":         table_lowe_single(force_recompute),
        "ransac_single":       table_ransac_single(force_recompute),
        "alpha_single":        table_alpha_single(force_recompute),
        "k_sweep":             table_k_sweep(force_recompute),
        "aggregator_sweep":    table_aggregator_sweep(force_recompute),
        "alpha_sweep":         table_alpha_sweep(force_recompute),
        "colour_matrix":       table_colour_space_matrix(force_recompute),
        "bins_matrix":         table_bins_matrix(force_recompute),
        "ransac_matrix":       table_ransac_matrix(force_recompute),
        "two_by_two":          table_2x2(force_recompute),
        "threshold":           table_threshold_plateau(force_recompute),
        "dev_scene_metrics":   table_dev_scene_metrics(force_recompute)
    }
