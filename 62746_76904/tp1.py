###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Shot boundary detection and recurring scenes
##
## 62746
## 76904
##
###############################################################################

import glob
import json
import os

import cv2
import numpy as np

import scene_grouping   # group_shots: union-find grouping of shots (Part 3, Task 7)
import metrics

INPUT_DIR = os.path.join("..", "input")
OUTPUT_DIR = os.path.join("..", "output")


def read_frames(path, size=(320, 180)):
    """Read every frame of a video, resized to `size`. Returns a list of BGR images."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames

def pixel_difference(f1, f2):
    return np.mean(cv2.absdiff(f1, f2))

def histogram_distance(h1, h2):
    sqr_diff = (h1 - h2) ** 2
    den = h1 + h2 + 0.00000001
    return np.sum(sqr_diff / den)


def edge_change_ratio(f1, f2):
    kernel = np.ones((5,5), np.uint8)

    e1 = f1 > 0
    e2 = f2 > 0
    d1 = cv2.dilate(f1, kernel) > 0
    d2 = cv2.dilate(f2, kernel) > 0

    n1 = np.sum(e1)
    n2 = np.sum(e2)

    if n1 == 0 or n2 == 0:
        return 0.0 if n1 == n2 else 1.0

    x_in = np.sum(e2 & ~d1)
    x_out = np.sum(e1 & ~d2)
    return max(x_in / n2, x_out / n1)

# aux for tasks 1 and 2
def compute_signals(frames):
    gray = []
    hist = []
    edges = []
    for n in frames:
        gray.append(cv2.cvtColor(n, cv2.COLOR_BGR2GRAY))
        edges.append(cv2.Canny(gray[-1], 100, 200))      # edge map, computed once per frame
        # colour histogram of the BGR frame, 8x8x8 bins (RGB turned out better than HSV on the dev videos)
        h = cv2.calcHist([n], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        h = h / h.sum()
        hist.append(h)

    d_pixel = [0]
    d_hist = [0]
    d_edge = [0]
    for t in range(1, len(gray)):
        d_pixel.append(pixel_difference(gray[t - 1], gray[t]))
        d_hist.append(histogram_distance(hist[t - 1], hist[t]))
        d_edge.append(edge_change_ratio(edges[t - 1], edges[t]))

    return d_pixel, d_hist, d_edge

# aux for task 3
def detect_cuts(signal, T, w):
    cuts = []
    for t in range(1, len(signal)):
        neighbours = signal[max(0, t - w): t + w + 1]
        if signal[t] > T and signal[t] == max(neighbours):
            cuts.append(t)
    return cuts

# aux for task 3
def combine_cuts(cut_lists, min_votes):
    count = {}
    for cuts in cut_lists:
        for t in cuts:
            count[t] = count.get(t, 0) + 1
    return sorted(t for t, c in count.items() if c >= min_votes)

# aux to get precision, recall and f1 of predicted cuts
def evaluate_cuts(cuts_by_video, gts):
    counts = []
    for name, cuts in cuts_by_video.items():
        gt_cuts = [t["start"] for t in gts[name]["transitions"]]
        counts.append(metrics.match_cuts(gt_cuts, cuts))
    tp, fp, fn = (sum(c[k] for c in counts) for k in range(3))
    return metrics.precision_recall_f1(tp, fp, fn)

# one poin per treshold for a signal over videos
def sweep_threshold(signals, gts, thresholds, w=2):
    points = []
    for T in thresholds:
        cuts = {name: detect_cuts(s, T, w) for name, s in signals.items()}
        p, r, _ = evaluate_cuts(cuts, gts)
        points.append((T, p, r))
    return points

# threshold of best point
def best_threshold(points):
    scored = [(T, p, r, 2 * p * r / (p + r) if p + r > 0 else 0.0) for T, p, r in points]
    return max(scored, key=lambda x: x[3])

# aux for task 4
def verify_cuts(cuts, frames, g, min_matches):
    kept = []
    for t in cuts:
        a = max(0, t - g)
        b = min(len(frames) - 1, t + g)
        if scene_grouping.sift_similarity(frames[a], frames[b]) < min_matches:
            kept.append(t)
    return kept


def process_video(frames, gt_shots):
    """Return (transitions, scenes) for one video.

    transitions: list of {"type": "cut", "start": t, "end": t}, t = first frame of the new shot (Parts 1 and 2)
    scenes:      dict {str(shot id): label} for the ground-truth shots (Part 3)
    """
    transitions, scenes = [], {}
    # TODO: Parts 1, 2 and 3. Everything else (writing the output, evaluation, summary table) is done for you.
    d_pixel, d_hist, d_edge = compute_signals(frames)
    # frame tolerance number when choosing cuts
    w = 2
    cuts_pixel = detect_cuts(d_pixel, 47, w)
    cuts_hist  = detect_cuts(d_hist, 0.22, w)
    cuts_edge  = detect_cuts(d_edge, 0.6, w)

    cuts = combine_cuts([cuts_pixel, cuts_hist, cuts_edge], 2)
    cuts = verify_cuts(cuts, frames, 4, 20)
    transitions = [{"type": "cut", "start": t, "end": t} for t in cuts]

    # part 3
    scenes = scene_grouping.scenes_from_shots(frames, gt_shots)

    return transitions, scenes



if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}                                    # video name -> counts from metrics.evaluate_video
    for video_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4"))):
        name = os.path.splitext(os.path.basename(video_path))[0]
        gt = json.load(open(os.path.join(INPUT_DIR, name + ".json")))
        frames = read_frames(video_path)
        transitions, scenes = process_video(frames, gt["shots"])
        pred = {"video": name + ".mp4", "transitions": transitions, "scenes": scenes}
        with open(os.path.join(OUTPUT_DIR, name + ".json"), "w") as f:
            json.dump(pred, f, indent=1, default=int)    # default=int: accepts NumPy integers
        results[name] = metrics.evaluate_video(gt, pred)

    # summary table (Section 8): one row per video, then the dev / test totals
    dev = [results[n] for n in sorted(results) if n.startswith("dev")]
    test = [results[n] for n in sorted(results) if n.startswith("test")]
    metrics.print_table([(n, results[n]) for n in sorted(results)]
                        + [("DEV", metrics.sum_counts(dev)), ("TEST", metrics.sum_counts(test))])
