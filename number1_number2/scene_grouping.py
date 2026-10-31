"""Scene grouping (Part 3: Recurring scenes)

Groups ground-truth shots into recurring scenes by:
  1. Extracting keyframes per shot
  2. Computing pairwise keyframe similarity (histogram + SIFT)
  3. Grouping shots with connected components
"""
import cv2 as cv
import numpy as np
import grouping

# ═══════════════════════════════════════════════════════════════
# 1. CONFIG — Every tunable parameter
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "method":      "hist",          # "hist" | "sift" | "combined"
    "kf_strategy": "uniform",        # "middle" | "uniform"
    "K":           5,               # keyframes per shot
    "aggregator":  "max",           # "max" | "mean" | "vote"
    "bins": (8, 8, 8),              # HSV 3D hist
    "lowe_ratio":  0.75,
    "nfeatures":   500,
    "use_ransac":   False,
    "transform":    "homography",   # "translation" | "similarity" | "affine" | "homography"
    "combined_alpha": 0.5,
    "threshold":   0.87,             # grouping threshold
}

# ═══════════════════════════════════════════════════════════════
# 2. Shot representation
# ═══════════════════════════════════════════════════════════════

def extract_keyframes(frames, shot, strategy="middle", K=3):
    """Return a list of BGR frames representing a shot."""
    s, e = shot["start"], shot["end"]
    if strategy == "middle":
        mid = (s + e) // 2
        return [frames[mid]]
    if strategy == "uniform":
        idxs = np.linspace(s, e, num=K).round().astype(int)
        return [frames[i] for i in np.unique(idxs)]
    raise ValueError(f"Unknown strategy: {strategy}")


def aggregate_similarities(sims, aggregator="max"):
    """Combine pairwise keyframe similarities into one value."""
    if not sims:
        return 0.0
    if aggregator == "max":
        return float(max(sims))
    if aggregator == "mean":
        return float(sum(sims) / len(sims))
    if aggregator == "vote":
        return float(sum(1 for s in sims if s > 0.5))
    raise ValueError(f"Unknown aggregator: {aggregator!r}")

# ═══════════════════════════════════════════════════════════════
# 3. Shot similarity
# ═══════════════════════════════════════════════════════════════
def _colour_histogram(bgr, space="hsv", bins=(8, 8, 8)):
    """Normalized 3D histogram in the given color space."""
    if space == "hsv":
        converted = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        ranges = [0, 180, 0, 256, 0, 256]
    elif space == "lab":
        converted = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        ranges = [0, 256, 0, 256, 0, 256]
    elif space == "rgb":
        converted = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        ranges = [0, 256, 0, 256, 0, 256]
    else:
        raise ValueError(f"Unknown colour space: {space!r}")

    hist = cv.calcHist([converted], [0, 1, 2], None, list(bins), ranges)
    hist = hist.flatten()
    total = hist.sum()
    return hist / total if total > 0 else hist


def _histogram_similarity_from_hist(h1, h2):
    """1 − χ² similarity between two normalized histograms, in [0, 1]."""
    denom = h1 + h2 + 1e-10
    chi2 = 0.5 * np.sum((h1 - h2) ** 2 / denom)
    return float(np.clip(1.0 - chi2, 0.0, 1.0))


def histogram_similarity(kf1, kf2, bins=(8, 8, 8)):
    """Similarity in [0,1] based on 1 - chi-square distance of 3D HSV histograms."""
    return _histogram_similarity_from_hist(
        _colour_histogram(kf1, "hsv", bins),
        _colour_histogram(kf2, "hsv", bins),
    )

def lab_similarity(kf1, kf2, bins=(8, 8, 8)):
    """Similarity in [0,1] based on 1 - chi-square distance of 3D Lab histograms."""
    return _histogram_similarity_from_hist(
        _colour_histogram(kf1, "lab", bins),
        _colour_histogram(kf2, "lab", bins),
    )


def rgb_similarity(kf1, kf2, bins=(8, 8, 8)):
    """Similarity in [0,1] based on 1 - chi-square distance of 3D RGB histograms."""
    return _histogram_similarity_from_hist(
        _colour_histogram(kf1, "rgb", bins),
        _colour_histogram(kf2, "rgb", bins),
    )

class SIFTMatcher:
    """SIFT-based keyframe matcher with cached detector."""

    _instances = {}

    def __init__(self, nfeatures=500):
        self.nfeatures = nfeatures
        self._sift = cv.SIFT_create(nfeatures=nfeatures)

    @classmethod
    def get(cls, nfeatures=500):
        """Return a cached SIFTMatcher for these parameters."""
        if nfeatures not in cls._instances:
            cls._instances[nfeatures] = cls(nfeatures)
        return cls._instances[nfeatures]

    def detect(self, kf):
        """Return (keypoints, descriptors) of a keyframe."""
        gray = cv.cvtColor(kf, cv.COLOR_BGR2GRAY)
        return self._sift.detectAndCompute(gray, None)

    @staticmethod
    def matches(kp1, des1, kp2, des2, ratio=0.75):
        """Match list surviving Lowe's ratio test."""
        if des1 is None or des2 is None:
            return []
        knn = cv.BFMatcher(cv.NORM_L2).knnMatch(des1, des2, k=2)
        return [pair[0] for pair in knn
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]

    def match_count(self, kp1, des1, kp2, des2, ratio=0.75,
                use_ransac=False, transform="homography"):
        """Count matches, optionally filtered by RANSAC inliers of a geometric model."""
        good = self.matches(kp1, des1, kp2, des2, ratio)
        if not use_ransac:
            return len(good)
        return self._ransac_inlier_count(good, kp1, kp2, transform=transform)

    @staticmethod
    def _ransac_inlier_count(matches, kp1, kp2, transform="homography", reproj=3.0):
        """Inliers of a geometric model fitted with RANSAC."""
        if len(matches) < 4:
            return 0
        src = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst = np.float32([kp2[m.trainIdx].pt for m in matches])

        if transform == "homography":
            _, mask = cv.findHomography(src, dst, cv.RANSAC, reproj)
        elif transform == "affine":
            _, mask = cv.estimateAffine2D(src, dst, method=cv.RANSAC,
                                           ransacReprojThreshold=reproj)
        elif transform == "similarity":
            _, mask = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC,
                                                  ransacReprojThreshold=reproj)
        elif transform == "translation":
            dx = dst[:, 0] - src[:, 0]
            dy = dst[:, 1] - src[:, 1]
            tx, ty = np.median(dx), np.median(dy)
            residual = np.sqrt((dx - tx) ** 2 + (dy - ty) ** 2)
            mask = (residual < reproj).astype(np.uint8)
        else:
            raise ValueError(f"Unknown transform: {transform!r}")

        return int(mask.sum()) if mask is not None else 0

    def similarity(self, kf1, kf2, ratio=0.75, use_ransac=False, transform="homography"):
        """Number of SIFT matches between two keyframes."""
        kp1, des1 = self.detect(kf1)
        kp2, des2 = self.detect(kf2)
        return self.match_count(kp1, des1, kp2, des2, ratio, use_ransac, transform)


def sift_similarity(kf1, kf2, ratio=0.75, nfeatures=500,
                    use_ransac=False, transform="homography"):
    """Number of SIFT matches between two keyframes."""
    return SIFTMatcher.get(nfeatures).similarity(
        kf1, kf2, ratio=ratio, use_ransac=use_ransac, transform=transform
    )

def combined_similarity(kf1, kf2, alpha=0.5, bins=(8, 8, 8),
                        ratio=0.75, nfeatures=500,
                        use_ransac=False, transform="homography",
                        kp_des1=None, kp_des2=None):
    """Weighted combination of color and SIFT similarities, in [0, 1]."""
    h = histogram_similarity(kf1, kf2, bins=bins) if alpha > 0 else 0.0
    if alpha >= 1.0:
        return float(h)

    matcher = SIFTMatcher.get(nfeatures)
    kp1, des1 = kp_des1 if kp_des1 is not None else matcher.detect(kf1)
    kp2, des2 = kp_des2 if kp_des2 is not None else matcher.detect(kf2)
    s = matcher.match_count(kp1, des1, kp2, des2, ratio, use_ransac, transform)

    n_min = min(len(kp1), len(kp2))
    s_norm = s / n_min if n_min > 0 else 0.0

    return float(alpha * h + (1.0 - alpha) * min(s_norm, 1.0))

# ═══════════════════════════════════════════════════════════════
# 4. Build Matrix
# ═══════════════════════════════════════════════════════════════

def _pairwise_keyframe_similarities(kfs_a, kfs_b, sift_a, sift_b, cfg):
    """Return the list of pairwise similarities between two shots' keyframes."""
    method = cfg["method"]
    bins = cfg["bins"]
    ratio = cfg["lowe_ratio"]
    use_ransac = cfg["use_ransac"]
    transform = cfg["transform"]
    alpha = cfg["combined_alpha"]

    if method in ("hist", "lab", "rgb"):
        sim_fn = {
            "hist": histogram_similarity,
            "lab": lab_similarity,
            "rgb": rgb_similarity,
        }[method]
        return [sim_fn(kfa, kfb, bins=bins) for kfa in kfs_a for kfb in kfs_b]

    # sift / combined: reuse the pre-detected descriptors
    matcher = SIFTMatcher.get(cfg["nfeatures"])
    values = []
    for idx_a, kfa in enumerate(kfs_a):
        kp_a, des_a = sift_a[idx_a]
        for idx_b, kfb in enumerate(kfs_b):
            kp_b, des_b = sift_b[idx_b]
            if method == "sift":
                s = matcher.match_count(kp_a, des_a, kp_b, des_b,
                                        ratio, use_ransac, transform)
                values.append(float(s))
            else:  # combined
                values.append(combined_similarity(
                    kfa, kfb,
                    alpha=alpha, bins=bins, ratio=ratio,
                    nfeatures=cfg["nfeatures"],
                    use_ransac=use_ransac, transform=transform,
                    kp_des1=(kp_a, des_a), kp_des2=(kp_b, des_b),
                ))
    return values


def build_similarity_matrix(frames, shots, config=None):
    """Compute the shot x shot similarity matrix."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    n = len(shots)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    method = cfg["method"]
    if method not in ("hist", "lab", "rgb", "sift", "combined"):
        raise ValueError(f"Unknown method: {method!r}")

    kfs_per_shot = [
        extract_keyframes(frames, shot, cfg["kf_strategy"], K=cfg["K"])
        for shot in shots
    ]

    sift_kp_des = None
    if method in ("sift", "combined"):
        matcher = SIFTMatcher.get(cfg["nfeatures"])
        sift_kp_des = [
            [matcher.detect(kf) for kf in kfs]
            for kfs in kfs_per_shot
        ]

    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            pair_values = _pairwise_keyframe_similarities(
                kfs_per_shot[i], kfs_per_shot[j],
                sift_kp_des[i] if sift_kp_des else None,
                sift_kp_des[j] if sift_kp_des else None,
                cfg,
            )
            v = aggregate_similarities(pair_values, cfg["aggregator"])
            sim[i, j] = v
            sim[j, i] = v
    return sim


# ═══════════════════════════════════════════════════════════════
# 5. Entry Point
# ═══════════════════════════════════════════════════════════════

def scenes_from_shots(frames, gt_shots, config=None):
    """Group the given shots into recurring scenes."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    sim = build_similarity_matrix(frames, gt_shots, cfg)
    labels = grouping.group_shots(sim, cfg["threshold"])
    return {str(s["id"]): int(l) for s, l in zip(gt_shots, labels)}
