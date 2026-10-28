"""Scene grouping (Part 3: Recurring scenes)

Groups ground-truth shots into recurring scenes by:
  1. Extracting keyframes per shot
  2. Computing pairwise keyframe similarity (histogram + SIFT)
  3. Grouping shots with connected components
"""
import cv2 as cv
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 1. CONFIG — Every tunable parameter
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "method":      "hist",          # "hist" | "sift"
    "kf_strategy": "middle",        # "middle" | "uniform"
    "K":           3,               # keyframes per shot
    "aggregator":  "max",           # "max" | "mean" | "vote"
    "bins": (8, 8, 8),              # HSV 3D hist
    "lowe_ratio":  0.75,
    "nfeatures":   500,
    "use_ransac":   False,
    "transform":    "homography",   # "translation" | "similarity" | "affine" | "homography"
    "combined_alpha": 0.5,
    "threshold":   0.5,             # grouping threshold
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

    def __init__(self, nfeatures=200):
        self.nfeatures = nfeatures
        self._sift = cv.SIFT_create(nfeatures=nfeatures)

    @classmethod
    def get(cls, nfeatures=200):
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
            # no OpenCV RANSAC for 2-DOF — median of displacements (robust estimator)
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
                        ratio=0.75, nfeatures=500):
    """Weighted combination of color and SIFT similarities, in [0, 1]."""
    h = histogram_similarity(kf1, kf2, bins=bins)

    matcher = SIFTMatcher.get(nfeatures)
    kp1, des1 = matcher.detect(kf1)
    kp2, des2 = matcher.detect(kf2)
    s = matcher.match_count(kp1, des1, kp2, des2, ratio)

    n_min = min(len(kp1), len(kp2))
    s_norm = s / n_min if n_min > 0 else 0.0

    return float(alpha * h + (1.0 - alpha) * min(s_norm, 1.0))

# ═══════════════════════════════════════════════════════════════
# 4. Build Matrix
# ═══════════════════════════════════════════════════════════════

def build_similarity_matrix(frames, shots, config=None):
    ...

# ═══════════════════════════════════════════════════════════════
# 5. Entry Point
# ═══════════════════════════════════════════════════════════════

def scenes_from_shots(frames, gt_shots, config=None):
    ...
