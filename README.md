# Computer Vision 2026-2027 - NOVA FCT

## Project Stage 1 - Shot boundary detection and recurring scenes

**Authors**:
- 62746, Guilherme Maurício, gb.mauricio@campus.fct.unl.pt
- 76904, Guilherme Belo, p213014@campus.fct.unl.pt

## 1. Introduction

In this project we have videos made of shots separated by cuts, with potential recurring scenes. The main goals of this preoject are to find the cuts and group the shots that show the same scene - this is difficult because there are random flashes, fast camera movement and different objects/camera perspectives.
We were given 4 videos, 2 for development and 2 for testing. We used the dev videos to choose and adjust the parameters and the test ones for testing. The project was divided in three parts: **Frame dissimilarity and hard cuts**, **Gradual transitions and false alarms** and **Recurring scenes**.

## 2. Part 1 - Frame dissimilarity and hard cuts

### 2.1 Task 1: Reading and preparing frames

To read the frames we used **`cv2.VideoCapture`** and resized every frame to **320x180** with `INTER_AREA`, as initially suggested by the teachers. This keeps the 16:9 aspect ratio, and since a hard cut changes the whole image, fine detail is not needed (averaging pixels when shrinking also reduces noise). To check the choice we compared it with other sizes:

- **Speed:** the *read time* in the table below (decoding the video and resizing each frame) is about the same at every size, between **7.6 s** and **10.1 s** for the two dev videos with no trend, because the full-resolution frame has to be decoded anyway. What does depend on the size is the time to compute the signals, which grows with the number of pixels: **1.5 s** at **80x45**, **9.2 s** at **320x180** and **34.4 s** at **640x360**. So going below **320x180** saves only **5 to 8 s**, while **640x360** costs about **25 s more**. The complete program (the four videos, including the verification of Task 4 and the scene grouping of Part 3) runs in about **80 s**, measured with `time python tp1.py`, well within the 5-minute limit.
- **Accuracy:** the best F1 of the pixel signal (about **0.86**) and of the histogram signal (**0.896**) is practically the same at every size, so they do not need more than a small frame. The ECR is the exception: it is best at **320x180** (F1 **0.657**) and worse at both smaller (**0.550** at **80x45** and **0.618** at **160x90**) and larger (**0.579** at **640x360**) sizes. A likely reason is that its Canny thresholds and the 5x5 dilation are fixed in pixels, so the same settings mean different things at different resolutions (we did not test this).

Since going smaller saves little time and hurts the ECR, and going larger only makes the signals slower, we kept **320x180**.

Result of the resolution experiment on the dev videos (`part1_exp.table_resolution()`; for each size, the threshold `T` of each signal was swept and the best `T` and its F1 are shown; the read time varies from run to run, as explained above):

```python
import sys

sys.path.insert(0, "62746_76904/experiments")

import tp1, metrics, scene_grouping as sg
import part1_exp, part2_exp, part3_exp
from render import show

```


```python
show(part1_exp.table_resolution())
```


| Size | Read time (s) | Signals time (s) | Pixel: T / F1 | Histogram: T / F1 | ECR: T / F1 |
|---|---|---|---|---|---|
| 80x45 | 9.1 | 1.5 | 43.136 / 0.870 | 0.246 / 0.896 | 0.340 / 0.550 |
| 160x90 | 10.1 | 4.0 | 45.678 / 0.863 | 0.221 / 0.896 | 0.614 / 0.618 |
| **320x180** | 7.6 | 9.2 | 46.949 / 0.863 | 0.221 / 0.896 | 0.597 / 0.657 |
| 640x360 | 9.3 | 34.4 | 46.949 / 0.863 | 0.221 / 0.896 | 0.807 / 0.579 |


### 2.2 Task 2: Three dissimilarity signals

For every frame we compute one number, d(t), that says how different frame t is from frame t-1 (d(0) = 0). We tried three ways of measuring it.

**Pixel difference.** The mean absolute difference between the two frames in grey levels (`cv2.absdiff` followed by `np.mean`). It is the simplest option and our baseline. Its weakness is that any change in brightness moves every pixel, so a flash looks like a cut.

**Histogram distance.** Each frame is summarised by a colour histogram, normalised so that its bins add up to 1, and two frames are compared with the χ² distance, the sum over all bins of (h1 - h2)² / (h1 + h2) (we add a tiny ε to the denominator so empty bins do not divide by zero). It goes from 0 (same colours) to 2 (nothing in common). A histogram does not care where things are in the image, so camera movement hardly changes it, but two shots with similar colours can look the same to it. We first expected the HSV space to work better than RGB, because separating the colour (hue and saturation) from the brightness should make the histogram less sensitive to lighting changes such as flashes, so we started with the hue and saturation of HSV with 8x8 bins. The next table compares that first choice with RGB and with other numbers of bins (dev videos; for each histogram we swept the threshold and show the best F1, and how many of the 12 flashes cause a false alarm at that threshold):


```python
show(part1_exp.table_colour_space())
```


| Histogram | best T | cut-P | cut-R | cut-F1 | flash-FA |
|---|---|---|---|---|---|
| **RGB, 8x8x8** | 0.221 | 0.821 | 0.986 | 0.896 | 12/12 |
| HSV (H,S,V), 8x8x8 | 0.522 | 0.784 | 0.986 | 0.873 | 12/12 |
| HSV (H,S), 4x4 | 0.171 | 0.793 | 0.986 | 0.879 | 11/12 |
| HSV (H,S), 8x8 | 0.271 | 0.773 | 0.971 | 0.861 | 12/12 |
| HSV (H,S), 16x16 | 0.672 | 0.823 | 0.929 | 0.872 | 12/12 |


Our expectation did not hold. RGB with 8x8x8 bins gives the best F1 on the dev videos (**0.896**), and our first choice, HSV (H,S) with 8x8 bins, is the lowest of the five (**0.861**). HSV also did not reduce the flash false alarms: with every histogram almost all flashes still cause a false alarm at the best threshold (**11/12 or 12/12**). Because the difference could be just a couple of cuts, we also tested it in the whole pipeline (2-of-3 vote plus the verification of Task 4, with the threshold of the histogram retuned, on the dev videos only): the F1 went from **0.971** with HSV to **0.979** with RGB. So we changed the histogram to **RGB with 8x8x8 bins** (threshold 0.22), and everything that follows uses it. The flashes are handled by the verification of Task 4, not by the colour space.

**Edge Change Ratio (ECR).** The idea is that inside a shot the edges of the image only move a few pixels, so almost every edge of frame t is close to an edge of frame t-1, while after a cut the edges are in unrelated places. We compute the Canny edges of both grey frames (thresholds 100 and 200), dilate each edge map with a 5x5 kernel, and count the edges of frame t that are not near any edge of frame t-1 (entering) and the edges of frame t-1 that are not near any edge of frame t (exiting). Each count is divided by the number of edges of its own frame and the ECR is the larger of the two ratios, so it goes from 0 to 1. If both frames have no edges we return 0 (nothing changed), and if only one of them has none we return 1 (everything appeared or disappeared). The Canny thresholds and the kernel size were set by hand and not tuned. The figure shows the method on two pairs of frames of `dev_01`:

![ECR on two pairs of consecutive frames of dev_01](./images/t2_ecr_example_dev_01.png)

*Figure — The Edge Change Ratio on two pairs of consecutive frames of `dev_01`. Right column: dilated edges of frame t-1 in grey; edges of frame t in green when they fall inside them (they moved little) and in red when they do not (entering edges). Top, inside a shot (frames 103 and 104): most edges stay green and ECR = 0.28 (the camera is moving). Bottom, across the first cut (frames 73 and 74): almost every edge is entering and ECR = 0.94.*

![Three signals over about 30 s of dev_01](./images/t3_signals_dev_01.png)

*Figure — Pixel difference, histogram χ² and ECR over frames 500 to 1250 of `dev_01` (about 30 s). Red dashed lines are the ground-truth cuts and orange bands the flashes. All three signals spike at every cut, and also at the two flashes (frames 860 and 958 to 960); the largest pixel-difference spike is the flash at 860. The ECR is noisier than the other two between frames 700 and 800.*

All three signals peak at every cut, so each of them can detect cuts with a threshold. They also peak at the flashes (orange bands), which none of them ignores, and the biggest spike of the pixel difference is precisely a flash. The ECR is noticeably noisier than the other two in some stretches, for example between frames 700 and 800, probably because of camera movement (we have not looked at those frames yet).

### 2.3 Task 3: From signal to cuts

We declare a cut at frame t when d(t) > T and d(t) is also the largest value in the window t-2 ... t+2. The second condition is there because one real cut often raises two or three consecutive values and we want to report it only once; the window of 2 frames is the same as the tolerance of the evaluation. T is chosen separately for each signal, because their ranges are very different: we tried 60 values on the dev videos, scored each one with `match_cuts` (adding the counts of the two videos before computing precision, recall and F1) and kept the one with the best F1. The last row of the table combines the signals: a cut is kept when at least two of the three signals detect it at the same frame, each one with its own threshold (rounded to 47, 0.22 and 0.6).


```python
show(part1_exp.table_thresholds())
```


| Signal | T | cut-P | cut-R | cut-F1 |
|---|---|---|---|---|
| pixel | 46.949 | 0.795 | 0.943 | 0.863 |
| histogram | 0.221 | 0.821 | 0.986 | 0.896 |
| ecr | 0.597 | 0.657 | 0.657 | 0.657 |
| vote 2 of 3 | 47 / 0.22 / 0.6 | 0.841 | 0.986 | 0.908 |


![Precision-recall curves of the three signals](./images/t3_pr_curve_dev.png)

*Figure — Precision-recall curves obtained by sweeping the threshold of each signal on the dev videos. The circles mark the threshold with the best F1: pixel T = 46.9 (F1 = 0.86), histogram T = 0.221 (F1 = 0.90) and ECR T = 0.597 (F1 = 0.66).*

The histogram is the best single signal (F1 **0.896**), the pixel difference is close behind (**0.863**) and the ECR is clearly worse (**0.657**). Voting two out of three is better than any signal alone (F1 **0.908**, recall **0.986**). In all cases the recall is high but the precision is limited (**0.841** for the vote): the PR curves show that to find every cut we have to accept many false alarms, and a lot of them are flashes, since with no extra check all **12/12** flashes of the dev videos cause a false detection. That is what Task 4 tries to fix.

## 3. Part 2 - Gradual transitions and false alarms

### 3.1 Task 4: Verification with local features


Many false alarms of Part 1 happen inside a single shot: a flash goes off, the camera moves, something crosses the frame. In those cases the scene before and after the candidate is the same one, so we check each candidate with local features. For a candidate cut at frame t we take frame t-4 and frame t+4 (the gap of 4 is there because a flash lasts up to 3 frames, so neither of the two frames can be a flash frame), detect SIFT keypoints (at most 500 per frame) and match them with Lowe's ratio test (ratio 0.75). If the two frames have at least `min_matches` = 20 matches, they show the same content and the candidate is rejected; if they have fewer, it is kept as a cut. We did not use RANSAC. The matching is the code of Part 3 (`scene_grouping.sift_similarity`).


```python
show(part2_exp.table_verification())
```


| Set | Verification | cut-P | cut-R | cut-F1 | flash-FA |
|---|---|---|---|---|---|
| DEV | without | 0.841 | 0.986 | 0.908 | 12/12 |
| DEV | with | 0.972 | 0.986 | 0.979 | 1/12 |
| TEST | without | 0.826 | 0.974 | 0.894 | 12/12 |
| TEST | with | 0.987 | 0.962 | 0.974 | 1/12 |


With verification the precision goes from **0.841** to **0.972** on DEV (**0.826** to **0.987** on TEST) and the F1 from **0.908** to **0.979** (**0.894** to **0.974**), and the number of flashes that cause a false alarm falls from **12/12** to **1/12** on both sets. The price is a small drop of the recall on TEST (**0.974** to **0.962**): a real cut between two shots that look alike can share enough matches to be rejected.


```python
show(part2_exp.table_min_matches())
```


| min_matches | cut-P | cut-R | cut-F1 | flash-FA |
|---|---|---|---|---|
| 1 | 1.000 | 0.214 | 0.353 | 0/12 |
| 5 | 0.962 | 0.714 | 0.820 | 1/12 |
| 10 | 0.970 | 0.929 | 0.949 | 1/12 |
| 15 | 0.972 | 0.986 | 0.979 | 1/12 |
| **20** | 0.972 | 0.986 | 0.979 | 1/12 |
| 25 | 0.972 | 0.986 | 0.979 | 1/12 |
| 30 | 0.972 | 0.986 | 0.979 | 1/12 |
| 40 | 0.972 | 0.986 | 0.979 | 1/12 |
| 50 | 0.972 | 0.986 | 0.979 | 1/12 |
| 75 | 0.932 | 0.986 | 0.958 | 4/12 |
| 100 | 0.920 | 0.986 | 0.952 | 5/12 |
| no verification | 0.841 | 0.986 | 0.908 | 12/12 |


`min_matches` was chosen on the dev videos. If it is too low (1 to 10) real cuts are rejected and the recall collapses, and if it is too high (75 or more) the flashes get through again. In between, from 15 to 50, the result does not change (F1 **0.979**), so the exact value is not critical, and we picked **20**, which is inside that range. The figure below shows one candidate that was rejected (a flash) and one that was kept (a real cut).

![A false alarm removed by verification](./images/t4_false_alarm_removed_dev_01.png)

*Figure — Verification of two candidate cuts of `dev_01`, showing frames t-4 and t+4 with the SIFT matches that pass Lowe's ratio test. Top: the candidate at t = 860 is a flash (frames 860 and 861); frames 856 and 864 share 218 matches, so the candidate is rejected. Bottom: the candidate at t = 74 is a real cut; frames 70 and 78 share 2 matches, so it is kept (`min_matches` = 20).*

## 4. Part 3 - Recurring scenes

### 4.1 Task 5: Shot representation by keyframes

Each shot has tens to hundreds of frames. Comparing every frame of one shot
with every frame of another is inefficient, so we reduce each shot to a small 
set of **keyframes**:

| Strategy | Rule | # keyframes |
|---|---|---|
| `middle` | `frames[(start + end) // 2]` | 1 |
| `uniform` | `K` frames evenly spaced via `np.linspace(start, end, K)` | K (3–5) |

`np.unique` on the indices avoids duplicates when a shot is shorter than `K`
(e.g. a 3-frame shot with K=10 returns only 3 keyframes).

**Choice of K.** Two constraints bound the initial range, chosen as a
*prior* before any results are seen:

- **Coverage** A shot usually has three visually distinct
  phases.
  K=3 is the smallest value that samples all three
  and K<3 has no redundancy against an outlier keyframe. This is a
  hypothesis about how shots are structured; the K-sweep of
  Section 5.1 tests it directly.
- **Cost.** The number of keyframe pairs between two shots grows as
  K², making SIFT much slower as K grows and risking the 5-minute
  budget.

We started with **K=3** and tune it in the K-sweep of Section 5.1
(K ∈ {1, 3, 5, 8}). As Section 5.1.1 shows, the "three phases"
heuristic is **not** confirmed by the data. K=1 already ties K=3 on
mean F1 and the sweep selects K=5 for a different reason: its plateau 
is the smallest K whose plateau passes the robustness criterion introduced 
in Section 5.1.4.

**Aggregating keyframe similarities.** For two shots with `K_a` and `K_b`
keyframes we get `K_a × K_b` pairwise similarities. These are collapsed into
one shot-to-shot value with:

- **`max`** — the strongest pair and the *Default for SIFT*
- **`mean`** — average
- **`vote`** — count of pairs with similarity > 0.5

The final choice is validated empirically in two places: the aggregator
sweep of Section 5.1.2 (which isolates the aggregator with everything
else fixed) and the 2×2 comparison of Section 5.2 (which tests the
aggregator inside the full pipeline).

![Middle vs Uniform keyframes](./images/t5_compare_shot0_dev_01.png)

*Figure 1 — Shot 0 of `dev_01` represented by `middle` (left, 1 frame) and `uniform K=3` (right, 3 frames). The uniform strategy samples the shot across its visual phases; the middle strategy captures only the central state. The K-sweep in 5.1 measures whether the extra coverage translates into better discrimination.*

**Implementation validation.** Two families of tests were run before
integrating the function into the pipeline:

- `aggregate_similarities` — 9 input→output cases covering the three
  aggregators across three list shapes (3-element, empty, 1-element),
  plus an unknown-aggregator case that must raise `ValueError`. All
  passed.
- `extract_keyframes` — two edge cases: (i) a 3-frame shot with `K=10`
  returns only 3 keyframes, confirming `np.unique` caps the output;
  (ii) an unknown strategy raises `ValueError` rather than silently
  returning `None`. Both passed.

### 4.2 Task 6: Shot similarity

#### 4.2.1 Colour similarity

Each keyframe is described by a 3D histogram. Similarity is one minus the
χ² distance:

$$
\chi^2(h_1, h_2) = \tfrac{1}{2}\sum_i \frac{(h_1[i] - h_2[i])^2}{h_1[i] + h_2[i]},
\qquad \mathrm{sim} = 1 - \chi^2 \in [0, 1]
$$

Both histograms are L1-normalised, so χ² ∈ [0, 1] and sim = 1
iff the distributions are identical. A small ε = 1e-10 in the denominator
avoids 0/0 on empty bins.

**Bin resolution** Each frame has 320 × 180 = 57 600
pixels, and a 3D histogram with `B` bins has on average 57 600 / B pixels
per bin. We sweep over four resolutions on both
development videos to choose the optimal number of bins:


```python
show(part3_exp.table_bins_single())
```


| Video | bins | same | diff | gap | pixels/bin |
|---|---|---|---|---|---|
| dev_01 | 8×8×8 | 0.5900 | 0.3777 | 0.2123 | 112.5 |
| dev_01 | 16×16×16 | 0.5079 | 0.3023 | 0.2056 | 14.1 |
| dev_01 | 30×32×32 | 0.4413 | 0.2598 | 0.1815 | 1.9 |
| dev_01 | 60×64×64 | 0.3639 | 0.2324 | 0.1316 | 0.23 |
| dev_02 | 8×8×8 | 0.8724 | 0.5862 | 0.2862 | 112.5 |
| dev_02 | 16×16×16 | 0.7874 | 0.4943 | 0.2931 | 14.1 |
| dev_02 | 30×32×32 | 0.6973 | 0.4221 | 0.2751 | 1.9 |
| dev_02 | 60×64×64 | 0.5677 | 0.3296 | 0.2381 | 0.23 |



The two videos differ in the shape of the curve. On dev_01 the gap
shrinks monotonically from `8³` onwards (0.2123 → 0.2056 → 0.1815 →
0.1316). On dev_02 it rises slightly from `8³` to `16³` (0.2862 →
0.2931) and then shrinks from `16³` onwards (0.2931 → 0.2751 →
0.2381). `8³` and `16³` are **practically
tied** on the mean gap (0.2493 vs 0.2494, a difference of 0.0001 —
well below any meaningful threshold) and we adopt `8³` because it
is cheaper to compute than `16³`.

`16³` is kept as a candidate to re-evaluate in the bin-resolution sweep
(Section 5.1.5), where every pair of shots is evaluated instead of a
single pair.

**Colour space: HSV vs. RGB.** The theoretical argument for HSV is that a
flash brightens the three RGB channels together (shifting every bin)
while in HSV only V shifts and H and S stay, so the metric is robust to
illumination changes. On the single test pair, the gap supports this
expectation:



```python
show(part3_exp.table_colour_space_single(include_lab=False))
```


| Video | Space | same | diff | gap |
|---|---|---|---|---|
| dev_01 | HSV | 0.5900 | 0.3777 | 0.2123 |
| dev_01 | RGB | 0.7529 | 0.6740 | 0.0789 |
| dev_02 | HSV | 0.8724 | 0.5862 | 0.2862 |
| dev_02 | RGB | 0.9323 | 0.7688 | 0.1636 |



HSV's gap is 2.7× wider on dev_01 and 1.75× wider on dev_02 than RGB's.
**But the gap is measured on a single pair, and this is not a conclusive
test**: the final choice is decided at matrix level in Section 5.1.4,
where the F1 and the plateau width are compared across all pairs of
shots. The matrix-level result there is counter-intuitive (RGB actually
achieves a *higher* F1 than HSV) but reveals a fragility that the gap
alone cannot capture.

![Histogram similarity example](./images/t6_hist_example_dev_01.png)

*Figure 2 — Colour similarity of two keyframes from the same scene (`dev_01`, scene `S00`). Top: the two keyframes. Bottom: the marginal distributions of H, S and V.*

**Decision.** We adopt **HSV with `8³`** as the production colour
representation. This is not the strictly optimal choice on any single
metric. `16³` ties `8³` on the mean gap (see above) and Lab has a
larger gap on dev_01 (Section 8.2), but it is the configuration that
**combines a competitive gap on both videos with the lowest
computational cost among the tested variants and it is the most
consistent colour space across the two development videos** (Lab's gap
differs by 0.10 between dev_01 and dev_02; HSV's by only 0.07). The
final matrix-level comparison with Lab is performed in Section 5.1.4,
where F1 and plateau width are computed on the full shot × shot
matrix instead of a single pair.


#### 4.2.2 Local-feature similarity (SIFT)

The colour similarity of 4.2.1 captures the overall distribution of pixels
across HSV cells but is blind to spatial structure. Two images with
inverted layouts may share an almost identical histogram. 

**Pipeline.** For each keyframe pair:

1. Detect keypoints and compute 128-dimensional descriptors on the
   grayscale keyframe (`cv2.SIFT_create(nfeatures=500)`).
2. `BFMatcher(NORM_L2).knnMatch(des1, des2, k=2)`: for each descriptor
   in `kf1`, find the two nearest descriptors in `kf2`.
3. **Lowe's ratio test**: keep a match only if
   `d(NN1) / d(NN2) < 0.75`.
4. The **similarity** is the number of surviving matches, an unbounded
   integer. Aggregation of the K_a × K_b pairwise counts into a shot-to-
   shot value then depends on the aggregator:
   - `max` and `mean` operate on the raw counts and return a value in
     **count units**; `group_shots` is therefore tuned with a threshold
     in those same units (see the sift uniform K=5 row of Section 5.2).
   - `vote` uses a **fixed internal threshold of 0.5 on the similarity
     values**. That threshold is meaningful only on a `[0, 1]` scale
     (like the colour-histogram output). On raw SIFT counts, on the
     other hand, every pair with at least one match exceeds 0.5, so
     `vote` would count every positive pair and discard the magnitude
     of the match count, the very signal that makes raw SIFT
     discriminative (Section 5.1.3). `vote` would therefore require
     normalising the SIFT counts to `[0, 1]` first
     (e.g. `s / min(len(kp1), len(kp2))`, as in Section 8.1).
   - For these reasons SIFT is evaluated only with `max` (Section 5.2),
     keeping the aggregator consistent with the colour-histogram
     pipeline of Section 5.1.2. `mean` would work in principle but is
     not tested with SIFT in this report.

![SIFT matches example](./images/t6_sift_matches_dev_01.png)

*Figure 3: SIFT matches between two keyframes from the same scene
(`dev_01`, scene `S00`) after Lowe's ratio test, with `nfeatures=500`.
Lines connect descriptor correspondences; the number of surviving
matches (15) is the similarity metric used by `sift_similarity`.*

**Parameter choices:** `nfeatures=500` and `ratio=0.75`. Both justified
empirically in **8.4**.

### 4.3 Task 7: Scene grouping

The two similarity measures of Section 4.2 produce a scalar value for
each pair of shots. The final step is to turn
this **N × N similarity matrix** into a **partition of the N shots
into scenes**.

**Pipeline.** The grouping itself is a two-stage process:

1. **`build_similarity_matrix(frames, shots, config)`**: produces the
   N × N float32 matrix `sim[i, j]`, where `sim[i, i] = 1.0` and
   `sim[i, j] == sim[j, i]`. The matrix is populated by extracting
   keyframes per shot (Section 4.1), computing per-keyframe similarities
   with the selected method (Section 4.2) and aggregating the K_a × K_b
   pairwise values into a single shot-to-shot score.

2. **`grouping.group_shots(sim, threshold)`** (given in the starter
   code).

The public entry point `scenes_from_shots(frames, gt_shots, config)`
wraps both stages and returns `{str(shot_id): int_label}`, in the exact
format that `metrics.evaluate_video` expects.

**Complexity consideration.** The pre-detection of SIFT descriptors
inside `build_similarity_matrix` is the single largest optimisation in
the pipeline. The matrix is built **per video**, so the relevant N is
the number of shots in the video (N=37 for dev_01, N=35 for dev_02),
not the total across both videos. Taking the larger of the two (N=37)
with K=5 keyframes per shot, the naive approach would run
`N · N · K · K · 2` SIFT detections:

    37² × 5² × 2 = 68 450

By detecting each keyframe exactly **once** and reusing its descriptors
across every pair it participates in, the total drops to `N · K`
detections:

    37 × 5 = 185

a **×370 reduction**. This keeps the whole run well inside the 5-minute
budget, even when the SIFT-based methods are used. (For dev_02 the
numbers scale to 61 250 naive and 175 with the cache.)

**Threshold selection.** The choice of `threshold` is the single most
consequential decision of Task 7. A low threshold over-merges and a high 
threshold under-merges. Because the optimal threshold depends
on the similarity distribution produced by the selected method, we
**defer the decision to Section 5.1**, where it is tuned empirically
using the sweep procedure described below. **Section 5.2** compares
the 2×2 combination of {hist, sift} × {1, K} keyframes and
**Section 5.1.3** evaluates whether the combined similarity of
Section 8.1 provides any benefit over the isolated signals.

![Similarity matrix vs ground truth](./images/t7_matrix_hist_dev_01.png)

*Figure 4 — Predicted shot × shot similarity matrix (left) and
ground-truth same-scene matrix (right), for `dev_01` under the
production configuration.*


## 5. Results (development and test)

### 5.1 Hyperparameter tuning on the development set

#### 5.1.1 K-sweep

The initial K=3 (Section 4.1) was chosen from coverage/cost bounds
alone. To justify it empirically, we swept K ∈ {1, 3, 5, 8} using
`method="hist"` and additionally compared to the `middle` strategy 
(single centered frame):



```python
show(part3_exp.table_k_sweep())
```


| Strategy | K | d01_F1 | d02_F1 | mean F1 | width (d01/d02) | time |
|---|---|---|---|---|---|---|
| middle | 1 | 0.889 | 0.923 | 0.906 | 0.08 / 0.00 | 393 ms |
| uniform | 1 | 1.000 | 0.955 | 0.977 | 0.09 / 0.02 | 382 ms |
| uniform | 3 | 1.000 | 0.950 | 0.975 | 0.08 / 0.14 | 4.1 s |
| uniform | 5 | 1.000 | 0.976 | 0.988 | 0.14 / 0.11 | 8.7 s |
| uniform | 8 | 1.000 | 0.976 | 0.988 | 0.15 / 0.12 | 24.5 s |



The `width` column reports the width of the F1 plateau from the threshold
sweep. A wide plateau means the exact choice of T matters less, because a 
range of thresholds achieves the maximum F1.

**The F1 curve is not monotone and the coverage prior is not
confirmed.** Going from K=3 to K=1 actually *raises* mean F1 slightly
(0.975 → 0.977) and K=3 is worse than K=1 on dev_02 specifically
(0.950 vs 0.955). So the "three phases" argument of Section 4.1 is not
what the data supports: **a single keyframe at the start of the shot
is as informative as three evenly-spaced keyframes.** A plausible
explanation is that the first frame is an *establishing frame* (camera
settled, no motion), whereas mid-shot keyframes often contain moving
objects or occlusions.

**Where K=5 actually wins: plateau width.** Applying the robustness
criterion of Section 5.1.4, *F1 within 0.02 of the best, plateau
width ≥ 0.10 on both videos, then prefer the simplest*, to the four
uniform rows of the sweep:

| K | mean F1 | Within 0.02 of best (0.988)? | Min width | Passes? |
|---|---|---|---|---|
| 1 | 0.977 | ✓ (0.011 < 0.02) | **0.02** | ✗ |
| 3 | 0.975 | ✓ (0.013 < 0.02) | **0.08** | ✗ |
| 5 | 0.988 | ✓ (best) | 0.11 | ✓ |
| 8 | 0.988 | ✓ (best) | 0.12 | ✓ |

**K=1 and K=3 are excluded by the width requirement** (0.02 on dev_02
and 0.08 on dev_01, respectively), not by their mean F1. **K=5 and K=8 both pass; 
K=5 is chosen because it is the cheaper of the two** (8.7 s vs 24.5 s, a 2.8× cost 
difference for no measurable gain).

**Decision.** Move from K=3 to **K=5** in the production configuration. 
The cost (8.7 s per video) remains well within the 5-minute budget when 
combined with the fastest method (hist).

#### 5.1.2 Aggregator sweep (max / mean / vote)

For two shots with K_a and K_b keyframes, we obtain K_a × K_b pairwise
similarities. These must be collapsed
into a single shot-to-shot value. Three aggregators are available:

- **`max`** — the strongest pair. A single good keyframe pair is enough
  to link two shots from the same scene.
- **`mean`** — the arithmetic average across all K² pairs.
- **`vote`** — the count of pairs whose similarity exceeds a fixed internal
  threshold (0.5).

We swept the three aggregators on both development videos, keeping
`method="hist"`, `uniform K=5`, and `bins=8³` fixed:


```python
show(part3_exp.table_aggregator_sweep())
```


| Aggregator | d01 F1 | d02 F1 | mean F1 | width (d01/d02) |
|---|---|---|---|---|
| max | 1.000 | 0.976 | 0.988 | 0.14 / 0.11 |
| mean | 0.952 | 0.952 | 0.952 | 0.00 / 0.03 |
| vote | 0.119 | 0.708 | 0.413 | 3.79 / 2.78 |


**`max` has the highest F1 on both videos.** Mean F1 is 0.988 vs
0.952 (`mean`) and 0.413 (`vote`). Among the aggregators on the same
scale (`max` vs `mean`, both producing similarities in `[0, 1]`),
`max` also has the wider plateau (0.14/0.11 vs 0.00/0.03). The
`vote` plateau widths (3.79/2.78) are in integer-count units (0 to
K²=25) and are therefore **not comparable** to the `[0, 1]` scale.

**Decision.** `max` is retained in the production configuration. Its
correctness is supported by the theoretical argument (a single good
keyframe pair is sufficient to identify a recurring scene) and by
three empirical signals: (i) the **highest mean F1** (0.988 vs 0.952
and 0.413), (ii) the **widest plateau among the aggregators on the
same scale** (0.14/0.11 vs 0.00/0.03 for `mean`), and (iii) no
dependence on a hidden second threshold (unlike `vote`, which has an
internal 0.5 that would also need tuning).

#### 5.1.3 Combined similarity 

Section 8.1 evaluated the blend `α · h + (1−α) · ŝ` on a single test
pair and deferred the choice of α to the full matrix. We now sweep
α ∈ {0.00, 0.25, 0.50, 0.75, 1.00} on the shot × shot matrix, keeping
`uniform K=5`, `aggregator=max`, and `bins=8³` fixed. 
`α=1.00` reproduces the histogram-only configuration of Section 5.1.2 exactly,
which validates the implementation. 

The timing row for `α=1.00` (~11 s) is **not** comparable to the
`method="hist"` rows of Sections 5.1.4 and 5.2 (~9–10 s): in
`build_similarity_matrix`, SIFT descriptors are pre-detected
**whenever `method="combined"`, regardless of `α`**. With `α=1.0`
the similarity function returns immediately after the histogram, but
the SIFT detection cost has already been paid upstream. The ~11 s
therefore measures SIFT detection (which is unavoidable in the
`combined` path) plus the histogram, not the histogram alone; the true
histogram-only runtime is the ~9–10 s reported in 5.1.4 and 5.2.
The α=0.25–0.75 rows cost ~30 s because they pay for both SIFT detection
**and** the K_a × K_b keyframe matching. `α=1.0` avoids the matching
(it returns the histogram immediately); `α=0.0` avoids only the
histogram multiplication, not the matching, so it still pays the
full SIFT cost.



```python
show(part3_exp.table_alpha_sweep())
```


| α | d01 F1 | d02 F1 | mean F1 | d01 width | d02 width | time |
|---|---|---|---|---|---|---|
| 0.00 | 0.889 | 0.895 | 0.892 | 0.07 | 0.01 | 12.8 s |
| 0.25 | 1.000 | 0.976 | 0.988 | 0.11 | 0.10 | 28.0 s |
| 0.50 | 1.000 | 0.976 | 0.988 | 0.21 | 0.13 | 28.5 s |
| 0.75 | 1.000 | 0.976 | 0.988 | 0.17 | 0.12 | 28.8 s |
| 1.00 | 1.000 | 0.976 | 0.988 | 0.14 | 0.11 | 11.2 s |



**A four-way tie in F1.** Every α ≥ 0.25 achieves the same mean F1 of
0.988. The blend does not improve accuracy over the histogram alone.
The opposite endpoint `α=0.00` corresponds to a **normalised** SIFT-only
configuration (`s / min(len(kp1), len(kp2))`), with the count capped at
the number of features detected in the smaller keyframe and its mean
F1 of 0.892 is 10% below the histogram-only value.

**The raw `sift_similarity` of Section 5.1.6 reaches 0.935** (only 5%
below the histogram). The gap between 0.892 and 0.935 isolates the
cost of normalising SIFT to `[0, 1]`. The normalisation
`s / min(len(kp1), len(kp2))` is inherently non-linear: a 15-match
pair between two keyframes detected at 146 features each becomes
0.103, while the same 15-match pair between two keyframes detected
at 300 features each becomes 0.050. The rescaling therefore changes
with the number of detected features, not with the match quality.
The `max` aggregator then compares these already-distorted values
across all pairs of shots and the distortion is largest precisely
when the smaller keyframe is feature-poor. Raw SIFT counts still depend 
on the number of detected features.
Section 8.4 shows that raising `nfeatures` from 100 to 500 raises the
dev_02 matches from 19 to 69 and the dependence is monotone (19 → 33 → 69 → 75 on dev_02),
whereas the normalisation above changes with the keyframe pair.

**Decision.** Production uses `method="hist"` (equivalent to
`combined` with `α=1.00`). The choice rests on three facts from the
table: (i) every α ≥ 0.25 ties on mean F1 (0.988), so the blend
provides no accuracy benefit; (ii) α=0 (normalised SIFT alone) is
10% below the histogram, while raw SIFT (§5.1.6) is only 5% below,
the normalisation costs 5% of F1; (iii) α=0.25–0.75 costs ~30 s per
video versus ~9–10 s for hist. Under the criterion of Section 5.1.4
(identical F1, prefer the simplest), hist is the unique choice. The
code's `combined_alpha` default of `0.5` is only used when
`method="combined"` is explicitly selected; under the production
configuration it is never read (see the `combined_alpha` row in the
final-configuration table of Section 5.1.7).

#### 5.1.4 Colour space on the matrix (HSV vs Lab vs RGB)

Section 8.2 compared HSV, Lab, and RGB on a single test pair and found
mixed evidence. That experiment measured the *discrimination gap* on one pair,
not the final grouping performance. To decide with the full shot × shot
matrix, we now repeat the comparison with `uniform K=5`, `aggregator=max`,
and the three colour spaces:



```python
show(part3_exp.table_colour_space_matrix())
```


| Method | d01 F1 | d02 F1 | mean F1 | d01 width | d02 width | time |
|---|---|---|---|---|---|---|
| hist (HSV) | 1.000 | 0.976 | 0.988 | 0.14 | 0.11 | 9.6 s |
| lab | 1.000 | 0.976 | 0.988 | 0.06 | 0.04 | 11.8 s |
| rgb | 1.000 | 1.000 | 1.000 | 0.06 | 0.00 | 5.6 s |



**RGB appears to win on F1**: a perfect 1.000 on both videos, higher
than HSV's 0.988. **But the RGB plateau on dev_02 has width 0.00**:
in the 100-point threshold sweep, only one value of T achieves the
maximum F1. A small drift in the similarity distribution (which is
likely on the test videos, whose footage never appears in the
development set) would move the optimum outside this single point,
and the F1 would drop. The RGB result on dev_02 is therefore too
sensitive to the exact choice of T to be trusted for unseen data.
The plateau width is what catches this failure mode, as the F1
metric alone does not.

**HSV and Lab tie on mean F1 (0.988) but HSV is more robust.**
HSV's plateau is 0.14/0.11, roughly 2-3× wider than Lab's (0.06/0.04).
A wider plateau means the exact T matters less: on unseen test videos,
the true optimal threshold is more likely to fall inside HSV's plateau
than inside Lab's narrower one.

**Decision.** We adopt a two-part criterion that will be used
consistently across the hyperparameter sweeps: **among configurations
whose F1 is within 0.02 of the best, require a plateau width ≥ 0.10 on
both videos to be considered robust. Among those that pass, prefer the
simplest.** The 0.02 tolerance allows slight F1 variations to be
considered "practically tied", treating 1.000 and 0.988 as the same
performance level, while still rejecting any F1 substantially below
the best.

Under this criterion:

| Config | mean F1 | Within 0.02 of best? | Min width | Passes width? |
|---|---|---|---|---|
| HSV | 0.988 | ✓ (0.988 ≥ 1.000 − 0.02) | 0.11 | ✓ |
| Lab | 0.988 | ✓ | 0.04 | ✗ (below 0.10) |
| RGB | 1.000 | ✓ (equal to best) | **0.00** | ✗ (single threshold) |

**Both Lab and RGB fail the width requirement.** Lab by a small margin
(0.04 < 0.10), RGB by the full collapse (0.00).
**HSV is the unique configuration that combines a competitive F1 with
substantial robustness to threshold drift.** HSV is therefore retained.

#### 5.1.5 Bin resolution on the matrix (4³ vs 8³ vs 16³ vs 30·32²)

Section 4.2.1 swept the histogram bin resolution on a single keyframe
pair and found that finer bins reduced the discrimination gap.
That analysis measured the *gap between the same-scene and different-scene means*
on one pair. We now repeat the sweep on the **full shot × shot matrix** to see 
whether the same conclusion holds when the metric is the F1 of the final grouping:



```python
show(part3_exp.table_bins_matrix())
```


| Bins | d01 F1 | d02 F1 | mean F1 | d01 width | d02 width | mean width | time |
|---|---|---|---|---|---|---|---|
| 4³ | 1.000 | 0.976 | 0.988 | 0.081 | 0.040 | 0.060 | 9.4 s |
| 8³ | 1.000 | 0.976 | 0.988 | 0.141 | 0.111 | 0.126 | 11.5 s |
| 16³ | 1.000 | 0.976 | 0.988 | 0.192 | 0.141 | 0.166 | 11.3 s |
| 30·32² | 1.000 | 0.976 | 0.988 | 0.232 | 0.130 | 0.181 | 10.3 s |

| Bins | d01 width | d02 width | Mean width | vs 8³ |
|---|---|---|---|---|
| 4³ | 0.081 | 0.040 | 0.060 | −52 % |
| 8³ | 0.141 | 0.111 | 0.126 | baseline |
| 16³ | 0.192 | 0.141 | 0.166 | +32 % |
| 30·32² | 0.232 | 0.130 | 0.181 | +44 % |



**A four-way F1 tie.** Every bin resolution
yields identical F1 (1.000 on dev_01, 0.976 on dev_02, mean 0.988). 
This is the most surprising result of the whole hyperparameter sweep, 
because it appears to contradict Section 4.2.1. 

**Section 4.2.1.** On a single pair, finer bins
push both the same-scene and the different-scene similarities
*downward* and the absolute gap follows them (on dev_01 from
0.2123 at `8³` to 0.1316 at `60³`; on dev_02 the effect is not
strictly monotone between `8³` and `16³`). On the full matrix,
the F1 is *identical* across all bin resolutions (first table above)
even though the absolute similarity values move. We did not measure
why: the experiment only reports F1 and plateau width, not how the
same-scene and different-scene similarities are ordered.
What matters for the grouping is not the absolute value of any single
similarity, but the **relative ordering** of the pairs: as long as
same-scene pairs keep scoring above different-scene pairs, the
threshold sweep can find a cut point that separates them and the F1
is preserved.

**A bonus that emerges from the wider-plateau effect.** Although F1 is
unchanged, the **mean plateau width grows with finer bins**. The per-video
widths are **not strictly monotone**. On dev_02 the width dips slightly
from 0.141 (16³) to 0.130 (30·32²), but the mean increases at every
step (the second of the two tables above).

The non-monotonicity on dev_02 is visible at a glance in the second table above:
the mean is monotone because the dev_01 and dev_02 widths move in opposite directions at the
last step (dev_01 +0.040, dev_02 −0.011). A wider plateau means the exact
threshold matters less: on unseen test videos, the true optimal threshold
is more likely to fall inside the plateau.

**Why coarser bins hurt.** The `4³` histogram yields the **narrowest
plateau** (0.060 mean width, vs 0.126 for `8³`). Note that this is a
different quantity from the discrimination gap of Section 4.2.1: the
gap is a single number computed on one keyframe pair, while the
plateau measures how wide a range of thresholds achieves the maximum
F1 on the full matrix. With only 4 bins per channel, each bin covers
~45 H levels (180/4) and ~64 S and V levels (256/4), grouping
together pixels that are perceptually distinct. A plausible
consequence, consistent with the observation, though not measured
directly, is that this coarser 3D histogram compresses the colour
information and makes the F1 more sensitive to the exact choice of T.
Finer histograms preserve more per-shot colour variation and produce
a wider plateau. We report this as a working hypothesis: the
experiment measures only F1 and plateau width, not the underlying
class overlap.

**Applying the robustness criterion of Section 5.1.4.** Four
configurations tie on F1: `4³`, `8³`, `16³`, `30·32²`. Of these, `4³`
fails the requirement (dev_02 width = 0.040, below 0.10) and is
excluded. The remaining three (`8³`, `16³`, `30·32²`) all pass the
width requirement on both videos. **Among those that pass, we prefer
the simplest**: `8³`, because it has the fewest histogram cells
(8³ = 512, vs 16³ = 4 096 and 30·32² = 30 720) and it is the configuration already
validated by every other hyperparameter sweep. Its measured
wall-clock time (11.5 s) is not the smallest of the three, but the
difference (~1 s) is within the run-to-run timing noise noted in
Section 5.1.7. The wider plateaus of `16³` and
`30·32²` are documented here as a robustness margin that is *available*
but not necessary: any threshold in `[0.838, 0.909]` achieves the
maximum F1 on both videos already under `8³`.

**Decision.** `bins=(8, 8, 8)` is retained in the production
configuration. It delivers the same F1 as any other tested resolution,
its plateau is already wide enough for practical purposes (both
per-video widths exceed the 0.10 requirement of Section 5.1.4), and it
has the smallest histogram (8³ = 512 cells) among the passing
configurations.

#### 5.1.6 RANSAC on the matrix — does geometric filtering help?

Section 8.3 explored RANSAC on a single test pair and found a
**regime-dependent effect**: on the dev_01 pair it raised the gap
(+3), on the clean dev_02 pair it degraded it.
The conservative decision from the single-pair experiment was to disable
RANSAC. We now repeat the comparison on the full shot × shot matrix to
see whether the same conclusion holds at distribution level. Since RANSAC
only applies to SIFT-based methods, the sweep uses `method="sift"` and
`method="combined"` (α=0.5), both with `uniform K=5` and `aggregator=max`:



```python
show(part3_exp.table_ransac_matrix())
```


| Method | RANSAC | d01 F1 | d02 F1 | mean F1 | d01 width | d02 width | time |
|---|---|---|---|---|---|---|---|
| sift | ✗ | 0.947 | 0.923 | 0.935 | 23.70 | 21.39 | 14.3 s |
| sift | ✓ | 0.947 | 0.950 | 0.949 | 31.60 | 0.00 | 22.4 s |
| combined | ✗ | 1.000 | 0.976 | 0.988 | 0.212 | 0.133 | 29.2 s |
| combined | ✓ | 1.000 | 0.976 | 0.988 | 0.192 | 0.097 | 37.2 s |



**Two counter-intuitive findings, both of which confirm the decision to
disable RANSAC in production.**

**Finding 1: RANSAC helps `sift` at F1 but destroys robustness.**
On `method="sift"`, RANSAC improves mean F1 from 0.935 to 0.949 (+1.4%).
This appears to *contradict* the Section 8.3 conclusion, which reported
RANSAC as degrading the dev_02 pair by −19 matches. The key is the
plateau width: on dev_02 the plateau **collapses to width = 0.00**.

**Finding 2: RANSAC is useless for `combined` and slightly harmful.**
On `method="combined"`, RANSAC leaves F1 unchanged (0.988 both ways)
while *narrowing* the plateau from 0.212/0.133 to 0.192/0.097 and *increasing* 
the runtime by ~27% (29.2 s → 37.2 s). RANSAC adds nothing
to the combined similarity. the SIFT term is a small perturbation
and RANSAC only makes it less diverse. 

**Decision.** `use_ransac=False` is retained in the production
configuration. For the production method (`hist`), RANSAC is
irrelevant. For `sift` and `combined`, the plateau width confirms the
single-pair conclusion of Section 8.3. 

#### 5.1.7 Final configuration

Sections 5.1.1 through 5.1.6 swept seven independent design choices.
The consolidated production configuration is:

| Parameter | Value | Justified in |
|---|---|---|
| `method` | `"hist"` (HSV χ²) | 5.1.3 (α=1.00 tie), 5.1.4 (HSV retained) |
| `kf_strategy` | `"uniform"` | 5.1.1 (outperforms `middle`) |
| `K` | `5` | 5.1.1 (smallest K whose plateau passes the width criterion; K=8 adds cost only) |
| `aggregator` | `"max"` | 5.1.2 (mean F1 = 0.988 vs 0.952 and 0.413) |
| `bins` | `(8, 8, 8)` | 5.1.5 (F1 tie, plateau already wide enough) |
| `lowe_ratio` | `0.75` | 8.4 (largest ratio with `same/diff` > 1.3 on the harder video; retains most matches among the equivalent options) |
| `nfeatures` | `500` | 8.4 (`same/diff` 8.25 → 11.5 on dev_02; `unlimited` adds only +6 matches) |
| `use_ransac` | `False` | 5.1.6 (plateau collapse with RANSAC on) |
| `transform` | `"homography"` | irrelevant with `use_ransac=False` |
| `combined_alpha` | `0.5` (code default for `method="combined"`; never read with `method="hist"`) | 5.1.3 (production = `hist`, equivalent to `α=1.00`) |
| `threshold` | `0.87` | plateau analysis of 5.1.1 (see below) |

The four selections above (`K`, `aggregator`, `method`, `bins`) all
follow the same criterion, introduced in Section 5.1.4: **among
configurations whose mean F1 is within 0.02 of the best, require a
plateau width ≥ 0.10 on both videos to be considered robust; among
those that pass the requirement, prefer the simplest.** This formalises what was already implicit in
the choices above: K=5 was preferred over K=8 (both pass the width 
requirement, but K=5 is ~2.8× faster) and `8³` was preferred over `16³`
and `30·32²` (all three pass, but `8³` has the fewest histogram cells
(8³ = 512, vs 16³ = 4 096 and 30·32² = 30 720)). For the
aggregator and colour-space sweeps the winner is unique among those
that pass the width requirement.

**Threshold selection.** Unlike the other parameters, the threshold is
not derived from a dedicated sweep but from the **plateau analysis of
the K-sweep**. For the chosen configuration (hist, uniform K=5, max
aggregation), the optimal-F1 threshold plateaus are:



```python
show(part3_exp.table_threshold_plateau())
```


| Video | Optimal T | Plateau width | Plateau range |
|---|---|---|---|
| dev_01 | 0.9091 | 0.1414 | [0.838, 0.980] |
| dev_02 | 0.8538 | 0.1106 | [0.798, 0.909] |



The **intersection** of the two plateaus is [0.838, 0.909] — a 0.071-wide
region where **both videos simultaneously achieve their maximum F1**.
Any threshold in this range is equally good on the development set; we chose `T = 0.87`, 
near the geometric centre of the intersection, to maximise the
distance to either boundary in case the optimal threshold on the test
videos drifts slightly. 

**Note on reported timings.** All per-video times in this section are
single-run wall-clock measurements on the development set. The same
configuration can vary between runs, the same hist, uniform K=5 row 
appears as 8.7 s in 5.1.1, 9.6 s in 5.1.4 and 9.8 s in 5.2. These
differences are considered measurement noise, not algorithm differences. Only the
ratios that differ by more than ~50% (e.g. `combined` vs `hist`, or
RANSAC on vs off) are large enough to be interpreted as real cost
differences.

### 5.2 Shot similarity — the 2×2 comparison

Section 4.2 introduced two shot similarity measures (colour histograms
and SIFT local features) and Section 4.1 introduced two keyframe
strategies (`middle` (1 frame) and `uniform` (K frames)). The assignment
requires their pairwise comparison, so we swept all four combinations
on both development videos with the tuned production parameters

`method` and `kf_strategy` according to the row, `uniform K=5`,  
`K=1 for middle rows`, `aggregator=max`, `bins=8³`. Two additional rows 
(combined similarity of Section 8.1) are shown as extras.

**Precision, recall and F1** are computed with the given
`metrics.pairwise_scene_counts` on the full shot × shot matrix, after
applying `grouping.group_shots` with the optimal threshold for each
configuration. The time column is the mean wall-clock time per video 
(both dev_01 and dev_02 included in each row).



```python
show(part3_exp.table_2x2())
```


| Method | KF | K | d01 P | d01 R | d01 F1 | d02 P | d02 R | d02 F1 | mean F1 | time |
|---|---|---|---|---|---|---|---|---|---|---|
| hist | `middle` | 1 | 1.000 | 0.800 | 0.889 | 1.000 | 0.857 | 0.923 | 0.906 | 367 ms |
| hist | `uniform` | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 | 0.976 | 0.988 | 9.8 s |
| sift | `middle` | 1 | 1.000 | 0.700 | 0.824 | 1.000 | 0.762 | 0.865 | 0.844 | 990 ms |
| sift | `uniform` | 5 | 1.000 | 0.900 | 0.947 | 1.000 | 0.857 | 0.923 | 0.935 | 13.4 s |
| combined | `middle` | 1 | 1.000 | 0.800 | 0.889 | 0.952 | 0.952 | 0.952 | 0.921 | 1.7 s |
| combined | `uniform` | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 | 0.976 | 0.988 | 29.1 s |



**Interpreting the 2×2.** Two conclusions emerge, both consistent with
the dedicated sweeps of Section 5.1:

1. **`uniform K=5` dominates `middle K=1` for every similarity method.**
   The 2×2 column pairs isolate the keyframe strategy effect:
   - `hist`: +0.082 mean F1 (0.906 → 0.988)
   - `sift`: +0.091 mean F1 (0.844 → 0.935)
   - `combined`: +0.067 mean F1 (0.921 → 0.988)

   The gain is consistent across both videos. This is the same conclusion
   already drawn in Section 5.1.1 with a larger K-sweep.

2. **`hist` and `combined` tie on F1 (0.988), but `hist` is ~3× faster**
   (9.8 s vs 29.1 s per video, consistent with the ~3× measured 
   independently in Section 5.1.3, within the run-to-run timing noise 
   noted in 5.1.7). The hist method therefore
   dominates the combined similarity at production scale: adding SIFT
   on top of the histogram does not improve accuracy and it multiplies
   the runtime by a factor of three.

**A weaker result for `sift` alone.** SIFT-`uniform` reaches only
0.935 mean F1 vs `hist`-`uniform`'s 0.988. A similar gap is also visible in the `middle` rows (0.844 vs 0.906),
so it does not depend on the keyframe strategy.

**Zero false positives for the `uniform K=5` configurations.** All
three `uniform K=5` rows (hist, sift, combined) achieve P=1.000 on
both videos: with the extra keyframes, no pair of different scenes is
ever merged and their errors are exclusively missed pairs (FN), not
spurious merges (FP). The only configuration with a non-perfect
precision is `combined middle` on dev_02 (P=0.952, one spurious
merge), where a single mid-shot keyframe occasionally produces a
false match. The `middle` rows for hist and sift still reach P=1.000,
so the spurious merge is a weakness specific to combining a noisy
signal (normalised SIFT) with a single non-establishing keyframe, not
a general failure of the pipeline. 

**Storyboard evidence.** The two storyboards below show the predictions
of the winning configuration on the two development videos. Each tile
is the middle keyframe of a shot, labelled with the scene id predicted
by `group_shots`. **Same-colour tiles that depict the same physical
scene are correct groupings; tiles that share a colour but depict
different content (or vice versa) are errors.**

![Storyboard dev_01](./images/t7_storyboard_hist_uniform_T0.87_dev_01.png)

*Figure 5: Storyboard of `dev_01` with the production configuration
(`hist`, `uniform K=5`, threshold 0.87). The system detects **28 scenes
in 37 shots**, an exact match with the 28 unique scenes in the ground
truth (F1 = 1.000).*

![Storyboard dev_02](./images/t7_storyboard_hist_uniform_T0.87_dev_02.png)

*Figure 6: Storyboard of `dev_02` with the production configuration.
The system detects **20 scenes in 35 shots** against 19 unique scenes
in the ground truth (F1 = 0.976).*

### 5.3 Shot boundary detection on dev 


The final detector (the pixel difference, the RGB histogram and the ECR with thresholds 47, 0.22 and 0.6, a cut when at least two of them agree, and then the verification of Task 4 with `min_matches` = 20) was applied to both development videos. The table shows the cut metrics, with the DEV row obtained by adding the counts of the two videos; the column TP / FP / FN gives those counts. Where each threshold and `min_matches` come from is explained in Sections 2.3 and 3.1.


```python
show(part2_exp.table_cuts("DEV"))
```


| Video | cut-P | cut-R | cut-F1 | TP / FP / FN | flash-FA |
|---|---|---|---|---|---|
| dev_01 | 1.000 | 0.972 | 0.986 | 35 / 0 / 1 | 0/6 |
| dev_02 | 0.944 | 1.000 | 0.971 | 34 / 2 / 0 | 1/6 |
| DEV | 0.972 | 0.986 | 0.979 | 69 / 2 / 1 | 1/12 |


Only one of the 70 real cuts of the development videos is missed (recall **0.986**), on `dev_01`, and there are 2 false positives, both on `dev_02` (precision **0.972**). The two false positives belong to the same flash, which is the **1/12** of the flash-FA column: the detector reports a cut at the start of the flash and another one when the image goes back to normal. All of these errors are analysed in Section 6.1.

### 5.4 Scene grouping on dev

The production configuration (`method="hist"`, `kf_strategy="uniform"`,
`K=5`, `aggregator="max"`, `bins=(8, 8, 8)`, `threshold=0.87`) was
applied to both development videos. The table below reports the
pairwise scene metrics at that threshold, with the DEV total obtained
by summing the true positives, false positives and false negatives
across the two videos.


```python
show(part3_exp.table_dev_scene_metrics())
```


| Video | Scenes (pred / GT) | Scene-P | Scene-R | Scene-F1 |
|---|---|---|---|---|
| dev_01 | 28 / 28 | 1.000 | 1.000 | 1.000 |
| dev_02 | 20 / 19 | 1.000 | 0.952 | 0.976 |
| DEV | 48 / 47 | 1.000 | 0.968 | 0.984 |


**dev_01 is an exact match.** The 37 shots are partitioned into
exactly the 28 scenes present in the ground truth (P = R = F1 = 1.000).
The storyboard of Figure 5 shows the visual evidence: tiles with the
same colour are the same physical scene, and the boundaries between
scene groups align with the ground-truth labels.

**dev_02 is one missed pair away from perfect.** The system detects 20
scenes against 19 in the ground truth, so precision stays at 1.000 
(no two different scenes are ever merged) but recall drops to 0.952:
one pair of shots from the same scene is not linked, because their
keyframes fall below the 0.87 threshold (Figure 6 shows the resulting
storyboard). 

**DEV total.** Summing the counts across the two videos keeps
precision at 1.000 (no spurious merge on either video) and gives a
recall of 0.968, between the two per-video values (dev_01 contributes
a perfect 1.000, dev_02 contributes 0.952). The DEV F1 is 0.984. The
single missed pair on dev_02 is the only error in the whole development
set.

### 5.5 Final results on test_01 + test_02


**Shot boundary detection.** The same detector, with the thresholds and `min_matches` that come from the development sweeps, was applied to the test videos:


```python
show(part2_exp.table_cuts("TEST"))
```


| Video | cut-P | cut-R | cut-F1 | TP / FP / FN | flash-FA |
|---|---|---|---|---|---|
| test_01 | 1.000 | 0.946 | 0.972 | 35 / 0 / 2 | 0/6 |
| test_02 | 0.976 | 0.976 | 0.976 | 40 / 1 / 1 | 1/6 |
| TEST | 0.987 | 0.962 | 0.974 | 75 / 1 / 3 | 1/12 |


On the test videos the F1 is **0.974**, very close to the **0.979** of the development videos, so the choices made on the development set also work on videos we did not use to choose them. Of the 78 real cuts, 3 are missed (recall **0.962**) and there is a single false positive (precision **0.987**), which is the flash of `test_02` counted in the **1/12** of the flash-FA column. The three missed cuts and the flash are analysed in Section 6.1.

## 6. Analysis and failure cases

This section walks through concrete failure cases of the two
subsystems, Shot boundary detection (6.1) and scene grouping
(6.2), with keyframes and figures. For each failure we identify
the frames involved and discuss the mechanism that caused the
error.

### 6.1 Shot boundary detection: failure cases



Over the four videos (148 real cuts) the final detector misses 4 cuts and makes 3 false positives. There ar two kinds of misses: three cuts are lost in the vote and one real cut is rejected by the verification. The three false positives are flashes that the verification does not reject. We will look at five of these errors.

**6.1.1 A cut between two similar scenes (`dev_01`, frame 2947).** Frames 2946 and 2947 show the inside of the same tram from two different points of view, with similar colours (people in dark clothes and a white-grey interior).

![Missed cut in dev_01 at frame 2947](./images/t6_failure_dev_01_2947_missed_cut_pixel_only.png)

*Figure — A real cut missed by the vote. Frames t-4, t-1, t and t+4 around the cut at frame 2947 of `dev_01`. The pixel difference is high (66.6, threshold 47) but the histogram (0.17, threshold 0.22) and the ECR (0.34, threshold 0.6) are below their thresholds.*

The pixel difference sees the change (66.6, above its threshold of 47), but the histogram (0.17) and the ECR (0.34) stay below theirs, so only one signal fires and the vote throws the cut away. This is the weakness we expected from the histogram: it only looks at the colours, not at where things are, and here the colours of both views are very similar. The ECR does not see it either. The verification would have kept this cut (only 6 SIFT matches between frames 2943 and 2951), so the cut was lost earlier, in the vote.

**6.1.2 A cut that only the histogram sees (`test_01`, frame 870).** The shot goes from a close-up of grass to a cow in a field.

![Missed cut in test_01 at frame 870](./images/t6_failure_test_01_870_missed_cut_hist_only.png)

*Figure — A real cut missed by the vote. Frames t-4, t-1, t and t+4 around the cut at frame 870 of `test_01`. Only the histogram is above its threshold (1.23 against 0.22); the pixel difference (28.8, threshold 47) and the ECR (0.56, threshold 0.6) are below theirs.*

Both shots are mostly green, so the pixel difference stays lower than for most cuts (28.8, threshold 47), probably because the average brightness is similar. The ECR (0.56) is just below its threshold of 0.6. Only the histogram (1.23) fires, so the vote rejects the cut. The missed cut at frame 357 of `test_02` has the same pattern in the signals (only the histogram above its threshold). In these three misses (frame 2947 of `dev_01`, 870 of `test_01` and 357 of `test_02`) the problem is the same: the vote asks for two signals, and for cuts like these only one of them reacts.

**6.1.3 A real cut rejected by the verification (`test_01`, frame 3073).** The shot goes from an aquarium to fireworks.

![Cut rejected by the verification in test_01 at frame 3073](./images/t6_failure_test_01_3073_cut_rejected_by_verification.png)

*Figure — A real cut rejected by the verification. Frames t-4, t-1, t and t+4 around the cut at frame 3073 of `test_01`. All three signals fire (59.2, 0.57 and 0.93), but frames 3069 and 3077 share 30 SIFT matches, more than `min_matches` = 20.*

All three signals fire (59.2, 0.57 and 0.93), so the candidate is correct, but frames 3069 and 3077 share **30** SIFT matches, above our limit of 20, and the cut is rejected as if it were a false alarm. The two scenes have nothing in common, so these matches must be accidental (probably because the fireworks frames are dark and noisy). We checked it with RANSAC: the 30 matches have **0** inliers of a homography, while the flash of the figure in Section 3.1 (frame 860 of `dev_01`) has 208 inliers out of 218 matches. Counting RANSAC inliers instead of all the matches (an option of Task 4 that we did not use) would have kept this cut. We only tried it on these frames, not on all the videos, so we do not know how it would change the rest of the results.

**6.1.4 Flashes that the verification does not reject (`dev_02`, frame 1867, and `test_02`, frame 2394).** In the first one the flash whitens frame 1867 for three frames:

![Flash not rejected in dev_02 at frame 1867](./images/t6_failure_dev_02_1867_flash_not_rejected.png)

*Figure — A flash that the verification does not reject. Frames t-4, t-1, t and t+4 around the flash at frame 1867 of `dev_02`. The three signals fire (98.1, 2.00 and 1.00), but frames 1863 and 1871 share only 3 SIFT matches.*

The three signals fire, as they do for any flash, and the verification compares frames 1863 and 1871, which are not flash frames. But they share only **3** matches, so the candidate is kept. The scene is made of sky and sea, with very little texture for SIFT to find, and the camera moves between the two frames (the pole is not in the same place). The detector also reports a second cut at frame 1870, when the image goes back to normal, so this flash produces two false positives.

![Flash not rejected in test_02 at frame 2394](./images/t6_failure_test_02_2394_flash_not_rejected_one_frame.png)

*Figure — A one-frame flash that the verification does not reject. Frames t-4, t-1, t and t+4 around the flash at frame 2394 of `test_02`. Frames 2390 and 2398 share 13 SIFT matches, less than `min_matches` = 20.*

In the second one (`test_02`) the flash lasts a single frame, but the people move fast: frames 2390 and 2398 only share **13** matches, below our limit of 20, so the candidate is kept (RANSAC finds 6 inliers). To reject this flash `min_matches` should be 13 or lower, but in the table of Section 3.1 we can see that with 10 the recall already falls to 0.929 on the development videos, because real cuts start to be rejected. The threshold of 20 is therefore a compromise: both failures come from the idea behind the verification, that two frames of the same scene have many matches, which does not hold when there is fast motion or almost no texture.

**6.1.5 What is hard, and which signal is the most robust.** The cuts that are lost in the vote are those between shots that look alike (similar colours and similar brightness, as in the tram and the grass), where fewer of the signals react. For motion, the histogram is the most robust signal: in the figure of the three signals it stays almost flat inside the shots, whereas the ECR is the noisiest one, even inside a shot with camera movement (0.28 in the example of Section 2.2, against 0.94 across a cut) and in the stretch between frames 700 and 800 of `dev_01`. For flashes, none of the signals is robust: all three spike at every flash (see the figure of the three signals in Section 2.3), and the only reason that only 1 of 12 flashes reaches the final result is the verification, which brings the false alarms from 12/12 down to 1/12. The two remaining flashes are the cases of 6.1.4, where there is fast motion or too little texture for SIFT.



### 6.2 Scene grouping: failure cases

**6.2.1 A spurious merge (false positive).**

The configuration `combined` with `kf_strategy="middle"` and `K=1` is
the only one in Table 5.2 that produces a non-perfect precision on
either development video, with P = 0.952 on dev_02, exactly one pair
of different scenes is merged. That pair is shots **0** (ground-truth
scene S00) and **28** (ground-truth scene S17), two shots that are far
apart in the video (shot 0 is the opening shot, shot 28 is close to the
end). At the configuration's best-F1 threshold (T = 0.317) the two shots
fall in the same predicted group, with a shot-to-shot similarity of
0.363.

Note that this is **not** the production threshold. The production
configuration (`hist`, `uniform K=5`) uses T = 0.87 and does not merge
this pair. The false positive only exists because the `combined middle`
configuration reaches its own maximum F1 at a much lower T (0.317), and
its plateau is a **single point** (width = 0.000 in the 100-point sweep
of §5.2). This is the same failure mode observed for RGB in 5.1.4: a
narrow plateau means the F1 at the reported optimum is not robust to a
small shift in the similarity distribution and the spurious merge is
the concrete manifestation of that fragility.

![Figure 7: Spurious merge between shot 0 (S00) and shot 28 (S17) under the `combined middle` configuration on dev_02. Both keyframes show night-time urban scenes with similar lighting, colour distributions and repetitive architectural structures.](./images/t6_fp_combined_middle_dev_02.png)

*Figure 7: Keyframes of shot 0 (left, frame 21, scene S00) and
shot 28 (right, frame 2848, scene S17). Both are outdoor nighttime
urban scenes, but they depict different physical locations in the
ground truth. The similarity between the two frames (sim = 0.363)
is above the configuration's best-F1 threshold (T = 0.317), so
`combined middle` merges them into the same predicted group.*

**6.2.2 A missed pair (false negative).**

Under the production configuration (`hist`, `uniform K=5`,
threshold = 0.87), dev_02 achieves P = 1.000 and R = 0.952, i.e.
exactly one pair of shots from the same scene is not linked.
That pair is shots **2** and **15**, both from ground-truth scene
**S02**. Their shot-to-shot similarity is `sim[2, 15] = 0.715`,
below the 0.87 threshold, so `group_shots` places them in
separate predicted groups.

![Figure 8: Missed pair (false negative) between shot 2 and shot 15 (scene S02) on dev_02. Top row: five uniform keyframes of shot 2. Bottom row: five uniform keyframes of shot 15.](./images/t6_fn_hist_uniform_dev_02_2_15.png)

*Figure 8: Five uniform keyframes of shot 2 (top) and shot 15
(bottom), both from ground-truth scene S02 of `dev_02`. The two
shots show the same rainy street from inside a car, but the camera
in shot 2 faces one direction while in
shot 15 it faces roughly the opposite direction and shot 15 contains 
sweeping wiper blades that occlude part of the view.*

This failure is worth contrasting with the false positive of
6.2.1. The spurious merge of 6.2.1 happened in a configuration
with a *narrow* plateau (`combined middle`, width = 0.000); the
missed pair of 6.2.2 happens in the configuration with the
*widest* plateau (`hist uniform K=5`, 0.14/0.11), which is the
one we retain. The production threshold of 0.87
was therefore placed inside the widest common plateau of both
dev videos (5.1.7) precisely to keep precision at 1.000 at the
cost of a single recall error.


**6.2.3 A hard scene pair that the system gets right.**

To understand the margin at which the production configuration operates,
we searched the dev_02 similarity matrix for *close calls* (pairs whose
similarity was within 0.05 of the 0.87 threshold). Only one same-scene
pair fell into this narrow band: shots **10** and **29** (both scene
S09), with a similarity of `sim[10, 29] = 0.910`. The system correctly
merges them, but by a margin of only 0.04 above the threshold. Among
different-scene pairs, **no** pair on dev_02 fell within 0.05 of the
threshold from below. The closest different-scene pair is well below
0.82. This means the system has a comfortable margin when rejecting
distinct scenes; the only rejection failure was the `combined middle`
configuration analysed in 6.2.1. In the production configuration, no
different-scene pair came close to being merged.

![Figure 9: Hard pair between shot 10 and shot 29 (scene S09) on dev_02. Top row: five uniform keyframes of shot 10. Bottom row: five uniform keyframes of shot 29.](./images/t6_hard_pair_dev_02_10_29.png)

*Figure 9: Five uniform keyframes of shot 10 (top, scene S09) and
shot 29 (bottom, scene S09) from `dev_02`. Both rows show a white
rally car on a muddy dirt track, surrounded by green grass and bare
trees under an overcast sky. The car's position and the camera angle
change substantially across the shots, but the background palette
remains the same.*


**6.2.4 Which scenes are hard.**

The failure modes above, combined with the close-call analysis, suggest a
taxonomy of scenes that are hard for the colour-histogram pipeline.

**Scenes filmed from opposing viewpoints.** The missed pair of 6.2.2
(shots 2 and 15, scene S02) shows that a 3D HSV histogram is
orientation-blind. When the camera reverses direction within the same
take, which is exactly the "recurring scenes filmed from a different
point of the same take" case mentioned in the assignment, the
proportion of colour in the frame changes even if the physical location
does not and the similarity can fall below the threshold. 

**Scenes with a dominant background colour.** Shot 28 (S17) in the false
positive of 6.2.1 is a modern building with large illuminated windows;
shot 0 (S00) is a lit street with dark metal barriers. Both are
night-time urban scenes with a similar grey/brown pavement. When a single
colour dominates the frame, the histogram summarises the background and
discards the foreground, so two visually different locations can look
similar in colour space.

**Scenes with strong motion or dominant foreground objects are
surprisingly *not* hard.** The rally car of 6.2.3 changes position
drastically across the keyframes, yet the pair still scores 0.910. 
The `max` aggregator combined with
`uniform K=5` is specifically designed to tolerate this kind of
intra-shot variation. A single keyframe where the car happens to leave
the background visible is enough to match the scene. The failure happens
only when the *background itself* changes (6.2.2) or when a *middle*
keyframe happens to be dominated by a moving object (6.2.1).

**What is *not* hard, empirically.** No different-scene pair on dev_02
came within 0.05 of the 0.87 threshold. This means that on this dataset,
two different scenes never produced a similarity high enough to be at
risk of merging. The threshold sits in a wide, empty gap in the
similarity distribution, which is why the DEV precision is a clean 1.000
rather than a value obtained by luck. This gap is a direct consequence
of the choices made in 5.1: using `hist` alone (rather than `combined`),
and using `uniform K=5` (rather than `middle K=1`), both push
different-scene pairs further away from the decision boundary.


## 7. Conclusion and self-evaluation

### 7.1 Conclusion

This project successfully developed a pipeline for shot boundary detection and recurring scene grouping, achieving solid results on both the development and test sets. The report addressed two main challenges: finding hard cuts while ignoring distractors like flashes and grouping visually similar shots into recurring scenes.

**Cut detection (Parts 1 and 2).** The combination of three dissimilarity signals (pixel difference, RGB histogram, and ECR) through a 2-of-3 voting scheme, followed by verification step using SIFT, achieved a test F1 score of **0.974** (P = 0.987, R = 0.962). The most significant finding was the impact of the verification step, which reduced false alarms caused by flashes from 12/12 to just 1/12 on both the development and test sets. The failure cases in Section 6.1 showed that the remaining errors are mostly cuts between visually identical shots and flashes where the background lacks texture or has fast motion, preventing SIFT from finding matches. The RGB histogram outperformed our initial HSV expectation, proving that, for cut detection, the full colour distribution is better than hue/saturation alone.

**Scene grouping (Part 3).** The production configuration achieved a perfect **F1 = 1.000** on the test set (P = 1.000, R = 1.000) and a DEV F1 of 0.984. This demonstrates that a conservative threshold (T = 0.87), chosen to maximize the plateau width, successfully eliminated over-merging. The system is highly robust: the `uniform K=5` strategy tolerates intra-shot variations like moving objects or camera angle changes by relying on the best-matching keyframe pair. While exploring the extras, we verified that blending SIFT into the colour histogram did not improve accuracy and multiplied the runtime by a factor of three. 

Overall, the production pipeline demonstrates that combining conservative, well-tuned features with a verification step can lead to robust and highly precise results in video structure analysis.

### 7.2 Self-Evaluation

We believe the project was completed successfully, meeting all the requirements of the assignment and achieving strong results. We maintained a rigorous methodology by tuning all thresholds strictly on the development set and reporting test results only once. 

We tried getting the errors in cut detection to 0, but that wasn't possible, maybe we could have tried a different approach to that side, but overall we are satisfied with the results and believe we chose the right parameters for the videos we were given.

For Part 3, the hyperparameter space was vast and we learned the importance of evaluating not just F1, but also the width of the threshold plateau to ensure the models generalize to unseen data. One limitation of the current approach is the reliance on global colour histograms for scene grouping, as scenes that differ only in spatial layout or texture are invisible to this metric.

Both group members contributed actively, dividing the work between Parts 1/2 (cut detection) and Part 3 (scene grouping), while collaborating on the final integration and analysis.



## 8. Extras

### 8.1 Combined similarity (colour + features)

**Motivation.** The assignment's extras list
proposes combining them into a single shot similarity.

**Method.** We blend the two normalised similarities with a convex weight
`α ∈ [0, 1]`:

$$
\mathrm{sim}(k_1, k_2) = \alpha \cdot h(k_1, k_2) + (1-\alpha) \cdot \hat{s}(k_1, k_2)
$$

where `h` is `histogram_similarity`, `s` is `sift_similarity`, and
`ŝ = s / min(len(kp1), len(kp2))` normalises the SIFT count to `[0, 1]`.

**Endpoint validation.** `α=1` reproduces `histogram_similarity` exactly,
which validates the implementation. The opposite endpoint `α=0` produces the **normalised**
SIFT count (`s / min(len(kp1), len(kp2))`) and **not** the raw match
count returned by `sift_similarity`. The two are different by
construction: `histogram_similarity` is bounded to `[0, 1]`, while
`sift_similarity` is unbounded, so a convex blend between them requires
normalising SIFT first. `α=0` is therefore a *normalised* SIFT-only
configuration, not `sift_similarity`. The matrix-level comparison of
the two is in Section 5.1.3 (α=0, F1 = 0.892) and Section 5.1.6
(raw SIFT, F1 = 0.935).

**Results** (dev_01 + dev_02 in [0, 1] units):



```python
show(part3_exp.table_alpha_single())
```


| Video | α=0.00 | α=0.25 | α=0.50 | α=0.75 | α=1.00 |
|---|---|---|---|---|---|
| dev_01 gap | 0.0627 | 0.1001 | 0.1375 | 0.1749 | 0.2123 |
| dev_02 gap | 0.1253 | 0.1656 | 0.2058 | 0.2460 | 0.2862 |



**Interpretation.** The gap grows **linearly** with α. On this specific test pair, 
the histogram dominates.

**But the test pair is not representative.** It compares the *first* and
*last* shot of the same scene. 
The blend's hypothesis (that SIFT and colour fail on disjoint regimes) 
can only be tested on the full shot × shot matrix. 
We therefore defer the α choice to the α sweep on the matrix in Section 5.1.3.
Production uses `method="hist"`, which is equivalent to `combined` with
`α=1.00` (Section 5.1.3); the code's `combined_alpha=0.5` is the default
only when `method="combined"` is selected explicitly and is never read
under the production configuration.


### 8.2 Colour space comparison (Lab vs HSV vs RGB)

**Motivation.** 4.2.1 justifies HSV through theoretical considerations
(chromaticity decoupled from luminance). But does Lab
discriminate better than HSV on real footage?

**Method.** The same χ² and 3D-histogram pipeline, applied in three
colour spaces. The colour space is a parameter of the shared helper
`_colour_histogram(bgr, space, bins)` and three public wrappers
(`histogram_similarity` for HSV `lab_similarity` for Lab and
`rgb_similarity` for RGB) call it with the corresponding `space`. In
the pipeline, the choice is made by the `method` field of the config
(`"hist"`, `"lab"` or `"rgb"`), which `_pairwise_keyframe_similarities`
maps to the corresponding wrapper.

**Results:**



```python
show(part3_exp.table_colour_space_single(include_lab=True))
```


| Video | Space | same | diff | gap |
|---|---|---|---|---|
| dev_01 | HSV | 0.5900 | 0.3777 | 0.2123 |
| dev_01 | Lab | 0.7836 | 0.5017 | 0.2819 |
| dev_01 | RGB | 0.7529 | 0.6740 | 0.0789 |
| dev_02 | HSV | 0.8724 | 0.5862 | 0.2862 |
| dev_02 | Lab | 0.9213 | 0.7427 | 0.1786 |
| dev_02 | RGB | 0.9323 | 0.7688 | 0.1636 |
| mean | HSV | — | — | 0.2493 |
| mean | Lab | — | — | 0.2302 |
| mean | RGB | — | — | 0.1212 |



**Interpretation.** The result is **mixed and symmetric**: Lab has the
larger gap on dev_01 (+33% over HSV), HSV has the larger gap on dev_02
(+60% over Lab). The two signals are of the same kind (gap between
same-scene and different-scene on a single keyframe pair) and each
colour space wins on exactly one video. **Neither is uniformly better
on this single pair.**

**What this experiment cannot decide.** A gap measured on a single
keyframe pair is too noisy to establish a ranking: the mean gap is 
HSV 0.2493 vs Lab 0.2302 (a difference of 0.019), well inside the 
variance one would expect from swapping the keyframe pair. The single
-pair gap is therefore reported as **evidence that both HSV and Lab
are plausible**, not as a decision criterion.

**Decision deferred to Section 5.1.4.** The final choice is made at
matrix level (F1 + plateau width on the full shot × shot matrix), where
HSV and Lab tie on mean F1 (0.988 both) but HSV has a 2–3× wider
plateau (0.14/0.11 vs 0.06/0.04). `lab_similarity` remains in the code
as a competitive alternative; Section 5.1.4 documents why HSV is
retained.


### 8.3 RANSAC transformation model 

**Scope.** The assignment suggests RANSAC + homography for **Part 2**, 
so we tested it in the Task 6 pipeline as an exploration of whether 
geometric filtering could further clean up SIFT matches.

**Method.** Four models, all fitted with RANSAC where applicable, on the
same three keyframes used in 4.2.2 (`nfeatures=500`, `ratio=0.75`):

| transform | DOF | 
|---|---|
| `none` (baseline) | — |
| `translation` | 2 | 
| `similarity` | 4 |
| `affine` | 6 | 
| `homography` | 8 |

**Results:**



```python
show(part3_exp.table_ransac_single())
```


| Video | transform | same | diff | gap | time |
|---|---|---|---|---|---|
| dev_01 | none | 15 | 11 | 4 | 1.4 ms |
| dev_01 | translation | 8 | 1 | 7 | 2.0 ms |
| dev_01 | similarity | 10 | 3 | 7 | 1.9 ms |
| dev_01 | affine | 11 | 4 | 7 | 2.0 ms |
| dev_01 | homography | 12 | 6 | 6 | 4.6 ms |
| dev_02 | none | 69 | 6 | 63 | 4.1 ms |
| dev_02 | translation | 1 | 0 | 1 | 5.5 ms |
| dev_02 | similarity | 49 | 3 | 46 | 5.0 ms |
| dev_02 | affine | 49 | 4 | 45 | 5.1 ms |
| dev_02 | homography | 49 | 5 | 44 | 6.1 ms |



**Interpretation.**:

1. **dev_01** Baseline has `same=15, diff=11` — barely
   discriminative. RANSAC raises the gap from 4 to 7. The **benefit is 
   real but small** in absolute terms.
2. **dev_02** Baseline has `same=69, diff=6`. RANSAC rejects 20 **true** 
   matches (`same`: 69 → 49) to remove only 1-3 false ones (`diff`: 6 → 3-5).
   The **cost is large**.

The three higher-DOF models (similarity, affine, homography) produce
**nearly identical** results. The extra degrees of freedom
do not add discriminative power here. `translation` fails catastrophically
on dev_02.

**Conclusion.** The conservative default is therefore to
**disable RANSAC for Task 6**. The RANSAC sweep on the matrix in Section 5.1.6 will
reveal the dataset's actual distribution and the decision can be revisited.


### 8.4 SIFT parameter sensitivity (nfeatures and Lowe's ratio)

**Motivation.** The assignment specifies SIFT + Lowe's ratio test for
Task 6 but does not fix the number of keypoints retained (`nfeatures`)
nor the ratio threshold. We swept both on the development videos to
justify the values used in 4.2.2.

**nfeatures**: maximum keypoints retained by SIFT. Sweep over
`{100, 200, 500, unlimited}`:



```python
show(part3_exp.table_nfeatures_single())
```


| Video | nfeatures | kp_a | kp_b | same | diff | gap | time |
|---|---|---|---|---|---|---|---|
| dev_01 | 100 | 100 | 100 | 7 | 5 | 2 | 61.7 ms |
| dev_01 | 200 | 200 | 146 | 11 | 9 | 2 | 45.1 ms |
| dev_01 | 500 | 307 | 146 | 15 | 11 | 4 | 45.4 ms |
| dev_01 | unlimited | 307 | 146 | 15 | 11 | 4 | 47.8 ms |
| dev_02 | 100 | 100 | 100 | 19 | 0 | 19 | 42.1 ms |
| dev_02 | 200 | 200 | 200 | 33 | 4 | 29 | 44.4 ms |
| dev_02 | 500 | 500 | 500 | 69 | 6 | 63 | 59.8 ms |
| dev_02 | unlimited | 545 | 678 | 75 | 6 | 69 | 71.4 ms |



`nfeatures=500` **more than doubles the discrimination gap on dev_02** relative
to the 200-cap (29 → 63) and the same/diff ratio jumps from 8.25
(33/4) to 11.5 (69/6). Going from 500 to `unlimited` adds only
+6 same-scene matches with no change in the different-scene count
(gap 63 → 69; same/diff 11.5 → 12.5), at a cost of ~20% more time
on dev_02 (59.8 ms vs 71.4 ms). Following the
`same/diff` criterion introduced below, the +6 matches are within the
natural variance of SIFT detection between frames and do not justify
removing the cap. On dev_01 the two configurations are identical
(only 307 and 146 keypoints are detected, both below 500). We therefore
keep **`nfeatures=500`** as an explicit, reproducible cap, as the
`unlimited` alternative gives no reliable gain over it.

**Lowe's ratio**: the ratio threshold. A sweep over `{0.50, 0.60,
0.70, 0.75, 0.80, 0.90}` with `nfeatures=500` gives the following
same-scene / different-scene match counts on the pair from Section
4.2.2:



```python
show(part3_exp.table_lowe_single())
```


| ratio | dev_01 (same / diff / gap) | dev_02 (same / diff / gap) |
|---|---|---|
| 0.50 | 8 / 1 / 7 | 32 / 0 / 32 |
| 0.60 | 8 / 2 / 6 | 45 / 1 / 44 |
| 0.70 | 10 / 7 / 3 | 59 / 3 / 56 |
| 0.75 | 15 / 11 / 4 | 69 / 6 / 63 |
| 0.80 | 22 / 21 / 1 | 81 / 11 / 70 |
| 0.90 | 74 / 68 / 6 | 176 / 68 / 108 |

| ratio | dev_01 (same/diff) | dev_02 (same/diff) |
|---|---|---|
| 0.50 | 8.00 | ∞ (0 false) |
| 0.60 | 4.00 | 45.00 |
| 0.70 | 1.43 | 19.67 |
| 0.75 | 1.36 | 11.50 |
| 0.80 | 1.05 | 7.36 |
| 0.90 | 1.09 | 2.59 |



**The dev_01 / dev_02 split explains the shape of the table.** The
`S00` keyframe on dev_01 has only 307 SIFT features (it does not
reach the `nfeatures=500` cap) and its two takes differ substantially
in viewpoint. As the ratio threshold grows, the number of same-scene
matches increases but the number of different-scene matches grows at
a similar pace, so the *separation* between the two classes collapses
above ratio 0.80. On dev_02, the same-scene count grows monotonically with the ratio
and the different-scene count stays at or below 11 up to ratio 0.80; only at
0.90 does it jump to 68, at which point the same/diff ratio collapses
from 7.36 (at 0.80) to 2.59.

**The `same/diff` ratio is the discriminating quantity, not the gap.**
The gap (`same − diff`) scales with the total number of matches: a
ratio threshold that lets 74 vs 68 matches through will have gap = 6,
but the classifier cannot separate same from different scenes when
`same ≈ diff`. The relevant ratios are:

**Three regimes emerge:**

- **0.50 and 0.60** give the cleanest separation (dev_01 ratio 8.00
  and 4.00) but retain only **8 matches** in the harder keyframe.
  With this few features, matrix-level thresholding is fragile:
  keyframes of different shots have very different feature counts
  (307 vs 146 on the `S00` pair), and an 8-match keyframe cannot be
  tuned reliably.

- **0.70 and 0.75 are equivalent in gap on both videos** (dev_01:
  3 vs 4; dev_02: 56 vs 63), with a solid `same/diff` ratio on
  dev_01 (1.43 vs 1.36). We keep **0.75** as the conventional
  value (Lowe's original recommendation), as it **retains more
  matches** (15 vs 10 on dev_01; 69 vs 59 on dev_02), which is
  preferable when the same keyframe is matched against many others
  at matrix level.

- **0.80 and 0.90 are rejected.** At 0.80 the dev_01 gap shrinks to
  1 and the `same/diff` ratio falls to **1.05**, the two classes
  overlap almost completely (22 same vs 21 diff). At 0.90 the
  *absolute* gap on dev_01 grows back to 6 (74 vs 68), but the
  `same/diff` ratio is only **1.09**: 68 different-scene matches
  slip through for every 74 same-scene matches. 

**Decision.** **We keep `ratio=0.75`.** The largest ratio with a
`same/diff` separation above 1.3 on the harder development video
and among the equivalent alternatives it is the one that retains
the most matches for matrix-level thresholding.

## 9. Use of AI tools

Throughout the project we used large language models as a supporting tool, not as a replacement for our own analysis. All design decisions, interpretations of the results and numerical values reported in this document are our own. Every output was manually verified against the assignment requirements, the ground-truth data and our own understanding of the algorithms. 

We used LLMs to:
- Get familiar with the syntax of Python and its libraries, as one of our members was not very familiar with Python.
- Debug the implementation of `scene_grouping.py` and `part3_exp.py`, by creating some tests.
- Review the report for internal consistency. Specifically, checking that the narrative claims were accurately supported by the tables generated by the code and that the reasoning in the hyperparameter sweeps was logically sound.
- Draft the structure of the some sections, which were then populated with the actual numerical values extracted from the pipeline.
- Improve the English phrasing, formality and correctness of specific paragraphs


