# Quick check — what α would recover #2/#15 and #10/#29?
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parent.parent / "number1_number2"))

import json
import scene_grouping as sg
from tp1 import read_frames

INPUT = Path("../input")
gt = json.load(open(INPUT / "dev_02.json"))
frames = read_frames(str(INPUT / "dev_02.mp4"))

# Extract keyframes for the 2 pairs
kf_2 = sg.extract_keyframes(frames, gt["shots"][2], "uniform", K=3)
kf_15 = sg.extract_keyframes(frames, gt["shots"][15], "uniform", K=3)
kf_10 = sg.extract_keyframes(frames, gt["shots"][10], "uniform", K=3)
kf_29 = sg.extract_keyframes(frames, gt["shots"][29], "uniform", K=3)

# Quick check — do Lab or RGB give higher values for these pairs?
kf_2_lab  = sg.lab_similarity(kf_2[0], kf_15[0])
kf_10_lab = sg.lab_similarity(kf_10[0], kf_29[0])
print(f"Lab pair #2/#15  : {kf_2_lab:.4f}")
print(f"Lab pair #10/#29 : {kf_10_lab:.4f}")