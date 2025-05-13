from __future__ import annotations
import random
import subprocess
import sys
import csv
from pathlib import Path
from typing import List, Tuple
import shutil
import re

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
DATA_PATH = Path("sample_tiktok.csv")
CPP_SRC   = Path("tiktok_knn")
CPP_EXE   = Path("tiktok_knn.exe" if sys.platform.startswith("win") else "tiktok_knn")
K_DEFAULT = 5  # neighbours

TAG_POOL: List[str] = [
    "dance", "funny", "cat", "dog", "challenge", "meme", "food", "travel",
    "fitness", "makeup", "gaming", "tutorial", "music", "comedy", "sports",
]

# ------------------------------------------------------------
# 1) Random dataset helper
# ------------------------------------------------------------

def generate_dataset(path: Path, n: int = 100) -> None:
    """Создать CSV `id,tags,views,likes,comments` с n строками."""
    print(f"[INFO] Creating random dataset → {path} ({n} rows)…")
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "tags", "views", "likes", "comments"])
        for i in range(1, n + 1):
            n_tags = random.randint(1, 4)
            tags   = random.sample(TAG_POOL, n_tags)
            views  = random.randint(1_000, 1_000_000)
            likes  = int(views * random.uniform(0.05, 0.2))
            comm   = int(views * random.uniform(0.01, 0.05))
            wr.writerow([i, ";".join(tags), views, likes, comm])

if not DATA_PATH.exists():
    generate_dataset(DATA_PATH)

# ------------------------------------------------------------
# 2) Compile C++ binary if missing
# ------------------------------------------------------------

def compile_cpp(src: Path, out: Path) -> None:
    print("[INFO] Compiling C++ source →", out)
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        sys.exit("[FATAL]  C++ compiler not found. Install g++ or clang++.")
    cmd = [compiler, "-O2", "-std=c++17", str(src), "-o", str(out)]
    subprocess.check_call(cmd)

if not CPP_EXE.exists():
    if not CPP_SRC.exists():
        sys.exit("[FATAL]  Missing tiktok_knn.cpp in working directory.")
    compile_cpp(CPP_SRC, CPP_EXE)

# ------------------------------------------------------------
# 3) Utils to load the whole dataset (for scatter‑plot)
# ------------------------------------------------------------

def load_dataset_metrics(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays: views, likes, comments (dtype=float)."""
    views, likes, comments = [], [], []
    with path.open(encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            views.append(float(row["views"]))
            likes.append(float(row["likes"]))
            comments.append(float(row["comments"]))
    return np.asarray(views), np.asarray(likes), np.asarray(comments)

ALL_VIEWS, ALL_LIKES, ALL_COMMS = load_dataset_metrics(DATA_PATH)

# ------------------------------------------------------------
# 4) Regex parsers for C++ stdout
# ------------------------------------------------------------
PRED_RE   = re.compile(r"Views:(\d+) Likes:(\d+) Comments:(\d+)")
NEIGH_RE  = re.compile(r"id=([^\s]+).*? (\d+) (\d+) (\d+)$")  # id views likes comments

# ------------------------------------------------------------
# 5) Main interactive loop
# ------------------------------------------------------------
print("\nTikTok‑KNN Visualiser (dual‑plot)\n" + "="*34)
print(f"Dataset : {DATA_PATH}  |  Binary : {CPP_EXE}\n")
print("Type semicolon‑separated tags (blank to exit).\n")

while True:
    raw = input("Enter tags > ").strip().lower()
    if not raw:
        print("👋  Bye!")
        break

    # -----------------------------------------------------------------
    # call C++ predictor
    # -----------------------------------------------------------------
    proc = subprocess.run(
        [str(CPP_EXE), str(K_DEFAULT), str(DATA_PATH), raw],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        print("[ERROR]", proc.stderr or proc.stdout)
        continue

    stdout = proc.stdout.strip()
    print("\n" + stdout + "\n")

    # -----------------------------------------------------------------
    # parse prediction & neighbour metrics
    # -----------------------------------------------------------------
    pm = PRED_RE.search(stdout)
    if not pm:
        print("[WARN] Could not parse prediction line → scatter skipped.")
        continue
    p_views, p_likes, p_comms = map(int, pm.groups())

    neigh_views, neigh_likes = [], []
    for line in stdout.splitlines():
        nm = NEIGH_RE.search(line)
        if nm:
            neigh_views.append(int(nm.group(2)))
            neigh_likes.append(int(nm.group(3)))

    # -----------------------------------------------------------------
    # Plot 1 — bar chart of prediction
    # -----------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(["Views", "Likes", "Comments"], [p_views, p_likes, p_comms])
    plt.title("Predicted engagement")
    plt.ylabel("Count")
    plt.tight_layout()

    # -----------------------------------------------------------------
    # Plot 2 — scatter of dataset + neighbours + prediction
    # -----------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.scatter(ALL_VIEWS, ALL_LIKES, s=10, alpha=0.3, label="Dataset")
    if neigh_views:
        plt.scatter(neigh_views, neigh_likes, s=40, alpha=0.8, label="Neighbours")
    plt.scatter([p_views], [p_likes], marker="*", s=200, label="Prediction")
    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.title("Views vs Likes (bubble ≈ Comments)")
    plt.legend()
    plt.tight_layout()

    plt.show(block=False)
    plt.pause(0.1)  # allow GUI loop to process

print("Done.")