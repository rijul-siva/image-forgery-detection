"""
Synthetic Benchmark Dataset Generator for PixelGuard.
Designed and maintained by Member C.

Generates reproducible positive (copy-move forged with known transforms)
and negative (authentic, repetitive textures) test images and ground-truth JSONs.
"""

import json
import math
import os
from pathlib import Path
import cv2
import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORIGINAL_DIR = DATA_DIR / "original"
FORGED_DIR = DATA_DIR / "forged"
GT_DIR = DATA_DIR / "ground_truth"
TEST_DIR = DATA_DIR / "test"


def make_asymmetric_object_patch(size: int = 140, seed: int = 42) -> np.ndarray:
    """Generate a rich, asymmetric object patch with diverse corners and gradients."""
    rng = np.random.default_rng(seed)
    patch = np.full((size, size, 3), 235, dtype=np.uint8)

    # Varied colorful circles
    for _ in range(35):
        cx = int(rng.integers(12, size - 12))
        cy = int(rng.integers(12, size - 12))
        r = int(rng.integers(4, 12))
        col = (int(rng.integers(20, 230)), int(rng.integers(20, 230)), int(rng.integers(20, 230)))
        cv2.circle(patch, (cx, cy), r, col, -1, lineType=cv2.LINE_AA)

    # Prominent asymmetric L-shape / P-shape to avoid rotational symmetry
    cv2.rectangle(patch, (18, 20), (45, size - 25), (20, 20, 190), -1)
    cv2.rectangle(patch, (18, 20), (size - 35, 55), (20, 20, 190), -1)
    cv2.rectangle(patch, (size - 45, 55), (size - 25, 95), (20, 20, 190), -1)

    # Angled distinct lines
    for _ in range(8):
        x1, y1 = int(rng.integers(0, size)), int(rng.integers(0, size))
        x2, y2 = int(rng.integers(0, size)), int(rng.integers(0, size))
        col = (int(rng.integers(10, 240)), int(rng.integers(10, 240)), int(rng.integers(10, 240)))
        cv2.line(patch, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

    return patch


def make_authentic_background(w: int = 800, h: int = 600, seed: int = 100) -> np.ndarray:
    """Create a realistic non-uniform gradient background with mild organic noise."""
    rng = np.random.default_rng(seed)
    # Smooth vertical/horizontal gradient
    y = np.linspace(140, 90, h, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(130, 110, w, dtype=np.float32).reshape(1, -1)
    base = np.clip(y + x * 0.2, 0, 255).astype(np.uint8)
    bg = cv2.merge([base, base, base])

    # Add subtle background textures
    for _ in range(20):
        cx = int(rng.integers(20, w - 20))
        cy = int(rng.integers(20, h - 20))
        r = int(rng.integers(15, 60))
        col = (int(rng.integers(80, 160)), int(rng.integers(80, 160)), int(rng.integers(80, 160)))
        cv2.circle(bg, (cx, cy), r, col, -1)
    bg = cv2.GaussianBlur(bg, (25, 25), 10.0)
    return bg


def make_window_facade(w: int = 800, h: int = 600) -> np.ndarray:
    """Create a negative test image: architectural facade with repetitive window lattice."""
    canvas = np.full((h, w, 3), 160, dtype=np.uint8)  # Brick/concrete color
    win_w, win_h = 70, 90
    gap_x, gap_y = 50, 45
    start_x, start_y = 60, 60

    # Draw regular grid of identical windows
    for y in range(start_y, h - win_h - 20, win_h + gap_y):
        for x in range(start_x, w - win_w - 20, win_w + gap_x):
            # Window frame
            cv2.rectangle(canvas, (x, y), (x + win_w, y + win_h), (50, 50, 60), -1)
            # Glass pane
            cv2.rectangle(canvas, (x + 6, y + 6), (x + win_w - 6, y + win_h - 6), (200, 220, 235), -1)
            # Pane crossbar
            cv2.line(canvas, (x + win_w // 2, y + 6), (x + win_w // 2, y + win_h - 6), (50, 50, 60), 2)
            cv2.line(canvas, (x + 6, y + win_h // 2), (x + win_w - 6, y + win_h // 2), (50, 50, 60), 2)

    return canvas


def make_brick_wall(w: int = 800, h: int = 600) -> np.ndarray:
    """Create a negative test image: repetitive brick wall texture."""
    canvas = np.full((h, w, 3), 180, dtype=np.uint8)  # Mortar
    brick_h = 24
    brick_w = 60
    mortar = 4

    for row, y in enumerate(range(10, h - brick_h, brick_h + mortar)):
        offset = (brick_w // 2) if (row % 2 == 1) else 0
        for x in range(-offset, w, brick_w + mortar):
            x1 = max(0, x)
            x2 = min(w, x + brick_w)
            y1 = y
            y2 = min(h, y + brick_h)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 65, 170), -1)
    return canvas


def generate_all_datasets():
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    FORGED_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print("[dataset] Generating synthetic authentic and forged dataset...")

    # --- Base Original Image ---
    bg_orig = make_authentic_background(800, 600, seed=101)
    patch_src = make_asymmetric_object_patch(size=140, seed=42)
    # Place pristine object in original image
    src_x, src_y = 120, 160
    patch_h, patch_w = patch_src.shape[:2]
    bg_orig[src_y:src_y + patch_h, src_x:src_x + patch_w] = patch_src

    orig_path = ORIGINAL_DIR / "original_001.png"
    cv2.imwrite(str(orig_path), bg_orig)

    # --- Test Scenarios ---
    scenarios = [
        {
            "id": "forged_001_clean_translation",
            "theta": 0.0,
            "scale": 1.0,
            "dx": 350.0,
            "dy": 120.0,
            "desc": "Clean translation without rotation or scaling",
            "blur": False,
            "quality": 95,
        },
        {
            "id": "forged_002_rotation_15deg",
            "theta": 15.0,
            "scale": 1.0,
            "dx": 320.0,
            "dy": 100.0,
            "desc": "Copy-move with +15.0 deg in-plane rotation",
            "blur": False,
            "quality": 95,
        },
        {
            "id": "forged_003_rotation_neg25deg",
            "theta": -25.0,
            "scale": 1.0,
            "dx": 300.0,
            "dy": -60.0,
            "desc": "Copy-move with -25.0 deg in-plane rotation",
            "blur": False,
            "quality": 95,
        },
        {
            "id": "forged_004_scaled_up",
            "theta": 0.0,
            "scale": 1.15,
            "dx": 340.0,
            "dy": 80.0,
            "desc": "Copy-move with 1.15x scale enlargement",
            "blur": False,
            "quality": 95,
        },
        {
            "id": "forged_005_scaled_down",
            "theta": 0.0,
            "scale": 0.85,
            "dx": 310.0,
            "dy": 140.0,
            "desc": "Copy-move with 0.85x scale reduction",
            "blur": False,
            "quality": 95,
        },
        {
            "id": "forged_006_combined_rot_scale",
            "theta": 12.0,
            "scale": 1.10,
            "dx": 330.0,
            "dy": 110.0,
            "desc": "Copy-move with +12.0 deg rotation and 1.10x scale",
            "blur": False,
            "quality": 95,
        },
        {
            "id": "forged_007_blurred_copy",
            "theta": 0.0,
            "scale": 1.0,
            "dx": 360.0,
            "dy": 90.0,
            "desc": "Copy-move with mild Gaussian blur applied to destination patch",
            "blur": True,
            "quality": 95,
        },
        {
            "id": "forged_008_jpeg_compressed",
            "theta": 10.0,
            "scale": 1.05,
            "dx": 320.0,
            "dy": 70.0,
            "desc": "Copy-move saved at JPEG quality 75 to test compression robustness",
            "blur": False,
            "quality": 75,
        },
    ]

    for sc in scenarios:
        img = bg_orig.copy()
        theta = sc["theta"]
        scale = sc["scale"]
        dx = sc["dx"]
        dy = sc["dy"]

        # Transform source patch
        center = (patch_w / 2.0, patch_h / 2.0)
        M_rot = cv2.getRotationMatrix2D(center, theta, scale)
        # Warp with transparent/border handling
        transformed_patch = cv2.warpAffine(
            patch_src, M_rot, (patch_w, patch_h),
            borderMode=cv2.BORDER_REFLECT
        )

        if sc["blur"]:
            transformed_patch = cv2.GaussianBlur(transformed_patch, (5, 5), 1.2)

        dst_x = int(round(src_x + dx))
        dst_y = int(round(src_y + dy))

        # Paste transformed patch onto destination
        img[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = transformed_patch

        ext = ".jpg" if sc["quality"] < 90 else ".png"
        out_name = f"{sc['id']}{ext}"
        out_path = FORGED_DIR / out_name

        if ext == ".jpg":
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, sc["quality"]])
        else:
            cv2.imwrite(str(out_path), img)

        # Ground truth distance
        dist = math.hypot(dx, dy)

        gt_data = {
            "dataset_type": "synthetic_benchmark",
            "source_image": "original_001.png",
            "forged_image": out_name,
            "is_forged": True,
            "scenario": sc["id"],
            "description": sc["desc"],
            "rotation_degrees": float(theta),
            "scale_factor": float(scale),
            "translation_x_px": float(dx),
            "translation_y_px": float(dy),
            "translation_distance_px": float(dist),
            "coordinate_system": "pixel_space (x: right, y: down, origin: top-left)",
            "source_region_bbox": [src_x, src_y, patch_w, patch_h],
            "target_region_bbox": [dst_x, dst_y, patch_w, patch_h],
        }

        gt_json_path = GT_DIR / f"{sc['id']}.json"
        with open(gt_json_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=4)

    # --- Negative Test Images ---
    # 1. Authentic Image (no copy-paste)
    neg_auth_path = TEST_DIR / "negative_authentic_001.png"
    cv2.imwrite(str(neg_auth_path), bg_orig)
    with open(GT_DIR / "negative_authentic_001.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_type": "synthetic_benchmark",
            "image_name": "negative_authentic_001.png",
            "is_forged": False,
            "scenario": "authentic_genuine",
            "description": "Authentic scene containing single object without copy-paste"
        }, f, indent=4)

    # 2. Repetitive Windows Facade
    neg_win = make_window_facade()
    neg_win_path = TEST_DIR / "negative_repetitive_windows_001.png"
    cv2.imwrite(str(neg_win_path), neg_win)
    with open(GT_DIR / "negative_repetitive_windows_001.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_type": "synthetic_benchmark",
            "image_name": "negative_repetitive_windows_001.png",
            "is_forged": False,
            "scenario": "repetitive_texture_windows",
            "description": "Architectural repetitive window grid for stress-testing false-positive suppression"
        }, f, indent=4)

    # 3. Repetitive Brick Wall Texture
    neg_brick = make_brick_wall()
    neg_brick_path = TEST_DIR / "negative_repetitive_bricks_002.png"
    cv2.imwrite(str(neg_brick_path), neg_brick)
    with open(GT_DIR / "negative_repetitive_bricks_002.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_type": "synthetic_benchmark",
            "image_name": "negative_repetitive_bricks_002.png",
            "is_forged": False,
            "scenario": "repetitive_texture_bricks",
            "description": "Periodic brick wall pattern for stress-testing false-positive suppression"
        }, f, indent=4)

    print(f"[dataset] Generation complete. Generated 1 original, 8 forged, 3 negative test cases.")


if __name__ == "__main__":
    generate_all_datasets()
