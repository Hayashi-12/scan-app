"""
analyze_burn.py
模擬臓器の焦げ領域・残存腫瘍を画像解析するスクリプト。

使い方:
  直接実行:
    python analyze_burn.py
    python analyze_burn.py input/S001_post.jpeg

  sync_download.py から自動呼び出し:
    環境変数 SCAN_IMAGE_PATH, SCAN_OUTPUT_DIR, SCAN_SUBJECT_ID で制御
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scan_validator import validate_or_abort, ScanStandard

# =====================
# 設定（環境変数があればそちらを優先）
# =====================
IMAGE_PATH = os.environ.get("SCAN_IMAGE_PATH", "input/IMG_0133.jpeg")
OUTPUT_DIR = Path(os.environ.get("SCAN_OUTPUT_DIR", "output"))
SUBJECT_ID = os.environ.get("SCAN_SUBJECT_ID", "unknown")

# コマンドライン引数があればそちらを使う
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    IMAGE_PATH = sys.argv[1]

OUTPUT_DIR.mkdir(exist_ok=True)

# 焦げ抽出パラメータ
DARK_V_MAX = 115
DARK_L_MAX = 115
MIN_BURN_AREA = 1200

# 残存腫瘍抽出パラメータ
TUMOR_GRAY_MIN = 140
TUMOR_S_MAX = 100
MIN_TUMOR_AREA = 500

# 3段階分類 V値
SEVERE_V_MAX = 45
MODERATE_V_MAX = 80
MILD_V_MAX = 100


# =====================
# ユーティリティ関数
# =====================
def remove_small_components(mask, min_area):
    """小さい連結成分を除去"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def keep_largest_components(mask, top_n=1):
    """上位N個の大きい連結成分のみ残す"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask

    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    keep_ids = [i for i, _ in areas[:top_n]]

    cleaned = np.zeros_like(mask)
    for i in keep_ids:
        cleaned[labels == i] = 255
    return cleaned


def calc_shape_metrics(mask):
    """マスクから面積・周囲長・円形度・重心を計算"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {
            "area": 0, "perimeter": 0, "circularity": np.nan,
            "centroid_x": np.nan, "centroid_y": np.nan,
        }

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else np.nan

    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid_x = M["m10"] / M["m00"]
        centroid_y = M["m01"] / M["m00"]
    else:
        centroid_x = centroid_y = np.nan

    return {
        "area": area, "perimeter": perimeter, "circularity": circularity,
        "centroid_x": centroid_x, "centroid_y": centroid_y,
    }


def calc_centroid_distance(m1, m2):
    """2つのメトリクス間の重心距離"""
    if np.isnan(m1["centroid_x"]) or np.isnan(m2["centroid_x"]):
        return np.nan
    dx = m1["centroid_x"] - m2["centroid_x"]
    dy = m1["centroid_y"] - m2["centroid_y"]
    return float(np.sqrt(dx ** 2 + dy ** 2))


def calc_burn_gradient(burn_mask, V_channel):
    """焦げ領域内のグラジュエーション（中心-周辺の濃さ相関）"""
    if burn_mask.sum() == 0:
        return {"burn_gradient_corr": np.nan, "burn_v_std": np.nan}

    y_coords, x_coords = np.where(burn_mask > 0)
    cy, cx = y_coords.mean(), x_coords.mean()
    distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    v_values = V_channel[burn_mask > 0].astype(float)

    if distances.std() == 0 or v_values.std() == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(distances, v_values)[0, 1])

    return {
        "burn_gradient_corr": corr,
        "burn_v_std": float(v_values.std()),
    }


# =====================
# メイン解析
# =====================
def analyze(image_path, output_dir, subject_id="unknown"):
    """1枚の画像を解析して結果を返す"""

    # バリデーション（strict=False で警告のみ、解析は続行）
    std = ScanStandard(brightness_min=80, brightness_max=220)
    try:
        img = validate_or_abort(image_path, standard=std, strict=False)
    except FileNotFoundError:
        print(f"[エラー] 画像が見つかりません: {image_path}")
        return None

    # ノイズ低減
    img = cv2.GaussianBlur(img, (7, 7), 0)
    original = img.copy()

    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L = lab[:, :, 0]

    # ─── 1. 模擬臓器領域抽出 ───
    # 模擬臓器の色相（赤橙系）に絞る
    # OpenCVのHue: 赤=0-10 & 170-180、橙=10-25
    # 木目(茶色)はHue=15-25でSが低め、臓器はSが高い
    red_low    = ((H <= 10) & (S > 60) & (V > 40))
    red_high   = ((H >= 170) & (S > 60) & (V > 40))
    orange     = ((H > 10) & (H <= 30) & (S > 80) & (V > 50))
    organ_candidate = (red_low | red_high | orange).astype(np.uint8) * 255

    kernel_large = np.ones((15, 15), np.uint8)
    kernel_mid   = np.ones((7, 7), np.uint8)
    organ_candidate = cv2.morphologyEx(organ_candidate, cv2.MORPH_CLOSE, kernel_large)
    organ_candidate = cv2.morphologyEx(organ_candidate, cv2.MORPH_OPEN, kernel_mid)

    contours, _ = cv2.findContours(organ_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("[エラー] 模擬臓器領域を検出できませんでした。")
        return None

    organ_contour = max(contours, key=cv2.contourArea)
    organ_mask = np.zeros_like(gray)
    cv2.drawContours(organ_mask, [organ_contour], -1, 255, -1)

    # ─── 2. 焦げ領域抽出 ───
    burn_candidate = (
        (organ_mask > 0) & (V < DARK_V_MAX) & (L < DARK_L_MAX)
    ).astype(np.uint8) * 255

    white_region = (
        (gray > TUMOR_GRAY_MIN) & (S < TUMOR_S_MAX) & (organ_mask > 0)
    ).astype(np.uint8) * 255
    burn_candidate[white_region > 0] = 0

    kernel_small = np.ones((5, 5), np.uint8)
    kernel_burn  = np.ones((11, 11), np.uint8)
    burn_candidate = cv2.morphologyEx(burn_candidate, cv2.MORPH_OPEN, kernel_small)
    burn_candidate = cv2.morphologyEx(burn_candidate, cv2.MORPH_CLOSE, kernel_burn)
    burn_candidate = remove_small_components(burn_candidate, MIN_BURN_AREA)
    burn_mask = keep_largest_components(burn_candidate, top_n=1)

    # ─── 3. 焦げ3段階分類 ───
    mild_mask     = ((burn_mask > 0) & (V >= MODERATE_V_MAX) & (V < MILD_V_MAX)).astype(np.uint8) * 255
    moderate_mask = ((burn_mask > 0) & (V >= SEVERE_V_MAX) & (V < MODERATE_V_MAX)).astype(np.uint8) * 255
    severe_mask   = ((burn_mask > 0) & (V < SEVERE_V_MAX)).astype(np.uint8) * 255

    # ─── 4. 残存腫瘍領域抽出 ───
    tumor_candidate = (
        (gray > TUMOR_GRAY_MIN) & (S < TUMOR_S_MAX) & (organ_mask > 0)
    ).astype(np.uint8) * 255
    tumor_candidate = cv2.morphologyEx(tumor_candidate, cv2.MORPH_OPEN, kernel_small)
    tumor_candidate = cv2.morphologyEx(tumor_candidate, cv2.MORPH_CLOSE, kernel_mid)
    tumor_candidate = remove_small_components(tumor_candidate, MIN_TUMOR_AREA)
    tumor_mask = keep_largest_components(tumor_candidate, top_n=1)

    # ─── 5. 指標計算 ───
    organ_m    = calc_shape_metrics(organ_mask)
    burn_m     = calc_shape_metrics(burn_mask)
    mild_m     = calc_shape_metrics(mild_mask)
    moderate_m = calc_shape_metrics(moderate_mask)
    severe_m   = calc_shape_metrics(severe_mask)
    tumor_m    = calc_shape_metrics(tumor_mask)

    oa = organ_m["area"]
    ba = burn_m["area"]

    # グラジュエーション（新規追加指標）
    grad = calc_burn_gradient(burn_mask, V)

    # 焦げの濃さ
    burn_mean_v    = float(np.mean(V[burn_mask > 0])) if ba > 0 else np.nan
    burn_mean_gray = float(np.mean(gray[burn_mask > 0])) if ba > 0 else np.nan

    result = {
        "subject_id": subject_id,

        "organ_area_px": oa,
        "organ_perimeter_px": organ_m["perimeter"],
        "organ_circularity": organ_m["circularity"],

        "burn_area_px": ba,
        "burn_ratio_to_organ": ba / oa if oa > 0 else 0,
        "burn_perimeter_px": burn_m["perimeter"],
        "burn_circularity": burn_m["circularity"],
        "burn_mean_v": burn_mean_v,
        "burn_mean_gray": burn_mean_gray,
        "burn_gradient_corr": grad["burn_gradient_corr"],
        "burn_v_std": grad["burn_v_std"],

        "mild_burn_area_px": mild_m["area"],
        "mild_ratio_in_burn": mild_m["area"] / ba if ba > 0 else 0,
        "moderate_burn_area_px": moderate_m["area"],
        "moderate_ratio_in_burn": moderate_m["area"] / ba if ba > 0 else 0,
        "severe_burn_area_px": severe_m["area"],
        "severe_ratio_in_burn": severe_m["area"] / ba if ba > 0 else 0,

        "residual_tumor_area_px": tumor_m["area"],
        "residual_tumor_ratio_to_organ": tumor_m["area"] / oa if oa > 0 else 0,
        "residual_tumor_circularity": tumor_m["circularity"],

        "burn_tumor_centroid_dist_px": calc_centroid_distance(burn_m, tumor_m),
        "burn_organ_centroid_dist_px": calc_centroid_distance(burn_m, organ_m),
        "tumor_organ_centroid_dist_px": calc_centroid_distance(tumor_m, organ_m),
    }

    # ─── 6. CSV保存 ───
    csv_path = output_dir / "burn_metrics.csv"
    df_new = pd.DataFrame([result])

    # 既存CSVがあれば追記、なければ新規作成
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ─── 7. 可視化 ───
    overlay = original.copy()
    overlay[mild_mask > 0]     = [0, 255, 255]    # 軽度: 黄
    overlay[moderate_mask > 0] = [0, 165, 255]    # 中等度: オレンジ
    overlay[severe_mask > 0]   = [0, 0, 255]      # 重度: 赤
    overlay[tumor_mask > 0]    = [255, 0, 0]      # 残存腫瘍: 青
    blended = cv2.addWeighted(original, 0.65, overlay, 0.35, 0)

    contour_img = original.copy()
    burn_contours, _ = cv2.findContours(burn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tumor_contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, burn_contours, -1, (0, 255, 0), 3)
    cv2.drawContours(contour_img, tumor_contours, -1, (255, 0, 0), 3)

    if not np.isnan(burn_m["centroid_x"]):
        cv2.circle(contour_img, (int(burn_m["centroid_x"]), int(burn_m["centroid_y"])), 8, (0, 255, 0), -1)
    if not np.isnan(tumor_m["centroid_x"]):
        cv2.circle(contour_img, (int(tumor_m["centroid_x"]), int(tumor_m["centroid_y"])), 8, (255, 0, 0), -1)

    # ─── 8. 画像保存 ───
    prefix = subject_id if subject_id != "unknown" else ""
    p = lambda name: str(output_dir / f"{prefix}_{name}" if prefix else output_dir / name)

    cv2.imwrite(p("organ_mask.png"), organ_mask)
    cv2.imwrite(p("burn_mask.png"), burn_mask)
    cv2.imwrite(p("burn_stage_overlay.png"), blended)
    cv2.imwrite(p("burn_contour_result.png"), contour_img)

    print(f"\n解析完了: {subject_id}")
    print(f"  焦げ率: {result['burn_ratio_to_organ']*100:.1f}%")
    print(f"  焦げ面積: {int(ba)} px")
    print(f"  残存腫瘍面積: {int(tumor_m['area'])} px")
    print(f"  CSV保存先: {csv_path}")

    return result


# =====================
# メイン実行
# =====================
if __name__ == "__main__":
    result = analyze(IMAGE_PATH, OUTPUT_DIR, SUBJECT_ID)
    if result:
        print("\n全指標:")
        for k, v in result.items():
            if k == "subject_id":
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
