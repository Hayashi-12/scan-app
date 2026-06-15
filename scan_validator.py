"""
scan_validator.py
撮影条件を自動チェックして、NGなら理由とともに解析を拒否するモジュール。
analyze_burn.py の冒頭で呼び出して使う。
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ScanStandard:
    """撮影条件の基準値（実験環境に合わせて調整してください）"""
    brightness_min: float = 100.0
    brightness_max: float = 200.0
    saturation_min: float = 40.0
    organ_ratio_min: float = 0.20
    organ_ratio_max: float = 0.75
    center_offset_max: float = 0.20
    blur_threshold: float = 80.0
    bg_uniformity_max: float = 40.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    value: float = float("nan")
    target: str = ""


@dataclass
class ValidationReport:
    passed: bool
    checks: list = field(default_factory=list)

    def print(self):
        print("\n" + "=" * 50)
        print("  撮影条件チェック結果")
        print("=" * 50)
        for c in self.checks:
            mark = "✓" if c.ok else "✗"
            val_str = f"  (計測値: {c.value:.1f})" if not np.isnan(c.value) else ""
            target_str = f"  [基準: {c.target}]" if c.target else ""
            print(f"  {mark} {c.name}: {c.message}{val_str}{target_str}")
        print("-" * 50)
        if self.passed:
            print("  → 撮影条件OK。解析を開始します。")
        else:
            print("  → 撮影条件NG。修正してから再撮影してください。")
        print("=" * 50 + "\n")


def validate_scan(img_bgr, standard=None):
    """撮影画像の品質を自動チェックする"""
    if standard is None:
        standard = ScanStandard()

    checks = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = gray.shape
    S, V = hsv[:, :, 1], hsv[:, :, 2]

    # 1. 明るさ
    brightness = float(np.mean(gray))
    rng = f"{standard.brightness_min:.0f}〜{standard.brightness_max:.0f}"
    if brightness < standard.brightness_min:
        checks.append(CheckResult("明るさ", False, "暗すぎます", brightness, rng))
    elif brightness > standard.brightness_max:
        checks.append(CheckResult("明るさ", False, "明るすぎます", brightness, rng))
    else:
        checks.append(CheckResult("明るさ", True, "OK", brightness, rng))

    # 2. ぼけ
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    tgt = f">{standard.blur_threshold:.0f}"
    if lap_var < standard.blur_threshold:
        checks.append(CheckResult("ピント", False, "ぼけています", lap_var, tgt))
    else:
        checks.append(CheckResult("ピント", True, "OK", lap_var, tgt))

    # 3. 臓器検出
    organ_candidate = ((S > 35) & (V > 25)).astype(np.uint8) * 255
    organ_candidate = cv2.morphologyEx(organ_candidate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    organ_candidate = cv2.morphologyEx(organ_candidate, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(organ_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        checks.append(CheckResult("臓器検出", False, "臓器が検出できません"))
    else:
        checks.append(CheckResult("臓器検出", True, "検出OK"))
        organ_contour = max(contours, key=cv2.contourArea)
        organ_mask = np.zeros_like(gray)
        cv2.drawContours(organ_mask, [organ_contour], -1, 255, -1)

        ratio = cv2.contourArea(organ_contour) / (h * w)
        rng = f"{standard.organ_ratio_min*100:.0f}〜{standard.organ_ratio_max*100:.0f}%"
        if ratio < standard.organ_ratio_min:
            checks.append(CheckResult("サイズ", False, "遠すぎます", ratio*100, rng))
        elif ratio > standard.organ_ratio_max:
            checks.append(CheckResult("サイズ", False, "近すぎます", ratio*100, rng))
        else:
            checks.append(CheckResult("サイズ", True, "OK", ratio*100, rng))

        M = cv2.moments(organ_contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            off = max(abs(cx - w/2) / w, abs(cy - h/2) / h)
            tgt = f"<{standard.center_offset_max*100:.0f}%"
            if off > standard.center_offset_max:
                checks.append(CheckResult("中心", False, "ずれています", off*100, tgt))
            else:
                checks.append(CheckResult("中心", True, "OK", off*100, tgt))

    all_ok = all(c.ok for c in checks)
    return ValidationReport(passed=all_ok, checks=checks)


def validate_or_abort(image_path, standard=None, strict=True):
    """画像を読み込んでバリデーション。NGならエラーを出して停止。"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")

    img_blur = cv2.GaussianBlur(img, (7, 7), 0)
    report = validate_scan(img_blur, standard)
    report.print()

    if strict and not report.passed:
        raise ValueError("撮影条件NG。再撮影してください。")

    return img
