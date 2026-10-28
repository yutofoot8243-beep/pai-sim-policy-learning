"""Gemini が指した「掴む点」を、画像上で物体の上面の中心に寄せる（ROSに依存しない）.

Gemini Robotics-ER の点は「どの物体か」は正確だが、上面の端や側面を指して 1〜2 cm
ずれることがある（5 cm のブロックではこれで掴み損ねる）. そこで、指された点の周りで
「同じ色の明るく一様な領域 = 上面」を flood fill で切り出し、その重心を掴む点にする.
判断（何を掴むか）は VLM、最後の数 mm は古典的な画像処理、という役割分担.
"""

import cv2
import numpy as np


def _top_face_component(bgr, seed, color_tol, mask):
    """seed から flood fill した領域 (画素マスク) を返す. mask は 2px 大きい作業用."""
    mask[:] = 0
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    tol = (color_tol,) * 3
    cv2.floodFill(bgr, mask, seed, 0, tol, tol, flags)
    return mask[1:-1, 1:-1] > 0


def refine_pick_pixel(bgr, u, v, search_px=20, max_shift_px=35, color_tol=12,
                      min_area=400, max_area=8000, min_saturation=80, min_fill=0.6):
    """(u, v) 付近の物体上面の重心 (u', v') と、補正できたかを返す.

    上面の条件: 面積が min_area〜max_area [px]、彩度が高い（机の木目・トレイの白縁を除く）、
    外接矩形に対して詰まっている（細長い影や縁を除く）. 側面は上面より暗いので、
    条件を満たす候補のうち最も明るい領域を選ぶ. 見つからなければ元の点を返す.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    work = np.zeros((h + 2, w + 2), np.uint8)
    visited = np.zeros((h, w), bool)
    best = None   # (平均明度, 重心u, 重心v)

    for dv in range(-search_px, search_px + 1, 5):
        for du in range(-search_px, search_px + 1, 5):
            su, sv = int(round(u + du)), int(round(v + dv))
            if not (0 <= su < w and 0 <= sv < h) or visited[sv, su]:
                continue
            region = _top_face_component(bgr, (su, sv), color_tol, work)
            visited |= region
            area = int(region.sum())
            if not min_area <= area <= max_area:
                continue
            ys, xs = np.nonzero(region)
            fill = area / float((np.ptp(xs) + 1) * (np.ptp(ys) + 1))
            sat = float(hsv[..., 1][region].mean())
            if fill < min_fill or sat < min_saturation:
                continue
            cu, cv_ = float(xs.mean()), float(ys.mean())
            if np.hypot(cu - u, cv_ - v) > max_shift_px:
                continue
            val = float(hsv[..., 2][region].mean())
            if best is None or val > best[0]:
                best = (val, cu, cv_)

    if best is None:
        return (float(u), float(v)), False
    return (best[1], best[2]), True
