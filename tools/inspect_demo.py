"""eval_node が record:=true で保存したデモ (episode_XXXX.npz) の中身を確かめる.

  python3 tools/inspect_demo.py ~/pai_eval/<dir>/demos/episode_0000.npz [--sheet out.jpg]

フレーム数・フレームレート（シミュレーション時間）・関節角の範囲を表示し、
--sheet を付けると等間隔に 8 枚並べた画像を保存する.
"""

import argparse
import json

import cv2
import numpy as np


def load_demo(path):
    d = np.load(path)
    offsets = d['jpeg_offsets']
    raw = d['jpeg_bytes']
    frames = [cv2.imdecode(raw[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR)
              for i in range(len(offsets) - 1)]
    return {'t': d['t'], 'qpos': d['qpos'], 'frames': frames,
            'joint_names': list(d['joint_names']), 'meta': json.loads(str(d['meta']))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--sheet', help='等間隔に 8 枚並べた画像の保存先')
    args = ap.parse_args()

    demo = load_demo(args.npz)
    t, qpos = demo['t'], demo['qpos']
    n = len(t)
    print(f"instruction: {demo['meta']['instruction']}  success: {demo['meta']['success']}")
    if n < 2:
        print(f'frames: {n} (too few)')
        return
    dt = np.diff(t)
    print(f'frames: {n}, duration {t[-1] - t[0]:.1f} s (sim), '
          f'{(n - 1) / (t[-1] - t[0]):.1f} fps, max gap {dt.max():.2f} s')
    print(f"image: {demo['frames'][0].shape}")
    for name, lo, hi in zip(demo['joint_names'], qpos.min(0), qpos.max(0)):
        print(f'  {name:24s} {lo:+.3f} .. {hi:+.3f} rad')

    if args.sheet:
        idx = np.linspace(0, n - 1, 8).astype(int)
        tiles = []
        for i in idx:
            img = demo['frames'][i].copy()
            cv2.putText(img, f't={t[i] - t[0]:.1f}s', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            tiles.append(img)
        sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        cv2.imwrite(args.sheet, sheet)
        print('wrote', args.sheet)


if __name__ == '__main__':
    main()
