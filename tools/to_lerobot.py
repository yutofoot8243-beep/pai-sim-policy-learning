"""eval_node が record:=true で保存したデモ (episode_XXXX.npz) を LeRobot のデータセットに変換する.

  python tools/to_lerobot.py <demos のフォルダ> [...] --root out/pai_oracle_n1 [--repo-id local/pai_oracle_n1]

lerobot の版によってデータセットの形式が変わるので、学習に使うのと同じ環境（Colab）で実行する.
lerobot 0.6.1 で確認した.

- observation.images.top: 真上カメラ (240, 320, 3) RGB
- observation.state:      qpos[t]   joint1〜4 とハンド [rad]
- action:                 qpos[t+1] 次のフレームの関節角（最後のフレームは qpos[T-1]）
- 成功したエピソード (meta.success) だけを使う
- --trim-idle RAD: 最初にアームが止まっているフレームを落とし、動き出す 1 フレーム前から使う.
  oracle は手前のブロックほど計画に時間がかかり home で長く止まるので、時刻を知らない ACT が
  「動かない」を覚えてしまう（10/28 の評価で、手前の 3 本が 40 s 動かなかった）

カメラのフレームはシミュレーション時間でほぼ 10 fps だが、取りこぼしで間隔が空くことがある.
LeRobot はフレーム番号 / fps を時刻とみなすので、1/fps 刻みの時刻に一番近いフレームを選び直す.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_demo import load_demo  # noqa: E402

IMAGE_KEY = 'observation.images.top'


def resample(t, fps):
    """1/fps 刻みの時刻ごとに、一番近いフレームの番号を返す."""
    grid = np.arange(t[0], t[-1] + 0.5 / fps, 1.0 / fps)
    idx = np.searchsorted(t, grid).clip(1, len(t) - 1)
    idx -= ((grid - t[idx - 1]) < (t[idx] - grid)).astype(idx.dtype)
    return idx


def first_motion(qpos, tol):
    """アーム (joint1〜4) が最初の姿勢から tol [rad] より動いたフレームの 1 つ前の番号."""
    moved = np.abs(qpos[:, :4] - qpos[0, :4]).max(axis=1) > tol
    return max(int(moved.argmax()) - 1, 0) if moved.any() else 0


def to_frames(demo, fps, trim_idle=0.0):
    """1 エピソード分を (画像 RGB, state, action) の列にする."""
    idx = resample(demo['t'], fps)
    if trim_idle > 0:
        idx = idx[first_motion(demo['qpos'][idx], trim_idle):]
    qpos = demo['qpos'][idx].astype(np.float32)
    action = np.concatenate([qpos[1:], qpos[-1:]])
    for k, i in enumerate(idx):
        rgb = cv2.cvtColor(demo['frames'][i], cv2.COLOR_BGR2RGB)
        yield rgb, qpos[k], action[k]


def features(joint_names, image_shape, use_videos):
    return {
        IMAGE_KEY: {'dtype': 'video' if use_videos else 'image', 'shape': image_shape,
                    'names': ['height', 'width', 'channels']},
        'observation.state': {'dtype': 'float32', 'shape': (len(joint_names),),
                              'names': joint_names},
        'action': {'dtype': 'float32', 'shape': (len(joint_names),), 'names': joint_names},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('demo_dirs', nargs='+', help='episode_XXXX.npz が入ったフォルダ')
    ap.add_argument('--root', required=True, help='データセットの出力先（既にあれば --overwrite が必要）')
    ap.add_argument('--repo-id', default='local/pai_oracle')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--images', action='store_true',
                    help='画像を動画にせず 1 枚ずつ保存する（ffmpeg がない環境向け. 容量は増える）')
    ap.add_argument('--trim-idle', type=float, default=0.0,
                    help='最初にアームが止まっているフレームを落とすしきい値 [rad]（0 なら落とさない）')
    ap.add_argument('--max-episodes', type=int, default=0, help='0 なら全部（動作確認用）')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    paths = sorted(p for d in args.demo_dirs for p in glob.glob(os.path.join(d, 'episode_*.npz')))
    if not paths:
        sys.exit(f'no episode_*.npz in {args.demo_dirs}')
    if os.path.exists(args.root):
        if not args.overwrite:
            sys.exit(f'{args.root} already exists (use --overwrite)')
        shutil.rmtree(args.root)

    dataset = None
    used, skipped, n_frames = 0, [], 0
    for path in paths:
        if args.max_episodes and used >= args.max_episodes:
            break
        demo = load_demo(path)
        name = os.path.basename(path)
        if not demo['meta']['success']:
            skipped.append((name, 'failed'))
            continue
        if len(demo['t']) < 2:
            skipped.append((name, 'too few frames'))
            continue
        if dataset is None:
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id, fps=args.fps, root=args.root, robot_type='crane_plus',
                features=features(demo['joint_names'], demo['frames'][0].shape, not args.images),
                use_videos=not args.images)
        for rgb, state, action in to_frames(demo, args.fps, args.trim_idle):
            dataset.add_frame({IMAGE_KEY: rgb, 'observation.state': state, 'action': action,
                               'task': demo['meta']['instruction']})
            n_frames += 1
        dataset.save_episode()
        used += 1
        print(f'[{used}] {name}: {len(demo["t"])} -> {dataset.meta.total_frames} frames total',
              flush=True)

    if dataset is None:
        sys.exit('no successful episodes')
    dataset.finalize()
    print(f'episodes: {used}, frames: {n_frames}, skipped: {len(skipped)}')
    for name, why in skipped:
        print(f'  skipped {name}: {why}')
    with open(os.path.join(args.root, 'source.json'), 'w') as f:
        json.dump({'demos': [os.path.abspath(d) for d in args.demo_dirs], 'fps': args.fps,
                   'trim_idle': args.trim_idle, 'episodes': used, 'skipped': skipped}, f, ensure_ascii=False, indent=1)
    print('wrote', args.root)


if __name__ == '__main__':
    main()
