"""学習した ACT を手元の CPU で動かし、デモ (npz) と比べる.

LeRobot のデータセットを通さず、npz の JPEG と関節角を直接入れる（ROS のノードで動かすときと同じ入れ方）.
各フレームで policy.reset() してチャンクの先頭を予測し、デモの次のフレームの関節角との差を見る.
あわせて 1 回の推論にかかる時間を測る.

    python tools/check_policy_cpu.py <pretrained_model> <demo.npz> [...] --plot out.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_demo import load_demo  # noqa: E402

from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402


def to_obs(frame_bgr, qpos, task):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return {'observation.images.top': img,
            'observation.state': torch.from_numpy(qpos.astype(np.float32)),
            'task': task}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('demos', nargs='+')
    ap.add_argument('--plot')
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.config.device = 'cpu'
    policy = policy.to('cpu').eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                         preprocessor_overrides={'device_processor': {'device': 'cpu'}})

    errs, times, first = [], [], None
    for path in args.demos:
        demo = load_demo(path)
        qpos = demo['qpos'].astype(np.float32)
        target = np.concatenate([qpos[1:], qpos[-1:]])
        task = demo['meta']['instruction']
        pred = []
        for i in range(len(qpos)):
            obs = to_obs(demo['frames'][i], qpos[i], task)
            policy.reset()
            t0 = time.perf_counter()
            with torch.no_grad():
                a = post(policy.select_action(pre(obs)))
            times.append(time.perf_counter() - t0)
            pred.append(a.cpu().numpy().ravel())
        pred = np.array(pred)
        err = np.abs(pred - target)
        errs.append(err)
        print(f'{Path(path).name}: frames={len(qpos)} mean|err|={err.mean(0).round(4).tolist()} '
              f'max|err|={err.max(0).round(3).tolist()}')
        if first is None:
            first = (Path(path).name, target, pred, demo['joint_names'])

    err = np.concatenate(errs)
    times = np.array(times[3:])  # 最初の数回は温まっていないので除く
    print('---')
    print('joints:', first[3])
    print('mean |err| [rad]:', err.mean(0).round(4).tolist())
    print('max  |err| [rad]:', err.max(0).round(3).tolist())
    print(f'inference [ms]: median={np.median(times) * 1e3:.0f} p90={np.percentile(times, 90) * 1e3:.0f} '
          f'max={times.max() * 1e3:.0f}  (threads={torch.get_num_threads()})')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        name, target, pred, joints = first
        fig, axes = plt.subplots(1, len(joints), figsize=(18, 3))
        for j, a in enumerate(axes):
            a.plot(target[:, j], label='oracle')
            a.plot(pred[:, j], '--', label='ACT (CPU)')
            a.set_title(joints[j].replace('crane_plus_', ''))
            a.set_xlabel('frame')
        axes[0].legend()
        fig.suptitle(name)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=100)
        print('plot:', args.plot)


if __name__ == '__main__':
    main()
