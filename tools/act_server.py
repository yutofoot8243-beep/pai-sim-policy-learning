"""学習した ACT の推論サーバー（lerobot が入った Python 3.12 以上の venv で動かす）.

ROS のノード (Python 3.10) からは lerobot を import できないので、推論だけを別のプロセスにする.
通信は multiprocessing.connection（127.0.0.1 のみ）. numpy の版が違っても受け渡せるように、
送り合うのは bytes・list・str だけにする.

  要求: {'jpeg': 320x240 の JPEG (BGR を符号化したもの), 'state': [5 floats], 'task': str}
  応答: {'actions': [[5 floats] x chunk_size], 'ms': 推論にかかった時間}

    ~/venvs/lerobot/bin/python -I tools/act_server.py <pretrained_model> [--port 5555]
    ~/venvs/lerobot/bin/python -I tools/act_server.py <pretrained_model> --selftest <demo.npz>
"""
import argparse
from multiprocessing.connection import Client, Listener
import time

import cv2
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

DEFAULT_PORT = 5555


class ActRunner:
    def __init__(self, ckpt):
        policy = ACTPolicy.from_pretrained(ckpt)
        policy.config.device = 'cpu'
        self.policy = policy.to('cpu').eval()
        self.pre, self.post = make_pre_post_processors(
            policy.config, pretrained_path=ckpt,
            preprocessor_overrides={'device_processor': {'device': 'cpu'}})

    def __call__(self, req):
        bgr = cv2.imdecode(np.frombuffer(req['jpeg'], np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        obs = {'observation.images.top': torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0,
               'observation.state': torch.tensor(req['state'], dtype=torch.float32),
               'task': req['task']}
        t0 = time.perf_counter()
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(self.pre(obs))   # (1, chunk, 5) 正規化済み
            actions = self.post(chunk)
        ms = (time.perf_counter() - t0) * 1e3
        return {'actions': actions.reshape(-1, actions.shape[-1]).tolist(), 'ms': ms}


def serve(runner, port):
    with Listener(('127.0.0.1', port)) as listener:
        print(f'[act_server] listening on 127.0.0.1:{port}', flush=True)
        while True:
            with listener.accept() as conn:
                print('[act_server] client connected', flush=True)
                n = 0
                try:
                    while True:
                        reply = runner(conn.recv())
                        conn.send(reply)
                        n += 1
                        if n % 20 == 0:
                            print(f'[act_server] {n} requests, last {reply["ms"]:.0f} ms', flush=True)
                except EOFError:
                    print(f'[act_server] client disconnected after {n} requests', flush=True)


def selftest(runner, npz, port):
    """デモの数フレームを、実際にソケットを通して送り、デモの次のフレームの関節角と比べる."""
    import json
    import threading
    d = np.load(npz)
    offsets, raw = d['jpeg_offsets'], d['jpeg_bytes']
    qpos = d['qpos']
    task = json.loads(str(d['meta']))['instruction']
    threading.Thread(target=serve, args=(runner, port), daemon=True).start()
    time.sleep(1.0)
    with Client(('127.0.0.1', port)) as conn:
        for i in (0, 50, 100, 150):
            conn.send({'jpeg': raw[offsets[i]:offsets[i + 1]].tobytes(),
                       'state': qpos[i].tolist(), 'task': task})
            reply = conn.recv()
            a = np.array(reply['actions'])
            print(f'frame {i}: chunk {a.shape}, {reply["ms"]:.0f} ms, '
                  f'|a[0] - qpos[{i + 1}]| = {np.abs(a[0] - qpos[i + 1]).round(3).tolist()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--selftest', metavar='NPZ')
    args = ap.parse_args()
    runner = ActRunner(args.ckpt)
    if args.selftest:
        selftest(runner, args.selftest, args.port)
    else:
        serve(runner, args.port)


if __name__ == '__main__':
    main()
