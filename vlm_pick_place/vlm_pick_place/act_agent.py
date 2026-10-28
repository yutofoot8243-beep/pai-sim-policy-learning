"""学習した ACT で1エピソードを閉ループで動かす（eval_node の agent:=act）.

推論は tools/act_server.py（lerobot が入った別の Python）に任せ、ここでは
  画像を縮めて JPEG にする → 推論サーバーへ送る → 返ってきたチャンクの先頭 N ステップを
  0.1 s 刻みの JointTrajectory 1本にしてアームとハンドのコントローラへ送る → N x 0.1 s 待つ
を決まった時間（シミュレーション時間）だけ繰り返す. 学習データは 10 fps で、action は次のフレームの関節角.
推論している間（実時間で 0.3〜0.7 s）アームは最後の点で止まって待つ.
"""

from multiprocessing.connection import Client
import time

import cv2
from builtin_interfaces.msg import Duration as DurationMsg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .arm_commander import ARM_JOINTS, HAND_JOINT

DT = 0.1   # 学習データのフレーム間隔 [s]


class ActAgent:
    def __init__(self, node):
        p = node.declare_parameter
        self._node = node
        self._port = p('act_port', 5555).value
        self._n_steps = p('act_n_action_steps', 10).value      # 1回の推論で実行するステップ数
        self._max_sec = p('act_max_sec', 40.0).value           # 1エピソードの長さ（シミュレーション時間）
        self._size = tuple(node.get_parameter('record_size').value)   # デモと同じ大きさに縮める
        # デモの action は実際の関節角なので、掴んでいるときはブロックに当たって止まった角度になり、
        # そのまま目標にすると握る力が出ない. 閉じる側の予測は oracle と同じ閉じる角度に置き換える.
        # 閾値は開始時 (0 付近) と掴んだとき (0.11〜0.12) の間. 開始時の値まで置き換えると、閉じた
        # ハンドを見た方策が閉じる側の予測を出し続け、開かないまま終わる（10/21 に -0.2 で起きた）.
        # 大きな値にすると置き換えない
        self._grip_snap = p('act_grip_snap_rad', 0.05).value
        self._arm_pub = node.create_publisher(
            JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self._hand_pub = node.create_publisher(
            JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self._conn = None

    def connect(self, timeout=60.0):
        """推論サーバーにつなぐ. 起動したばかりでモデルを読み込み中のこともあるので少し待つ."""
        deadline = time.monotonic() + timeout
        while self._conn is None:
            try:
                self._conn = Client(('127.0.0.1', self._port))
            except ConnectionRefusedError:
                if time.monotonic() > deadline:
                    raise RuntimeError(f'act_server が 127.0.0.1:{self._port} にいません') from None
                time.sleep(2.0)

    @staticmethod
    def _trajectory(names, rows):
        msg = JointTrajectory()
        msg.joint_names = names
        for k, row in enumerate(rows):
            t = (k + 1) * DT
            msg.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in row],
                time_from_start=DurationMsg(sec=int(t), nanosec=int(round((t % 1) * 1e9)))))
        return msg

    def rollout(self, obs_fn, sleep_sim, sim_now, task, grip_close_rad):
        """1エピソード分動かす. obs_fn(after) は stamp >= after の (stamp, bgr, qpos) を返す."""
        log = self._node.get_logger()
        conn = self._conn
        t_end = sim_now() + self._max_sec
        infer_ms, chunks = [], 0
        while sim_now() < t_end:
            stamp, bgr, qpos = obs_fn(sim_now() - 0.05)
            small = cv2.resize(bgr, (self._size[1], self._size[0]), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError('JPEG encode failed')
            t0 = time.monotonic()
            conn.send({'jpeg': jpeg.tobytes(), 'state': [float(v) for v in qpos], 'task': task})
            reply = conn.recv()
            infer_ms.append((time.monotonic() - t0) * 1e3)
            actions = reply['actions'][:self._n_steps]
            hand = [grip_close_rad if a[4] > self._grip_snap else a[4] for a in actions]
            self._arm_pub.publish(self._trajectory(ARM_JOINTS, [a[:4] for a in actions]))
            self._hand_pub.publish(self._trajectory([HAND_JOINT], [[h] for h in hand]))
            chunks += 1
            if chunks % 10 == 1:
                log.info(f'  act chunk {chunks}: t={stamp:.1f}s, infer {infer_ms[-1]:.0f} ms, '
                         f'hand {actions[0][4]:+.2f} -> {hand[0]:+.2f}')
            sleep_sim(len(actions) * DT)
        infer_ms.sort()
        return {'chunks': chunks, 'n_action_steps': self._n_steps,
                'infer_ms_median': round(infer_ms[len(infer_ms) // 2]) if infer_ms else None,
                'infer_ms_max': round(infer_ms[-1]) if infer_ms else None}
