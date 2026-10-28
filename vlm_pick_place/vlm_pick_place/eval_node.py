"""ランダムなシーンで「同じ色のトレイに片付ける」を N 回続けて実行し、成功率と失敗の内訳を測る.

  ros2 launch vlm_pick_place eval.launch.py episodes:=20 agent:=oracle record:=true

方策 (agent):
  oracle  Gazebo の正解位置を使う先生役. 画像は見ない. 学習用デモの収集と、動作部分
          （把持・運搬）だけの性能上限を測るのに使う
  gemini  vlm_agent と同じ「画像 → Gemini → 次の一手」. 無料枠は1日20リクエストなので少数で
  act     学習した ACT で関節の目標を出し続ける（閉ループ）. 決まった時間だけ動かして最後に判定する.
          推論サーバー tools/act_server.py を先に起動しておくこと

出力 (output_dir):
  episodes.jsonl   1行1エピソード（シーン・各手の結果・成否）
  summary.json / summary.md
  demos/episode_XXXX.npz   record:=true のとき. 真上カメラの JPEG と関節角の時系列
"""

from datetime import datetime
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
import tf2_ros

from . import sim_world
from .arm_commander import ARM_JOINTS, HAND_JOINT, ArmCommander
from .evaluation import classify_attempt, cubes_sorted, format_summary, summarize
from .geometry import (check_reachable, normalized_yx_to_pixel, pixel_to_table_point,
                       quaternion_to_matrix)
from .refine import refine_pick_pixel
from .scene import COLORS, sample_scene
from .skills import declare_motion_params, pick_and_place, read_motion_params
from .vlm_agent_node import image_msg_to_bgr

INSTRUCTION = 'ブロックを同じ色のトレイに片付けて'
ALL_CUBES = [f'{c}_cube' for c in COLORS]
ALL_TRAYS = [f'{c}_tray' for c in COLORS]


class OracleAgent:
    """正解位置から次の一手を決める. まだ片付いていないブロックを1つずつ運ぶ."""

    def next_step(self, scene, poses, image_fn):
        done = cubes_sorted(scene, poses)
        for cube in scene.cubes:
            if not done[cube.name]:
                tray = scene.tray_of(cube.color)
                return poses[cube.name][:2], (tray.x, tray.y), f'{cube.name} -> {tray.name}'
        return None


class GeminiAgent:
    """vlm_agent_node と同じ判断（画像 → Gemini → 掴む点を上面の中心に補正 → 机上の座標）."""

    def __init__(self, node):
        from .gemini_brain import DEFAULT_FALLBACK_MODEL_IDS, DEFAULT_MODEL_ID, GeminiBrain
        p = node.declare_parameter
        self._node = node
        self._brain = GeminiBrain(
            model_id=p('model_id', DEFAULT_MODEL_ID).value,
            thinking_budget=p('thinking_budget', 1024).value,
            temperature=p('temperature', 0.5).value,
            fallback_model_ids=p('fallback_model_ids', list(DEFAULT_FALLBACK_MODEL_IDS)).value,
            log=node.get_logger().warn)
        self._pick_z = p('pick_surface_z', 0.05).value
        self._place_z = p('place_surface_z', 0.005).value
        self._refine = p('refine_pick', True).value
        self.history = []

    def reset(self):
        self.history = []

    def next_step(self, scene, poses, image_fn):
        header, bgr, cam_k, cam_tf = image_fn()
        _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        decision = self._brain.next_step(jpeg.tobytes(), INSTRUCTION, self.history)
        self._node.get_logger().info(f'Gemini: {decision.raw}')
        if decision.done:
            return None
        h, w = bgr.shape[:2]
        pick_uv = normalized_yx_to_pixel(decision.pick.point_yx, w, h)
        place_uv = normalized_yx_to_pixel(decision.place.point_yx, w, h)
        if self._refine:
            uv, ok = refine_pick_pixel(bgr, *pick_uv)
            if ok:
                pick_uv = uv
        t, rot = cam_tf
        pick = pixel_to_table_point(*pick_uv, cam_k, t, rot, self._pick_z)
        place = pixel_to_table_point(*place_uv, cam_k, t, rot, self._place_z)
        if pick is None or place is None:
            raise RuntimeError('Gemini point does not hit the table')
        label = f"pick '{decision.pick.label}' -> place '{decision.place.label}'"
        return pick[:2], place[:2], label

    def report(self, label, msg):
        self.history.append(f'{label}: {msg}')


class DemoRecorder:
    """エピソード中の真上カメラ画像と関節角をシミュレーション時間で記録する（学習用デモ）."""

    def __init__(self, size_hw=(240, 320), jpeg_quality=90):
        self._size = size_hw
        self._quality = jpeg_quality
        self._lock = threading.Lock()
        self._active = False
        self._clear()

    def _clear(self):
        self.t, self.qpos, self.jpegs = [], [], []

    def start(self):
        with self._lock:
            self._clear()
            self._active = True

    def stop(self):
        with self._lock:
            self._active = False

    def add(self, stamp, bgr, qpos):
        with self._lock:
            if not self._active or qpos is None:
                return
            small = cv2.resize(bgr, (self._size[1], self._size[0]), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            if ok:
                self.t.append(stamp)
                self.qpos.append(qpos)
                self.jpegs.append(jpeg.tobytes())

    def save(self, path, meta):
        """JPEG は連結したバイト列と各フレームの開始位置で持つ（pickle を使わずに読める）."""
        with self._lock:
            offsets = np.cumsum([0] + [len(j) for j in self.jpegs]).astype(np.int64)
            np.savez(path,
                     t=np.asarray(self.t, np.float64),
                     qpos=np.asarray(self.qpos, np.float32).reshape(-1, len(ARM_JOINTS) + 1),
                     jpeg_bytes=np.frombuffer(b''.join(self.jpegs), np.uint8),
                     jpeg_offsets=offsets,
                     joint_names=np.array(ARM_JOINTS + [HAND_JOINT]),
                     meta=np.array(json.dumps(meta, ensure_ascii=False)))
            return len(self.t)


class EvalNode(Node):
    def __init__(self):
        super().__init__('vlm_agent')   # params.yaml の vlm_agent セクションをそのまま読む
        p = self.declare_parameter
        p('agent', 'oracle')
        p('episodes', 10)
        p('episodes_per_run', 0)         # 0 以外なら、この数だけ新しく実行したら終わる（定期的に起動し直す用）
        p('seed', 0)                     # エピソード i のシーンは seed + i で決まる
        p('n_cubes', 0)                  # 0 なら 1〜3 個をランダムに
        p('yaw_jitter_deg', 20.0)
        p('physics_step', 0.004)         # 0 ならワールドの設定のまま
        p('max_steps', 6)
        p('max_failures', 3)             # 連続でこれだけ失敗したら打ち切る
        p('output_dir', '')
        p('record', False)
        p('record_size', [240, 320])
        p('base_frame', 'base_link')
        p('camera_frame', 'camera_color_optical_frame')
        p('reach_r_min', 0.14)
        p('reach_r_max', 0.30)
        p('velocity_scaling', 0.6)
        declare_motion_params(self)
        gp = self.get_parameter

        name = gp('agent').value
        if name == 'oracle':
            self._agent = OracleAgent()
        elif name == 'gemini':
            self._agent = GeminiAgent(self)
        elif name == 'act':
            from .act_agent import ActAgent
            self._agent = ActAgent(self)
        else:
            raise ValueError(f'unknown agent: {name}')
        out = gp('output_dir').value or os.path.expanduser(
            f'~/pai_eval/{datetime.now():%Y%m%d_%H%M%S}_{name}')
        self._out = out
        os.makedirs(out, exist_ok=True)
        self._recorder = DemoRecorder(tuple(gp('record_size').value)) if gp('record').value else None
        if self._recorder:
            os.makedirs(os.path.join(out, 'demos'), exist_ok=True)

        self._arm = ArmCommander(self, base_frame=gp('base_frame').value,
                                 velocity_scaling=gp('velocity_scaling').value,
                                 acceleration_scaling=gp('velocity_scaling').value)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._lock = threading.Lock()
        self._image = None
        self._camera_info = None
        self._qpos = None
        self.create_subscription(Image, 'image_raw', self._on_image, 10)
        self.create_subscription(CameraInfo, 'camera_info', self._on_camera_info, 10)
        self.create_subscription(JointState, 'joint_states', self._on_joint_states, 10)
        self.done = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    # ------------------------------------------------------------ コールバック
    def _on_camera_info(self, msg):
        self._camera_info = msg

    def _on_joint_states(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in ARM_JOINTS + [HAND_JOINT]):
            self._qpos = [pos[n] for n in ARM_JOINTS + [HAND_JOINT]]

    def _on_image(self, msg):
        bgr = image_msg_to_bgr(msg)
        with self._lock:
            self._image = (msg.header, bgr)
        if self._recorder:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._recorder.add(stamp, bgr, self._qpos)

    # ------------------------------------------------------------ 補助
    def _sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _sleep_sim(self, sec):
        end = self.get_clock().now() + Duration(seconds=sec)
        while rclpy.ok() and self.get_clock().now() < end:
            time.sleep(0.02)

    def _camera_snapshot(self):
        """(header, bgr, K, (平行移動, 回転行列)) を返す. GeminiAgent 用."""
        self._sleep_sim(1.0)   # アームが止まってからの画像を使う
        with self._lock:
            header, bgr = self._image
        tf = self._tf_buffer.lookup_transform(
            self.get_parameter('base_frame').value, self.get_parameter('camera_frame').value,
            rclpy.time.Time(), timeout=Duration(seconds=2.0))
        t, q = tf.transform.translation, tf.transform.rotation
        return (header, bgr, self._camera_info.k,
                (np.array([t.x, t.y, t.z]), quaternion_to_matrix(q.x, q.y, q.z, q.w)))

    def _observation(self, after, timeout=30.0):
        """シミュレーション時間 after 以降に撮った画像と、そのときの関節角 (stamp, bgr, qpos). ActAgent 用."""
        deadline = time.monotonic() + timeout
        while rclpy.ok():
            with self._lock:
                header, bgr = self._image
            stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
            if stamp >= after:
                return stamp, bgr, list(self._qpos)
            if time.monotonic() > deadline:
                raise sim_world.SimError(f'no camera image after t={after:.1f}s')
            time.sleep(0.01)
        raise RuntimeError('shutdown')

    def _poses(self, scene):
        return sim_world.get_poses([c.name for c in scene.cubes] + [t.name for t in scene.trays])

    # ------------------------------------------------------------ 評価ループ
    def _run(self):
        log = self.get_logger()
        gp = self.get_parameter
        try:
            if not self._arm.wait_for_server(timeout=120.0):
                log.error('move_group (/move_action) が見つかりません')
                return
            while self._camera_info is None or self._image is None or self._qpos is None:
                log.info('waiting for camera and joint_states...', throttle_duration_sec=5.0)
                time.sleep(0.2)
            if hasattr(self._agent, 'connect'):
                self._agent.connect()
            if gp('physics_step').value > 0:
                sim_world.set_physics_step(gp('physics_step').value)

            # 同じ output_dir で起動し直すと、記録済みのエピソードを飛ばして続きから行う
            path = os.path.join(self._out, 'episodes.jsonl')
            episodes = []
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    episodes = [json.loads(line) for line in f if line.strip()]
                if episodes:
                    log.info(f'resuming: {len(episodes)} episodes already in {path}')
            if any(e['seed'] != gp('seed').value + e['index'] for e in episodes):
                raise RuntimeError(f'{path} was recorded with a different seed')
            done = {e['index'] for e in episodes}
            budget = gp('episodes_per_run').value or gp('episodes').value
            for i in range(gp('episodes').value):
                if not rclpy.ok() or budget == 0:
                    break
                if i in done:
                    continue
                budget -= 1
                ep = self._run_episode(i, gp('seed').value + i)
                episodes.append(ep)
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(ep, ensure_ascii=False) + '\n')
                rate = sum(e['success'] for e in episodes) / len(episodes)
                log.info(f'episode {i}: {"SUCCESS" if ep["success"] else "FAIL"} '
                         f'({ep["end_reason"]}, {len(ep["attempts"])} attempts, '
                         f'{ep["sim_duration"]:.0f}s sim) | success rate {rate:.0%}')

            summary = summarize(episodes)
            summary['agent'] = gp('agent').value
            summary['params'] = {k: gp(k).value for k in (
                'seed', 'n_cubes', 'yaw_jitter_deg', 'physics_step', 'max_steps')}
            with open(os.path.join(self._out, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            md = format_summary(summary, f"評価結果 (agent={gp('agent').value})")
            with open(os.path.join(self._out, 'summary.md'), 'w', encoding='utf-8') as f:
                f.write(md + '\n')
            log.info('\n' + md + f'\n\nsaved to {self._out}')
        except sim_world.SimError as e:
            # そのエピソードは記録しない. tools/run_eval.sh が起動し直して続きから行う
            log.error(f'simulator not responding, stopping (episode not recorded): {e}')
        except Exception as e:  # noqa: BLE001  スレッドが黙って死なないようにする
            log.error(f'evaluation aborted: {e!r}')
        finally:
            self.done.set()

    def _run_episode(self, index, seed):
        gp = self.get_parameter
        log = self.get_logger().info
        cfg = read_motion_params(self)
        scene = sample_scene(seed, gp('n_cubes').value, gp('yaw_jitter_deg').value)
        log(f'--- episode {index} (seed {seed}): '
            + ', '.join(f'{c.name}@({c.x:.3f},{c.y:.3f})' for c in scene.cubes))

        self._arm.go_named('home')
        # 方策が前のエピソードを閉じたハンドで終えても、デモと同じく開いた状態から始める
        self._arm.gripper(math.radians(cfg['gripper_open_deg']))
        sim_world.apply_scene(scene, ALL_CUBES)
        self._sleep_sim(1.0)
        if hasattr(self._agent, 'reset'):
            self._agent.reset()
        poses = self._poses(scene)
        start_poses = poses
        attempts, failures, end_reason = [], 0, 'max_steps'
        t0 = self._sim_now()
        if self._recorder:
            self._recorder.start()
        policy_info = None
        try:
            if hasattr(self._agent, 'rollout'):
                policy_info = self._agent.rollout(
                    self._observation, self._sleep_sim, self._sim_now, INSTRUCTION,
                    math.radians(cfg['gripper_close_deg']))
                end_reason = 'time_limit'
            else:
                for _ in range(gp('max_steps').value):
                    step = self._agent.next_step(scene, poses, self._camera_snapshot)
                    if step is None:
                        end_reason = 'agent_done'
                        break
                    pick_xy, place_xy, label = step
                    reach = [check_reachable(xy, gp('reach_r_min').value, gp('reach_r_max').value)
                             for xy in (pick_xy, place_xy)]
                    if all(ok for ok, _ in reach):
                        exec_ok, msg = pick_and_place(self._arm, cfg, list(pick_xy), list(place_xy),
                                                      log=self.get_logger().debug)
                    else:
                        exec_ok, msg = False, 'SKIPPED: ' + '; '.join(m for ok, m in reach if not ok)
                    self._sleep_sim(0.5)
                    after = self._poses(scene)
                    result = classify_attempt(scene, poses, after, pick_xy, place_xy, exec_ok)
                    result.update(label=label, exec=msg,
                                  pick=[round(v, 4) for v in pick_xy],
                                  place=[round(v, 4) for v in place_xy])
                    attempts.append(result)
                    log(f'  {label}: {result["outcome"]} ({msg})')
                    if hasattr(self._agent, 'report'):
                        self._agent.report(label, msg)
                    poses = after
                    failures = 0 if result['outcome'] == 'success' else failures + 1
                    if failures >= gp('max_failures').value:
                        end_reason = 'consecutive_failures'
                        break
        except sim_world.SimError:
            raise
        except Exception as e:  # noqa: BLE001  1エピソードの失敗で評価全体を止めない
            self.get_logger().error(f'episode {index} error: {e!r}')
            end_reason = f'error: {e}'
            self._arm.go_named('home')
        finally:
            if self._recorder:
                self._recorder.stop()

        sim_duration = self._sim_now() - t0
        final = self._poses(scene)
        sorted_ = cubes_sorted(scene, final)
        success = all(sorted_.values())
        if success and end_reason == 'max_steps':
            end_reason = 'max_steps_but_sorted'
        ep = {'index': index, 'seed': seed, 'agent': gp('agent').value,
              'scene': scene.to_dict(), 'attempts': attempts, 'sorted': sorted_,
              'success': success, 'end_reason': end_reason,
              'sim_duration': round(sim_duration, 2),
              'start_poses': {k: [round(v, 4) for v in p] for k, p in start_poses.items()},
              'final_poses': {k: [round(v, 4) for v in p] for k, p in final.items()}}
        if policy_info:
            ep['policy'] = policy_info
        if self._recorder:
            path = os.path.join(self._out, 'demos', f'episode_{index:04d}.npz')
            n = self._recorder.save(path, {'instruction': INSTRUCTION, 'seed': seed,
                                           'success': success, 'scene': scene.to_dict()})
            ep['demo'] = {'file': os.path.basename(path), 'frames': n}
        return ep


def main():
    rclpy.init()
    node = EvalNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.done.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
