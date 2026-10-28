"""Gazebo (Ignition Fortress) のワールドを直接読み書きする（評価用. 方策からは使わない）.

Fortress には Python バインディングが無く、ros_gz_bridge で Pose_V を流すとモデル名が
落ちるので、ign コマンドを呼ぶ. 位置の読み取りは1回 2 秒ほどかかるので、1手ごとに使う程度にする.
座標の入出力は base_link 基準（ワールドの z から TABLE_Z を引いたもの）.
"""

import math
import re
import subprocess
import time

from .scene import CUBE_SIZE, PARK_XY, TABLE_Z, TRAY_THICK

WORLD = 'default'


class SimError(RuntimeError):
    """Gazebo が応答しない（固まった・重すぎる）. 方策の失敗ではないので評価には数えない."""


def parse_pose_info(text):
    """`ign topic -e -t /world/<w>/pose/info` の1メッセージ分のテキストを解析する.

    {モデル名: (x, y, z, yaw)} (ワールド座標) を返す. 0 の項目は省略されて出力されるので
    既定値 0 で埋める. 同名のエントリ（各モデルの "link" など）は最初のものを残す.
    """
    poses = {}
    for block in re.split(r'^pose \{', text, flags=re.M)[1:]:
        name = re.search(r'^\s*name: "([^"]*)"', block, re.M)
        if name is None or name.group(1) in poses:
            continue
        p = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        q = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        for section, values in (('position', p), ('orientation', q)):
            m = re.search(section + r' \{([^}]*)\}', block)
            if m:
                for key, val in re.findall(r'(\w): (\S+)', m.group(1)):
                    values[key] = float(val)
        yaw = math.atan2(2 * (q['w'] * q['z'] + q['x'] * q['y']),
                         1 - 2 * (q['y'] ** 2 + q['z'] ** 2))
        poses[name.group(1)] = (p['x'], p['y'], p['z'], yaw)
    return poses


def _ign(args, timeout):
    return subprocess.run(['ign'] + args, capture_output=True, text=True, timeout=timeout,
                          check=False).stdout


def get_poses(names, timeout=10.0):
    """指定したモデルの位置 {name: (x, y, z, yaw)} を base_link 基準で返す."""
    for i in range(3):
        try:
            text = _ign(['topic', '-e', '-n', '1', '-t', f'/world/{WORLD}/pose/info'], timeout)
        except subprocess.TimeoutExpired:
            text = ''
        world = parse_pose_info(text)
        missing = [n for n in names if n not in world]
        if not missing:
            break
        time.sleep(1.0 + i)
    else:
        raise SimError(f'models not found in /world/{WORLD}/pose/info: {missing}')
    return {n: (world[n][0], world[n][1], world[n][2] - TABLE_Z, world[n][3]) for n in names}


def _call(service, reqtype, req, timeout=5.0, attempts=3):
    # シミュレーションが重いと応答が空で返ることがある（数十回に1回）ので、やり直す
    for i in range(attempts):
        try:
            out = _ign(['service', '-s', f'/world/{WORLD}/{service}', '--reqtype', reqtype,
                        '--reptype', 'ignition.msgs.Boolean',
                        '--timeout', str(int(timeout * 1000)), '--req', req], timeout + 2.0)
        except subprocess.TimeoutExpired:
            out = ''
        if 'data: true' in out:
            return
        time.sleep(1.0 + i)
    raise SimError(f'{service} failed {attempts} times: {req} -> {out.strip()!r}')


def set_pose(name, x, y, z, yaw=0.0):
    """モデルを base_link 基準の (x, y, z, yaw) に置く（静的なトレイも動かせる）."""
    _call('set_pose', 'ignition.msgs.Pose',
          f'name: "{name}" position: {{x: {x} y: {y} z: {z + TABLE_Z}}} '
          f'orientation: {{z: {math.sin(yaw / 2)} w: {math.cos(yaw / 2)}}}')


def set_physics_step(step):
    """物理の刻み幅 [s] を変える. 既定の 1 ms は CPU では実時間の 1/6 ほどしか出ない."""
    _call('set_physics', 'ignition.msgs.Physics',
          f'max_step_size: {step} real_time_factor: 1.0')


def apply_scene(scene, all_cube_names):
    """scene のとおりにトレイとブロックを置き、使わないブロックは机の外へ退避する."""
    for i, name in enumerate(all_cube_names):   # 先に全部どけてから置く（置き場所で衝突しない）
        set_pose(name, PARK_XY[0] + 0.2 * i, PARK_XY[1], CUBE_SIZE / 2 - TABLE_Z)
    for t in scene.trays:
        set_pose(t.name, t.x, t.y, TRAY_THICK / 2, t.yaw)
    for c in scene.cubes:
        set_pose(c.name, c.x, c.y, CUBE_SIZE / 2 + 0.002, c.yaw)
