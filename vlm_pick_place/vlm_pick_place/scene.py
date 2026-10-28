"""評価用シーンのランダム生成と、ブロックがトレイに入っているかの判定（ROSに依存しない）.

座標はすべて base_link（= 机の天板の高さ、ロボットの根元）基準 [m]. 寸法は
tools/scene_layout.py と同じ値. ワールドには色ごとにブロック1個・トレイ1枚が置いてあり、
エピソードごとに位置を入れ替える（使わないブロックは机の外へ退避する）.
"""

from dataclasses import asdict, dataclass, field
import math

import numpy as np

TABLE_Z = 1.015          # base_link のワールド高さ
CUBE_SIZE = 0.05
TRAY_SIZE = 0.08
TRAY_THICK = 0.005
COLORS = ('red', 'green', 'blue')

# 配置の範囲. ブロックは内側の輪、トレイは外側の輪（上から置くので内側のブロックに当たらない）.
# 方位角の範囲は真上カメラの視野と、アームの joint1 の可動域で決めている
CUBE_R = (0.16, 0.22)
TRAY_R = (0.255, 0.285)
ANGLE_DEG = (-35.0, 70.0)
MIN_CUBE_GAP = 0.09      # ブロック中心どうし. これより近いと開いた指で隣を倒す
MIN_TRAY_GAP = 0.10      # トレイ中心どうし（8 cm 角が重ならない）
MIN_CUBE_TRAY_GAP = 0.08   # ブロックがトレイの縁に乗らない（元の配置の最小値 0.081 と同程度）

PARK_XY = (2.0, 2.0)     # 使わないブロックの退避先（ワールド座標. カメラに映らない床の上）


@dataclass
class Placement:
    name: str            # Gazebo のモデル名 (red_cube, blue_tray, ...)
    color: str
    x: float
    y: float
    yaw: float           # [rad]


@dataclass
class Scene:
    seed: int
    cubes: list = field(default_factory=list)   # 机の上に置くブロックだけ
    trays: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def tray_of(self, color):
        return next(t for t in self.trays if t.color == color)


def _polar(r, deg):
    return r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))


def _far_enough(xy, others, gap):
    return all(math.hypot(xy[0] - o[0], xy[1] - o[1]) >= gap for o in others)


def sample_scene(seed, n_cubes=0, yaw_jitter_deg=20.0, max_tries=1000):
    """seed から決まるシーンを返す. n_cubes=0 なら 1〜3 個をランダムに選ぶ.

    ブロックの向きは「根元から見た方向 ± yaw_jitter_deg」. 指は根元から見て左右から
    挟むので、この角度が大きいほど掴みにくくなる（難しさのつまみ）.
    """
    rng = np.random.default_rng(seed)
    if n_cubes == 0:
        n_cubes = int(rng.integers(1, len(COLORS) + 1))
    if not 1 <= n_cubes <= len(COLORS):
        raise ValueError(f'n_cubes must be 0..{len(COLORS)}: {n_cubes}')

    for _ in range(max_tries):
        trays, cubes = [], []
        for color in rng.permutation(COLORS):
            deg = rng.uniform(*ANGLE_DEG)
            xy = _polar(rng.uniform(*TRAY_R), deg)
            if not _far_enough(xy, [(t.x, t.y) for t in trays], MIN_TRAY_GAP):
                break
            trays.append(Placement(f'{color}_tray', str(color), *xy, math.radians(deg)))
        else:
            for color in rng.choice(COLORS, n_cubes, replace=False):
                deg = rng.uniform(*ANGLE_DEG)
                xy = _polar(rng.uniform(*CUBE_R), deg)
                if not (_far_enough(xy, [(c.x, c.y) for c in cubes], MIN_CUBE_GAP)
                        and _far_enough(xy, [(t.x, t.y) for t in trays], MIN_CUBE_TRAY_GAP)):
                    break
                yaw = math.radians(deg + rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
                cubes.append(Placement(f'{color}_cube', str(color), *xy, yaw))
            else:
                return Scene(seed=seed, cubes=cubes, trays=trays)
    raise RuntimeError(f'could not sample a scene for seed {seed}')


def in_tray(cube_xyz, tray):
    """ブロックの中心 (base_link) がトレイの上に載っているか.

    水平方向はトレイの座標系で 8 cm 角の内側、高さはトレイ上面に載った高さ付近（倒れて
    いても可. 落下して机の上にある・掴まれたまま宙にある、は不可）.
    """
    dx, dy = cube_xyz[0] - tray.x, cube_xyz[1] - tray.y
    c, s = math.cos(tray.yaw), math.sin(tray.yaw)
    u, v = c * dx + s * dy, -s * dx + c * dy
    half = TRAY_SIZE / 2
    # 中心の高さ: 机の上 0.025、トレイの上 0.030
    return abs(u) <= half and abs(v) <= half and 0.027 <= cube_xyz[2] <= 0.045
