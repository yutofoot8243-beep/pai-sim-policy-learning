"""1手ごとの失敗の分類と、エピソード・評価全体の集計（ROSに依存しない）.

タスクは「ブロックを同じ色のトレイに片付ける」. 正解は Gazebo のモデル位置 (poses:
モデル名 → (x, y, z, yaw), base_link 基準) で判定し、方策（Gemini・学習済みモデル）には渡さない.

1手の結果:
  success     狙ったブロックが正しいトレイに入った
  perception  掴む点がどのブロックにも当たっていない / 置く点がどのトレイにも当たっていない
  reasoning   ブロックを違う色のトレイに置こうとした
  planning    IK が解けない・動作計画に失敗した（アームが動けなかった）
  grasp       アームは動いたが、ブロックがほとんど動いていない（掴み損ね）
  place       ブロックは動いたが、トレイに入らなかった（落とした・ずれた）
"""

from collections import Counter
import math

from .scene import in_tray

OUTCOMES = ('success', 'perception', 'reasoning', 'planning', 'grasp', 'place')
PICK_TOL = 0.025         # 掴む点とブロック中心の許容距離（ブロックの半分の幅）
PLACE_MARGIN = 0.01      # 置く点がトレイの外側にはみ出してよい量
MOVED_TOL = 0.02         # これ以上動いていれば「掴んで運んだ」とみなす
COLLATERAL_TOL = 0.01    # 狙っていないブロックがこれ以上動いたら「巻き添え」


def _dist_xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _tray_at(xy, scene):
    """xy を含むトレイ（縁から PLACE_MARGIN の余裕あり）. 無ければ None."""
    for tray in scene.trays:
        dx, dy = xy[0] - tray.x, xy[1] - tray.y
        c, s = math.cos(tray.yaw), math.sin(tray.yaw)
        u, v = c * dx + s * dy, -s * dx + c * dy
        half = 0.04 + PLACE_MARGIN
        if abs(u) <= half and abs(v) <= half:
            return tray
    return None


def classify_attempt(scene, before, after, pick_xy, place_xy, exec_ok):
    """1回のピック&プレースを分類して dict を返す.

    exec_ok は skills.pick_and_place の成否（False = 計画・実行の失敗）. 届かないと判断して
    実行しなかった場合も False を渡す.
    """
    target = min(scene.cubes, key=lambda c: _dist_xy(pick_xy, before[c.name]))
    pick_err = _dist_xy(pick_xy, before[target.name])
    tray = _tray_at(place_xy, scene)
    moved = _dist_xy(before[target.name], after[target.name])
    # 狙っていないブロックの移動量 {名前: 距離 [m]}（ぶつけたのか、わずかにずれただけかを後で見分ける）
    collateral = {c.name: round(_dist_xy(before[c.name], after[c.name]), 4)
                  for c in scene.cubes if c is not target
                  and _dist_xy(before[c.name], after[c.name]) > COLLATERAL_TOL}
    result = {'target': target.name, 'tray': tray.name if tray else None,
              'pick_err': round(pick_err, 4), 'moved': round(moved, 4),
              'collateral': collateral}

    if pick_err > PICK_TOL or tray is None:
        outcome = 'perception'
    elif tray.color != target.color:
        outcome = 'reasoning'
    elif in_tray(after[target.name], tray):
        outcome = 'success'
    elif not exec_ok:
        outcome = 'planning'
    elif moved < MOVED_TOL:
        outcome = 'grasp'
    else:
        outcome = 'place'
    result['outcome'] = outcome
    return result


def cubes_sorted(scene, poses):
    """各ブロックが同じ色のトレイに入っているか {ブロック名: bool}."""
    return {c.name: in_tray(poses[c.name], scene.tray_of(c.color)) for c in scene.cubes}


def summarize(episodes):
    """episodes (eval_node が書く episodes.jsonl の各行) から集計 dict を作る."""
    n = len(episodes)
    attempts = [a for e in episodes for a in e['attempts']]
    outcome_counts = Counter(a['outcome'] for a in attempts)
    durations = [e['sim_duration'] for e in episodes if e['success']]
    steps = [len(e['attempts']) for e in episodes if e['success']]
    n_cubes = sum(len(e['scene']['cubes']) for e in episodes)
    return {
        'episodes': n,
        'success_rate': sum(e['success'] for e in episodes) / n if n else 0.0,
        'cube_success_rate': (sum(sum(e['sorted'].values()) for e in episodes) / n_cubes
                              if n_cubes else 0.0),
        'attempts': len(attempts),
        'outcomes': {k: outcome_counts.get(k, 0) for k in OUTCOMES},
        'collateral_attempts': sum(1 for a in attempts if a['collateral']),
        'end_reasons': dict(Counter(e['end_reason'] for e in episodes)),
        'mean_sim_duration_success': sum(durations) / len(durations) if durations else None,
        'mean_attempts_success': sum(steps) / len(steps) if steps else None,
    }


def format_summary(summary, title='評価結果'):
    """summarize の結果を Markdown の表にする."""
    s = summary
    lines = [f'## {title}', '',
             '| 指標 | 値 |', '|---|---|',
             f"| エピソード数 | {s['episodes']} |",
             f"| タスク成功率 | {s['success_rate']:.1%} |",
             f"| ブロック単位の成功率 | {s['cube_success_rate']:.1%} |",
             f"| 試行回数（ピック&プレース） | {s['attempts']} |"]
    if s['mean_sim_duration_success'] is not None:
        lines.append(f"| 成功時の平均所要時間（シミュレーション時間） | "
                     f"{s['mean_sim_duration_success']:.1f} s |")
        lines.append(f"| 成功時の平均試行回数 | {s['mean_attempts_success']:.2f} |")
    lines += ['', '| 1手の結果 | 回数 | 割合 |', '|---|---|---|']
    total = max(s['attempts'], 1)
    for k in OUTCOMES:
        v = s['outcomes'][k]
        lines.append(f'| {k} | {v} | {v / total:.1%} |')
    lines.append(f"| （隣のブロックを動かした） | {s['collateral_attempts']} | "
                 f"{s['collateral_attempts'] / total:.1%} |")
    lines += ['', '終了理由: ' + ', '.join(f'{k}={v}' for k, v in s['end_reasons'].items())]
    return '\n'.join(lines)
