"""最終課題の再提出用の動画: 前半は提出版 (Gemini, 2 倍速)、後半は ACT の発展 (2 倍速).
  python make_final_video.py <pai_eval_data> <提出版の 2 倍速 mp4> <出力 mp4>
録画 (npz) は 10 fps なので、20 fps の動画に 1 コマずつ並べると 2 倍速になる.
"""
import json
import math
import sys

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA, OLD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H, FPS = 1280, 720, 20
FONT_B = 'C:/Windows/Fonts/BIZ-UDGothicB.ttc'
FONT_R = 'C:/Windows/Fonts/BIZ-UDGothicR.ttc'
JA = {'green': '緑', 'blue': '青', 'red': '赤'}
BG = (24, 24, 28)
WHITE, GRAY, DIM = (240, 240, 240), (200, 200, 200), (165, 165, 165)
ACCENT = (120, 190, 255)
RUN_20K = 'act300_20k_n1_s3000'
RUN_40K = 'act300_40k_open_n1_s3000'
ORACLE = 'oracle_n1_s2100'


def font(size, bold=True):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


def load(run, seed):
    eps = [json.loads(l) for l in open(f'{DATA}/{run}/episodes.jsonl', encoding='utf-8') if l.strip()]
    e = next(x for x in eps if x['seed'] == seed)
    z = np.load(f"{DATA}/{run}/demos/{e['demo']['file']}")
    off, jb = z['jpeg_offsets'], z['jpeg_bytes']
    frames = [cv2.imdecode(jb[off[k]:off[k + 1]], cv2.IMREAD_COLOR) for k in range(len(off) - 1)]
    return frames, z['t'] - z['t'][0], z['qpos'], e


def outcome(e, qpos):
    if e['success']:
        return '成功', (70, 200, 90)
    cube = e['scene']['cubes'][0]
    s, f = e['start_poses'][cube['name']], e['final_poses'][cube['name']]
    if math.dist(s[:2], f[:2]) > 0.08:
        trays = {t['color']: e['final_poses'][t['name']][:2] for t in e['scene']['trays']}
        near = min(trays, key=lambda k: math.dist(f[:2], trays[k]))
        if near != cube['color']:
            return f'違うトレイ（{JA[near]}）', (230, 90, 70)
        return 'トレイの外', (230, 160, 60)
    if not (qpos[:, 4] > 0.05).any() and math.dist(s[:2], f[:2]) < 0.01:
        return '動かない', (230, 90, 70)
    return '掴み損ね', (230, 90, 70)


class Writer:
    def __init__(self, path):
        self.c = av.open(path, 'w')
        self.s = self.c.add_stream('libx264', rate=FPS, options={'crf': '20'})
        self.s.width, self.s.height, self.s.pix_fmt = W, H, 'yuv420p'
        self.n = 0

    def add(self, rgb, n=1):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format='rgb24')
        for _ in range(n):
            for p in self.s.encode(frame):
                self.c.mux(p)
        self.n += n

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()


def card(wr, lines, sec):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    total = sum(sz * 1.6 for _, sz, _ in lines)
    y = (H - total) / 2
    for text, sz, col in lines:
        f = font(sz, sz >= 34)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=col)
        y += sz * 1.6
    wr.add(np.array(im), int(sec * FPS))


def to_rgb(bgr, w, h):
    return cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)


def badge(d, x, y, text, col, size=30):
    w = d.textlength(text, font=font(size))
    d.rounded_rectangle((x, y, x + w + 24, y + size + 18), 10, fill=col)
    d.text((x + 12, y + 7), text, font=font(size), fill=(255, 255, 255))


def part_label(d, text):
    d.rectangle((0, 0, W, 44), fill=(10, 10, 12))
    d.text((16, 8), text, font=font(24), fill=ACCENT)
    d.text((W - 110, 10), '2倍速', font=font(22), fill=WHITE)


def old_video(wr):
    """提出版 (1280x560, 20 fps, 2 倍速の表示入り) を帯つきで入れる."""
    c = av.open(OLD)
    for f in c.decode(video=0):
        im = Image.new('RGB', (W, H), BG)
        im.paste(Image.fromarray(f.to_ndarray(format='rgb24')), (0, 80))
        d = ImageDraw.Draw(im)
        d.text((16, 22), '前半　Gemini Robotics-ER が画像を見て「次の1手」を毎回決める（10/17 の提出版）',
               font=font(26), fill=ACCENT)
        d.text((16, 652), 'AI は推論のみ（学習なし）。掴む点・置く点を画像上で指し、MoveIt でアームを動かす。'
               'ずれた青を自分で見つけて置き直している',
               font=font(20, False), fill=GRAY)
        wr.add(np.array(im))
    c.close()


def single(wr, run, seed, title, note, label):
    """左に映像 (960x720 のうち上 44 は帯)、右に説明."""
    frames, t, qpos, e = load(run, seed)
    cube = e['scene']['cubes'][0]['color']
    res, rcol = outcome(e, qpos)
    for k in range(len(frames)):
        im = Image.new('RGB', (W, H), BG)
        im.paste(Image.fromarray(to_rgb(frames[k], 904, 678)), (0, 44))
        d = ImageDraw.Draw(im)
        part_label(d, label)
        x = 928
        d.text((x, 70), title, font=font(28), fill=WHITE)
        d.text((x, 116), f'{JA[cube]}のブロック → {JA[cube]}のトレイ', font=font(22, False), fill=GRAY)
        y = 170
        for line in note:
            d.text((x, y), line, font=font(21, False), fill=DIM)
            y += 32
        d.text((x, H - 110), f't = {t[k]:4.1f} s', font=font(22, False), fill=DIM)
        if k >= len(frames) - 3 * 10:
            badge(d, x, H - 72, res, rcol)
        wr.add(np.array(im))
    wr.add(np.array(im), FPS // 2)


def oracle_clip(wr, index):
    z = np.load(f'{DATA}/{ORACLE}/demos/episode_{index:04d}.npz')
    off, jb, t = z['jpeg_offsets'], z['jpeg_bytes'], z['t'] - z['t'][0]
    for k in range(len(off) - 1):
        im = Image.new('RGB', (W, H), BG)
        im.paste(Image.fromarray(to_rgb(cv2.imdecode(jb[off[k]:off[k + 1]], cv2.IMREAD_COLOR), 904, 678)), (0, 44))
        d = ImageDraw.Draw(im)
        part_label(d, '後半　発展: お手本を真似る学習（ACT）')
        x = 928
        d.text((x, 70), 'お手本（oracle）', font=font(28), fill=WHITE)
        for i, line in enumerate(['Gazebo の正解の位置を使う', '決め打ちのプログラム', '',
                                  'ブロック1個の配置を', '乱数で変えて 300 本録画', '（すべて成功したもの）', '',
                                  '真上の画像と関節角を', '記録する']):
            d.text((x, 120 + i * 32), line, font=font(21, False), fill=DIM)
        d.text((x, H - 110), f't = {t[k]:4.1f} s', font=font(22, False), fill=DIM)
        wr.add(np.array(im))


def compare(wr):
    seeds = [3003, 3008]
    rows = [('2 万ステップ', RUN_20K), ('4 万ステップ', RUN_40K)]
    cells = {(r, c): load(run, s) for r, (_, run) in enumerate(rows) for c, s in enumerate(seeds)}
    n = max(len(v[0]) for v in cells.values())
    tw, th, x0, y0 = 452, 338, 376, 44
    for k in range(n + FPS):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        part_label(d, '後半　学習を長くすると（同じお手本 300 本・同じ配置）')
        for r, (label, _) in enumerate(rows):
            d.text((24, y0 + r * th + th / 2 - 40), label, font=font(30), fill=WHITE)
            d.text((24, y0 + r * th + th / 2 + 4), '約 2.9 エポック' if r == 0 else '約 5.6 エポック',
                   font=font(20, False), fill=DIM)
            for c, s in enumerate(seeds):
                frames, t, qpos, e = cells[(r, c)]
                i = min(k, len(frames) - 1)
                x, y = x0 + c * tw, y0 + r * th
                im.paste(Image.fromarray(to_rgb(frames[i], tw, th)), (x, y))
                cube = e['scene']['cubes'][0]['color']
                d.rectangle((x, y, x + tw, y + 32), fill=(0, 0, 0))
                d.text((x + 10, y + 5), f'seed {s}　{JA[cube]}のブロック', font=font(20), fill=WHITE)
                if k >= len(frames) - 3 * 10:
                    res, rcol = outcome(e, qpos)
                    badge(d, x + tw - 200, y + th - 58, res, rcol, 26)
        for seg in ((x0, y0, x0, H), (x0, y0 + th, W, y0 + th), (x0 + tw, y0, x0 + tw, H)):
            d.line(seg, fill=(80, 80, 80), width=2)
        wr.add(np.array(im))


wr = Writer(OUT)
card(wr, [('言葉で片付けるロボットアーム', 46, WHITE),
          ('Gemini Robotics-ER × CRANE+ V2（ROS 2 / Gazebo / MoveIt）', 26, GRAY),
          ('', 16, BG),
          ('前半: VLM が毎手考えて動かす　　後半: 発展 ― お手本を真似る学習（ACT）', 24, DIM),
          ('全編 2 倍速', 24, ACCENT)], 3.5)
old_video(wr)
card(wr, [('発展: 学習した方策で動かす', 42, WHITE),
          ('Gemini は 1 日 20 回まで → お手本集めには使えない', 26, GRAY),
          ('Gazebo の正解位置で動く「お手本」を 300 本集め、LeRobot の ACT を Colab で学習', 26, GRAY),
          ('入力は真上の画像と関節角だけ。出力は次の関節角（位置の計算もプランナーも使わない）', 24, DIM),
          ('（このパートはブロック 1 個の片付け）', 22, DIM)], 5)
oracle_clip(wr, 120)
compare(wr)
single(wr, RUN_40K, 3001, 'うまくいかない例 1', ['掴んで運べるが、', '色の違うトレイに置く', '',
                                                 '運んだうち正しい色は', '約半分（13/26）', '',
                                                 '色の対応づけを', 'まだ学べていない'],
       '後半　うまくいかない例（4 万ステップ）')
single(wr, RUN_40K, 3000, 'うまくいかない例 2', ['ブロックが手前にあると', 'home のまま動かない', '',
                                                 'お手本は手前のときほど', '動き出すまで長く', '止まっていた',
                                                 '（2.8 秒 / 遠いと 0.8 秒）', '',
                                                 '時刻を知らない ACT が', '「止まる」を真似たと', 'みられる'],
       '後半　うまくいかない例（4 万ステップ）')
card(wr, [('学習した方策の評価（10 本・配置は乱数）', 36, WHITE),
          ('成功 2 ・ 違うトレイ 3 ・ 動かない 3 ・ 掴み損ね 2', 30, GRAY),
          ('2 万ステップでは成功 0（お手本の oracle はブロック 1〜3 個の 30 本で全部成功）', 24, DIM),
          ('', 14, BG),
          ('評価の途中で見つけた不具合（ハンドが開き直されない）も修正済み', 24, DIM),
          ('次: お手本の最初の止まっている時間を落として学習し直す（準備済み）', 24, ACCENT)], 6)
wr.close()
print('wrote', OUT, f'{wr.n / FPS:.1f} s')
