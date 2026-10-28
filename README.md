# シミュレーションで評価・学習するロボットアーム — CRANE+ V2

Physical AI 応用1講座の最終課題「言葉で片付けるロボットアーム」
（Gemini Robotics-ER × CRANE+ V2, [デモ動画](https://youtu.be/PHHKitfAc7g)）の発展版です。
ハードウェアを使わず、Gazebo の中で次の2つを進めます。

1. **評価基盤**: ブロックの数・位置・向きをランダムに変えたシーンで「同じ色のトレイに片付ける」を
   N 回続けて実行し、成功率と失敗の内訳（認識・推論・計画・把持・配置）を自動で測る
   （正解位置を使う先生役 oracle のベースライン: ブロック 1〜3 個・30 エピソードで成功率 100%, `results/`）
2. **方策の学習**: oracle のデモ（ブロック 1 個、300 本）から LeRobot の ACT を学習し、
   真上の画像と関節角だけでアームを動かす。乱数の配置 10 本で成功 2/10（下の「方策の学習（ACT）」）

元の課題の内容（Gemini による1手ずつの計画）はそのまま動きます。以下の「しくみ」〜「調整のポイント」は元の課題の説明です。

## しくみ

```
/instruction (日本語の指示)
        │
        ▼
┌───────────────────────── vlm_agent ノード ─────────────────────────┐
│ 1. 観察: アームを home に戻し、真上のカメラ画像 (image_raw) を取る    │
│ 2. 思考: Gemini Robotics-ER に画像・指示・これまでの履歴を渡し、     │
│          次の1手 {pick 点, place 点} か「完了」を JSON で受け取る     │
│ 3. 変換: 掴む点を上面の中心に補正 → 画素 → tf で base_link → 机と交差 │
│ 4. 行動: MoveIt (/move_action) で 掴む→持ち上げ→運ぶ→置く           │
│ 5. 1 に戻る（失敗しても次の観察で Gemini が気づいてやり直す）        │
└────────────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
 Gazebo 上の CRANE+ V2              /vlm/annotated_image（動画用の可視化）
```

### 講義・サンプルからの発展点

| ベース | このリポジトリで加えたこと |
|---|---|
| 第4回: Gazebo + CRANE+ V2 + MoveIt | カメラ付きモデルと独自ワールド（色ブロック3個・トレイ3枚）|
| `crane_plus_examples/color_detection` (HSV しきい値で赤だけ検出) | VLM で任意の物体・言語指示に対応。tf を使った任意平面への逆投影 |
| `crane_plus_examples/pick_and_place_tf` (置き場所固定) | 置き場所も VLM が画像から決定。多段タスクを1手ずつ計画 |
| サンプル「Gemini Robotics-ER を試す」(画像に点を打つだけ) | 点を実際のロボット動作につなげ、観察→思考→行動のループで失敗から回復 |
| Gemini の点をそのまま使う | 点が上面の端を指して 1〜2 cm ずれ、掴み損ねていた。指された点から同色の上面を flood fill で切り出し重心に補正（誤差 1 mm 以下に）|

## リポジトリ構成

```
vlm_pick_place/                  ROS 2 (ament_python) パッケージ
  vlm_pick_place/
    vlm_agent_node.py            メインノード（観察→思考→行動ループ）
    gemini_brain.py              Gemini 呼び出し・プロンプト・JSON 解析（ROS 非依存）
    geometry.py                  画素→机上座標、把持姿勢、到達判定（ROS 非依存）
    refine.py                    Gemini の掴む点をブロック上面の中心に寄せる（ROS 非依存）
    arm_commander.py             MoveGroup アクションで arm / gripper を動かす
    skills.py                    ピック&プレースの動作手順
    pick_test_node.py            Gemini なしで把持だけ試すノード
    recorder_node.py             デモ動画 (MP4) の録画
    eval_node.py                 ランダムなシーンで連続実行し評価・デモ収集（oracle / gemini / act）
    act_agent.py                 学習した ACT を推論サーバー経由で動かす（agent:=act）
    scene.py                     シーンのランダム生成、トレイに入ったかの判定（ROS 非依存）
    evaluation.py                1手ごとの失敗分類と集計（ROS 非依存）
    sim_world.py                 ign コマンドでモデル配置・正解位置の取得
    annotate.py                  動画用オーバーレイ
  launch/vlm_pick_place.launch.py  全部入りの起動
  launch/sim.launch.py           Gazebo + CRANE+ (カメラスタンド付き) + MoveIt（headless:=true で画面なし）
  launch/eval.launch.py          画面なしの Gazebo + 評価ノード（終わったら全部止める）
  worlds/vlm_table.sdf           tools/make_world.py で生成（vlm_table_eval.sdf は斜めカメラなし）
  config/params.yaml             高さ・オフセット・モデルなどの調整値
  test/                          単体テスト（ROS なしで実行可）
tools/
  scene_layout.py                物体配置（ワールド生成と検証で共有）
  make_world.py                  ワールド SDF の生成
  offline_check.py               Gazebo なしで Gemini の指差し精度を検証
  sync_to_docker.sh              コードを Docker コンテナへ反映してビルド
  stop_sim.sh                    コンテナ内のシミュレーションを全停止
  inspect_demo.py                収集したデモ (npz) の確認
  run_eval.sh / run_eval_detached.sh  評価を回す（固まったら起動し直して続きから. detached はコンテナ内で回し続ける）
  to_lerobot.py                  デモ (npz) を LeRobot のデータセットに変換
  act_server.py                  ACT の推論サーバー（Python 3.12 以上の lerobot の venv で動かす）
  check_policy_cpu.py            取り分けたデモで ACT の予測を手元の CPU で確かめる
notebooks/
  train_act_colab.ipynb          Colab (T4) で変換→学習→予測の確認. Drive に保存して途中から再開できる
```

## セットアップ（第4回の Docker 環境）

講義の `airobotbook/ros2-desktop-ai-robot-book-humble` コンテナ（名前 `pai_ros2`）で動かします。
この中の crane_plus は古い版なので、ワールドとカメラは `launch/sim.launch.py` と
`worlds/vlm_table.sdf` で自前で用意しています（crane_plus 側の変更は不要）。

```bash
# --- Windows / Mac 側（このリポジトリのフォルダで）---
docker start pai_ros2
bash tools/sync_to_docker.sh            # コードをコンテナの ~/pai_ws/src にコピーしてビルド
                                        # （提出版のコピーは ~/pai_ws/src_disabled へ退避される）

# --- コンテナ側（ブラウザで http://localhost:6080 → ターミナル）---
pip3 install --user google-genai        # 最初の1回だけ
echo 'export GEMINI_API_KEY=ここにキー' >> ~/.bashrc   # 最初の1回だけ（Google AI Studio で発行）
echo 'source ~/pai_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

コードを書き換えたら、もう一度 `bash tools/sync_to_docker.sh` を実行します。

## 実行（コンテナのデスクトップで）

```bash
# Gazebo + MoveIt + VLM エージェント + 可視化ウィンドウを起動（15秒後にエージェントが立ち上がる）
ros2 launch vlm_pick_place vlm_pick_place.launch.py

# 別ターミナルから指示を送る
ros2 topic pub --once /instruction std_msgs/msg/String "data: 'ブロックを同じ色のトレイに片付けて'"
ros2 topic pub --once /instruction std_msgs/msg/String "data: '一番右のブロックを赤いトレイに置いて'"

# 起動と同時に実行する（動画撮影向け）
ros2 launch vlm_pick_place vlm_pick_place.launch.py auto_instruction:='ブロックを同じ色のトレイに片付けて'
```

### デモ動画を録画する

```bash
ros2 launch vlm_pick_place vlm_pick_place.launch.py record:=true video:=$HOME/videos/demo.mp4     auto_instruction:='ブロックを同じ色のトレイに片付けて'
```

左に斜めカメラ、右に Gemini の判断を重ねた真上カメラ、上に指示と進捗（日本語）を並べた
1280x560 の MP4 (H.264) を保存し、タスクが終わると自動で閉じます（シミュレーション時間で 10 fps）。
倍速版は `ffmpeg -i demo.mp4 -vf setpts=0.5*PTS -r 20 demo_2x.mp4` で作れます（動画内に倍速を明記すること）。

進捗は `ros2 topic echo /vlm/status`、判断の可視化は rqt_image_view（`/vlm/annotated_image`）で見られます。
止めるときは Ctrl+C。ゾンビが残ったら Windows 側で `bash tools/stop_sim.sh`。

### Gemini の無料枠について

`gemini-robotics-er-2-preview` の無料枠は **1日20リクエスト**（1回の片付けで 4〜5 リクエスト）。
使い切ると自動で `fallback_model_ids`（`gemini-3.5-flash` など、枠はモデルごとに別）に切り替わります。
動画撮影の日は事前のテストを控えるか、Google AI Studio で課金を有効にしておくと安心です。

## 調整のポイント

まず Gazebo なしで Gemini 部分だけ確認できます（Windows/Mac でも可）:

```bash
pip install google-genai pillow numpy
python tools/offline_check.py                       # 合成画像で指差し誤差 [mm] を表示
python tools/offline_check.py --instruction "ブロックを同じ色のトレイに片付けて"
python tools/offline_check.py --image gazebo_camera.png --instruction "..."   # 実際のカメラ画像で
```

把持だけを Gemini を使わずに試せます（API 枠を消費しない）:

```bash
ros2 launch vlm_pick_place vlm_pick_place.launch.py use_agent:=false
ros2 run vlm_pick_place pick_test --ros-args   --params-file $(ros2 pkg prefix vlm_pick_place)/share/vlm_pick_place/config/params.yaml   -p use_sim_time:=true -p pick:="[0.1414, 0.1414]" -p place:="[0.2338, 0.1350]"
```

Gazebo で掴めないときは `config/params.yaml` を調整します:

| 症状 | 調整するパラメータ |
|---|---|
| 掴みに行く途中で隣の物体を倒す | `approach_z` を高く |
| 物体の手前/奥で空振りする | `gripper_offset`（link4 から把持点までの距離, 既定 0.13） |
| 上から押しつぶす/下を掴む | `grasp_z` |
| 掴めているのに滑り落ちる | `gripper_close_deg` を大きく |
| `too far` / `too close` で飛ばされる | 物体配置（`tools/scene_layout.py` → `make_world.py`）か `reach_r_*` |
| Gemini が遅い | `thinking_budget` を小さく（0 で思考なし） |

座標変換は `ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame` で確認できます。

## 評価（ランダムなシーンで連続実行）

```bash
# コンテナ内. 画面なしで Gazebo を起動し、終わったら自動で止まる
ros2 launch vlm_pick_place eval.launch.py episodes:=30 seed:=1000 agent:=oracle record:=true
```

| 引数 | 既定値 | 意味 |
|---|---|---|
| `agent` | `oracle` | `oracle`: Gazebo の正解位置を使う先生役（画像は見ない）／ `gemini`: 元の課題と同じ判断（無料枠に注意） |
| `episodes`, `seed` | 10, 0 | エピソード i のシーンは `seed + i` で決まる（同じ seed なら同じシーン） |
| `n_cubes` | 0 | ブロックの数. 0 なら 1〜3 個をランダムに |
| `yaw_jitter_deg` | 20 | ブロックの向きのばらつき（大きいほど掴みにくい） |
| `physics_step` | 0.004 | 物理の刻み幅 [s]. ワールドの既定 1 ms だと CPU では実時間の 1/6 ほどしか出ない |
| `record` | false | 学習用デモ（真上カメラ 320x240 JPEG と5関節の角度, 10 fps）を保存 |
| `output_dir` | `~/pai_eval/<日時>_<agent>` | 出力先 |

出力は `episodes.jsonl`（1行1エピソード: シーン・各手の結果・成否）、`summary.md` / `summary.json`、
`demos/episode_XXXX.npz`。デモの中身は `python3 tools/inspect_demo.py <npz> --sheet sheet.jpg` で確かめられます。

1手の結果の分類（`vlm_pick_place/evaluation.py`）:

| 結果 | 意味 |
|---|---|
| success | 狙ったブロックが同じ色のトレイに入った |
| perception | 掴む点がどのブロックにも当たっていない／置く点がどのトレイにも当たっていない |
| reasoning | 違う色のトレイに置こうとした |
| planning | IK が解けない・動作計画に失敗した |
| grasp | アームは動いたが、ブロックがほとんど動いていない（掴み損ね） |
| place | ブロックは動いたが、トレイに入らなかった |

しくみ: 評価ノードはエピソードごとに `ign service .../set_pose` でトレイとブロックを置き直し
（使わないブロックは机の外へ退避）、各手の前後に `ign topic .../pose/info` で正解位置を読んで判定します。
正解位置は方策には渡しません。CPU だけの環境（Intel Iris Xe, ソフトウェア描画）では、
刻み 4 ms・真上カメラ1台で実時間の 0.4〜0.7 倍、1手あたり実時間で約1分です。

## 方策の学習（ACT）

1. **デモを集める**（コンテナ内）: `bash tools/run_eval_detached.sh oracle_n1_s2000 episodes:=100 seed:=2000 n_cubes:=1 record:=true`
   （成功したエピソードだけが学習に使われる. 300 本で使ったのは seed 2000〜2099 と 2100〜2299）
2. **学習する**（Colab）: `tools/to_lerobot.py`・`tools/inspect_demo.py` とデモを Google Drive の `pai_sim/` に置き、
   `notebooks/train_act_colab.ipynb` を T4 で「すべてのセルを実行」. lerobot 0.6.1 の ACT（既定の設定、batch 8）。
   2,500 ステップごとに Drive に保存するので、無料枠で切れても続きから再開できる（4 万ステップで約 1 時間 40 分）。
   `TRIM_IDLE` を 0 より大きくすると、各デモの最初にアームが止まっている部分を落とす（`to_lerobot.py --trim-idle`）。
3. **動かす**: lerobot は Python 3.12 以上が必要で、講義の ROS 2 環境（3.10）には入らないので、推論だけを別のプロセスにする。

```bash
# コンテナ内の Python 3.12 の venv（uv などで作り、lerobot==0.6.1 と CPU 版 torch を入れる）
~/venvs/lerobot/bin/python -I tools/act_server.py <pretrained_model のフォルダ>      # 127.0.0.1:5555 で待つ
# 別のターミナルで評価（seed 3000〜3009 の 10 本）
bash tools/run_eval_detached.sh act_eval episodes:=10 seed:=3000 n_cubes:=1 agent:=act record:=true
```

`act_agent.py` は 1 回の推論（CPU で約 0.3 秒）で返る 100 ステップのうち先頭 `act_n_action_steps`（既定 10 = 1 秒分）を
関節の軌道として送り、`act_max_sec`（既定 40 秒）で打ち切って、最後にブロックの位置で判定します。
エピソードの開始時にはアームを home に戻し、ハンドを開きます。

### 結果（同じ 10 通りの配置: seed 3000〜3009）

| お手本 | 学習ステップ | 成功 | 運んだ | うち正しい色 |
|---|---|---|---|---|
| 100 本 | 2 万 | 1/10 | 8 | 4 |
| 114 本 | 2.75 万 | 1/10 | 7 | 3 |
| 300 本 | 2 万 | 0/10 | 3 | 2 |
| 300 本 | 4 万 | 2/10 | 3 | 2 |
| 300 本 | 4 万（開始時にハンドを開くよう修正したあと） | **2/10** | 5 | 2 |

「運んだ」はブロックが 8 cm 以上動いた本数。上の 4 行は、前のエピソードの終わりのハンドの状態が残る不具合があったときの評価。
失敗は主に 2 種類:

- **動かない**: ブロックがベースの手前（0.17 m 未満）にあると home のまま止まる. oracle のデモは手前のブロックほど
  動き出すまで長く止まっていた（中央値 2.8 秒、遠いと 0.8 秒）ので、それを真似たとみられる → `--trim-idle` で落として学習し直す予定
- **違う色のトレイ**: 置く位置は 1〜3 cm と正確だが、運んだうち正しい色は約半分（5 回の評価の合計 13/26）

## テスト

```bash
cd vlm_pick_place && python -m pytest test     # ROS なしで動く（GitHub Actions でも実行）
```

## ライセンス

Apache-2.0。CRANE+ V2 のモデル・サンプルは [rt-net/crane_plus](https://github.com/rt-net/crane_plus)（Apache-2.0）を利用しています。
