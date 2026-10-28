#!/usr/bin/env bash
# 評価を最後まで回す. シミュレータが固まって止まったら、起動し直して続きから行う.
#   bash tools/run_eval.sh <名前> episodes:=30 seed:=1000 record:=true [他の eval.launch.py 引数]
# 結果はコンテナの ~/pai_eval/<名前>/ に溜まる. 途中で止めても、同じコマンドで続きから再開できる.
set -uo pipefail
NAME=${1:?usage: run_eval.sh <name> episodes:=N [launch args...]}
shift
CONTAINER=${CONTAINER:-pai_ros2}
EPISODES_PER_RUN=${EPISODES_PER_RUN:-10}   # この数ごとにシミュレータを起動し直す（だんだん重くなるため）
MAX_RESTARTS=${MAX_RESTARTS:-20}
OUT=/home/ubuntu/pai_eval/$NAME
export MSYS_NO_PATHCONV=1   # Git Bash が /home/... を Windows パスに変換しないように

TOTAL=$(printf '%s\n' "$@" | sed -n 's/^episodes:=//p')
TOTAL=${TOTAL:-10}
DIR=$(dirname "$0")

recorded() {
  docker exec "$CONTAINER" bash -c "[ -f $OUT/episodes.jsonl ] && grep -c . $OUT/episodes.jsonl || echo 0"
}

for run in $(seq 1 "$MAX_RESTARTS"); do
  n=$(recorded)
  echo "[run_eval] $(date +%H:%M:%S) run $run: $n / $TOTAL episodes recorded"
  [ "$n" -ge "$TOTAL" ] && break
  bash "$DIR/stop_sim.sh" "$CONTAINER" > /dev/null
  docker exec -u ubuntu "$CONTAINER" bash -c "unset DISPLAY; source /opt/ros/humble/setup.bash; \
    source ~/pai_ws/install/setup.bash; mkdir -p $OUT; \
    ros2 launch vlm_pick_place eval.launch.py output_dir:=$OUT episodes_per_run:=$EPISODES_PER_RUN $* \
    >> $OUT/launch.log 2>&1"
done
bash "$DIR/stop_sim.sh" "$CONTAINER" > /dev/null
n=$(recorded)
echo "[run_eval] finished: $n / $TOTAL episodes"
docker exec "$CONTAINER" cat "$OUT/summary.md" 2>/dev/null
[ "$n" -ge "$TOTAL" ]
