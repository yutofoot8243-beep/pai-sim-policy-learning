#!/usr/bin/env bash
# run_eval.sh と同じことを、起動し直しの繰り返しごとコンテナの中で動かす（docker exec -d）.
# Windows 側のウィンドウや Claude Code が止まっても続く. 長いデモ収集向け.
#   bash tools/run_eval_detached.sh <名前> episodes:=N [他の eval.launch.py 引数]
# 進み具合: docker exec pai_ros2 tail -f /home/ubuntu/pai_eval/<名前>/run_eval.log
# 止める:   docker exec pai_ros2 pkill -f '[r]un_eval_loop' してから bash tools/stop_sim.sh
set -uo pipefail
NAME=${1:?usage: run_eval_detached.sh <name> episodes:=N [launch args...]}
shift
CONTAINER=${CONTAINER:-pai_ros2}
EPISODES_PER_RUN=${EPISODES_PER_RUN:-10}
MAX_RESTARTS=${MAX_RESTARTS:-40}
OUT=/home/ubuntu/pai_eval/$NAME
export MSYS_NO_PATHCONV=1

TOTAL=$(printf '%s\n' "$@" | sed -n 's/^episodes:=//p')
TOTAL=${TOTAL:-10}

if docker exec "$CONTAINER" pgrep -f '[r]un_eval_loop' > /dev/null; then
  echo 'already running (run_eval_loop)'; exit 1
fi
bash "$(dirname "$0")/stop_sim.sh" "$CONTAINER" > /dev/null

# 下のスクリプトの中に「ros2 launch」などをそのまま書くと、pkill -f（stop_sim.sh も）が
# このループ自身にもマッチして止めてしまうので、"launch" と引用符で区切っている.
# 引数の先頭 run_eval_loop は $0 になり、pkill -f '[r]un_eval_loop' で見つけるための目印.
docker exec -d -u ubuntu "$CONTAINER" bash -c '
  OUT=$1; TOTAL=$2; PER_RUN=$3; MAX=$4; shift 4
  mkdir -p "$OUT"
  exec >> "$OUT/run_eval.log" 2>&1
  recorded() { [ -f "$OUT/episodes.jsonl" ] && grep -c . "$OUT/episodes.jsonl" || echo 0; }
  stop() {
    for p in "[r]os2 launch" "[i]gn gazebo" "[m]ove_group" "[p]arameter_bridge" \
             "[r]obot_state_publisher" "[s]pawner"; do pkill -9 -f "$p"; done
    sleep 1
  }
  for run in $(seq 1 "$MAX"); do
    n=$(recorded)
    echo "[run_eval] $(date +%H:%M:%S) run $run: $n / $TOTAL episodes recorded"
    [ "$n" -ge "$TOTAL" ] && break
    stop
    (unset DISPLAY; source /opt/ros/humble/setup.bash; source ~/pai_ws/install/setup.bash
     ros2 "launch" vlm_pick_place eval.launch.py output_dir:="$OUT" episodes_per_run:="$PER_RUN" "$@" \
       >> "$OUT/launch.log" 2>&1)
  done
  stop
  echo "[run_eval] $(date +%H:%M:%S) finished: $(recorded) / $TOTAL episodes"
  cat "$OUT/summary.md" 2>/dev/null
' run_eval_loop "$OUT" "$TOTAL" "$EPISODES_PER_RUN" "$MAX_RESTARTS" "$@"

sleep 3
if docker exec "$CONTAINER" pgrep -f '[r]un_eval_loop' > /dev/null; then
  echo "started. log: docker exec $CONTAINER tail -f $OUT/run_eval.log"
else
  echo 'failed to start'; exit 1
fi
