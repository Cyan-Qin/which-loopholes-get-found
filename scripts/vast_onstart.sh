#!/usr/bin/env bash
# Resume the main experiment after the Vast container restarts (findings section 2).
# Vast runs /root/onstart.sh at every container start; install this script there once:
#   echo 'bash /workspace/which-loopholes-get-found/scripts/vast_onstart.sh' >> /root/onstart.sh
# It does nothing unless runs/main/AUTO_RESUME exists (create it to opt in, remove it to opt out).
# Then it starts two detached tmux sessions: "queue" (python -m src.train.queue, which skips what is
# done and repeats the interrupted run) and "autostop" (scripts/stop_when_done.sh), unless a session of
# that name is already running. Attach with `tmux attach -t queue`. Output also goes to runs/main/*.out.

REPO=/workspace/which-loopholes-get-found
cd "$REPO" || exit 0
[ -f runs/main/AUTO_RESUME ] || exit 0
ENV="source /workspace/loopholes-env.sh"
start() {
  local name=$1 cmd=$2
  if tmux has-session -t "$name" 2>/dev/null; then echo "tmux session $name already running"; return; fi
  tmux new-session -d -s "$name" "cd $REPO && $ENV && $cmd 2>&1 | tee -a runs/main/$name.out; exec bash"
  echo "[$(date '+%F %T')] started tmux session $name"
}
start queue "python -m src.train.queue"
sleep 60   # let the queue start before the watcher looks for it
start autostop "bash scripts/stop_when_done.sh"
