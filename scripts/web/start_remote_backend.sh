#!/usr/bin/env bash
# Called over SSH on a login node. Only the srun step performs computation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ $# == 4 ]] || { echo "Expected job-id node api-port frontend-origin" >&2; exit 2; }
job=$1 node=$2 port=$3 origin=$4
[[ "$job" =~ ^[0-9]+(_[0-9]+)?$ && "$node" =~ ^[a-zA-Z0-9.-]+$ && "$port" =~ ^[0-9]{1,5}$ ]] || exit 2
((10#$port >= 1 && 10#$port <= 65535)) || exit 2
[[ "$origin" =~ ^https?://[a-zA-Z0-9.:-]+$ ]] || exit 2
# Revalidate on the server: never start on another user's allocation or a login node.
allowed=false
while IFS='|' read -r identity nodes gres; do
    [[ "$identity" == "$job" && "$gres" == *gpu* ]] || continue
    while IFS= read -r member; do [[ "$member" == "$node" ]] && allowed=true; done < <(scontrol show hostnames "$nodes")
done < <(squeue --noheader --me --states=RUNNING --format='%i|%N|%b')
[[ "$allowed" == true ]] || { echo "Selected node is not in your running GPU allocation." >&2; exit 1; }
load_backend_config
launch_dir="$("$WEB_PYTHON" -c 'from pathlib import Path; import yaml; print(Path(yaml.safe_load(Path("auditory_cortex/config.yml").read_text())["cache_dir"]) / "web/launches")')"
umask 077
mkdir -p -- "$launch_dir"
instance="$launch_dir/$job-$node-$port"
if flock -n "$instance.lock" true; then
    rm -f -- "$instance.step"
    step_command='printf "%s.%s\n" "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$1"; cd "$2"; export AUDITORY_FRONTEND_ORIGIN="$3"; exec ./start-backend.sh --port "$4"'
    nohup flock -n "$instance.lock" srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
        bash -c "$step_command" auditory-backend "$instance.step" "$WEB_REPO_ROOT" "$origin" "$port" \
        > "$instance.log" 2>&1 < /dev/null &
    client=$!
    for ((attempt=0; attempt<15; attempt++)); do
        [[ -s "$instance.step" ]] && break
        kill -0 "$client" 2>/dev/null || { cat "$instance.log" >&2; exit 1; }
        sleep 1
    done
    [[ -s "$instance.step" ]] || { echo "Backend step is waiting; inspect $instance.log" >&2; exit 1; }
    echo "Backend launch requested in existing allocation."
else
    echo "Reusing this allocation's existing backend launcher (origins unchanged)."
fi
printf 'Remote backend log: %s.log\n' "$instance"
if [[ -f "$instance.step" ]]; then printf 'Backend Slurm step: %s (allocation is not cancelled on disconnect)\n' "$(cat "$instance.step")"; fi
