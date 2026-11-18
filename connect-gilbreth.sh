#!/usr/bin/env bash
# Run on your laptop with Bash 4+, OpenSSH, and curl (Git Bash or WSL on Windows).
set -euo pipefail
login="${GILBRETH_LOGIN_HOST:-gilbreth}"
repo="${GILBRETH_REPO:-/home/ahmedb/projects/project-phase-2}"
origin="${GILBRETH_FRONTEND_ORIGIN:-http://127.0.0.1:5173}"
ports="${GILBRETH_LOCAL_PORTS:-28000 28001 28002 18000 8000}"
job_id=""
requested_node=""
api_port=8000
usage() {
    cat <<'EOF'
Usage: bash connect-gilbreth.sh [options]
  --login ALIAS             Existing SSH login-node alias (default: gilbreth)
  --repo PATH               Repository path on Gilbreth
  --job-id ID               Existing running GPU allocation; required if ambiguous
  --node NODE               Choose a node/allocation, e.g. gilbreth-n6
  --frontend-origin URL     Frontend origin, e.g. https://USERNAME.github.io
  --ports "28000 28001 ..." Ordered laptop ports also configured in the frontend
  --api-port PORT           Compute-node API port (default: 8000)
No allocation is requested or cancelled. Backend remains running after disconnect.
EOF
}
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --login|--repo|--job-id|--node|--frontend-origin|--ports|--api-port)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --login) login="$2" ;; --repo) repo="$2" ;; --job-id) job_id="$2" ;;
                --node) requested_node="$2" ;; --frontend-origin) origin="$2" ;;
                --ports) ports="$2" ;; --api-port) api_port="$2" ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
((BASH_VERSINFO[0] >= 4)) || { echo "Use Bash 4+ (Git Bash or WSL)." >&2; exit 2; }
for program in ssh curl awk grep tr mktemp; do command -v "$program" >/dev/null || { echo "Missing command: $program" >&2; exit 2; }; done
[[ "$login" != -* && "$login" != *[$'\r\n\t ']* && -n "$login" ]] || { echo "Invalid SSH alias" >&2; exit 2; }
[[ "$repo" == /* && "$repo" != *[$'\r\n']* ]] || { echo "--repo must be an absolute remote path" >&2; exit 2; }
[[ -z "$job_id" || "$job_id" =~ ^[0-9]+(_[0-9]+)?$ ]] || { echo "Invalid job ID" >&2; exit 2; }
[[ -z "$requested_node" || "$requested_node" =~ ^[a-zA-Z0-9.-]+$ ]] || { echo "Invalid node name" >&2; exit 2; }
[[ "$origin" =~ ^https?://[a-zA-Z0-9.:-]+$ ]] || { echo "Use an HTTP(S) origin without a path or trailing slash" >&2; exit 2; }
valid_port() { [[ "$1" =~ ^[0-9]{1,5}$ ]] && ((10#$1 >= 1 && 10#$1 <= 65535)); }
valid_port "$api_port" || { echo "Invalid API port" >&2; exit 2; }
read -r -a candidates <<< "$ports"
((${#candidates[@]})) || { echo "No laptop ports configured" >&2; exit 2; }
for port in "${candidates[@]}"; do valid_port "$port" || { echo "Invalid laptop port: $port" >&2; exit 2; }; done
normalize_node() {
    local short="${1%%.*}"
    if [[ "$short" =~ ^([a-zA-Z-]+)([0-9]+)$ ]]; then
        printf '%s%d' "${BASH_REMATCH[1]}" "$((10#${BASH_REMATCH[2]}))"
    else printf '%s' "$short"; fi
}
work="$(mktemp -d)"
tunnel_pid=""
tunnel_input=""
tunnel_output=""
cleanup() {
    if [[ -n "$tunnel_input" ]]; then exec {tunnel_input}>&-; fi
    if [[ -n "$tunnel_output" ]]; then exec {tunnel_output}<&-; fi
    if [[ -n "$tunnel_pid" ]]; then
        kill "$tunnel_pid" 2>/dev/null || true
        wait "$tunnel_pid" 2>/dev/null || true
    fi
    rm -rf -- "$work"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Discovering running GPU allocations through %s...\n' "$login"
ssh -T "$login" 'bash -s' > "$work/jobs" <<'REMOTE'
set -euo pipefail
printf 'AUDITORY_USER|%s\n' "$(id -un)"
while IFS='|' read -r job nodes gres; do
    [[ "$gres" == *gpu* ]] || continue
    while IFS= read -r node; do printf 'AUDITORY_JOB|%s|%s\n' "$job" "$node"; done < <(scontrol show hostnames "$nodes")
done < <(squeue --noheader --me --states=RUNNING --format='%i|%N|%b')
REMOTE
remote_user="$(awk -F '|' '$1=="AUDITORY_USER" {print $2; exit}' "$work/jobs")"
[[ "$remote_user" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Could not read remote username" >&2; exit 1; }
selected=()
while IFS='|' read -r marker found_job node; do
    [[ "$marker" == AUDITORY_JOB && "$found_job" =~ ^[0-9]+(_[0-9]+)?$ && "$node" =~ ^[a-zA-Z0-9.-]+$ ]] || continue
    [[ -z "$job_id" || "$job_id" == "$found_job" ]] || continue
    [[ -z "$requested_node" || "$(normalize_node "$requested_node")" == "$(normalize_node "$node")" ]] || continue
    selected+=("$found_job|$node")
done < "$work/jobs"
if ((${#selected[@]} != 1)); then
    echo "Select exactly one running GPU allocation/node with --job-id and/or --node." >&2
    awk -F '|' '$1=="AUDITORY_JOB" {print "  Job " $2 ": " $3}' "$work/jobs" >&2
    exit 1
fi
IFS='|' read -r job_id node <<< "${selected[0]}"
printf 'Using job %s on %s.\n' "$job_id" "$node"
printf -v remote_command '%q ' bash "$repo/scripts/web/start_remote_backend.sh" "$job_id" "$node" "$api_port" "$origin"
ssh -T "$login" "$remote_command"
# Each failed bind uses a separate SSH process. No local socket probes or public binds.
for port in "${candidates[@]}"; do
    printf 'Trying laptop port %s...\n' "$port"
    coproc TUNNEL { exec ssh -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -J "$login" -l "$remote_user" -L "127.0.0.1:$port:127.0.0.1:$api_port" "$node" \
        "printf '%s\\n' AUDITORY_TUNNEL_READY; exec cat" 2> "$work/tunnel.log"; }
    tunnel_pid=$!
    tunnel_input=${TUNNEL[1]}
    tunnel_output=${TUNNEL[0]}
    connected=false
    for ((attempt=0; attempt<180; attempt++)); do
        if IFS= read -r -t 1 -u "$tunnel_output" line; then
            if [[ "$line" == AUDITORY_TUNNEL_READY ]]; then connected=true; break; fi
        elif ! kill -0 "$tunnel_pid" 2>/dev/null; then break; fi
    done
    if [[ "$connected" == true ]]; then break; fi
    exec {tunnel_input}>&-; tunnel_input=""
    exec {tunnel_output}<&-; tunnel_output=""
    kill "$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
    tunnel_pid=""
    if ! grep -Eq 'bind .*:|cannot listen to port|Could not request local forwarding' "$work/tunnel.log"; then
        cat "$work/tunnel.log" >&2
        echo "SSH failed; authentication/host-key errors are not port failures." >&2
        exit 1
    fi
    cat "$work/tunnel.log" >&2
done
[[ "$connected" == true ]] || { echo "All configured laptop ports are unavailable." >&2; exit 1; }
url="http://127.0.0.1:$port"
printf 'Waiting for backend at %s...\n' "$url"
healthy=false
for ((attempt=0; attempt<90; attempt++)); do
    kill -0 "$tunnel_pid" 2>/dev/null || { cat "$work/tunnel.log" >&2; echo "Tunnel closed" >&2; exit 1; }
    if curl --silent --fail --connect-timeout 1 --max-time 2 "$url/healthz?identify=true" > "$work/health" && grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "$work/health" && grep -Eq '"service"[[:space:]]*:[[:space:]]*"auditory-cortex-control"' "$work/health"; then healthy=true; break; fi
    sleep 1
done
[[ "$healthy" == true ]] || { echo "Backend did not become healthy. Inspect the remote launch log printed above." >&2; exit 1; }
if ! curl --silent --show-error --fail --max-time 5 -D "$work/cors" -o /dev/null -X OPTIONS "$url/api/v1/capabilities" \
    -H "Origin: $origin" -H 'Access-Control-Request-Method: GET' -H 'Access-Control-Request-Headers: authorization'; then
    echo "Backend rejected the frontend origin $origin. Check the remote launch log and CORS configuration." >&2
    exit 1
fi
allowed_origin="$(tr -d '\r' < "$work/cors" | awk 'tolower($1)=="access-control-allow-origin:" {sub(/^[^:]+:[[:space:]]*/, ""); print; exit}')"
if [[ "$allowed_origin" != "$origin" ]]; then
    echo "Backend does not allow $origin. Restart its launcher with the correct frontend origin." >&2; exit 1
fi
printf '\nConnected: %s (job %s, node %s)\nKeep this terminal open; Ctrl+C closes only this tunnel.\n' "$url" "$job_id" "$node"
wait "$tunnel_pid"
