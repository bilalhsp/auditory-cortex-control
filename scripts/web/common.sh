#!/usr/bin/env bash
# Shared setup for the local launchers. Source this file; do not run it directly.
WEB_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WEB_PYTHON="$WEB_REPO_ROOT/.venv/bin/python"
if [[ ! -x "$WEB_PYTHON" ]]; then
    echo "Existing Python environment missing: $WEB_PYTHON" >&2
    return 1
fi
cd "$WEB_REPO_ROOT"

load_backend_config() {
    local config_file config_dir config_temp operator_name password_hash dependency_dir database_path setting
    config_file="${AUDITORY_ENV_FILE:-}"
    if [[ -z "$config_file" ]]; then
        config_file="$("$WEB_PYTHON" -c 'from pathlib import Path; import yaml; print(Path(yaml.safe_load(Path("auditory_cortex/config.yml").read_text())["cache_dir"]) / "web/local.env")')"
    fi
    if [[ ! -f "$config_file" ]]; then
        if [[ "${1:-}" != setup || ! -t 0 ]]; then
            echo "Backend configuration missing: $config_file" >&2
            echo "Run ./start-backend.sh interactively once to configure your password." >&2
            return 1
        fi
        read -r -p "Operator username [operator]: " operator_name
        operator_name="${operator_name:-operator}"
        password_hash="$("$WEB_PYTHON" -m backend.auth)" || return 1
        database_path="${AUDITORY_DATABASE_PATH:-}"
        if [[ -z "$database_path" ]]; then
            database_path="$("$WEB_PYTHON" -c 'from pathlib import Path; import yaml; print(Path(yaml.safe_load(Path("auditory_cortex/config.yml").read_text())["cache_dir"]) / "web/sessions.sqlite3")')"
        fi
        config_dir="$(dirname -- "$config_file")"
        mkdir -p -- "$config_dir"
        config_temp="$(umask 077; mktemp "$config_dir/.local.env.XXXXXX")"
        {
            printf 'export AUDITORY_OPERATOR=%q\n' "$operator_name"
            printf 'export AUDITORY_PASSWORD_HASH=%q\n' "$password_hash"
            printf 'export AUDITORY_DATABASE_PATH=%q\n' "$database_path"
            printf 'export AUDITORY_ALLOWED_ORIGINS=%q\n' "${AUDITORY_ALLOWED_ORIGINS:-http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173,http://127.0.0.1:15173,http://localhost:15173,https://bilalhsp.github.io}"
            for setting in AUDITORY_NEURAL_DATA_DIR AUDITORY_RECORDINGS_METADATA AUDITORY_UPLOAD_DIR AUDITORY_FIT_DIR AUDITORY_MAX_UPLOAD_BYTES; do
                if [[ -n "${!setting:-}" ]]; then
                    printf 'export %s=%q\n' "$setting" "${!setting}"
                fi
            done
        } > "$config_temp"
        # Do not overwrite configuration another launcher may have created.
        if ! ln -- "$config_temp" "$config_file"; then
            rm -f -- "$config_temp"
            echo "Configuration was created concurrently. Run this command again." >&2
            return 1
        fi
        rm -f -- "$config_temp"
        echo "Saved private backend configuration: $config_file"
    fi
    # This is the operator's trusted local shell configuration, never frontend data.
    source "$config_file"
    : "${AUDITORY_OPERATOR:?Set AUDITORY_OPERATOR in $config_file}"
    : "${AUDITORY_PASSWORD_HASH:?Set AUDITORY_PASSWORD_HASH in $config_file}"
    export AUDITORY_OPERATOR AUDITORY_PASSWORD_HASH
    if [[ -n "${AUDITORY_FRONTEND_ORIGIN:-}" ]]; then
        export AUDITORY_ALLOWED_ORIGINS="${AUDITORY_ALLOWED_ORIGINS:+$AUDITORY_ALLOWED_ORIGINS,}$AUDITORY_FRONTEND_ORIGIN"
    fi
    export PYTHONPATH="$WEB_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    if [[ -f "$WEB_REPO_ROOT/.venv/runtime.env" ]]; then
        source "$WEB_REPO_ROOT/.venv/runtime.env"
        return
    fi
    for dependency_dir in "$WEB_REPO_ROOT/../deepspeech.pytorch" "$WEB_REPO_ROOT/../Wav2Letter" "$WEB_REPO_ROOT/../utils_jgm" "$WEB_REPO_ROOT/../pycochleagram"; do
        if [[ -d "$dependency_dir" ]]; then
            export PYTHONPATH="$PYTHONPATH:$dependency_dir"
        fi
    done
}
