#!/usr/bin/env bash

set -o pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_file="$script_dir/terminal_execution.log"

if (( $# == 0 )); then
    echo "사용법: ./run_logged.sh <실행할 명령어> [인자 ...]"
    exit 2
fi

started_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
started_seconds=$SECONDS

{
    echo
    echo "================================================================"
    echo "실행 시작: $started_at"
    echo "작업 위치: $PWD"
    printf '실행 명령:'
    printf ' %q' "$@"
    echo
    echo "----------------------------------------------------------------"
} | tee -a "$log_file"

"$@" 2>&1 | tee -a "$log_file"
exit_code=${PIPESTATUS[0]}

elapsed_seconds=$((SECONDS - started_seconds))
finished_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"

{
    echo "----------------------------------------------------------------"
    echo "실행 종료: $finished_at"
    echo "종료 코드: $exit_code"
    printf '경과 시간: %02d:%02d:%02d\n' \
        $((elapsed_seconds / 3600)) \
        $(((elapsed_seconds % 3600) / 60)) \
        $((elapsed_seconds % 60))
    echo "================================================================"
} | tee -a "$log_file"

exit "$exit_code"
