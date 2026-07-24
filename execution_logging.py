"""프로젝트 CLI 출력을 터미널과 로그 파일에 동시에 기록한다."""

from __future__ import annotations

import atexit
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


LOG_PATH = Path(__file__).resolve().parent / "terminal_execution.log"
_LOG_STARTED = False


class _Tee:
    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def start_automatic_logging() -> None:
    """현재 Python 프로그램의 stdout과 stderr를 프로젝트 로그에 누적한다."""
    global _LOG_STARTED
    if _LOG_STARTED:
        return
    _LOG_STARTED = True

    log_file = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)

    print()
    print("================================================================")
    print(f"자동 로그 시작: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"작업 위치: {Path.cwd()}")
    print(f"실행 명령: {shlex.join(sys.argv)}")
    print("----------------------------------------------------------------")

    def finish_logging() -> None:
        print("----------------------------------------------------------------")
        print(f"자동 로그 종료: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
        print("================================================================")
        log_file.flush()

    atexit.register(finish_logging)
