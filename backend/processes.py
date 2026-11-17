"""Owned compute subprocesses: cancellation waits for exit before releasing GPUs."""
import ctypes
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Callable


def require_parent(expected_parent: int) -> None:
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != expected_parent:
        raise SystemExit('Compute worker parent exited')


def run_compute(module: str, workspace, cancelled: Callable[[], bool], env: dict, tick=lambda: None, lock_fd=None) -> int:
    if cancelled(): return -signal.SIGTERM
    with (workspace / 'diagnostics.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-m', module, str(workspace.resolve()), str(os.getpid())],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=() if lock_fd is None else (lock_fd,))
        try:
            (workspace / 'process.json').write_text(json.dumps({'pid': process.pid, 'module': module}))
            while process.poll() is None:
                if cancelled():
                    process.terminate()
                    try: return process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        return process.wait()
                tick()
                time.sleep(.1)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            try:
                (workspace / 'process.json').write_text(json.dumps({'pid': process.pid, 'module': module, 'exit_code': process.returncode}))
            except OSError:
                logging.getLogger(__name__).exception('Could not record compute exit status')
