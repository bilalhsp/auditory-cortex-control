"""Start and monitor one API and one compute worker in the same allocation."""
import argparse
import os
import signal
import subprocess
import sys
import time


def stop_services(processes):
    # Each service owns a process group, including any native compute children.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--api-only', action='store_true')
    args, api_args = parser.parse_known_args()
    api_command = [sys.executable, '-m', 'uvicorn', 'backend.app:create_app', '--factory', *api_args]
    if args.api_only or '--help' in api_args:
        os.execv(sys.executable, api_command)
    # Use a short-lived process so the supervisor holds no CUDA context.
    result = subprocess.call([sys.executable, '-m', 'backend.gpu'])
    if result:
        return result if result > 0 else 128 - result
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {kind: signal.signal(kind, request_stop) for kind in (signal.SIGINT, signal.SIGTERM)}
    processes = []
    try:
        for command in ([sys.executable, '-m', 'backend.worker'], api_command):
            processes.append(subprocess.Popen(command, start_new_session=True))
        print('Backend API and compute worker started. Ctrl+C stops both.', flush=True)
        while not stopping:
            for name, process in zip(('Compute worker', 'API'), processes):
                result = process.poll()
                if result is not None:
                    print('%s exited (%d); stopping the backend.' % (name, result), file=sys.stderr, flush=True)
                    return result if result > 0 else (128 - result if result < 0 else 1)
            time.sleep(.2)
        return 0
    finally:
        stop_services(processes)
        for kind, handler in previous.items():
            signal.signal(kind, handler)


if __name__ == '__main__':
    raise SystemExit(main())
