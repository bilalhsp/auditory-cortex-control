"""Exercise the laptop CLI with controlled OpenSSH/Slurm command adapters."""
import os
import select
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class LaptopConnectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repo = Path(__file__).resolve().parents[3]
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.ssh_log = self.root / 'ssh-calls'
        self.config = self.root / 'jobs'
        self.config.write_text('AUDITORY_USER|operator\nAUDITORY_JOB|123|gilbreth-n006\n')
        self.make_command('ssh', r'''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CHECK_SSH_LOG"
case "$*" in
  *"bash -s"*) cat "$CHECK_JOBS" ;;
  *start_remote_backend.sh*) echo 'Remote backend log: test.log' ;;
  *)
    if [[ "$CHECK_MODE" == auth_failure ]]; then echo 'Permission denied (publickey)' >&2; exit 255; fi
    if [[ "$CHECK_MODE" == all_ports || "$*" == *127.0.0.1:28000:* ]]; then echo 'bind [127.0.0.1]:28000: Permission denied' >&2; exit 255; fi
    printf '%s\n' AUDITORY_TUNNEL_READY
    cat ;;
esac
''')
        self.make_command('curl', r'''#!/usr/bin/env bash
while (($#)); do
  if [[ "$1" == '-D' ]]; then
    allowed=https://operator.github.io
    [[ "$CHECK_MODE" != cors_failure ]] || allowed=https://other.github.io
    printf 'HTTP/1.1 200 OK\r\naccess-control-allow-origin: %s\r\n' "$allowed" > "$2"; shift 2; continue
  fi
  if [[ "$1" == '-o' ]]; then shift 2; continue; fi
  shift
done
printf '%s' '{"status":"ok","service":"auditory-cortex-control"}'
''')
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        CHECK_SSH_LOG=str(self.ssh_log), CHECK_JOBS=str(self.config), CHECK_MODE='normal')
        self.command = ['bash', str(self.repo / 'connect-gilbreth.sh'), '--login', 'cluster-alias',
                        '--frontend-origin', 'https://operator.github.io']

    def make_command(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def test_missing_allocation_stops_before_backend_or_tunnel(self):
        self.config.write_text('AUDITORY_USER|operator\n')
        result = subprocess.run(self.command, env=self.env, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Select exactly one', result.stderr)
        self.assertNotIn('start_remote_backend', self.ssh_log.read_text())

    def test_multiple_allocations_require_selection(self):
        with self.config.open('a') as handle:
            handle.write('AUDITORY_JOB|456|gilbreth-n007\n')
        result = subprocess.run(self.command, env=self.env, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Job 123', result.stderr)
        self.assertIn('Job 456', result.stderr)
        self.assertNotIn('start_remote_backend', self.ssh_log.read_text())

    def test_node_alias_and_blocked_port_fallback_connect_without_cancelling_allocation(self):
        process = subprocess.Popen(self.command + ['--node', 'gilbreth-n6'], env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True)
        output = ''
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                ready, _, _ = select.select([process.stdout], [], [], .2)
                if ready:
                    chunk = os.read(process.stdout.fileno(), 4096).decode()
                    output += chunk
                    if 'Connected:' in output:
                        break
                    if not chunk:
                        break
            self.assertIn('Connected: http://127.0.0.1:28001', output)
            self.assertIn('node gilbreth-n006', output)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=5)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        calls = self.ssh_log.read_text()
        self.assertIn('-J cluster-alias', calls)
        self.assertIn('127.0.0.1:28000:127.0.0.1:8000', calls)
        self.assertIn('127.0.0.1:28001:127.0.0.1:8000', calls)
        self.assertNotIn('scancel', calls)
        self.assertNotIn('salloc', calls)

    def test_authentication_failure_does_not_try_another_port(self):
        env = dict(self.env, CHECK_MODE='auth_failure')
        result = subprocess.run(self.command, env=env, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('authentication/host-key', result.stderr)
        self.assertNotIn('127.0.0.1:28001:', self.ssh_log.read_text())

    def test_all_ports_blocked_reports_failure_without_cancelling_backend(self):
        result = subprocess.run(self.command + ['--ports', '28000 28001'],
            env=dict(self.env, CHECK_MODE='all_ports'), capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('All configured laptop ports are unavailable', result.stderr)
        self.assertNotIn('scancel', self.ssh_log.read_text())

    def test_wrong_cors_origin_does_not_report_success(self):
        result = subprocess.run(self.command, env=dict(self.env, CHECK_MODE='cors_failure'),
                                capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not allow', result.stderr)
        self.assertNotIn('Connected:', result.stdout)
