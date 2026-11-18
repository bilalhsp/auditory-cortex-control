"""Backend launch acceptance through the process and HTTP boundaries."""
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

from backend.auth import hash_password


class BackendLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password_hash = hash_password('launch-test-password')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = Path(__file__).resolve().parents[2]
        self.config = self.root / 'local.env'
        values = {'AUDITORY_OPERATOR': 'operator', 'AUDITORY_PASSWORD_HASH': self.password_hash,
                  'AUDITORY_DATABASE_PATH': str(self.root / 'sessions.sqlite3')}
        self.config.write_text(''.join('export %s=%s\n' % (key, shlex.quote(value)) for key, value in values.items()))
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.port = listener.getsockname()[1]

    def launch(self, *args):
        return subprocess.Popen([str(self.repo / 'start-backend.sh'), '--port', str(self.port), *args],
            cwd=str(self.root), env=dict({key: value for key, value in os.environ.items()
                if key != 'AUDITORY_FRONTEND_ORIGIN'}, AUDITORY_ENV_FILE=str(self.config), CUDA_VISIBLE_DEVICES='-1'),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def stop(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if process.stdout:
            process.stdout.close()

    def test_default_launch_rejects_unusable_cuda_before_serving_api(self):
        process = self.launch()
        try:
            try:
                output, _ = process.communicate(timeout=25)
            except subprocess.TimeoutExpired:
                self.fail('Default backend stayed running without a usable GPU worker')
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('GPU preflight failed', output)
            with socket.socket() as connection:
                self.assertNotEqual(connection.connect_ex(('127.0.0.1', self.port)), 0)
        finally:
            self.stop(process)

    def test_api_only_launch_serves_authenticated_metadata_without_cuda(self):
        import base64
        import json
        import time
        import urllib.request
        process = self.launch('--api-only')
        try:
            url = 'http://127.0.0.1:%d' % self.port
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                self.assertIsNone(process.poll(), 'API-only launch exited unexpectedly')
                try:
                    with urllib.request.urlopen(url + '/healthz', timeout=1) as response:
                        self.assertEqual(json.loads(response.read()), {'status': 'ok'})
                    break
                except OSError:
                    time.sleep(.2)
            else:
                self.fail('API-only launch did not become healthy')
            authorization = 'Basic ' + base64.b64encode(b'operator:launch-test-password').decode()
            request = urllib.request.Request(url + '/api/v1/sessions', headers={'Authorization': authorization})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read()), [])
        finally:
            self.stop(process)
        with socket.socket() as connection:
            self.assertNotEqual(connection.connect_ex(('127.0.0.1', self.port)), 0)

    def test_default_launcher_allows_bilalhsp_pages_origin(self):
        import time
        import urllib.request
        process = self.launch('--api-only')
        try:
            request = urllib.request.Request('http://127.0.0.1:%d/api/v1/capabilities' % self.port,
                method='OPTIONS', headers={'Origin': 'https://bilalhsp.github.io',
                    'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'authorization'})
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                self.assertIsNone(process.poll(), 'API-only launch exited unexpectedly')
                try:
                    with urllib.request.urlopen(request, timeout=1) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'https://bilalhsp.github.io')
                    break
                except OSError:
                    time.sleep(.2)
            else:
                self.fail('Default launcher did not allow the configured Pages origin')
        finally:
            self.stop(process)
