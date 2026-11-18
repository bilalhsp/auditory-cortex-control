"""Runtime setup acceptance through its CLI with controlled package/Git adapters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RuntimeSetupTests(unittest.TestCase):
    def test_failed_source_setup_can_be_retried_and_all_caches_use_selected_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            commands = root / 'bin'
            commands.mkdir()
            log = root / 'calls.jsonl'
            fail_once = root / 'fail-once'
            fail_once.touch()
            adapter = '''#!{python}
import json, os, pathlib, sys
with open(os.environ['RUNTIME_TEST_LOG'], 'a') as log:
    log.write(json.dumps({{'command': pathlib.Path(sys.argv[0]).name, 'args': sys.argv[1:],
        'caches': {{key: os.environ.get(key) for key in ['UV_CACHE_DIR', 'TMPDIR',
            'UV_PYTHON_INSTALL_DIR', 'PIP_CACHE_DIR', 'XDG_CACHE_HOME', 'CUPY_CACHE_DIR', 'CUDA_CACHE_PATH']}}}})+'\\n')
if pathlib.Path(sys.argv[0]).name == 'uv':
    target=pathlib.Path(os.environ['UV_PROJECT_ENVIRONMENT'])
    (target/'lib/python3.8/site-packages/nvidia').mkdir(parents=True, exist_ok=True)
    (target/'bin').mkdir(exist_ok=True)
    python=target/'bin/python'
    python.write_text('#!/usr/bin/env bash\\nexit 0\\n')
    python.chmod(0o755)
elif 'apply' in sys.argv:
    sys.stdin.buffer.read()
    source=pathlib.Path(sys.argv[sys.argv.index('-C')+1])
    patched=source/'patched'
    if patched.exists():
        sys.exit(2)
    patched.touch()
    fail=pathlib.Path(os.environ['RUNTIME_TEST_FAIL'])
    if fail.exists():
        fail.unlink()
        sys.exit(1)
'''.format(python=sys.executable)
            for name in ('uv', 'git'):
                command = commands / name
                command.write_text(adapter)
                command.chmod(0o755)
            repo = Path(__file__).resolve().parents[3]
            env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                       RUNTIME_TEST_LOG=str(log), RUNTIME_TEST_FAIL=str(fail_once),
                       AUDITORY_BOOTSTRAP_PYTHON=sys.executable)
            command = [str(repo / 'setup-runtime.sh'), '--cache-dir', str(cache)]
            failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(failed.returncode, 0)
            self.assertNotIn('Validated runtime:', failed.stdout)
            retried = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(retried.returncode, 0, retried.stdout + retried.stderr)
            self.assertIn('Validated runtime:', retried.stdout)
            for call in map(json.loads, log.read_text().splitlines()):
                for path in call['caches'].values():
                    self.assertIsNotNone(path)
                    self.assertIn(cache, Path(path).parents)
                if call['command'] == 'uv':
                    self.assertIn('--locked', call['args'])


if __name__ == '__main__':
    unittest.main()
