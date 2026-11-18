"""Build the Python 3.8.5 runtime and its separate CUDA 12 toolkit on scratch."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
PATCHES = REPO / '.scratch/web-migration/dependency-patches'


def run(command, env, cwd=REPO, **kwargs):
    subprocess.run([str(item) for item in command], env=env, cwd=str(cwd), check=True, **kwargs)


def assemble_toolkit(packages, toolkit):
    """Expose NVIDIA wheels as the toolkit layout understood by CuPy 12."""
    for component in sorted(packages.iterdir()):
        for source in component.rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            relative = source.relative_to(component)
            if relative.parts[0] not in ('include', 'lib', 'bin', 'nvvm'):
                continue
            directory = 'lib64' if relative.parts[0] == 'lib' else relative.parts[0]
            destination = toolkit / directory / Path(*relative.parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.symlink_to(source)


def materialize_sources(root, env):
    paths = []
    for source in json.loads((REPO / 'runtime/scientific-sources.json').read_text()):
        destination = root / 'sources' / source['name']
        marker = destination / '.auditory-ready'
        if not marker.exists():
            destination.mkdir(parents=True, exist_ok=True)
            run(['git', 'init', '-q', destination], env)
            run(['git', '-C', destination, 'fetch', '--depth', '1', source['url'], source['revision']], env)
            run(['git', '-C', destination, 'checkout', '--detach', 'FETCH_HEAD'], env)
            if 'patch' in source:
                record = json.loads((PATCHES / source['patch']).read_text())
                patch = base64.b64decode(record['content'], validate=True)
                if hashlib.sha256(patch).hexdigest() != record['sha256']:
                    raise RuntimeError('Scientific source patch checksum mismatch')
                run(['git', '-C', destination, 'apply', '--binary', '-'], env, input=patch)
            marker.write_text(source['revision'] + '\n')
        paths.append(str(destination))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, help='Defaults to cache_dir in auditory_cortex/config.yml')
    parser.add_argument('--activate', action='store_true', help='After validation, repoint the repository .venv symlink')
    parser.add_argument('--plan', action='store_true', help='Print storage paths without downloading or installing')
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 8, 5):
        parser.error('Use Python 3.8.5; set AUDITORY_BOOTSTRAP_PYTHON to its executable.')
    if args.cache_dir is None:
        import yaml
        args.cache_dir = Path(yaml.safe_load((REPO / 'auditory_cortex/config.yml').read_text())['cache_dir'])
    cache = args.cache_dir.expanduser().resolve()
    if cache == Path.home() or Path.home() in cache.parents:
        parser.error('Select a cache outside the home directory (scratch is configured for this repository).')
    inputs = [REPO / name for name in ('pyproject.toml', 'uv.lock', 'runtime/cuda12/pyproject.toml',
              'runtime/cuda12/uv.lock', 'runtime/scientific-sources.json', 'scripts/web/setup_runtime.py')]
    inputs.extend(sorted(PATCHES.glob('*.patch.json')))
    fingerprint = hashlib.sha256(str(REPO).encode() + b''.join(path.read_bytes() for path in inputs)).hexdigest()[:16]
    root = cache / 'runtime' / fingerprint
    env = dict(os.environ, UV_CACHE_DIR=str(cache / 'uv'), TMPDIR=str(cache / 'tmp'),
               UV_PYTHON_INSTALL_DIR=str(cache / 'python'), PIP_CACHE_DIR=str(cache / 'pip'),
               XDG_CACHE_HOME=str(cache / 'xdg'), UV_NO_MANAGED_PYTHON='1',
               UV_PYTHON_DOWNLOADS='never')
    if args.plan:
        print(json.dumps({'runtime': str(root), 'python': sys.version.split()[0],
                          'storage': {key: env[key] for key in ('UV_CACHE_DIR', 'TMPDIR',
                                      'UV_PYTHON_INSTALL_DIR', 'PIP_CACHE_DIR', 'XDG_CACHE_HOME')}}, indent=2))
        return
    for directory in (root, cache / 'tmp', cache / 'uv'):
        directory.mkdir(parents=True, exist_ok=True)
    with (root / 'setup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = root / 'python/bin/python'
        if not (root / 'ready.json').exists():
            for project, target in ((REPO, root / 'python'), (REPO / 'runtime/cuda12', root / 'cuda12')):
                install_env = dict(env, UV_PROJECT_ENVIRONMENT=str(target))
                run(['uv', 'sync', '--locked', '--no-install-project', '--project', project,
                     '--python', sys.executable], install_env)
            toolkit = root / 'toolkit'
            assemble_toolkit(root / 'cuda12/lib/python3.8/site-packages/nvidia', toolkit)
            sources = materialize_sources(root, env)
            runtime = dict(UV_CACHE_DIR=env['UV_CACHE_DIR'], TMPDIR=env['TMPDIR'],
                           UV_PYTHON_INSTALL_DIR=env['UV_PYTHON_INSTALL_DIR'],
                           PIP_CACHE_DIR=env['PIP_CACHE_DIR'], XDG_CACHE_HOME=env['XDG_CACHE_HOME'],
                           CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit),
                           LD_LIBRARY_PATH=str(toolkit / 'lib64'),
                           CUPY_CACHE_DIR=str(cache / 'cupy'),
                           PYTHONPATH=os.pathsep.join([str(REPO)] + sources))
            (root / 'python/runtime.env').write_text(''.join('export %s=%s\n' % (key, shlex.quote(value))
                                                           for key, value in runtime.items()))
            # Always exercise the real GPU before marking an installation ready.
            run([python, '-m', 'backend.gpu'], dict(env, **runtime))
            (root / 'ready.json').write_text(json.dumps({'python': '3.8.5', 'fingerprint': fingerprint,
                                                        'bootstrap_python': sys.executable}, indent=2) + '\n')
        else:
            run(['bash', '-c', 'source "$1"; exec "$2" -m backend.gpu', 'runtime-check',
                 root / 'python/runtime.env', python], env)
        if args.activate:
            link = REPO / '.venv'
            if not link.is_symlink():
                parser.error('Activation only replaces an existing .venv symlink; the existing directory was preserved.')
            previous = str(link.resolve())
            temporary = REPO / '.venv.next'
            temporary.symlink_to(root / 'python')
            temporary.replace(link)
            print('Activated %s\nPrevious environment preserved: %s' % (root, previous))
        else:
            print('Validated runtime: %s\nTo activate after stopping services: ./setup-runtime.sh --activate' % root)


if __name__ == '__main__':
    main()
