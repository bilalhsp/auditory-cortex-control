import io
import unittest
import zipfile
from uuid import uuid4
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.tests import test_generations
from backend.worker import run_worker


class BatchApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_generations.GenerationApiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.settings = self.fixture.settings
        self.capacity = 3
        self.client = TestClient(create_app(self.settings, lambda: self.capacity))
        self.client.auth = ('operator', 'test-password')
        self.url = '/api/v1/fits/' + self.fixture.fit['id'] + '/batches'

    def body(self, count=2):
        return {'request_id': str(uuid4()), 'gpu_count': count, 'duration_seconds': 1, 'clip_count': 1,
                'targets': [{'channel_id': '12.03', 'objective': 'stretch'}, {'channel_id': '-2.02', 'objective': 'one-hot'}]}

    def test_batch_reserves_requested_capacity_and_deduplicates_submission(self):
        body = self.body()
        response = self.client.post(self.url, json=body)
        self.assertEqual(response.status_code, 202, response.text)
        batch = response.json()
        self.assertEqual(batch['gpu_count'], 2)
        self.assertEqual(len(batch['targets']), 2)
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 1)
        self.assertEqual(self.client.post(self.url, json=body).json()['id'], batch['id'])
        self.assertEqual(self.client.post(self.url, json=self.body()).status_code, 409)
        self.assertEqual(self.client.get('/api/v1/jobs/' + batch['job_id']).json()['id'], batch['id'])

    def test_bounded_waves_partial_results_and_reservation_release(self):
        import threading
        body = self.body()
        body['targets'] += [{'channel_id': '12.03', 'objective': 'one-hot'}, {'channel_id': '-2.02', 'objective': 'stretch'}]
        batch = self.client.post(self.url, json=body).json()
        active, seen, overlaps, directories = set(), set(), [], set()
        lock, started, finish = threading.Lock(), threading.Event(), threading.Event()
        peak = [0]
        def compute(task, progress):
            progress('generating')
            with lock:
                if task.device_index in active: overlaps.append(task.device_index)
                active.add(task.device_index);seen.add(task.device_index);directories.add(task.workspace)
                peak[0] = max(peak[0], len(active))
                if len(active) == 2: started.set()
            try:
                if not finish.wait(10): raise RuntimeError('Controlled compute timed out')
                self.fixture.compute(task, progress)
                return 7 if (task.generation.channel_id, task.generation.objective) == ('-2.02', 'one-hot') else 0
            finally:
                with lock: active.remove(task.device_index)
        worker = threading.Thread(target=run_worker, args=(self.settings,), kwargs={'once': True, 'generation_runner': compute, 'gpu_count': lambda: 3})
        worker.start()
        try:
            self.assertTrue(started.wait(5))
            current = self.client.get('/api/v1/batches/' + batch['id']).json()
            self.assertEqual(current['state'], 'running')
            self.assertEqual(sum(target['state'] == 'running' for target in current['targets']), 2)
            self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 1)
            self.assertEqual(self.client.post(self.url, json=self.body()).status_code, 409)
        finally:
            finish.set();worker.join(15)
        result = self.client.get('/api/v1/batches/' + batch['id']).json()
        self.assertEqual(result['state'], 'partial')
        self.assertEqual((peak[0], seen, overlaps, len(directories)), (2, {0, 1}, [], 4))
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 3)
        response = self.client.get('/api/v1/batches/' + batch['id'] + '/download')
        self.assertEqual(response.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        self.assertEqual(len(archive.namelist()), 4)
        failed = next(target for target in result['targets'] if target['state'] == 'failed')
        self.assertFalse(any(name.startswith(failed['id'] + '/') for name in archive.namelist()))
        self.assertEqual(self.client.get('/api/v1/generations/' + failed['id'] + '/download').status_code, 409)
        self.assertIn('manifest.json', archive.namelist())
        workspace = self.settings.database_path.parent / 'batches' / batch['id']
        self.assertEqual(list(workspace.glob('download-*.zip')), [])

    def test_fewer_targets_reserve_requested_gpus_and_report_idle_devices(self):
        batch = self.client.post(self.url, json=self.body(3)).json()
        self.assertEqual((batch['used_gpu_count'], batch['unused_gpu_count']), (2, 1))
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 0)
        devices = []
        def compute(task, progress):
            devices.append(task.device_index)
            return self.fixture.compute(task, progress)
        run_worker(self.settings, once=True, generation_runner=compute, gpu_count=lambda: 3)
        self.assertEqual(self.client.get('/api/v1/batches/' + batch['id']).json()['state'], 'succeeded')
        self.assertTrue(set(devices) <= {0, 1})
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 3)

    def test_changed_capacity_and_visibility_fail_without_launch_or_silent_reduction(self):
        import os
        from unittest.mock import patch
        launched = []
        def compute(task, progress): launched.append(task);return 0
        batch = self.client.post(self.url, json=self.body(3)).json()
        self.capacity = 2
        run_worker(self.settings, once=True, generation_runner=compute, gpu_count=lambda: self.capacity)
        self.assertEqual(self.client.get('/api/v1/batches/' + batch['id']).json()['state'], 'failed')
        self.assertEqual(launched, [])
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 2)
        self.capacity = 3
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-A,GPU-B,GPU-C'}):
            batch = self.client.post(self.url, json=self.body()).json()
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-B,GPU-A,GPU-C'}):
                run_worker(self.settings, once=True, generation_runner=compute, gpu_count=lambda: 3)
        self.assertEqual(self.client.get('/api/v1/batches/' + batch['id']).json()['state'], 'failed')
        self.assertEqual(launched, [])

    def test_invalid_targets_authentication_and_competing_atomic_requests(self):
        from concurrent.futures import ThreadPoolExecutor
        anonymous = TestClient(create_app(self.settings, lambda: 3))
        self.assertEqual(anonymous.post(self.url, json=self.body()).status_code, 401)
        for targets in [[], [{'channel_id': 'unknown', 'objective': 'stretch'}], self.body()['targets'] * 2]:
            body = self.body();body['targets'] = targets
            self.assertEqual(self.client.post(self.url, json=body).status_code, 422)
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(lambda _: self.client.post(self.url, json=self.body()), range(2)))
        self.assertEqual(sorted(response.status_code for response in responses), [202, 409])
        batch = next(response.json() for response in responses if response.status_code == 202)
        self.assertEqual(anonymous.get('/api/v1/batches/' + batch['id']).status_code, 401)
        self.assertEqual(anonymous.get('/api/v1/batches/' + batch['id'] + '/download').status_code, 401)
        self.assertNotIn(str(self.settings.database_path.parent), str(batch))

    def test_real_subprocess_launcher_uses_only_reserved_runtime_devices(self):
        import os
        from pathlib import Path
        from unittest.mock import patch
        import json
        fixture = self.settings.database_path.parent / 'controlled-compute'
        package = fixture / 'backend';package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        (package / 'generation_compute.py').write_text("""
import json,os,sys,time,wave
from pathlib import Path
root=Path(sys.argv[1]);job=json.loads((root/'input.json').read_text())['generation']
(root/'device.txt').write_text(os.environ['CUDA_VISIBLE_DEVICES'])
time.sleep(.2)
for i in range(job['clip_count']):
    with wave.open(str(root/('clip-%03d.wav'%(i+1))), 'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000);audio.writeframes(b'\\x01\\x00'*15392)
""")
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-A,GPU-B,GPU-C'}):
            batch = self.client.post(self.url, json=self.body()).json()
            previous = Path.cwd()
            try:
                os.chdir(fixture)
                run_worker(self.settings, once=True, gpu_count=lambda: 3)
            finally: os.chdir(previous)
        result = self.client.get('/api/v1/batches/' + batch['id']).json()
        self.assertEqual(result['state'], 'succeeded')
        workspace = self.settings.database_path.parent / 'batches' / batch['id']
        self.assertEqual({(workspace / target['id'] / 'device.txt').read_text() for target in result['targets']}, {'GPU-A', 'GPU-B'})

    def test_worker_restart_preserves_completed_targets_and_releases_reservation(self):
        import multiprocessing
        import time
        batch = self.client.post(self.url, json=self.body(1)).json()
        context = multiprocessing.get_context('fork')
        started = context.Event()
        def compute(task, progress):
            if task.generation.objective == 'one-hot':
                started.set();time.sleep(20)
            return self.fixture.compute(task, progress)
        process = context.Process(target=run_worker, args=(self.settings,), kwargs={'once': True, 'generation_runner': compute, 'gpu_count': lambda: 3})
        process.start()
        try:
            self.assertTrue(started.wait(8))
            current = self.client.get('/api/v1/batches/' + batch['id']).json()
            self.assertEqual([target['state'] for target in current['targets']], ['succeeded', 'running'])
        finally:
            process.terminate();process.join(5)
        run_worker(self.settings, once=True, generation_runner=self.fixture.compute, gpu_count=lambda: 3)
        result = self.client.get('/api/v1/batches/' + batch['id']).json()
        self.assertEqual(result['state'], 'interrupted')
        self.assertEqual([target['state'] for target in result['targets']], ['succeeded', 'interrupted'])
        self.assertTrue(result['artifacts_ready'])
        response = self.client.get('/api/v1/batches/' + batch['id'] + '/download')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(zipfile.ZipFile(io.BytesIO(response.content)).namelist()), 2)
        self.assertEqual(self.client.get('/api/v1/capabilities').json()['gpu']['available_count'], 3)


if __name__ == '__main__': unittest.main()
