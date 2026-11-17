import io
import json
import unittest
import zipfile
from uuid import uuid4

from backend.tests import test_fits
from backend.worker import run_worker


class GenerationApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_fits.FitApiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.client, self.settings = self.fixture.client, self.fixture.settings
        self.fit = self.client.post(self.fixture.prefix + '/fits', json=self.fixture.request()).json()
        run_worker(self.settings, once=True, runner=self.fixture.output)
        self.url = '/api/v1/fits/' + self.fit['id'] + '/generations'

    def body(self, **values):
        return dict({'request_id': str(uuid4()), 'channel_id': '12.03', 'objective': 'stretch'}, **values)

    def test_submit_generation_pins_fit_defaults_and_is_idempotent(self):
        body = self.body()
        response = self.client.post(self.url, json=body)
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual(job['fit_id'], self.fit['id'])
        self.assertEqual(job['duration_seconds'], 2)
        self.assertEqual(job['clip_count'], 4)
        self.assertEqual(job['state'], 'queued')
        self.assertEqual(self.client.post(self.url, json=body).json()['id'], job['id'])
        self.assertEqual(self.client.get('/api/v1/jobs/' + job['job_id']).json()['id'], job['id'])
        self.assertEqual(self.client.get('/api/v1/generations/' + job['id'] + '/download').status_code, 409)

    @staticmethod
    def compute(task, progress):
        import wave
        progress('generating')
        frames = {1: 15392, 2: 35872, 3: 51232, 4: 66592}[task.generation.duration_seconds]
        for index in range(task.generation.clip_count):
            with wave.open(str(task.workspace / ('clip-%03d.wav' % (index + 1))), 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b'\x01\x00' * frames)
        return 0

    def test_success_download_contains_only_validated_clips_and_preserves_prior_generation(self):
        first = self.client.post(self.url, json=self.body()).json()
        run_worker(self.settings, once=True, generation_runner=self.compute)
        result = self.client.get('/api/v1/generations/' + first['id']).json()
        self.assertEqual(result['state'], 'succeeded')
        self.assertEqual(result['exit_code'], 0)
        download_url = '/api/v1/generations/' + first['id'] + '/download'
        download = self.client.get(download_url)
        self.assertEqual(download.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(download.content))
        self.assertEqual(archive.namelist(), ['clip-001.wav', 'clip-002.wav', 'clip-003.wav', 'clip-004.wav'])
        second = self.client.post(self.url, json=self.body(objective='one-hot', duration_seconds=1, clip_count=1)).json()
        run_worker(self.settings, once=True, generation_runner=self.compute)
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual(self.client.get(download_url).content, download.content)
        workspace = self.settings.database_path.parent / 'generations' / first['id']
        (workspace / 'clip-001.wav').write_bytes(b'broken')
        self.assertEqual(self.client.get(download_url).status_code, 409)

    def test_invalid_stale_missing_and_unauthenticated_requests_do_not_launch(self):
        from fastapi.testclient import TestClient
        from backend.app import create_app
        for body in [self.body(channel_id=''), self.body(channel_id='unknown'), self.body(channel_id='7.01'),
                     self.body(clip_count=3), self.body(duration_seconds=0), self.body(duration_seconds=5),
                     self.body(clip_count=True), self.body(objective='other'), self.body(device='cuda:99')]:
            self.assertEqual(self.client.post(self.url, json=body).status_code, 422)
        anonymous = TestClient(create_app(self.settings, lambda: 1))
        self.assertEqual(anonymous.post(self.url, json=self.body()).status_code, 401)
        no_gpu = TestClient(create_app(self.settings, lambda: 0))
        no_gpu.auth = ('operator', 'test-password')
        self.assertEqual(no_gpu.post(self.url, json=self.body()).status_code, 409)
        self.client.post(self.fixture.prefix + '/recording-selection', json={'recording_id': self.fixture.recordings[1]['id']})
        self.assertEqual(self.client.post(self.url, json=self.body()).status_code, 409)

    def test_nonzero_exit_missing_and_invalid_output_never_publish_download(self):
        def unsuccessful(task, progress):
            self.compute(task, progress)
            return 7
        def no_files(task, progress):
            return 0
        def bad_format(task, progress):
            for index in range(task.generation.clip_count):
                (task.workspace / ('clip-%03d.wav' % (index + 1))).write_bytes(b'not a WAV')
            return 0
        for runner in [unsuccessful, no_files, bad_format]:
            job = self.client.post(self.url, json=self.body()).json()
            run_worker(self.settings, once=True, generation_runner=runner)
            result = self.client.get('/api/v1/generations/' + job['id']).json()
            self.assertEqual(result['state'], 'failed')
            self.assertFalse(result['artifacts_ready'])
            self.assertEqual(result['clips'], [])
            self.assertEqual(self.client.get('/api/v1/generations/' + job['id'] + '/download').status_code, 409)

    def test_lost_response_reuses_identity_but_changed_settings_are_rejected(self):
        body = self.body()
        job = self.client.post(self.url, json=body).json()
        body['objective'] = 'one-hot'
        self.assertEqual(self.client.post(self.url, json=body).status_code, 409)
        workspace = self.settings.database_path.parent / 'fits' / self.fit['id']
        (workspace / 'model.pkl').unlink()
        self.assertEqual(self.client.post(self.url, json=self.body()).status_code, 409)
        run_worker(self.settings, once=True, generation_runner=self.compute)
        self.assertEqual(self.client.get('/api/v1/generations/' + job['id']).json()['state'], 'failed')

    def test_preserves_requested_gpus_while_only_one_target_runs(self):
        import threading
        from fastapi.testclient import TestClient
        from backend.app import create_app
        client = TestClient(create_app(self.settings, lambda: 3))
        client.auth = ('operator', 'test-password')
        experiment = client.post('/api/v1/experiments', json={'gpu_count': 3}).json()
        prefix = '/api/v1/experiments/' + experiment['id']
        selected = client.post(prefix + '/recording-selection', json={'recording_id': self.fixture.recordings[0]['id']}).json()
        fit = client.post(prefix + '/fits', json={'request_id': str(uuid4()), 'recording_id': selected['recording']['id'], 'recording_revision': 1}).json()
        run_worker(self.settings, once=True, runner=self.fixture.output)
        job = client.post('/api/v1/fits/' + fit['id'] + '/generations', json=self.body(clip_count=1)).json()
        self.assertEqual(job['configuration']['gpu_count'], 3)
        self.assertEqual((job['used_gpu_count'], job['unused_gpu_count']), (1, 2))
        started, finish = threading.Event(), threading.Event()
        def compute(task, progress):
            progress('generating')
            started.set()
            if not finish.wait(10): raise RuntimeError('Controlled compute timed out')
            return self.compute(task, progress)
        thread = threading.Thread(target=run_worker, args=(self.settings,), kwargs={'once': True, 'generation_runner': compute})
        thread.start()
        try:
            self.assertTrue(started.wait(5))
            self.assertEqual(client.get('/api/v1/capabilities').json()['gpu']['available_count'], 2)
            restarted = TestClient(create_app(self.settings, lambda: 3))
            restarted.auth = ('operator', 'test-password')
            self.assertEqual(restarted.get('/api/v1/generations/' + job['id']).json()['state'], 'running')
            anonymous = TestClient(create_app(self.settings, lambda: 3))
            self.assertEqual(anonymous.get('/api/v1/generations/' + job['id']).status_code, 401)
            self.assertEqual(anonymous.get('/api/v1/generations/' + job['id'] + '/download').status_code, 401)
        finally:
            finish.set()
            thread.join(10)
        self.assertEqual(client.get('/api/v1/generations/' + job['id']).json()['state'], 'succeeded')


if __name__ == '__main__':
    unittest.main()
