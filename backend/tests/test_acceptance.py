"""Release matrix at the agreed HTTP seam, with controlled external compute."""
import io
import unittest
import zipfile
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.tests import test_fits, test_generations, test_uploads
from backend.worker import run_worker


class ReleaseAcceptanceTests(unittest.TestCase):
    def test_both_families_and_sources_retain_scoped_results_after_restart(self):
        fixture = test_fits.FitApiTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        client, settings = fixture.client, fixture.settings
        saved = []
        for family in ['TIMIT', 'mVocs']:
            for source in ['existing', 'upload']:
                with self.subTest(family=family, source=source):
                    response = client.post('/api/v1/experiments', json={
                        'stimulus_family': family, 'gpu_count': 1})
                    self.assertEqual(response.status_code, 201)
                    experiment = response.json()
                    prefix = '/api/v1/experiments/' + experiment['id']
                    if source == 'existing':
                        recording = client.get(prefix + '/recordings').json()['recordings'][0]
                        response = client.post(prefix + '/recording-selection', json={'recording_id': recording['id']})
                    else:
                        payload = test_uploads.UploadApiTests().recording_bytes(family='mvox' if family == 'mVocs' else family)
                        response = client.post(prefix + '/recording-upload', content=payload, headers={
                            'Content-Type': 'application/octet-stream', 'X-Recording-Filename': 'same.mat'})
                    self.assertIn(response.status_code, [200, 201], response.text)
                    experiment = response.json()['experiment'] if source == 'upload' else response.json()
                    response = client.post(prefix + '/fits', json={
                        'request_id': str(uuid4()), 'recording_id': experiment['recording']['id'],
                        'recording_revision': experiment['recording_revision'], 'seed': 42})
                    self.assertEqual(response.status_code, 202, response.text)
                    fit = response.json()
                    run_worker(settings, once=True, runner=test_fits.FitApiTests.output)
                    ready = client.get('/api/v1/fits/' + fit['id']).json()
                    self.assertEqual(ready['state'], 'succeeded')
                    self.assertEqual(ready['configuration']['stimulus_family'], family)
                    self.assertEqual(ready['recording']['source'], source)
                    self.assertEqual(ready['best_channel']['channel_id'], '-100.0')
                    generations = []
                    for objective in ['stretch', 'one-hot', 'stretch']:
                        response = client.post('/api/v1/fits/' + fit['id'] + '/generations', json={
                            'request_id': str(uuid4()), 'channel_id': '12.03', 'objective': objective,
                            'duration_seconds': 1, 'clip_count': 1})
                        self.assertEqual(response.status_code, 202, response.text)
                        generation = response.json()
                        run_worker(settings, once=True, generation_runner=test_generations.GenerationApiTests.compute)
                        url = '/api/v1/generations/' + generation['id'] + '/download'
                        response = client.get(url)
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(zipfile.ZipFile(io.BytesIO(response.content)).namelist(), ['clip-001.wav'])
                        generations.append(generation['id'])
                        saved.append((url, response.content))
                    self.assertEqual(len(set(generations)), 3)
                    self.assertEqual(client.post('/api/v1/sessions/' + experiment['session_id'] + '/close').json()['status'], 'closed')
        restarted = TestClient(create_app(settings, lambda: 0))
        restarted.auth = ('operator', 'test-password')
        for url, expected in saved:
            response = restarted.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, expected)
            self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertGreaterEqual(len(restarted.get('/api/v1/sessions').json()), 4)
