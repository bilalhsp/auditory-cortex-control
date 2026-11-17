import json
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.fit_compute import FitOutput
from backend.tests import test_uploads
from backend.worker import run_worker


class UploadedFitApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_uploads.UploadApiTests()
        self.fixture.setUpClass()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.settings = self.fixture.settings
        self.client = TestClient(create_app(self.settings, lambda: 1))
        self.client.auth = ('operator', 'test-password')
        self.experiment = self.client.post('/api/v1/experiments', json={'gpu_count': 1}).json()
        self.prefix = '/api/v1/experiments/' + self.experiment['id']

    def upload(self):
        response = self.fixture.upload(self.experiment['id'], filename='same.mat', client=self.client)
        self.assertEqual(response.status_code, 201)
        self.experiment = response.json()['experiment']
        return self.experiment['recording']

    def submit(self):
        return self.client.post(self.prefix + '/fits', json={
            'request_id': str(uuid4()), 'recording_id': self.experiment['recording']['id'],
            'recording_revision': self.experiment['recording_revision'], 'seed': 42,
        })

    def test_uploaded_recording_fits_through_shared_workflow(self):
        recording = self.upload()
        response = self.submit()
        self.assertEqual(response.status_code, 202, response.text)
        fit = response.json()
        accepted = next((self.fixture.root / 'uploads').rglob('*.mat')).read_bytes()

        def compute(task, progress):
            snapshot = json.loads((task.workspace / 'input.json').read_text())
            self.assertEqual((task.workspace / snapshot['recording_file']).read_bytes(), accepted)
            self.assertEqual(task.fit.recording.source, 'upload')
            progress('fitting')
            (task.workspace / 'model.pkl').write_bytes(b'uploaded-fit-model')
            (task.workspace / 'trf_config.json').write_text('{"session_id":0}')
            return FitOutput({'-2.02': .1, '12.03': .7}, {'executor': 'controlled'})

        run_worker(self.settings, once=True, runner=compute)
        result = self.client.get('/api/v1/fits/' + fit['id']).json()
        self.assertEqual(result['state'], 'succeeded')
        self.assertEqual(result['recording']['id'], recording['id'])
        self.assertEqual(result['best_channel']['channel_id'], '12.03')
        self.assertNotIn(str(self.fixture.root), json.dumps(result))

    def test_same_named_uploads_keep_distinct_queued_fits_and_models(self):
        first_recording = self.upload()
        first = self.submit().json()
        second_recording = self.upload()
        second = self.submit().json()
        self.assertNotEqual(first_recording['id'], second_recording['id'])
        self.assertNotEqual(first['id'], second['id'])

        def compute(task, progress):
            progress('writing')
            (task.workspace / 'model.pkl').write_bytes(task.fit.recording.id.encode())
            (task.workspace / 'trf_config.json').write_text('{"session_id":0}')
            return FitOutput({'-2.02': .3}, {})

        run_worker(self.settings, once=True, runner=compute)
        saved = self.client.get('/api/v1/fits/' + first['id']).json()
        run_worker(self.settings, once=True, runner=compute)
        self.assertEqual(self.client.get('/api/v1/fits/' + first['id']).json(), saved)
        for fit, recording in [(first, first_recording), (second, second_recording)]:
            result = self.client.get('/api/v1/jobs/' + fit['job_id']).json()
            self.assertEqual(result['state'], 'succeeded')
            workspace = self.fixture.root / 'fits' / fit['id']
            self.assertEqual((workspace / 'model.pkl').read_bytes(), recording['id'].encode())
            self.assertEqual(json.loads((workspace / 'trf_config.json').read_text())['session_id'], 0)

    def test_invalid_missing_changed_and_foreign_uploads_cannot_be_fitted(self):
        invalid = self.fixture.upload(self.experiment['id'], data=b'not MAT', client=self.client)
        self.assertEqual(invalid.status_code, 422)
        self.assertIsNone(self.client.get(self.prefix).json()['recording'])
        recording = self.upload()
        other = self.client.post('/api/v1/experiments', json={'gpu_count': 1}).json()
        foreign = self.client.post('/api/v1/experiments/' + other['id'] + '/fits', json={
            'request_id': str(uuid4()), 'recording_id': recording['id'], 'recording_revision': 1,
        })
        self.assertEqual(foreign.status_code, 409)
        path = next((self.fixture.root / 'uploads').rglob('*.mat'))
        path.write_bytes(b'changed after validation')
        self.assertEqual(self.submit().status_code, 409)
        self.assertEqual(list((self.fixture.root / 'fits').iterdir()), [])
        path.unlink()
        self.assertEqual(self.submit().status_code, 404)

    def test_uploaded_model_loading_or_fitting_failure_is_not_success(self):
        for stage in ['loading', 'fitting']:
            self.upload()
            fit = self.submit().json()

            def broken(task, progress):
                progress(stage)
                raise ValueError('private recording content /depot/secret.mat')

            run_worker(self.settings, once=True, runner=broken)
            response = self.client.get('/api/v1/fits/' + fit['id'])
            result = response.json()
            self.assertEqual(result['state'], 'failed')
            self.assertFalse(result['artifacts_ready'])
            self.assertEqual(result['correlations'], [])
            self.assertNotIn('/depot', response.text)
            self.assertNotIn('private recording content', response.text)


if __name__ == '__main__':
    unittest.main()
