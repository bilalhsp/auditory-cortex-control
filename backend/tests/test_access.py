"""Access checks at the accepted authenticated HTTP seam."""
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.tests import test_api


class AccessApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_api.ExperimentApiTests.setUpClass()

    def setUp(self):
        self.fixture = test_api.ExperimentApiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.client = self.fixture.client

    def test_protected_metadata_and_errors_cannot_be_cached(self):
        for auth, status in [(None, 401), (("operator", "test-password"), 200)]:
            response = self.client.get("/api/v1/capabilities", auth=auth)
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_every_api_route_rejects_missing_and_wrong_credentials(self):
        app = create_app(self.fixture.settings, lambda: 3)
        client = TestClient(app)
        for route in app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/api/"):
                continue
            path = path.replace("{identity}", str(uuid4()))
            for method in getattr(route, "methods", set()):
                for auth in [None, ("operator", "wrong"), ("wrong", "test-password")]:
                    with self.subTest(method=method, path=path, auth=auth):
                        response = client.request(method, path, auth=auth)
                        self.assertEqual(response.status_code, 401, response.text)
                        self.assertEqual(response.headers.get("www-authenticate"), "Basic")

    def test_health_is_public_but_files_and_documentation_are_not_served(self):
        client = TestClient(create_app(self.fixture.settings, lambda: 3))
        self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
        for path in ["/docs", "/openapi.json", "/static/model.pkl", "/artifacts/clip-001.wav",
                     "/sessions.sqlite3", "/api/v1/generations/../../model.pkl"]:
            self.assertEqual(client.get(path).status_code, 404)

    def test_saved_download_stays_private_after_close_and_api_restart(self):
        from backend.tests.test_generations import GenerationApiTests
        from backend.worker import run_worker
        fixture = GenerationApiTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        generation = fixture.client.post(fixture.url, json=fixture.body(
            duration_seconds=1, clip_count=1)).json()
        run_worker(fixture.settings, once=True, generation_runner=fixture.compute)
        url = '/api/v1/generations/' + generation['id'] + '/download'
        saved = fixture.client.get(url)
        self.assertEqual(saved.status_code, 200)
        session = fixture.client.get('/api/v1/sessions').json()[0]
        self.assertEqual(fixture.client.post('/api/v1/sessions/' + session['id'] + '/close').status_code, 200)
        restarted = TestClient(create_app(fixture.settings, lambda: 0))
        for auth in [None, ('operator', 'wrong'), ('wrong', 'test-password')]:
            self.assertEqual(restarted.get(url, auth=auth).status_code, 401)
        response = restarted.get(url, auth=('operator', 'test-password'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, saved.content)
        self.assertEqual(response.headers['cache-control'], 'no-store')
