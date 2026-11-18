import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import Settings, create_app
from backend.auth import hash_password


class ExperimentApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password_hash = hash_password("test-password")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = Settings(
            database_path=Path(self.directory.name) / "sessions.sqlite3",
            operator="operator",
            password_hash=self.password_hash,
            allowed_origins=("https://frontend.example",),
        )
        self.gpus = 3
        self.client = TestClient(create_app(self.settings, lambda: self.gpus))
        self.client.auth = ("operator", "test-password")

    def test_public_discovery_identifies_application_without_revealing_configuration(self):
        anonymous = TestClient(create_app(self.settings, lambda: self.gpus))
        response = anonymous.get("/healthz?identify=true")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "auditory-cortex-control"})

    def test_operator_can_read_supported_settings_and_local_gpu_capacity(self):
        response = self.client.get("/api/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["settings"]["stimulus_families"], ["TIMIT", "mVocs"])
        self.assertEqual(body["settings"]["datasets"], ["ucdavisBAK"])
        self.assertEqual(body["settings"]["models"], ["deepspeech2"])
        self.assertEqual(body["settings"]["layers"], [2])
        self.assertEqual(body["settings"]["bin_width_ms"], 50)
        self.assertEqual(body["settings"]["receptive_field_ms"], 200)
        self.assertEqual(body["gpu"]["available_count"], 3)
        self.assertTrue(body["gpu"]["observed_at"])

    def test_experiment_can_be_created_read_and_recovered_after_restart(self):
        response = self.client.post("/api/v1/experiments", json={
            "stimulus_family": "mVocs", "gpu_count": 2,
        })
        self.assertEqual(response.status_code, 201)
        experiment = response.json()
        self.assertEqual(experiment["configuration"]["stimulus_family"], "mVocs")
        self.assertEqual(experiment["configuration"]["gpu_count"], 2)
        self.assertEqual(experiment["configuration"]["layer_id"], 2)
        restarted = TestClient(create_app(self.settings, lambda: self.gpus))
        response = restarted.get("/api/v1/experiments/" + experiment["id"],
                                 auth=("operator", "test-password"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), experiment)

    def test_detection_failure_does_not_make_api_or_saved_metadata_unavailable(self):
        saved = self.client.post("/api/v1/experiments", json={"gpu_count": 1}).json()

        def broken_detector():
            raise OSError("CUDA library unavailable")

        client = TestClient(create_app(self.settings, broken_detector))
        client.auth = ("operator", "test-password")
        self.assertEqual(client.get("/healthz").status_code, 200)
        self.assertEqual(client.get("/api/v1/experiments/" + saved["id"]).status_code, 200)
        response = client.get("/api/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["gpu"]["status"], "unavailable")

    def test_zero_gpu_backend_can_save_setup_without_claiming_cpu_generation(self):
        self.gpus = 0
        capacity = self.client.get("/api/v1/capabilities").json()["gpu"]
        self.assertEqual(capacity["available_count"], 0)
        self.assertEqual(capacity["status"], "no_gpu")
        response = self.client.post("/api/v1/experiments", json={"gpu_count": 0})
        self.assertEqual(response.status_code, 201)

    def test_one_gpu_backend_accepts_one_gpu(self):
        self.gpus = 1
        self.assertEqual(self.client.get("/api/v1/capabilities").json()["gpu"]["available_count"], 1)
        self.assertEqual(self.client.post("/api/v1/experiments", json={"gpu_count": 1}).status_code, 201)

    def test_stale_gpu_request_is_rejected_without_reducing_requested_count(self):
        self.assertEqual(self.client.get("/api/v1/capabilities").json()["gpu"]["available_count"], 3)
        self.gpus = 1
        response = self.client.post("/api/v1/experiments", json={"gpu_count": 3})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["available_count"], 1)

    def test_invalid_gpu_counts_are_rejected(self):
        for value in [-1, 0, 1.5, True, "2", None]:
            with self.subTest(value=value):
                response = self.client.post("/api/v1/experiments", json={"gpu_count": value})
                self.assertEqual(response.status_code, 422)

    def test_unsupported_scientific_settings_are_rejected(self):
        for field, value in {"stimulus_family": "other", "dataset_name": "ucsf",
                             "model_name": "whisper", "layer_id": 3, "bin_width_ms": 20,
                             "receptive_field_ms": 300, "unknown_field": True}.items():
            with self.subTest(field=field):
                response = self.client.post("/api/v1/experiments", json={"gpu_count": 1, field: value})
                self.assertEqual(response.status_code, 422)

    def test_credentials_required_for_capabilities_creation_and_reading(self):
        identity = self.client.post("/api/v1/experiments", json={"gpu_count": 1}).json()["id"]
        for auth in [None, ("operator", "wrong"), ("wrong", "test-password")]:
            with self.subTest(auth=auth):
                self.assertEqual(self.client.get("/api/v1/capabilities", auth=auth).status_code, 401)
                self.assertEqual(self.client.post("/api/v1/experiments", json={"gpu_count": 1}, auth=auth).status_code, 401)
                self.assertEqual(self.client.get("/api/v1/experiments/" + identity, auth=auth).status_code, 401)

    def test_missing_experiment_is_not_found(self):
        self.assertEqual(self.client.get("/api/v1/experiments/00000000-0000-0000-0000-000000000000").status_code, 404)

    def test_cors_allows_only_configured_frontend(self):
        headers = {"Origin": "https://frontend.example", "Access-Control-Request-Method": "POST",
                   "Access-Control-Request-Headers": "authorization,content-type"}
        response = self.client.options("/api/v1/experiments", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://frontend.example")
        headers["Origin"] = "https://other.example"
        self.assertEqual(self.client.options("/api/v1/experiments", headers=headers).status_code, 400)


if __name__ == "__main__":
    unittest.main()
