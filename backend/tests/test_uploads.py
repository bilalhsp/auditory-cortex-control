import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import numpy as np
from scipy.io import savemat
from fastapi.testclient import TestClient

from backend.app import Settings, create_app
from backend.auth import hash_password


class UploadApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password_hash = hash_password("test-password")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.settings = Settings(self.root / "sessions.sqlite3", "operator", self.password_hash,
                                 upload_dir=self.root / "uploads", max_upload_bytes=10000)
        (self.root / "datasets/ucdavisBAK/Data").mkdir(parents=True)
        (self.root / "catalog.yml").write_text("sessions_info: []\n")
        self.settings = replace(self.settings, neural_data_dir=self.root / "datasets", recording_metadata_path=self.root / "catalog.yml")
        self.client = self.new_client()

    def new_client(self, settings=None):
        client = TestClient(create_app(settings or self.settings, lambda: 0))
        client.auth = ("operator", "test-password")
        return client

    def create_experiment(self, family="TIMIT"):
        return self.client.post("/api/v1/experiments", json={"gpu_count": 0, "stimulus_family": family}).json()["id"]

    def recording_bytes(self, family="TIMIT", neural=True, onsets=True):
        trial = {"StimulusName": np.array(["C:\\Stimuli\\{}\\{}.wav".format(family, name) for name in ["unique", "repeat", "repeat", "repeat"]], dtype=object),
                 "StimulusType": np.array(["S1T"] * 4, dtype=object)}
        if onsets:
            trial["StimulusTimeOn"] = np.array([0., 1., 2., 3.])
        data = {"TrialStimData": trial}
        if neural:
            data["BAK1"] = {"times": np.array([.1, 1.1, 2.1, 3.1])}
        buffer = io.BytesIO()
        savemat(buffer, data)
        return buffer.getvalue()

    def upload(self, experiment, data=None, filename="session.mat", client=None):
        return (client or self.client).post(
            "/api/v1/experiments/{}/recording-upload".format(experiment),
            content=self.recording_bytes() if data is None else data,
            headers={"Content-Type": "application/octet-stream", "X-Recording-Filename": quote(filename, safe="")},
        )

    def test_valid_upload_attaches_stable_recording_and_survives_restart(self):
        experiment = self.create_experiment()
        payload = self.recording_bytes()
        response = self.upload(experiment, payload)
        self.assertEqual(response.status_code, 201)
        result = response.json()
        self.assertEqual(result["validation_status"], "valid")
        recording = result["experiment"]["recording"]
        UUID(recording["id"])
        self.assertEqual(recording["source"], "upload")
        self.assertEqual(recording["display_name"], "session.mat")
        self.assertEqual(result["experiment"]["recording_revision"], 1)
        recovered = self.new_client().get("/api/v1/experiments/" + experiment).json()
        self.assertEqual(recovered, result["experiment"])
        self.assertNotIn(str(self.root), response.text)
        files = list((self.root / "uploads").rglob("*.mat"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_bytes(), payload)
        self.assertNotEqual(files[0].name, "session.mat")

    def test_same_names_are_isolated_and_replacement_retains_previous_artifact(self):
        experiment = self.create_experiment()
        first = self.upload(experiment, filename="../../same.mat").json()["experiment"]
        second = self.upload(experiment, filename="C:\\uploads\\same.mat").json()["experiment"]
        self.assertNotEqual(first["recording"]["id"], second["recording"]["id"])
        self.assertEqual(first["recording"]["display_name"], "same.mat")
        self.assertEqual(second["recording"]["display_name"], "same.mat")
        self.assertEqual(second["recording_revision"], 2)
        files = list((self.root / "uploads").rglob("*.mat"))
        self.assertEqual(len(files), 2)
        self.assertNotEqual(files[0].parent, files[1].parent)
        other = self.create_experiment()
        self.assertIsNone(self.client.get("/api/v1/experiments/" + other).json()["recording"])
        body = {"recording_id": first["recording"]["id"]}
        self.assertEqual(self.client.post("/api/v1/experiments/" + other + "/recording-selection", json=body).status_code, 404)
        restored = self.client.post("/api/v1/experiments/" + experiment + "/recording-selection", json=body).json()
        self.assertEqual(restored["recording"]["id"], first["recording"]["id"])
        self.assertEqual(restored["recording_revision"], 3)

    def test_invalid_uploads_preserve_selection_and_remove_staged_files(self):
        experiment = self.create_experiment()
        saved = self.upload(experiment).json()["experiment"]
        invalid = [(b"", "empty.mat", 422), (b"not a mat", "bad.mat", 422),
                   (b"pickle payload", "model.pkl", 415),
                   (self.recording_bytes(neural=False), "missing-spikes.mat", 422),
                   (self.recording_bytes(onsets=False), "missing-onsets.mat", 422),
                   (self.recording_bytes("mvox"), "wrong-family.mat", 409)]
        for data, name, status in invalid:
            with self.subTest(name=name):
                response = self.upload(experiment, data, name)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["detail"]["validation_status"], "invalid")
                self.assertNotIn(str(self.root), response.text)
                self.assertEqual(self.client.get("/api/v1/experiments/" + experiment).json(), saved)
        self.assertEqual(len(list((self.root / "uploads").rglob("*.mat"))), 1)
        self.assertEqual(list((self.root / "uploads").rglob("*.part")), [])

    def test_size_limit_is_enforced_for_declared_and_chunked_uploads(self):
        experiment = self.create_experiment()
        client = self.new_client(replace(self.settings, max_upload_bytes=32))
        response = self.upload(experiment, client=client)
        self.assertEqual(response.status_code, 413)
        response = client.post("/api/v1/experiments/{}/recording-upload".format(experiment),
                               content=iter([b"x" * 20, b"y" * 20]),
                               headers={"Content-Type": "application/octet-stream", "X-Recording-Filename": "large.mat"})
        self.assertEqual(response.status_code, 413)
        self.assertIsNone(client.get("/api/v1/experiments/" + experiment).json()["recording"])
        self.assertEqual(list((self.root / "uploads").rglob("*.part")), [])

    def test_upload_requires_authentication_and_a_real_experiment(self):
        experiment = self.create_experiment()
        anonymous = TestClient(create_app(self.settings, lambda: 0))
        self.assertEqual(self.upload(experiment, client=anonymous).status_code, 401)
        self.assertEqual(self.upload("00000000-0000-0000-0000-000000000000").status_code, 404)
        self.assertEqual(list((self.root / "uploads").rglob("*.mat")), [])

    def test_mvocs_recordings_can_be_uploaded_for_mvocs_experiment(self):
        response = self.upload(self.create_experiment("mVocs"), self.recording_bytes("mvox"))
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["experiment"]["recording"]["stimulus_families"], ["mVocs"])

    def test_capabilities_and_cors_expose_configured_upload_contract(self):
        self.assertEqual(self.client.get("/api/v1/capabilities").json()["upload"],
                         {"max_bytes": 10000, "formats": [".mat"]})
        response = self.client.options("/api/v1/experiments/00000000-0000-0000-0000-000000000000/recording-upload", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,x-recording-filename",
        })
        self.assertEqual(response.status_code, 200)

    def test_storage_failure_has_actionable_error_without_selecting_an_upload(self):
        (self.root / "uploads").write_text("not a directory")
        experiment = self.create_experiment()
        response = self.upload(experiment)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["validation_status"], "error")
        self.assertNotIn(str(self.root), response.text)
        self.assertIsNone(self.client.get("/api/v1/experiments/" + experiment).json()["recording"])

    def test_nonfinite_and_matrix_trial_fields_are_rejected(self):
        experiment = self.create_experiment()
        for onsets in [np.array([0., np.nan, 2., 3.]), np.array([[0., 1.], [2., 3.]])]:
            with self.subTest(onsets=onsets):
                buffer = io.BytesIO()
                savemat(buffer, {"TrialStimData": {
                    "StimulusName": np.array(["C:\\TIMIT\\a.wav"] * 4, dtype=object),
                    "StimulusType": np.array(["S1T"] * 4, dtype=object),
                    "StimulusTimeOn": onsets,
                }, "BAK1": {"times": np.array([.1, 1.1, 2.1, 3.1])}})
                response = self.upload(experiment, buffer.getvalue())
                self.assertEqual(response.status_code, 422)
        self.assertIsNone(self.client.get("/api/v1/experiments/" + experiment).json()["recording"])


if __name__ == "__main__":
    unittest.main()
