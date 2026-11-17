import tempfile
import unittest
from pathlib import Path
from uuid import UUID

import numpy as np
import scipy.io
import yaml
from fastapi.testclient import TestClient

from backend.app import Settings, create_app
from backend.auth import hash_password


class RecordingApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password_hash = hash_password("test-password")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data_dir = self.root / "datasets/ucdavisBAK/Data"
        self.data_dir.mkdir(parents=True)
        self.metadata_path = self.root / "sessions.yml"
        self.metadata_path.write_text(yaml.safe_dump({"sessions_info": [
            {"id": 0, "session": "monkey_2026-01-14a_recording.mat", "num_repeats": 3},
            {"id": 1, "session": "monkey_2026-02-09b_recording.mat", "num_repeats": 3},
        ]}))
        self.write_recording("monkey_2026-01-14a_recording.mat", ["TIMIT", "mvox"])
        self.write_recording("monkey_2026-02-09b_recording.mat", ["TIMIT"])
        self.settings = Settings(
            self.root / "sessions.sqlite3", "operator", self.password_hash,
            neural_data_dir=self.root / "datasets", recording_metadata_path=self.metadata_path,
        )
        self.client = self.new_client()

    def new_client(self):
        client = TestClient(create_app(self.settings, lambda: 0))
        client.auth = ("operator", "test-password")
        return client

    def write_recording(self, name, families):
        scipy.io.savemat(str(self.data_dir / name), {"TrialStimData": {
            "StimulusName": np.array(["C:\\Stimuli\\{}\\sound.wav".format(family) for family in families], dtype=object),
            "StimulusType": np.array(["S1T"] * len(families), dtype=object),
        }})

    def create_experiment(self, family="TIMIT"):
        response = self.client.post("/api/v1/experiments", json={"stimulus_family": family, "gpu_count": 0})
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def test_catalog_filters_by_actual_stimulus_family_and_has_stable_opaque_ids(self):
        experiment = self.create_experiment("mVocs")
        url = "/api/v1/experiments/{}/recordings".format(experiment)
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        records = response.json()["recordings"]
        self.assertEqual(len(records), 1)
        self.assertIn("2026-01-14a", records[0]["display_name"])
        UUID(records[0]["id"])
        self.assertEqual(records[0]["stimulus_families"], ["TIMIT", "mVocs"])
        self.assertNotIn(str(self.root), response.text)
        self.assertEqual(self.new_client().get(url).json(), response.json())

    def test_selection_persists_and_is_isolated_from_other_experiments(self):
        first = self.create_experiment()
        second = self.create_experiment()
        prefix = "/api/v1/experiments/" + first
        records = self.client.get(prefix + "/recordings").json()["recordings"]
        response = self.client.post(prefix + "/recording-selection", json={"recording_id": records[0]["id"]})
        self.assertEqual(response.status_code, 200)
        selected = response.json()
        self.assertEqual(selected["recording"], records[0])
        self.assertEqual(selected["recording_revision"], 1)
        self.assertEqual(self.new_client().get(prefix).json(), selected)
        self.assertIsNone(self.client.get("/api/v1/experiments/" + second).json()["recording"])
        repeated = self.client.post(prefix + "/recording-selection", json={"recording_id": records[0]["id"]})
        self.assertEqual(repeated.json()["recording_revision"], 1)
        changed = self.client.post(prefix + "/recording-selection", json={"recording_id": records[1]["id"]})
        self.assertEqual(changed.json()["recording_revision"], 2)
        self.assertEqual(changed.json()["recording"], records[1])
        self.assertTrue((self.data_dir / "monkey_2026-01-14a_recording.mat").exists())

    def test_corrupt_and_missing_files_do_not_hide_healthy_recordings(self):
        (self.data_dir / "monkey_2026-02-09b_recording.mat").write_bytes(b"")
        metadata = yaml.safe_load(self.metadata_path.read_text())
        metadata["sessions_info"].append({"id": 2, "session": "missing.mat", "num_repeats": 3})
        self.metadata_path.write_text(yaml.safe_dump(metadata))
        experiment = self.create_experiment()
        response = self.client.get("/api/v1/experiments/{}/recordings".format(experiment))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["recordings"]), 1)
        self.assertEqual(response.json()["unavailable_count"], 2)

    def test_incompatible_or_removed_recording_cannot_replace_selection(self):
        timit = self.create_experiment()
        mvocs = self.create_experiment("mVocs")
        records = self.client.get("/api/v1/experiments/{}/recordings".format(timit)).json()["recordings"]
        prefix = "/api/v1/experiments/" + mvocs
        self.assertEqual(self.client.post(prefix + "/recording-selection", json={"recording_id": records[1]["id"]}).status_code, 409)
        selected = self.client.post(prefix + "/recording-selection", json={"recording_id": records[0]["id"]}).json()
        (self.data_dir / "monkey_2026-01-14a_recording.mat").unlink()
        response = self.client.post(prefix + "/recording-selection", json={"recording_id": records[0]["id"]})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get(prefix).json(), selected)

    def test_empty_and_unavailable_catalog_are_distinct(self):
        prefix = "/api/v1/experiments/" + self.create_experiment()
        for path in self.data_dir.glob("*.mat"):
            path.unlink()
        self.assertEqual(self.client.get(prefix + "/recordings").json()["recordings"], [])
        self.metadata_path.unlink()
        response = self.client.get(prefix + "/recordings")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(str(self.root), response.text)
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get(prefix).status_code, 200)

    def test_recording_endpoints_require_authentication_and_reject_paths(self):
        prefix = "/api/v1/experiments/" + self.create_experiment()
        record = self.client.get(prefix + "/recordings").json()["recordings"][0]
        anonymous = TestClient(create_app(self.settings, lambda: 0))
        self.assertEqual(anonymous.get(prefix + "/recordings").status_code, 401)
        self.assertEqual(anonymous.post(prefix + "/recording-selection", json={"recording_id": record["id"]}).status_code, 401)
        for body in [{"recording_id": "/etc/passwd"}, {"recording_id": record["id"], "path": "/etc/passwd"}]:
            self.assertEqual(self.client.post(prefix + "/recording-selection", json=body).status_code, 422)
        self.assertEqual(self.client.post(prefix + "/recording-selection", json={"recording_id": "00000000-0000-0000-0000-000000000000"}).status_code, 404)

    def test_symlink_outside_recording_storage_is_not_exposed(self):
        original = self.data_dir / "monkey_2026-02-09b_recording.mat"
        outside = self.root / "private.mat"
        original.rename(outside)
        original.symlink_to(outside)
        prefix = "/api/v1/experiments/" + self.create_experiment()
        response = self.client.get(prefix + "/recordings")
        self.assertEqual(len(response.json()["recordings"]), 1)
        self.assertEqual(response.json()["unavailable_count"], 1)

    def test_target_only_stimuli_are_not_offered_as_training_recordings(self):
        scipy.io.savemat(str(self.data_dir / "monkey_2026-02-09b_recording.mat"), {
            "TrialStimData": {"StimulusName": "C:\\Stimuli\\TIMIT\\target.wav", "StimulusType": "S2"}
        })
        prefix = "/api/v1/experiments/" + self.create_experiment()
        records = self.client.get(prefix + "/recordings").json()["recordings"]
        self.assertEqual(len(records), 1)

    def test_missing_experiment_does_not_access_catalog(self):
        self.metadata_path.unlink()
        prefix = "/api/v1/experiments/00000000-0000-0000-0000-000000000000"
        self.assertEqual(self.client.get(prefix + "/recordings").status_code, 404)
        self.assertEqual(self.client.post(prefix + "/recording-selection", json={"recording_id": "00000000-0000-0000-0000-000000000000"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
