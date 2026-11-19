import unittest
import json
import threading
import multiprocessing
import time
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.tests import test_recordings
from backend.worker import run_worker
from backend.fit_compute import FitOutput


class FitApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_recordings.RecordingApiTests()
        self.fixture.setUpClass()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.settings = self.fixture.settings
        self.client = TestClient(create_app(self.settings, lambda: 1))
        self.client.auth = ("operator", "test-password")
        experiment = self.client.post("/api/v1/experiments", json={"gpu_count": 1}).json()
        self.prefix = "/api/v1/experiments/" + experiment["id"]
        self.recordings = self.client.get(self.prefix + "/recordings").json()["recordings"]
        self.experiment = self.client.post(self.prefix + "/recording-selection", json={"recording_id": self.recordings[0]["id"]}).json()

    def request(self):
        return {"request_id": str(uuid4()), "recording_id": self.experiment["recording"]["id"],
                "recording_revision": self.experiment["recording_revision"], "seed": 42}

    def test_guided_recording_change_preserves_accepted_fit_in_original_experiment(self):
        accepted = self.client.post(self.prefix + "/fits", json=self.request()).json()
        response = self.client.post(
            self.prefix + "/recording-selection?preserve_fits=true",
            json={"recording_id": self.recordings[1]["id"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        replacement = response.json()
        self.assertNotEqual(replacement["id"], self.experiment["id"])
        self.assertEqual(replacement["session_id"], self.experiment["session_id"])
        original = self.client.get(self.prefix).json()
        self.assertEqual(original["recording"], self.experiment["recording"])
        self.assertEqual(self.client.get(self.prefix + "/work").json()["fits"][0]["id"], accepted["id"])
        self.assertEqual(self.client.get("/api/v1/experiments/" + replacement["id"] + "/work").json()["fits"], [])

    def test_submission_returns_persisted_immutable_job_without_running_compute(self):
        body = self.request()
        response = self.client.post(self.prefix + "/fits", json=body)
        self.assertEqual(response.status_code, 202)
        fit = response.json()
        self.assertEqual(fit["state"], "queued")
        self.assertNotEqual(fit["id"], fit["job_id"])
        self.assertFalse(fit["artifacts_ready"])
        self.client.post(self.prefix + "/recording-selection", json={"recording_id": self.recordings[1]["id"]})
        self.assertEqual(self.client.get("/api/v1/fits/" + fit["id"]).json(), fit)
        self.assertEqual(self.client.get("/api/v1/jobs/" + fit["job_id"]).json(), fit)
        repeated = self.client.post(self.prefix + "/fits", json=body)
        self.assertEqual(repeated.json()["id"], fit["id"])
        self.assertNotIn(str(self.fixture.root), response.text)

    @staticmethod
    def output(task, progress):
        progress("writing")
        (task.workspace / "model.pkl").write_bytes(b"controlled-compute-artifact")
        (task.workspace / "trf_config.json").write_text('{"model_name":"deepspeech2"}')
        return FitOutput({"-2.02": .1, "12.03": .7, "-100.0": .9, "7.01": float("nan")}, {"executor": "controlled"})

    def test_worker_exposes_running_and_sorted_results_from_pinned_input(self):
        original = (self.fixture.data_dir / "monkey_2026-01-14a_recording.mat").read_bytes()
        body = self.request()
        fit = self.client.post(self.prefix + "/fits", json=body).json()
        (self.fixture.data_dir / "monkey_2026-01-14a_recording.mat").write_bytes(b"changed after acceptance")
        started, finish = threading.Event(), threading.Event()

        def compute(task, progress):
            snapshot = json.loads((task.workspace / "input.json").read_text())
            self.assertEqual((task.workspace / snapshot["recording_file"]).read_bytes(), original)
            progress("fitting")
            started.set()
            if not finish.wait(8):
                raise RuntimeError("Controlled computation timed out")
            return self.output(task, progress)

        worker = threading.Thread(target=run_worker, args=(self.settings,), kwargs={"once": True, "runner": compute})
        worker.start()
        try:
            self.assertTrue(started.wait(5))
            active = self.client.get("/api/v1/jobs/" + fit["job_id"]).json()
            self.assertEqual((active["state"], active["stage"]), ("running", "fitting"))
            self.assertFalse(active["artifacts_ready"])
            self.assertEqual(self.client.get("/api/v1/capabilities").json()["gpu"]["available_count"], 0)
            self.assertEqual(self.client.post(self.prefix + "/fits", json=body).json()["id"], fit["id"])
        finally:
            finish.set()
            worker.join(10)
        result = self.client.get("/api/v1/fits/" + fit["id"]).json()
        self.assertEqual(result["state"], "succeeded")
        self.assertTrue(result["artifacts_ready"])
        self.assertEqual([row["channel_id"] for row in result["correlations"]], ["-100.0", "12.03", "-2.02", "7.01"])
        self.assertIsNone(result["correlations"][-1]["correlation"])
        self.assertEqual(result["best_channel"]["channel_id"], "-100.0")

    def test_failure_never_exposes_private_errors_or_ready_artifacts(self):
        fit = self.client.post(self.prefix + "/fits", json=self.request()).json()

        def broken(task, progress):
            raise RuntimeError("private recording /depot/secret.mat")

        run_worker(self.settings, once=True, runner=broken)
        response = self.client.get("/api/v1/fits/" + fit["id"])
        self.assertEqual(response.json()["state"], "failed")
        self.assertFalse(response.json()["artifacts_ready"])
        self.assertNotIn("/depot", response.text)
        self.assertEqual(response.json()["correlations"], [])

    def test_missing_artifacts_cannot_be_reported_as_success(self):
        fit = self.client.post(self.prefix + "/fits", json=self.request()).json()
        run_worker(self.settings, once=True, runner=lambda task, progress: FitOutput({"1": .9}, {}))
        result = self.client.get("/api/v1/fits/" + fit["id"]).json()
        self.assertEqual(result["state"], "failed")
        self.assertFalse(result["artifacts_ready"])

    def test_distinct_fits_retain_prior_results_and_do_not_overwrite_artifacts(self):
        first = self.client.post(self.prefix + "/fits", json=self.request()).json()
        run_worker(self.settings, once=True, runner=self.output)
        saved = self.client.get("/api/v1/fits/" + first["id"]).json()
        second = self.client.post(self.prefix + "/fits", json=self.request()).json()
        run_worker(self.settings, once=True, runner=self.output)
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(self.client.get("/api/v1/fits/" + first["id"]).json(), saved)
        for fit in [first, second]:
            workspace = self.settings.database_path.parent / "fits" / fit["id"]
            self.assertEqual((workspace / "model.pkl").read_bytes(), b"controlled-compute-artifact")
            provenance = json.loads((workspace / "provenance.json").read_text())
            self.assertEqual(provenance["fit_id"], fit["id"])
            self.assertEqual(provenance["recording_id"], fit["recording"]["id"])

    def test_no_gpu_and_reused_request_with_different_context_are_rejected(self):
        body = self.request()
        no_gpu = TestClient(create_app(self.settings, lambda: 0))
        no_gpu.auth = ("operator", "test-password")
        self.assertEqual(no_gpu.post(self.prefix + "/fits", json=body).status_code, 409)
        fit = self.client.post(self.prefix + "/fits", json=body).json()
        body["seed"] = 43
        self.assertEqual(self.client.post(self.prefix + "/fits", json=body).status_code, 409)
        anonymous = TestClient(create_app(self.settings, lambda: 1))
        self.assertEqual(anonymous.get("/api/v1/fits/" + fit["id"]).status_code, 401)
        self.assertEqual(anonymous.get("/api/v1/jobs/" + fit["job_id"]).status_code, 401)

    def test_stale_context_missing_recording_and_authentication(self):
        body = self.request()
        anonymous = TestClient(create_app(self.settings, lambda: 1))
        self.assertEqual(anonymous.post(self.prefix + "/fits", json=body).status_code, 401)
        self.client.post(self.prefix + "/recording-selection", json={"recording_id": self.recordings[1]["id"]})
        self.assertEqual(self.client.post(self.prefix + "/fits", json=body).status_code, 409)
        self.client.post(self.prefix + "/recording-selection", json={"recording_id": self.recordings[0]["id"]})
        current = self.client.get(self.prefix).json()
        body["recording_revision"] = current["recording_revision"]
        (self.fixture.data_dir / "monkey_2026-01-14a_recording.mat").unlink()
        self.assertEqual(self.client.post(self.prefix + "/fits", json=body).status_code, 404)

    def test_worker_restart_interrupts_old_work_and_requires_new_attempt(self):
        fit = self.client.post(self.prefix + "/fits", json=self.request()).json()
        context = multiprocessing.get_context("fork")
        started = context.Event()

        def wait_forever(task, progress):
            started.set()
            time.sleep(20)
            return self.output(task, progress)

        process = context.Process(target=run_worker, args=(self.settings,), kwargs={"once": True, "runner": wait_forever})
        process.start()
        try:
            self.assertTrue(started.wait(5))
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                run_worker(self.settings, once=True, runner=self.output)
        finally:
            process.terminate()
            process.join(5)
        run_worker(self.settings, once=True, runner=self.output)
        result = self.client.get("/api/v1/fits/" + fit["id"]).json()
        self.assertEqual(result["state"], "interrupted")
        self.assertFalse(result["artifacts_ready"])
        retry = self.client.post(self.prefix + "/fits", json=self.request()).json()
        self.assertNotEqual(retry["id"], fit["id"])


if __name__ == "__main__":
    unittest.main()
