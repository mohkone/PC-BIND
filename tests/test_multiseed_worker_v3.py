"""Operational failures must halt this queue without starting research runs."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multiseed_infrastructure_v3 import require_fresh_destinations
from multiseed_protocol_v1 import clean_environment
from run_multiseed_dev_v3 import execute_serial, run_queue


JOBS = [{"training_seed": seed, "arm": arm, "partner_transport": arm == "ot"}
        for seed in (2102, 2103, 2104) for arm in ("ot", "noot")]


class WorkerTests(unittest.TestCase):
    def run_engine(self, exit_index=None, validation_index=None, preflight_index=None):
        events = []
        def preflight(index, job):
            events.append(("preflight", index))
            if index == preflight_index:
                raise ValueError("changed immutable fold/source/data/runtime")
        def launch(index, job, live, state):
            events.append(("launch", index))
            return 7 if index == exit_index else 0
        def validate(index, job):
            events.append(("validate", index))
            if index == validation_index:
                raise ValueError("partial OOF or corrupt checkpoint")
            return {"status": "passed"}
        def aggregate():
            events.append(("aggregate", 6))
            return {"status": "complete"}
        return execute_serial(copy.deepcopy(JOBS), preflight, launch, validate, aggregate), events

    def test_exact_order_and_validation_before_next_launch(self):
        state, events = self.run_engine()
        expected = [(step, i) for i in range(6) for step in ("preflight", "launch", "validate")]
        self.assertEqual(events, expected + [("aggregate", 6)])
        self.assertEqual(state["status"], "complete")
        self.assertTrue(all(job["status"] == "complete" for job in state["jobs"]))

    def test_every_nonzero_exit_stops_later_arms_and_analysis(self):
        for index in range(6):
            with self.subTest(index=index):
                state, events = self.run_engine(exit_index=index)
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["jobs"][index]["status"], "failed")
                self.assertNotIn(("validate", index), events)
                self.assertNotIn(("aggregate", 6), events)
                self.assertTrue(all(j["status"] == "queued" for j in state["jobs"][index+1:]))

    def test_zero_exit_with_bad_artifacts_stops_later_arms(self):
        for index in range(6):
            with self.subTest(index=index):
                state, events = self.run_engine(validation_index=index)
                self.assertEqual(state["jobs"][index]["status"], "failed_validation")
                self.assertNotIn(("aggregate", 6), events)
                self.assertNotIn(("launch", index+1), events)

    def test_changed_preflight_never_launches_arm(self):
        state, events = self.run_engine(preflight_index=2)
        self.assertEqual(state["status"], "failed")
        self.assertNotIn(("launch", 2), events)
        self.assertNotIn(("aggregate", 6), events)

    def test_clear_inherited_case_insensitive_ppi_settings(self):
        self.assertEqual(clean_environment({"PPI_SEED": "2102"},
            {"PPI_SEED": "1", "ppi_hack": "1", "PATH": "safe"}),
            {"PATH": "safe", "PPI_SEED": "2102"})

    def test_output_and_aggregate_collision_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"jobs": [{"output_dir": str(root / "run")}], "aggregate_output_dir": str(root / "aggregate")}
            require_fresh_destinations(manifest)
            (root / "run").mkdir()
            with self.assertRaises(ValueError):
                require_fresh_destinations(manifest)
            (root / "run").rmdir()
            (root / "aggregate").mkdir()
            with self.assertRaises(ValueError):
                require_fresh_destinations(manifest)

    def test_retained_lock_refuses_concurrent_worker_before_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            (root / "queue.lock").write_text("existing worker", encoding="utf-8")
            manifest = {"process_dir": str(root), "jobs": [], "aggregate_output_dir": str(root / "aggregate")}
            with patch("run_multiseed_dev_v3.verify_manifest", return_value=(manifest, {}, {})), \
                    patch("run_multiseed_dev_v3.run_logged") as child:
                with self.assertRaises(FileExistsError):
                    run_queue(manifest_path)
                child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
