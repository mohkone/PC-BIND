"""The real preflight report must serialize NumPy-derived coverage counts."""
import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_pcbind_prereqs as prerequisites
from test_data_integrity import valid_sample


class PrerequisiteProvenanceTests(unittest.TestCase):
    def test_successful_gate_writes_complete_json_with_native_integer_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            with (data / "synthetic.pkl").open("wb") as stream:
                pickle.dump([valid_sample()], stream)
            report = root / "prerequisite_provenance.json"
            args = ["check_pcbind_prereqs.py", "--data-dir", str(data),
                    "--require-plm", "--require-partner", "--require-pair",
                    "--require-partner-encoder", "--min-partner-encoder-coverage", "1.0",
                    "--provenance-output", str(report), "synthetic.pkl"]
            with mock.patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prerequisites.main(), 0)
            recorded = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(recorded["status"], "passed")
            self.assertEqual(recorded["dataset_selection"]["resolved_data_dir"], str(data.resolve()))
            self.assertEqual(recorded["coverage_rows"], [["synthetic.pkl", 1, 1, 1, 1, 1, 1, 1, 1]])
            self.assertTrue(all(type(count) is int for count in recorded["coverage_rows"][0][1:]))
            self.assertEqual(recorded["loaded_datasets"]["synthetic.pkl"]["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
