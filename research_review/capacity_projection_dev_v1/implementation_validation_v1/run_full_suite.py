"""Record the complete repository regression suite against the new source freeze."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from projection_protocol_v1 import DEFAULT_MANIFEST, verify_manifest
from research_provenance import file_record
from run_multiseed_dev_v1 import write_new


if __name__ == "__main__":
    manifest, protocol = verify_manifest(DEFAULT_MANIFEST)
    directory = Path(__file__).resolve().parent
    test_sources = [file_record(path) for path in sorted((ROOT / "tests").glob("test*.py"))]
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test*.py")
    with (directory / "full_tests.log").open("x", encoding="utf-8", newline="\n") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    for source in manifest["source_files"] + test_sources:
        if file_record(source["path"]) != source:
            raise ValueError("Source changed during the full regression suite")
    record = {"status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "execution_manifest": file_record(DEFAULT_MANIFEST), "source_files": manifest["source_files"],
        "test_source_files": test_sources, "test_log": file_record(directory / "full_tests.log"),
        "runner": file_record(__file__), "training_started": False}
    write_new(directory / "full_tests.json", record)
    print(json.dumps({key: record[key] for key in ("status", "tests_run", "failures", "errors", "skipped")}))
    sys.exit(0 if result.wasSuccessful() else 1)
