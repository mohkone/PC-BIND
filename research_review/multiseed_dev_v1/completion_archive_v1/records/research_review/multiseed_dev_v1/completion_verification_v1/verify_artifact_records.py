"""Check saved final artifact/metadata records against all completed arm records."""
import json
from pathlib import Path
import hashlib
from datetime import datetime, timezone

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def main():
    verification_path = EVIDENCE / "verification.json"
    verification = read(verification_path)
    assert verification["status"] == "passed" and verification["validate_only"]["exit_code"] == 0
    queue = read(ROOT / "outputs_multiseed_dev_v3_process/queue_state.json")
    final = read(queue["comparison"]["artifacts"]["comparison"]["path"])
    checked = []
    for original, final_arm in zip(queue["jobs"], final["artifacts"]):
        completed_validation = read(original["validation"]["path"])
        assert completed_validation["artifacts"] == final_arm["artifacts"]
        actual_run = read(final_arm["artifacts"]["run_provenance"]["path"])
        assert final_arm["run"] == actual_run
        assert actual_run["runtime"] == final["runtime"]
        assert actual_run["cv_splits"] == final["cv_splits"]
        assert actual_run["dataset_selection"] == final["dataset_selection"]
        checked.append({"training_seed": original["training_seed"], "arm": original["arm"],
                        "validation": record(original["validation"]["path"]),
                        "saved_final_artifact_dictionary_equals_completed_validation": True,
                        "saved_final_run_dictionary_equals_current_run_json": True,
                        "runtime_fold_and_dataset_selection_equal_final_binding": True})
    assert len(checked) == 6
    result = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "verification_source": record(__file__), "full_validation_evidence": record(verification_path),
              "arms": checked, "bootstrap_recomputed": False, "training": False,
              "output_originals_modified": False}
    output = EVIDENCE / "artifact_record_consistency.json"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "evidence": record(output)}, indent=2))


if __name__ == "__main__":
    main()
