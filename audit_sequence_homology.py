import argparse
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_TESTS = (
    "Test60.pkl",
    "Test287.pkl",
    "Test70.pkl",
    "TestB25.pkl",
    "TestUB25.pkl",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit target-sequence homology between Train335 and test sets."
    )
    parser.add_argument("--data-dir", default=os.path.join("data", "geo"))
    parser.add_argument("--train", default="Train335.pkl")
    parser.add_argument("--tests", nargs="*", default=list(DEFAULT_TESTS))
    parser.add_argument("--output-json", default="sequence_homology_audit.json")
    parser.add_argument("--output-md", default="sequence_homology_audit.md")
    return parser.parse_args()


def load_records(path):
    with open(path, "rb") as handle:
        samples = pickle.load(handle)
    records = []
    for index, sample in enumerate(samples):
        sequence = str(sample.get("residue_sequence", "")).strip().upper()
        if not sequence:
            continue
        records.append(
            {
                "index": index,
                "complex_code": str(sample.get("complex_code", "")).strip().upper(),
                "chain": str(sample.get("pdb_chain", "")).strip(),
                "sequence": sequence,
            }
        )
    return records


def write_fasta(path, records, prefix):
    with open(path, "w", encoding="ascii") as handle:
        for record in records:
            handle.write(f">{prefix}_{record['index']}\n{record['sequence']}\n")


def run_checked(command):
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def best_hits(blast_output):
    hits = {}
    for line in blast_output.splitlines():
        fields = line.split("\t")
        if len(fields) != 12:
            continue
        query_id, subject_id = fields[:2]
        row = {
            "query_id": query_id,
            "subject_id": subject_id,
            "identity": float(fields[2]),
            "alignment_length": int(fields[3]),
            "query_length": int(fields[4]),
            "subject_length": int(fields[5]),
            "evalue": float(fields[6]),
            "bitscore": float(fields[7]),
            "query_start": int(fields[8]),
            "query_end": int(fields[9]),
            "subject_start": int(fields[10]),
            "subject_end": int(fields[11]),
        }
        query_span = abs(row["query_end"] - row["query_start"]) + 1
        subject_span = abs(row["subject_end"] - row["subject_start"]) + 1
        row["query_coverage"] = 100.0 * query_span / row["query_length"]
        row["subject_coverage"] = 100.0 * subject_span / row["subject_length"]
        row["min_coverage"] = min(row["query_coverage"], row["subject_coverage"])
        current = hits.get(query_id)
        if current is None or row["bitscore"] > current["bitscore"]:
            hits[query_id] = row
    return hits


def summarize(records, hits):
    rows = []
    for record in records:
        query_id = f"query_{record['index']}"
        hit = hits.get(query_id)
        rows.append({**record, "best_hit": hit})

    def count(identity, coverage=80.0):
        return sum(
            row["best_hit"] is not None
            and row["best_hit"]["identity"] >= identity
            and row["best_hit"]["min_coverage"] >= coverage
            for row in rows
        )

    return {
        "queries": len(rows),
        "with_blast_hit": sum(row["best_hit"] is not None for row in rows),
        "identity_ge_30_min_coverage_ge_80": count(30.0),
        "identity_ge_40_min_coverage_ge_80": count(40.0),
        "identity_ge_50_min_coverage_ge_80": count(50.0),
        "identity_ge_90_min_coverage_ge_80": count(90.0),
        "best_hits": rows,
    }


def write_markdown(path, report):
    lines = [
        "# Train-Test Sequence Homology Audit",
        "",
        "BLASTP was run from each target sequence against Train335. Counts require aligned coordinate spans covering at least 80% of both query and subject.",
        "",
        "| Test set | Records | Sequences | >=30% identity | >=40% | >=50% | >=90% | Exact duplicates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["datasets"].items():
        lines.append(
            f"| {name.removesuffix('.pkl')} | {summary['records']} | {summary['queries']} | "
            f"{summary['identity_ge_30_min_coverage_ge_80']} | "
            f"{summary['identity_ge_40_min_coverage_ge_80']} | "
            f"{summary['identity_ge_50_min_coverage_ge_80']} | "
            f"{summary['identity_ge_90_min_coverage_ge_80']} | "
            f"{summary['exact_sequence_duplicates']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit measures target-sequence similarity only. It does not establish structural independence, family-level independence, or partner-pair independence. Test70 must not be described as sequence-disjoint when exact duplicates are present.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    makeblastdb = shutil.which("makeblastdb")
    blastp = shutil.which("blastp")
    if not makeblastdb or not blastp:
        raise RuntimeError("NCBI makeblastdb and blastp must be available on PATH")

    train_records = load_records(data_dir / args.train)
    train_sequences = {record["sequence"] for record in train_records}
    report = {
        "purpose": "Audit target-sequence homology between Train335 and evaluation sets.",
        "method": "BLASTP best single-HSP hit; identity thresholds require >=80% aligned-span coverage of both query and subject.",
        "train_records": len(train_records),
        "train_unique_sequences": len(train_sequences),
        "datasets": {},
    }

    with tempfile.TemporaryDirectory(prefix="pcbind_homology_") as temp_dir:
        temp_dir = Path(temp_dir)
        train_fasta = temp_dir / "train.fasta"
        db_prefix = temp_dir / "train_db"
        write_fasta(train_fasta, train_records, "train")
        run_checked(
            [
                makeblastdb,
                "-in",
                str(train_fasta),
                "-dbtype",
                "prot",
                "-out",
                str(db_prefix),
            ]
        )

        for filename in args.tests:
            records = load_records(data_dir / filename)
            query_fasta = temp_dir / f"{Path(filename).stem}.fasta"
            write_fasta(query_fasta, records, "query")
            output = run_checked(
                [
                    blastp,
                    "-query",
                    str(query_fasta),
                    "-db",
                    str(db_prefix),
                    "-outfmt",
                    "6 qseqid sseqid pident length qlen slen evalue bitscore qstart qend sstart send",
                    "-max_target_seqs",
                    "1",
                    "-max_hsps",
                    "1",
                    "-evalue",
                    "1e-3",
                ]
            )
            summary = summarize(records, best_hits(output))
            with open(data_dir / filename, "rb") as handle:
                summary["records"] = len(pickle.load(handle))
            summary["exact_sequence_duplicates"] = sum(
                record["sequence"] in train_sequences for record in records
            )
            report["datasets"][filename] = summary
            print(
                f"{filename}: queries={summary['queries']} "
                f">=30%/80%={summary['identity_ge_30_min_coverage_ge_80']} "
                f"exact={summary['exact_sequence_duplicates']}"
            )

    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.output_md, report)
    print(f"Wrote {Path(args.output_json).resolve()}")
    print(f"Wrote {Path(args.output_md).resolve()}")


if __name__ == "__main__":
    main()
