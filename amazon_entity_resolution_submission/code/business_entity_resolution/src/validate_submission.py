from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def split_ids(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()

    issues = []

    matching = read_tsv(Path(args.matching))
    candidate = read_tsv(Path(args.candidate))
    test_dir = Path(args.test_dir)

    required_matching = {"source1_entity_id", "matched_entity_ids"}
    required_candidate = {"source1_entity_id", "candidate_entity_ids"}

    if set(matching.columns) != required_matching:
        issues.append("matching_results.tsv must contain exactly the required columns.")

    if set(candidate.columns) != required_candidate:
        issues.append("candidate_pairs.tsv must contain exactly the required columns.")

    test_s1 = read_tsv(test_dir / "test_source1.tsv")
    test_s2 = read_tsv(test_dir / "test_source2.tsv")
    test_s3 = read_tsv(test_dir / "test_source3.tsv")

    s1_ids = set(test_s1["entity_id"])
    target_ids = set(test_s2["entity_id"]) | set(test_s3["entity_id"])

    if len(matching) != len(s1_ids):
        issues.append("matching_results.tsv must contain exactly one row for every Source 1 entity.")

    if len(candidate) != len(s1_ids):
        issues.append("candidate_pairs.tsv must contain exactly one row for every Source 1 entity.")

    if matching["source1_entity_id"].duplicated().any():
        issues.append("matching_results.tsv contains duplicate Source 1 entity IDs.")

    if candidate["source1_entity_id"].duplicated().any():
        issues.append("candidate_pairs.tsv contains duplicate Source 1 entity IDs.")

    if set(matching["source1_entity_id"]) != s1_ids:
        issues.append("matching_results.tsv Source 1 IDs do not exactly match the test Source 1 IDs.")

    if set(candidate["source1_entity_id"]) != s1_ids:
        issues.append("candidate_pairs.tsv Source 1 IDs do not exactly match the test Source 1 IDs.")

    candidate_map = {
        row["source1_entity_id"]: set(split_ids(row["candidate_entity_ids"]))
        for _, row in candidate.iterrows()
    }

    for _, row in matching.iterrows():
        source_id = row["source1_entity_id"]
        ids = split_ids(row["matched_entity_ids"])

        if len(ids) != len(set(ids)):
            issues.append(f"{source_id} contains duplicate matched IDs.")

        invalid_prefix = [
            x for x in ids
            if not (x.startswith("S2-") or x.startswith("S3-"))
        ]
        if invalid_prefix:
            issues.append(f"{source_id} contains non-S2/S3 matched IDs.")

        missing = [x for x in ids if x not in target_ids]
        if missing:
            issues.append(f"{source_id} contains IDs that do not exist in the test target sources.")

        outside_candidates = [
            x for x in ids
            if x not in candidate_map.get(source_id, set())
        ]
        if outside_candidates:
            issues.append(
                f"{source_id} has matches that are not present in candidate_pairs.tsv."
            )

    for _, row in candidate.iterrows():
        source_id = row["source1_entity_id"]
        ids = split_ids(row["candidate_entity_ids"])

        if len(ids) != len(set(ids)):
            issues.append(f"{source_id} contains duplicate candidate IDs.")

        invalid_prefix = [
            x for x in ids
            if not (x.startswith("S2-") or x.startswith("S3-"))
        ]
        if invalid_prefix:
            issues.append(f"{source_id} contains non-S2/S3 candidate IDs.")

        missing = [x for x in ids if x not in target_ids]
        if missing:
            issues.append(f"{source_id} contains candidate IDs that do not exist in the test sources.")

    if issues:
        print("VALIDATION FAILED")
        for i, issue in enumerate(sorted(set(issues)), 1):
            print(f"{i}. {issue}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
