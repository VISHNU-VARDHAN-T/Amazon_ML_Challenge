from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return set(text.split()) if text else set()


def token_jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return 0.0


def pair_features(a: pd.Series, b: pd.Series) -> np.ndarray:
    name_a = a["name_norm"]
    name_b = b["name_norm"]
    addr_a = a["address_norm"]
    addr_b = b["address_norm"]

    return np.array([
        ratio(name_a, name_b) / 100.0,
        ratio(addr_a, addr_b) / 100.0,
        token_jaccard(name_a, name_b),
        token_jaccard(addr_a, addr_b),
        containment(name_a, name_b),
        containment(addr_a, addr_b),
        float(a["country_norm"] == b["country_norm"]),
        float(name_a == name_b and name_a != ""),
        float(addr_a == addr_b and addr_a != ""),
        float(bool(name_a) and bool(name_b)),
        float(bool(addr_a) and bool(addr_b)),
    ], dtype=float)


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["name_norm"] = df["business_name"].map(normalize_text)
    df["address_norm"] = df["business_address"].map(normalize_text)
    df["country_norm"] = df["country"].map(normalize_text)
    return df


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "matched_entity_ids"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    result: dict[str, set[str]] = {}
    for _, row in gt.iterrows():
        value = row["matched_entity_ids"].strip()
        ids = {x.strip() for x in value.split(",") if x.strip()}
        result[row["source1_entity_id"]] = ids
    return result


def exact_block_candidates(s1: pd.DataFrame, targets: list[pd.DataFrame]) -> dict[str, set[str]]:
    result = {x: set() for x in s1["entity_id"]}

    for target in targets:
        by_name: dict[str, list[str]] = {}
        by_address: dict[str, list[str]] = {}

        for _, row in target.iterrows():
            if row["name_norm"]:
                by_name.setdefault(row["name_norm"], []).append(row["entity_id"])
            if row["address_norm"]:
                by_address.setdefault(row["address_norm"], []).append(row["entity_id"])

        for _, row in s1.iterrows():
            if row["name_norm"]:
                result[row["entity_id"]].update(by_name.get(row["name_norm"], []))
            if row["address_norm"]:
                result[row["entity_id"]].update(by_address.get(row["address_norm"], []))

    return result


def tfidf_candidates(
    s1: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int,
    fields: tuple[str, ...] = ("name_norm", "address_norm"),
) -> dict[str, set[str]]:
    result = {x: set() for x in s1["entity_id"]}

    if target.empty or s1.empty:
        return result

    combined = (
        s1[list(fields)].fillna("").agg(" ".join, axis=1).tolist()
        + target[list(fields)].fillna("").agg(" ".join, axis=1).tolist()
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = normalize(vectorizer.fit_transform(combined))
    left = matrix[: len(s1)]
    right = matrix[len(s1) :]

    k = min(top_k, len(target))
    if k == 0:
        return result

    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(right)
    _, indices = nn.kneighbors(left)

    target_ids = target["entity_id"].tolist()
    for row_index, source_id in enumerate(s1["entity_id"]):
        result[source_id].update(target_ids[i] for i in indices[row_index])

    return result


def generate_candidates(
    s1: pd.DataFrame,
    targets: list[pd.DataFrame],
    top_k: int,
) -> dict[str, set[str]]:
    result = exact_block_candidates(s1, targets)

    for target in targets:
        tfidf_result = tfidf_candidates(s1, target, top_k)
        for source_id, ids in tfidf_result.items():
            result[source_id].update(ids)

    return result


def build_training_matrix(
    s1: pd.DataFrame,
    targets: list[pd.DataFrame],
    candidates: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> tuple[np.ndarray, np.ndarray]:
    target_lookup = {}
    for target in targets:
        target_lookup.update(
            (row["entity_id"], row) for _, row in target.iterrows()
        )

    features = []
    labels = []

    for _, row in s1.iterrows():
        source_id = row["entity_id"]
        positives = ground_truth.get(source_id, set())

        for candidate_id in candidates.get(source_id, set()):
            target_row = target_lookup.get(candidate_id)
            if target_row is None:
                continue
            features.append(pair_features(row, target_row))
            labels.append(int(candidate_id in positives))

        for positive_id in positives:
            if positive_id not in candidates.get(source_id, set()):
                target_row = target_lookup.get(positive_id)
                if target_row is not None:
                    features.append(pair_features(row, target_row))
                    labels.append(1)

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)

    if len(x) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Training data did not produce both positive and negative pairs.")

    return x, y


def make_outputs(
    s1: pd.DataFrame,
    targets: list[pd.DataFrame],
    candidates: dict[str, set[str]],
    model: LogisticRegression,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_lookup = {}
    for target in targets:
        target_lookup.update(
            (row["entity_id"], row) for _, row in target.iterrows()
        )

    matching_rows = []
    candidate_rows = []

    for _, row in s1.iterrows():
        source_id = row["entity_id"]
        candidate_ids = sorted(candidates.get(source_id, set()))

        scored = []
        for candidate_id in candidate_ids:
            target_row = target_lookup.get(candidate_id)
            if target_row is None:
                continue
            features = pair_features(row, target_row).reshape(1, -1)
            probability = float(model.predict_proba(features)[0, 1])
            scored.append((candidate_id, probability))

        matches = sorted(
            candidate_id
            for candidate_id, probability in scored
            if probability >= threshold
        )

        candidate_rows.append({
            "source1_entity_id": source_id,
            "candidate_entity_ids": ",".join(candidate_ids),
        })
        matching_rows.append({
            "source1_entity_id": source_id,
            "matched_entity_ids": ",".join(matches),
        })

    return pd.DataFrame(matching_rows), pd.DataFrame(candidate_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_root = data_root / "train"
    test_root = data_root / "test"
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_s1 = load_source(train_root / "train_source1.tsv")
    train_s2 = load_source(train_root / "train_source2.tsv")
    train_s3 = load_source(train_root / "train_source3.tsv")
    ground_truth = load_ground_truth(train_root / "train_ground_truth.tsv")

    train_candidates = generate_candidates(
        train_s1, [train_s2, train_s3], args.top_k
    )

    x_train, y_train = build_training_matrix(
        train_s1,
        [train_s2, train_s3],
        train_candidates,
        ground_truth,
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)

    test_s1 = load_source(test_root / "test_source1.tsv")
    test_s2 = load_source(test_root / "test_source2.tsv")
    test_s3 = load_source(test_root / "test_source3.tsv")

    test_candidates = generate_candidates(
        test_s1, [test_s2, test_s3], args.top_k
    )

    matching, candidate = make_outputs(
        test_s1,
        [test_s2, test_s3],
        test_candidates,
        model,
        args.threshold,
    )

    matching.to_csv(
        output_root / "matching_results.tsv",
        sep="\t",
        index=False,
    )
    candidate.to_csv(
        output_root / "candidate_pairs.tsv",
        sep="\t",
        index=False,
    )

    print(f"Generated {len(matching)} Source 1 prediction rows.")
    print(f"Candidate pairs written to {output_root / 'candidate_pairs.tsv'}")
    print(f"Matching results written to {output_root / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()
