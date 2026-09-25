Business Entity Resolution Challenge

Project overview

This project implements an end-to-end baseline pipeline for business entity resolution across three independent sources.

Source 1 is treated as the reference source. The pipeline finds possible matching records in Source 2 and Source 3, scores the candidate pairs, and produces:

output/matching_results.tsv
output/candidate_pairs.tsv

The pipeline is designed so the dataset can be added later without changing the source code.

Expected dataset structure

dataset/
    train/
        train_source1.tsv
        train_source2.tsv
        train_source3.tsv
        train_ground_truth.tsv
    test/
        test_source1.tsv
        test_source2.tsv
        test_source3.tsv

The training files contain Source 1, Source 2, Source 3, and the ground truth matches.

The test files contain Source 1, Source 2, and Source 3 without ground truth.

Dataset files are intentionally not included in this archive. Add the files supplied by the team leader to the corresponding folders.

Requirements

Python 3.10 or newer is recommended.

Install the required packages from the business_entity_resolution directory:

python -m pip install -r requirements.txt

Run the complete pipeline

From inside code/business_entity_resolution:

python src/run_pipeline.py

The pipeline will:

1. Load the training data.
2. Normalize business names and addresses.
3. Generate candidate pairs using character TF-IDF nearest-neighbor blocking and exact normalized-field blocking.
4. Build labelled training pairs from the provided ground truth.
5. Train a logistic regression pair classifier using string similarity and field agreement features.
6. Load the test data.
7. Generate candidates for every Source 1 test entity.
8. Score the candidates.
9. Apply the configurable match threshold.
10. Write matching_results.tsv.
11. Write candidate_pairs.tsv.

The default threshold is intentionally configurable and should be tuned using a validation split before the final submission.

Run with a custom threshold

python src/run_pipeline.py --threshold 0.70

Run with a custom number of candidates per source

python src/run_pipeline.py --top-k 20

Run with custom paths

python src/run_pipeline.py     --data-root dataset     --output-dir output

Validate the generated submission

From code/business_entity_resolution:

python src/validate_submission.py     --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv     --test-dir dataset/test

The validator checks the required submission structure and confirms that final matches are contained in the candidate set.

Important challenge constraints

Only the supplied challenge data is used.

The pipeline does not use external business databases, APIs, geocoding services, web lookups, or external identity information.

The country field is treated as an open-set string field.

Source 1 records are never used as final matches. Final matches can only contain Source 2 and Source 3 entity IDs.

Every Source 1 test entity receives one row in each output file.

Empty match lists are retained for entities for which no candidate passes the matching threshold.

Methodology

Preprocessing

Business names and addresses are converted to lowercase, Unicode-normalized, punctuation-normalized text. Common whitespace variations are removed. The original values remain available for feature calculation.

Candidate generation

Candidate generation combines two approaches.

The first approach uses exact normalized-field blocks. Records sharing normalized names or strong normalized address tokens are considered candidates.

The second approach represents names and addresses with character TF-IDF features and retrieves a limited number of nearest records from Source 2 and Source 3. Character n-grams help tolerate spelling differences, punctuation changes, abbreviations, transliteration differences, and small textual variations.

The union of these candidates forms the candidate set passed to the matching model.

Matching features

The pair classifier uses features derived only from the two records being compared:

name similarity
address similarity
country equality
name containment
address containment
token overlap
character similarity
exact normalized-name agreement
exact normalized-address agreement

Model

A logistic regression classifier is trained using positive pairs from train_ground_truth.tsv and negative pairs sampled from generated training candidates that are not present in the ground truth.

The probability produced by the classifier is used as the final match score.

Outputs

matching_results.tsv contains:

source1_entity_id
matched_entity_ids

candidate_pairs.tsv contains:

source1_entity_id
candidate_entity_ids

The candidate file represents the final candidate set passed to the matching model.

Files

src/run_pipeline.py
Main end-to-end training, candidate generation, matching, and output script.

src/validate_submission.py
Submission validator based on the challenge requirements.

requirements.txt
Python dependencies.

dataset/train/
Place the training TSV files here.

dataset/test/
Place the test TSV files here.

output/
Generated submission files are written here.

Tuning

The default configuration is a starting point only. The team should create a validation split from the training data and tune:

candidate top-k
candidate blocking rules
match threshold
feature weights or model
negative sampling strategy

The challenge uses F_0.5, which gives more weight to precision than recall. Validation should therefore examine false merges and singleton predictions in addition to overall matching performance.

Before final submission, run the validator and inspect both generated TSV files.

The final archive should contain the required output files, runnable code, dependencies, and methodology documentation required by the challenge.
