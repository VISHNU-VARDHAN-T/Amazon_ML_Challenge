# Methodology

## Objective

The pipeline resolves business entities across Source 1, Source 2, and Source 3 using only the supplied challenge data.

## Preprocessing

Business names and addresses are normalized using Unicode normalization, lowercase conversion, replacement of ampersands with the word and, punctuation removal, and whitespace normalization.

Country values are normalized for comparison but are not restricted to a fixed list.

## Candidate Generation

Candidate generation is performed before final matching.

Exact normalized business names and exact normalized addresses are used as blocking signals.

A character TF-IDF representation is also built from normalized business names and addresses. Nearest-neighbor retrieval produces a configurable number of candidate records for every Source 1 record from Source 2 and Source 3.

The union of these candidates is passed to the matching model.

## Matching Features

The matching model receives:

- Character-level business-name similarity
- Character-level address similarity
- Name token Jaccard similarity
- Address token Jaccard similarity
- Name containment
- Address containment
- Country equality
- Exact normalized-name agreement
- Exact normalized-address agreement
- Presence indicators for name and address values

## Model

A logistic regression classifier is trained using positive pairs from the supplied training ground truth and negative pairs from generated training candidates that are not ground-truth matches.

The predicted probability is used as the match score.

## Output

The final matching output contains one row for every Source 1 test entity.

The candidate output contains the final candidate set passed to the matching model.

Final matches are always selected from the candidate set.

## Tuning

The team should create a validation split from the training data and tune candidate top-k, threshold, and feature/model settings against the challenge F_0.5 metric.

No external data lookup or external identity-resolution service is used.
