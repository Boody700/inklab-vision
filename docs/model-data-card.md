# Model and data card

## Intended use

Educational computer-vision inspection: normalized digit classification, model comparisons, robustness demonstrations and browser inference. Not production handwriting/document OCR, identity inference or a service with a promised accuracy level.

## Dataset and terms

E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. DOI 10.24432/C50P49; source https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits. UCI licenses the dataset under CC BY 4.0. Retain attribution and indicate adaptations.

There are 5,620 labelled examples in the two official feature files. Training: 3,823 from 30 writers. Test: 1,797 from 13 other writers. Features are counts of on-pixels in nonoverlapping 4 × 4 blocks of normalized 32 × 32 bitmaps: 64 integer counts in [0,16]. The target is a digit in 0–9. The project does not use personal identities or demographics. All training feature vectors are unique; no exact vector crosses the official train/test boundary. Source/file SHA-256 hashes are retained.

## Split and selection

Stratified training-row split: random_state 42, validation fraction 0.2; 3,058 fit / 765 validation. Individual writer IDs are unavailable within the training file, so this validation split is not writer-independent. The final official test is writer-disjoint as described by the source.

Model selection maximizes validation accuracy, then validation macro F1. After selection, all three candidates are refitted on the entire official training set and scored on the same official test. Reported test scores are never reused for model or threshold selection.

## Fixed candidates

- **Linear softmax:** scikit-learn LogisticRegression, C=10, max_iter=2000, tol=1e-6, random_state=42; feature counts /16.
- **3-nearest neighbors:** uniform votes over Euclidean distances; distance ties resolve by official training-row order, class vote ties by smallest digit. This deterministic reference implementation is exported directly.
- **Neural:** MLPClassifier, 64-ReLU-10-softmax, Adam, alpha=.001, batch=128, learning_rate_init=.001, max_iter=350, tol=1e-5, n_iter_no_change=30, early_stopping=False, random_state=42; inputs /16. Both validation fit and final refit reach the fixed 350-iteration limit without convergence. This comparison has a stated compute budget; no test-based optimizer adjustment was made.

## Outcomes

Selected: 3-nearest neighbors. Validation accuracy 0.9895425. Official-test accuracy 0.9782972; macro F1 0.9783038; log loss 0.3292360; multiclass Brier 0.0352439; 39 errors. Full scores and per-class/confusion reports remain in `reports/evaluation.json`.

Approximate per-image Wilson 95% accuracy interval: [0.9704702, 0.9840836]. Images from the same writer are not independent, and writer IDs are unavailable for cluster bootstrap. This interval is conditional and descriptive, not an estimate of guaranteed external-domain performance.

## Review policy and uncertainty

Candidate thresholds: 0,.5,.6,.7,.8,.9,.95,.99. For the selected model, choose the lowest validation threshold yielding at least 99% accepted accuracy and 50% coverage if available; otherwise use preset .8. This chooses .7, which for three uniform votes requires unanimity. Validation accepted accuracy is 99.73% at 95.69% coverage. Official-test accepted accuracy is 99.24% at 95.27% coverage (1,712 accepted, including 13 mistakes).

The browser allows exploratory threshold changes. Other model selectors initialize at an illustrative .8, not the neighbor-derived policy. Vote fractions and softmax values are uncalibrated. Passing a score threshold is not evidence that an arbitrary user image is in-distribution or correct.

## Robustness experiment

All test examples are transformed and classified by the selected model. Uniform noise uses a shared 32-bit linear-congruential generator, seed 42+test-row-index, applied after contrast and translation, then rounded half-up and clipped to [0,16]. Shifts clip at the image edges without wraparound.

| Condition | Test accuracy |
|---|---:|
| Original | 97.83% |
| Right one pixel | 65.83% |
| Left one pixel | 68.50% |
| Uniform noise ±2 counts | 97.83% |
| Uniform noise ±4 counts | 97.50% |
| Half contrast | 93.77% |

These conditions were fixed before scoring, rather than chosen from a larger search for dramatic cases. They are narrow controlled tests, not exhaustive adversarial or external-domain validation.

## Interface limits

The pixel editor bypasses original document segmentation and bitmap normalization. Paint changes one count to 0,8 or16. It is a feature-space sandbox, not a complete scanned-handwriting workflow. Blank/nearly blank inputs are suppressed by a simple sum-of-counts rule (<8); this is an interface heuristic, not a validated out-of-distribution detector. Models still produce scores for those inputs.

The three visual neighbors are the actual voting references for the neighbor classifier. For linear and neural classifiers they remain a separate distance-based comparison, not a feature attribution. Browser timing is one local run and excludes loading; it is not a cross-device performance benchmark.
