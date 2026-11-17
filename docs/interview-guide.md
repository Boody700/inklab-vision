# Explain InkLab in your own words

## A short introduction

“I built a small-image classification lab with three exported classifiers and a real editable input. A deterministic three-neighbor baseline beat my linear and neural candidates on training-only validation and reached 97.8% on an official test from different writers. I verified all browser probabilities against Python and made every test mistake inspectable. A one-pixel shift reduced accuracy to 65.8%, which makes the preprocessing limitations concrete.”

## Questions to prepare for

**Is this MNIST?** No. It is UCI Optdigits: 8 × 8 block counts derived from normalized 32 × 32 binary bitmaps. The official training/test split contains 3,823/1,797 examples. Using sklearn's convenient `load_digits()` test subset alone would be a different experiment.

**Why did a simple neighbor model win?** This small normalized feature space supports strong distance-based matching. The winner was chosen on validation results. Model complexity is not evidence of better performance. A larger neural architecture was not tuned against the test to force a preferred outcome.

**How do the splits protect evaluation?** Internal validation uses only official training rows. Selection and the review threshold are frozen before official test scoring. The official split uses different writers, though internal writer grouping is unavailable. No identical feature vector crosses the official boundary.

**Does a score of 100% mean certainty?** No. For 3-NN it means all three references voted for the same class. Thirteen mistaken test predictions are still unanimous. Softmax scores are also uncalibrated. Log loss exposes weaknesses that accuracy alone can hide.

**Why keep the neural convergence warning?** It reached a fixed compute budget. That limits the conclusion about that architecture. I reported the limitation and score rather than adjusting optimization after seeing the held-out result. A new training-budget experiment would need fresh validation and a clearly separated final evaluation.

**Does painting a digit validate general OCR?** No. The editor changes block-count features directly, bypassing original image normalization and segmentation. Arbitrary doodles can differ from both train and test data. The blank-input rule is a UI heuristic, not reliable novelty detection.

**What makes browser inference trustworthy numerically?** Exact preprocessing, exported weight orientation, stable softmax and deterministic neighbor tie rules. Tests compare all 5,391 classifier/test-image predictions and their full probability arrays, not just top labels in screenshots.

**Are the displayed neighbors explanations for the neural network?** They are not causal model attributions. They are separate nearest-image comparisons. They directly explain the votes only for the neighbor classifier.

**What would you add next?** A documented raw-image preprocessing pipeline, more writers, a new external-domain test, calibrated scores, writer-aware uncertainty estimates and realistic camera/scanning conditions. Keep a fresh final set for any further tuning.

## A five-minute walkthrough

Load a real example; switch classifiers; paint or erase a cell; shift one pixel; reset; open the mistakes gallery; click a confusion cell; inspect a confident mistake; export/reimport an experiment. Explain what is known from the benchmark and what remains unvalidated. Acknowledge AI assistance and be able to read the core inference and split code yourself.
