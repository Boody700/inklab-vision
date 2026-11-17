# InkLab: make model behavior tangible

## Problem

An impressive accuracy number can obscure preprocessing assumptions and failure modes. A useful junior computer-vision portfolio project should let a reader inspect both the working system and its limits.

InkLab connects an editable input, three real classifiers, an official different-writer test and a complete error gallery. It adopts a creative playground layout: lavender workbench, large pixel editor and prediction, tool palette, real-sample strip and stress controls. It is deliberately different from the earlier commercial dashboards.

## Method

The official UCI Optdigits training file has 3,823 examples from 30 writers. Its separate 1,797-example test file uses 13 different writers. Inside training, a fixed stratified 80/20 row split yields 3,058 fitting and 765 validation images. Writer IDs are not available for grouping that internal split.

Three fixed candidates compete: multinomial logistic regression, deterministic Euclidean 3-nearest neighbors and a 64-unit ReLU neural network. Inputs for linear/neural models are divided by 16; neighbors use the raw counts. Validation accuracy selects the winner, with macro F1 as a tie-break. Test results do not alter model identity or review threshold.

The neighbor model wins at validation accuracy 98.95%. Refitted on the full training file, it scores 97.83% on the official test: 1,758 correct and 39 incorrect. All mistakes remain accessible. The neural optimizer reaches its fixed 350-iteration cap without convergence; this limitation is disclosed and its score remains in the comparison.

## Engineering

Model weights, references and full-probability test fixtures are exported to JSON. Small JavaScript inference functions implement matrix layers, stable softmax and nearest-neighbor votes. All 5,391 classifier/test-image predictions match Python; maximum probability difference is 2.55e-15.

The editor paints the actual 64-value input rather than showing an unrelated decorative drawing. Keyboard painting and undo are supported. Effects act on a copy, preserving the source pixels. Nearest references explain neighbor votes and are labelled as a separate visual comparison for other classifiers. Imports validate numeric bounds and do not trust saved prediction claims.

## Findings

One pixel of rightward translation reduces selected-model accuracy to 65.83%. The result is a useful warning about a normalized benchmark: translation invariance has not been established. A unanimous neighbor vote can also be wrong. At the default review threshold, thirteen mistaken test predictions still pass.

The project establishes reproducible performance on this official small-image split and parity between two inference implementations. It does not establish general OCR accuracy, calibrated confidence, robust recognition of new doodles or production fitness. An extension should validate a real preprocessing pipeline and collect broader external/writer-specific examples before making those claims.
