# Architecture and reproduction

`download.py` → official feature files + source documentation + hashes

`train.py` → training-only validation → fixed candidate comparison → frozen selection/review policy → full-training refits → official test + stress tests → portable models, samples and probability references

`vision.mjs` → numeric input checks, dense layers/softmax, deterministic nearest neighbors, matching seeded transforms and validated experiment parsing

`app.mjs` → editable pixel buttons, pointer/keyboard input, undo, real-time scores/references, gallery, confusion filters and local JSON files

The site is static HTML/CSS/ES modules. No server-side inference, credentials, analytics, authentication or framework dependencies are needed. Three JSON assets load at startup; the public training-reference vectors support local nearest-neighbor scoring. Google Fonts optionally supplies the typefaces; system fallbacks remain available.

## Inference contract

Pixels are 64 finite values in [0,16]. Linear/neural features are divided by 16. Dense weights are stored input-major; the neural hidden layer applies ReLU and the output applies numerically stable softmax. Neighbors use squared Euclidean distances; a stable comparator orders distance then reference index. Class ties sort by ascending digit. All exported weights retain JSON double precision.

The probability parity test checks all 1,797 official test inputs against all three Python models. The largest absolute difference was 2.55e-15. Robustness parity checks every test image under six conditions. This is numerical verification, not a claim of browser DOM coverage on every sample.

## Editor and state

The editor is a set of 64 native buttons with per-pixel labels/pressed state, arrow navigation and Space painting. Pointer drawing updates cells in place so focus is retained. Undo holds the last 30 source matrices. Transform controls operate on a copy, so reset does not discard painting. Page navigation preserves experiment state; refresh clears it.

JSON exports include source pixels, effect settings, model/threshold and a prediction snapshot. Imports whitelist supported schema/model values, enforce numeric/count/seed bounds, copy pixels and recompute inference instead of trusting saved prediction fields. Files are limited to 100 KB and parsed locally. Malformed input leaves the prior experiment intact.

## Reproduce and extend

The pinned Python environment is in `requirements.txt`; actual versions and hashes are in the report. Node.js 22+ serves and tests the app without installation. Training is deterministic for the reference environment but may exhibit tiny library/platform differences. The fixed-budget neural convergence warning is documented.

The main missing piece for a practical OCR product is real-image preprocessing and external validation. Add segmentation, original bitmap normalization, writer-aware validation and a broader held-out corpus before claiming real-document performance. Consider calibrated models and explicit distribution-shift detection rather than equating a high model score with reliability.
