# Validation record

## Automated

Seven Python unittest checks pass: official shapes/count ranges, source hashes, cross-split uniqueness, training-only validation partition, model/threshold selection, metric reconciliation, deterministic ties and transforms.

Ten Node checks pass. Full inference parity covers all three models × 1,797 official test images (5,391 predictions); maximum absolute probability difference is 2.5535e-15. All 10,782 model/image stress evaluations across the six stated conditions reproduce Python correct counts. Other cases cover blank handling, stable softmax, invalid images/effects, edge clipping, reproducible noise, gallery reconciliation and validated local experiment imports.

The neural candidate reaches its fixed 350-iteration budget without convergence in the reference run. This is documented in the interface and model card. No test-based parameter adjustment was made.

## Browser

The final desktop/phone review and captured views are retained under `reports/`. Browser smoke checks complement numerical tests; they are not an exhaustive cross-browser certification.

Observed desktop interactions: pointer painting, arrow-key/Space painting, clearing/blank-label suppression, undo, real-sample loading, model switching, a one-pixel shift, effect reset, confusion-cell drill-down, gallery filters and two-page pagination. A neural 8 changed to a high-score 1 after a right shift, illustrating the documented stress sensitivity. A confusion cell for true 8/predicted 1 correctly opened five neural-model test examples.

The valid local experiment fixture imported successfully, restoring its classifier, pixels and effects. Export prepared the experiment JSON. Ten automated checks separately validate inference, transform and import data contracts; browser smoke checks do not substitute for those numeric checks.

At 390 × 844, playground, evaluation and error-gallery pages had equal 375-pixel client/scroll widths. At 320 pixels, both playground and evaluation retained equal 305-pixel client/scroll widths; the editor reduced to approximately 223 pixels. Wide evaluation tables scroll inside their containers. The final browser console had no warnings or errors. Screenshots include desktop playground/gallery and phone playground/gallery.

## Final collection polish

Undo now restores source identity and edited-state metadata as well as pixel values. A Restore control returns to the loaded sample and clears effects. The live provenance line counts cells that differ from the original; an original dataset label is no longer easy to mistake for a verified label on edited pixels. Exploratory review thresholds are labelled separately from the validation default. Range controls use a consistent lavender track/thumb style.

Browser checks painted one cell (the provenance line reported 1/64 changed), restored the original, cleared and undid the image (original sample #0009/label 8 returned), and changed the threshold (the note switched to “Your exploratory threshold”). Final phone checks again confirmed no page overflow at 390 and 320 pixels. The ten Node inference/data checks pass; model weights and benchmark results are unchanged.
