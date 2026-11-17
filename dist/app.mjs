import { predict, nearest, perturb, validatePixels, validateExperiment, matrixSVG } from "./vision.mjs";
const $ = (id) => document.getElementById(id), pct = (n) => n === null ? "—" : (100 * n).toFixed(1) + "%", esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
let models, samples, report, painting = false, paintValue = 16, lastPaint = -1;
const state = { page: "play", model: null, pixels: Array(64).fill(0), brush: 16, undo: [], source: null, modified: false, dx: 0, dy: 0, noise: 0, contrast: 1, seed: 42, threshold: 0.7, actual: "all", predicted: "all", show: "errors", galleryPage: 0, message: "" };
const currentInput = () => perturb(state.pixels, { dx: state.dx, dy: state.dy, noise: state.noise, contrast: state.contrast, seed: state.seed });
const names = () => Object.entries(report.names).map(([k, v]) => `<option value="${k}" ${state.model === k ? "selected" : ""}>${v}</option>`).join("");
const hero = (title, description, kicker = "A LITTLE COMPUTER VISION PLAYGROUND") => `<section class="hero"><div><div class="eyebrow">${kicker}</div><h1>${title}</h1><p>${description}</p></div><div class="hero-art" aria-hidden="true"><div class="art-orbit"></div><span class="art-digit art-a">8</span><span class="art-digit art-b">2</span><span class="art-digit art-c">4</span><span class="art-spark">✳</span></div></section>`;
const modelControl = () => `<label class="model-select">CLASSIFIER<select id="classifier" data-model aria-label="Classifier">${names()}</select></label>`;
function notify(message) {
  state.message = message;
  $("status").textContent = message;
  if ($("file-message")) $("file-message").textContent = message;
}
function resetEffects() {
  state.dx = 0;
  state.dy = 0;
  state.noise = 0;
  state.contrast = 1;
  state.seed = 42;
}
function useSample(id) {
  const s = samples[id];
  state.pixels = s.pixels.slice();
  state.source = id;
  state.modified = false;
  state.undo = [];
  resetEffects();
  state.page = "play";
  if (location.hash !== "#play") location.hash = "play";
  else render();
  notify(`Loaded official test sample ${id + 1}, labelled ${s.label}.`);
}
function pixelButton(v, i) {
  return `<button class="pixel" data-pixel="${i}" style="background:rgba(143,110,188,${v / 16})" aria-label="Row ${Math.floor(i / 8) + 1}, column ${i % 8 + 1}, intensity ${v} of 16" aria-pressed="${v > 0}"></button>`;
}
function sliders() {
  return [["dx", "Horizontal shift", -2, 2, 1, "−2 px", "+2 px"], ["dy", "Vertical shift", -2, 2, 1, "−2 px", "+2 px"], ["noise", "Noise amount", 0, 6, 1, "None", "±6 counts"], ["contrast", "Ink strength", 0.25, 1.5, 0.05, "25%", "150%"]].map(([key, label, min, max, step, left, right]) => `<label><span class="slider-label">${label}<output id="${key}-value" for="${key}">${effectValue(key)}</output></span><input type="range" aria-label="${label}" id="${key}" min="${min}" max="${max}" step="${step}" value="${state[key]}"><span class="slider-endpoints"><span>${left}</span><span>${right}</span></span></label>`).join("");
}
function effectValue(key) {
  return key === "contrast" ? Math.round(state.contrast * 100) + "%" : key === "noise" ? "±" + state.noise : state[key] + " px";
}
function reviewStats() {
  const selected = samples.filter((s) => s.scores[state.model] >= state.threshold), correct = selected.filter((s) => s.label === s.predictions[state.model]).length;
  return { coverage: selected.length / samples.length, accuracy: selected.length ? correct / selected.length : null };
}
function inputProvenance(input) {
  if (state.source === null) return "Custom input · outside the evaluated test set.";
  const source = samples[state.source], changed = input.filter((v, i) => v !== source.pixels[i]).length;
  return changed ? `${changed} / 64 cells differ from the original. Its reference label is not a label for your edits.` : `Original test image · dataset label ${source.label}.`;
}
function thresholdNote() {
  const reference = state.model === report.selected ? report.reviewThreshold : 0.8;
  return Math.abs(state.threshold - reference) < 1e-8 ? state.model === report.selected ? "Default selected on validation only." : "Preset 80% for this model; not validation-selected." : `Your exploratory threshold. ${state.model === report.selected ? "Validation default" : "Preset"}: ${pct(reference)}.`;
}
function resultHTML() {
  const start = performance.now(), input = currentInput(), p = predict(input, models[state.model]), duration = performance.now() - start;
  const review = p.blank || p.score < state.threshold;
  return `<div class="result-top"><span class="result-caption">THE MODEL'S BEST GUESS</span><span class="live-pill">LOCAL INFERENCE</span></div><div class="prediction-wrap"><div class="prediction ${review ? "blur" : ""}">${p.blank ? "?" : p.label}</div><div class="prediction-context"><strong>${p.blank ? "A little more ink?" : review ? "Worth another look." : "A possible " + p.label + "."}</strong>${p.blank ? "The input is nearly empty." : review ? "Below your review threshold." : "Passes your score threshold."}<small>${p.blank ? "Try a sample, or paint a few cells." : "A model vote or softmax score is not certainty."}</small></div></div><div class="result-detail"><div class="input-preview">${matrixSVG(input, { size: 45, color: "#9872bc", background: "#e7ddf3", label: "The transformed 8 by 8 input seen by the model" })}<span>WHAT THE MODEL SEES<br>64 pixel counts / 0–16</span></div><span class="model-time">${duration.toFixed(1)} ms<br>this device / one run</span></div><p class="input-provenance">${inputProvenance(input)}</p><div class="scores">${p.ranking.slice(0, 3).map((x) => `<div class="score-row"><strong>${x.label}</strong><span class="score-track"><span class="score-fill" style="width:${x.score * 100}%"></span></span><span>${pct(x.score)}</span></div>`).join("")}</div><p class="score-foot">${state.model === "neighbors" ? "Bars show the fraction of three nearest references voting for each digit." : "Bars show uncalibrated softmax model scores."} ${p.blank ? "The model still returns scores on blank images; the interface suppresses its label." : "These scores are not validated probabilities of correctness."}</p>`;
}
function neighborsHTML() {
  const input = currentInput(), rows = nearest(input, models.neighbors, 3);
  return rows.map((n, i) => `<article class="neighbor">${matrixSVG(models.neighbors.pixels[n.index], { size: 60, label: "Training reference labelled " + n.label })}<div><h3>${i + 1}. A training-set ${n.label}</h3><p>Reference #${n.index + 1} · distance ${Math.sqrt(n.distance).toFixed(1)}<br>${state.model === "neighbors" ? "One of the three voting neighbors." : "Visual comparison; not this model’s explanation."}</p></div></article>`).join("");
}
function sourceLabel() {
  return state.source === null ? "YOUR PIXEL EXPERIMENT" : `${state.modified ? "EDITED FROM" : "OFFICIAL TEST SAMPLE"} #${String(state.source + 1).padStart(4, "0")} · ORIGINAL LABEL ${samples[state.source].label}`;
}
function play() {
  const stats = reviewStats();
  return `${hero("A little ink.<br><em>A lot to learn.</em>", "Make a mark. Meet three real classifiers. Find out what a small image can teach us about machine learning.")}<section class="workbench"><div class="bench-top"><span class="section-label"><span class="step">1</span> Make your mark.</span>${modelControl()}</div><div class="bench-body"><section class="drawing-zone" aria-label="Digit editor"><div class="source-line" id="source-line">${sourceLabel()}</div><div class="pixel-editor" id="editor" aria-label="Editable 8 by 8 pixel image">${state.pixels.map(pixelButton).join("")}</div><div class="tools" role="group" aria-label="Drawing tools"><button class="tool" data-brush="16" aria-pressed="${state.brush === 16}">Ink</button><button class="tool" data-brush="8" aria-pressed="${state.brush === 8}">Soft ink</button><button class="tool" data-brush="0" aria-pressed="${state.brush === 0}">Erase</button><span class="tool-divider"></span><button class="tool" id="undo" ${state.undo.length ? "" : "disabled"} aria-label="Undo last stroke">↶</button><button class="tool" id="clear">Clear</button><button class="tool" id="restore-sample" ${state.source === null ? "disabled" : ""}>Restore</button></div><p class="drawing-note">Paint or tap cells. Arrow keys move; Space paints.<br>Edits happen directly in the model's 8 × 8 feature space.</p></section><section class="result-zone" id="result" aria-label="Live prediction">${resultHTML()}</section></div><div class="examples-strip"><div class="examples-title"><strong>Try a real digit.</strong>First test example of each class.</div><div class="example-list">${Array.from({ length: 10 }, (_, label) => samples.find((s) => s.label === label)).map((s) => `<button class="example ${state.source === s.id ? "active" : ""}" data-sample="${s.id}" aria-label="Try digit ${s.label}, official test sample ${s.id + 1}">${matrixSVG(s.pixels, { size: 40, label: "Example " + s.label })}${s.label}</button>`).join("")}</div></div></section><section><div class="section-head"><div><h2>Small changes. Different answers?</h2><p>Stress the input and watch the model react.</p></div><button class="button" id="reset-effects">Reset changes ↶</button></div><div class="perturb-panel">${sliders()}</div><div class="perturb-note"><span>Shifts clip at the edges. Seeded uniform noise is added after contrast, then counts are rounded and clipped to 0–16.</span><span class="page-actions"><button class="button plain" id="noise-seed">New noise pattern ↻</button></span></div></section><section><div class="section-head"><div><h2>A few familiar faces.</h2><p>Closest training references by Euclidean pixel distance.</p></div><span class="section-label"><span class="step">3</span> COMPARE</span></div><div class="neighbor-grid" id="neighbors">${neighborsHTML()}</div></section><section class="review-control"><div><h3>Leave room for “I'm not sure.”</h3><p id="review-stat">At this threshold, the official test has ${pct(stats.coverage)} coverage and ${pct(stats.accuracy)} accuracy among accepted predictions. This describes the test set, not your edited input.</p></div><div><label for="threshold">Review below <output id="threshold-value">${pct(state.threshold)}</output></label><input id="threshold" type="range" min="0" max="1" step=".01" value="${state.threshold}"><small id="threshold-note">${thresholdNote()} Scores remain uncalibrated.</small></div></section><div class="section-head"><div><h3>Keep your experiment.</h3><p id="file-message">${esc(state.message || "JSON export keeps your pixels, effects and model settings. No image is uploaded.")}</p></div><div class="page-actions"><button class="button" id="import">Import</button><button class="button purple" id="export">Export ↗</button><input id="import-file" type="file" accept="application/json,.json" hidden></div></div><p class="wide-notice">This is a small-image classification experiment, not general handwriting OCR. The original dataset uses counts from normalized 32 × 32 bitmaps; the editor works directly on its 8 × 8 features. New doodles can differ substantially from the test data. <a href="#about">Read the limits →</a></p>`;
}
function filteredSamples() {
  return samples.filter((s) => (state.show !== "errors" || s.label !== s.predictions[state.model]) && (state.actual === "all" || s.label === Number(state.actual)) && (state.predicted === "all" || s.predictions[state.model] === Number(state.predicted)));
}
function gallery() {
  const all = filteredSamples(), pages = Math.max(1, Math.ceil(all.length / 24));
  state.galleryPage = Math.min(state.galleryPage, pages - 1);
  const rows = all.slice(state.galleryPage * 24, (state.galleryPage + 1) * 24), test = report.models[state.model].test;
  return `${hero("The wrong answers<br><em>are part of the story.</em>", "Inspect every mistake from the official different-writer test. Load any image into the playground and ask what changed.", "THE MISTAKES / NOTHING HIDDEN")}<div class="highlight-banner"><span class="highlight-value">${test.errors}</span><p><strong>${report.names[state.model]} mistakes</strong> out of ${test.total.toLocaleString()} official test images.<br>Errors are preserved, rather than removed from the demo.</p></div><div class="gallery-tools">${modelControl()}<label>ACTUAL DIGIT<select id="actual"><option value="all">Any digit</option>${Array.from({ length: 10 }, (_, i) => `<option ${state.actual === String(i) ? "selected" : ""}>${i}</option>`).join("")}</select></label><label>PREDICTED DIGIT<select id="predicted"><option value="all">Any prediction</option>${Array.from({ length: 10 }, (_, i) => `<option ${state.predicted === String(i) ? "selected" : ""}>${i}</option>`).join("")}</select></label><label>SHOW<select id="show"><option value="errors" ${state.show === "errors" ? "selected" : ""}>Mistakes only</option><option value="all" ${state.show === "all" ? "selected" : ""}>All test images</option></select></label><span class="gallery-count">${all.length.toLocaleString()} matching images</span></div>${rows.length ? `<div class="sample-grid">${rows.map((s) => `<button class="sample-card ${s.label !== s.predictions[state.model] ? "wrong" : ""}" data-sample="${s.id}" aria-label="Inspect sample ${s.id + 1}, true ${s.label}, predicted ${s.predictions[state.model]}"><span class="sample-id">TEST / ${String(s.id + 1).padStart(4, "0")} ↗</span>${matrixSVG(s.pixels, { size: 95, label: "Digit sample " + (s.id + 1) })}<span class="sample-labels"><span>True <strong>${s.label}</strong></span><span>Guess <strong>${s.predictions[state.model]}</strong></span></span><span class="sample-score">Model score ${pct(s.scores[state.model])}</span></button>`).join("")}</div>` : '<div class="empty">No test images match these filters. Try another digit or show all images.</div>'}<div class="pagination"><span>Page ${state.galleryPage + 1} / ${pages} · 24 images per page</span><div><button class="button" id="prev" ${state.galleryPage === 0 ? "disabled" : ""}>← Previous</button><button class="button" id="next" ${state.galleryPage >= pages - 1 ? "disabled" : ""}>Next →</button></div></div><p class="small-note">Predictions were made by models fitted to the official training set. The gallery does not train on these test labels. Filters help inspect errors; they do not change reported benchmark results.</p>`;
}
function confusion() {
  const c = report.models[state.model].test.confusion, max = Math.max(...c.flat());
  return `<div class="confusion" aria-label="Confusion matrix, rows actual digit, columns predicted digit"><span></span>${Array.from({ length: 10 }, (_, i) => `<span>${i}</span>`).join("")}${c.map((row, i) => `<span>${i}</span>${row.map((v, j) => `<button class="${i === j ? "diagonal" : ""}" data-confusion="${i},${j}" style="background:rgba(${i === j ? "183,161,211" : "223,173,196"},${v ? 0.15 + 0.65 * v / max : 0.06})" aria-label="True ${i}, predicted ${j}, ${v} images">${v || "·"}</button>`).join("")}`).join("")}</div>`;
}
function bench() {
  const test = report.models[report.selected].test;
  return `${hero("Pretty pixels.<br><em>Proper checks.</em>", "Three fixed candidates. A training-only validation split. A separate official test from different writers.", "UNDER THE HOOD / MEASUREMENT FIRST")}<div class="section-head"><div><h2>Simple models earn their place.</h2><p>Chosen by validation accuracy, then macro F1 as a tie-break.</p></div><button class="button" id="export-benchmark">Export report ↗</button></div><div class="table-wrap"><table><caption class="sr-only">Classifier validation and official test performance</caption><thead><tr><th>CLASSIFIER</th><th class="num">VAL ACCURACY</th><th class="num">TEST ACCURACY</th><th class="num">MACRO F1</th><th class="num">LOG LOSS</th></tr></thead><tbody>${Object.entries(report.models).map(([k, m]) => `<tr class="${k === report.selected ? "selected" : ""}"><td>${m.label}${k === report.selected ? " · selected" : ""}</td><td class="num">${pct(m.validation.accuracy)}</td><td class="num">${pct(m.test.accuracy)}</td><td class="num">${m.test.macroF1.toFixed(3)}</td><td class="num">${m.test.logLoss.toFixed(3)}</td></tr>`).join("")}</tbody></table></div><p class="small-note">The official split uses 30 training writers and 13 different test writers. Validation is a stratified row split within the training file; individual writer IDs are unavailable there. All models are refitted on all 3,823 training rows before testing. The neural candidate reached its fixed 350-iteration budget before convergence; its results remain visible.</p><div class="highlight-banner"><span class="highlight-value">${pct(test.accuracy)}</span><p><strong>Official-test accuracy / ${report.names[report.selected]}</strong><br>${test.correct} correct out of ${test.total}. Approximate per-image Wilson 95% interval: ${pct(test.wilson95[0])}–${pct(test.wilson95[1])}; not an interval over unseen writers or doodles.</p></div><div class="bench-grid"><section class="panel"><div class="model-picker"><div><div class="eyebrow">TRUE ↓ / PREDICTED →</div><h2>Where digits get mixed up.</h2></div>${modelControl()}</div><p>Click a cell to inspect its test images.</p>${confusion()}<div class="axis-note">Rows = reference labels · columns = model predictions.<br>Diagonal cells are correct. Pink cells are mistakes.</div></section><section class="panel"><div class="eyebrow">ROBUSTNESS / SELECTED VALIDATION WINNER</div><h2>Move one pixel.<br>Lose a lot.</h2><p>Fixed perturbations applied to all 1,797 test images. Results below use ${report.names[report.selected]}, regardless of the matrix selector.</p>${report.robustness.map((v) => `<div class="stress-row"><span>${esc(v.label)}</span><span class="stress-track"><i style="width:${v.accuracy * 100}%"></i></span><strong>${pct(v.accuracy)}</strong></div>`).join("")}<p class="small-note">Shifted pixels outside the frame are discarded. Noise uses the same seeded generator and rounding as the playground. These stress tests were specified before inspecting results and do not exhaust real-world variation.</p></section></div><div class="prose-grid" style="margin-top:23px"><section class="panel prose"><div class="eyebrow">MODEL SCORES / NOT CERTAINTY</div><h2>A unanimous vote can be wrong.</h2><p>Nearest-neighbor bars are vote fractions; linear and neural bars are softmax scores. Neither has been calibrated to the chance that a fresh input is correct. A mistaken unanimous neighbor vote can receive a score of 100%.</p><p>The default review threshold for the selected model is ${pct(report.reviewThreshold)}, chosen on validation to target at least 99% accepted accuracy with at least 50% coverage if feasible. That target is not a guarantee. Other models start at an illustrative 80% threshold.</p></section><section class="panel prose"><div class="eyebrow">WHAT THESE NUMBERS SUPPORT</div><h2>A useful small benchmark.</h2><p>Macro F1 weights each digit equally. Accuracy counts correct test labels. Log loss penalizes assigning low scores to the correct class; hard neighbor votes can be confidently wrong, so accuracy alone misses part of the story.</p><p>The test set contains only 13 writers and normalized feature vectors. An image-independent Wilson interval does not account for clustering by writer. New drawings, scripts, camera images, backgrounds and preprocessing can change performance substantially.</p></section></div>`;
}
function about() {
  return `${hero("Tiny images.<br><em>Honest experiments.</em>", "A complete computer-vision study you can inspect—from source pixels to browser predictions.", "ABOUT THE LAB / BUILT TO BE UNDERSTOOD")}<div class="prose-grid"><section class="panel prose"><div class="eyebrow">01 / SOURCE & SCOPE</div><h2>Real handwriting, reduced to counts.</h2><p><a href="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits" target="_blank" rel="noopener">Optical Recognition of Handwritten Digits</a>, by E. Alpaydin and C. Kaynak (1998), is published by UCI under <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a>. DOI: 10.24432/C50P49.</p><p>The official files contain 3,823 training and 1,797 test images, each with 64 integer values and one digit label. They summarize normalized 32 × 32 binary bitmaps as 8 × 8 grids of counts from nonoverlapping 4 × 4 blocks. Each count ranges from 0 to 16.</p><p>Thirty writers contributed training samples; thirteen different writers contributed the test set. There are no identical 64-value vectors across the two official files. This is a digit benchmark, not a general OCR dataset.</p></section><section class="panel prose"><div class="eyebrow">02 / TRAINING & PORTABILITY</div><h2>Three models. No inference API.</h2><div class="process"><div><strong>Split</strong>training-only validation</div><span>→</span><div><strong>Compare</strong>linear / neighbors / neural</div><span>→</span><div><strong>Export</strong>weights & references</div><span>→</span><div><strong>Inspect</strong>local browser inference</div></div><p>Linear softmax and a 64-unit ReLU neural net use inputs divided by 16. A deterministic three-neighbor classifier uses Euclidean pixel distance and row-index tie-breaks. The winner is selected using validation only.</p><p>The static app runs the exported models in JavaScript. The project verifies browser inference against Python probabilities for every official test image, not just a chosen example. No drawing or question is sent to a server for classification.</p></section><section class="panel prose"><div class="eyebrow">03 / THE EDITOR IS AN EXPERIMENT</div><h2>Feature-space painting.</h2><p>You are editing the 64 counts directly. This does not reproduce the original document segmentation, bitmap normalization or 32 × 32 preprocessing. Painting a digit here can produce inputs far outside the original handwriting distribution.</p><p>Brushes set a cell to 16, 8 or 0. Arrow keys move focus; Space paints with the selected brush. Undo stores the last 30 strokes. Shifts, contrast and seeded noise act on a copy of the source image, so resetting effects preserves your drawing.</p><p>Closest references explain the neighbor model's votes. For the linear and neural models, they are a separate visual comparison, not a causal attribution or explanation of the classifier.</p></section><section class="panel prose"><div class="eyebrow">04 / PRIVACY & REPRODUCTION</div><h2>A small, self-contained lab.</h2><p>Models and public reference data load from this site; inference, edits, imports and exports run locally. No account, analytics or inference service is used. Refreshing clears your experiment; export JSON if you want to keep it. Optional Google Fonts requests load the typefaces.</p><p>The source includes the official-data downloader, fixed training script, score and split manifests, robustness evaluation, full test parity fixtures, tests, a case study and an interview guide. No training package is needed just to open the prebuilt app through its local server.</p><p>The study does not claim validated accuracy for arbitrary user drawings or production document recognition. Broader writer samples, realistic preprocessing and external-domain evaluation are needed for those claims.</p></section></div><div class="provenance">Dataset citation: Alpaydin, E. & Kaynak, C. (1998). <em>Optical Recognition of Handwritten Digits</em>. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C50P49" target="_blank" rel="noopener">doi.org/10.24432/C50P49</a>. Adaptations: exported classifiers, pixel editor, evaluation and perturbed examples. No endorsement by UCI or the dataset creators is implied.</div>`;
}
function updatePrediction() {
  if (!$("result")) return;
  $("result").innerHTML = resultHTML();
  $("neighbors").innerHTML = neighborsHTML();
  $("source-line").textContent = sourceLabel();
  $("undo").disabled = !state.undo.length;
  $("status").textContent = $("result").querySelector(".prediction-context").textContent;
}
function remember() {
  state.undo.push({ pixels: state.pixels.slice(), source: state.source, modified: state.modified });
  if (state.undo.length > 30) state.undo.shift();
}
function paint(index) {
  if (index === lastPaint) return;
  lastPaint = index;
  state.pixels[index] = paintValue;
  state.modified = true;
  const b = document.querySelector(`[data-pixel="${index}"]`);
  b.style.background = `rgba(143,110,188,${paintValue / 16})`;
  b.setAttribute("aria-label", `Row ${Math.floor(index / 8) + 1}, column ${index % 8 + 1}, intensity ${paintValue} of 16`);
  b.setAttribute("aria-pressed", String(paintValue > 0));
  updatePrediction();
}
function exportJSON(name, obj) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" })), a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1e3);
  notify("Prepared " + name);
}
function bind() {
  document.querySelectorAll("[data-model]").forEach((el) => el.addEventListener("change", () => {
    state.model = el.value;
    state.threshold = state.model === report.selected ? report.reviewThreshold : 0.8;
    state.galleryPage = 0;
    render();
  }));
  document.querySelectorAll("[data-sample]").forEach((el) => el.addEventListener("click", () => useSample(Number(el.dataset.sample))));
  document.querySelectorAll("[data-brush]").forEach((el) => el.addEventListener("click", () => {
    state.brush = Number(el.dataset.brush);
    document.querySelectorAll("[data-brush]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.brush) === state.brush)));
  }));
  const editor = $("editor");
  if (editor) {
    editor.addEventListener("pointerdown", (e) => {
      const b = e.target.closest("[data-pixel]");
      if (!b || e.button > 0) return;
      e.preventDefault();
      remember();
      painting = true;
      paintValue = state.brush;
      lastPaint = -1;
      b.focus({ preventScroll: true });
      paint(Number(b.dataset.pixel));
    });
    editor.addEventListener("pointermove", (e) => {
      if (!painting) return;
      const b = document.elementFromPoint(e.clientX, e.clientY)?.closest("[data-pixel]");
      if (b && editor.contains(b)) paint(Number(b.dataset.pixel));
    });
    editor.addEventListener("click", (e) => {
      if (e.detail !== 0) return;
      const b = e.target.closest("[data-pixel]");
      if (b) {
        remember();
        paintValue = state.brush;
        lastPaint = -1;
        paint(Number(b.dataset.pixel));
      }
    });
    editor.addEventListener("keydown", (e) => {
      const b = e.target.closest("[data-pixel]");
      if (!b) return;
      const i = Number(b.dataset.pixel), offset = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -8, ArrowDown: 8 }[e.key];
      if (offset !== void 0) {
        e.preventDefault();
        const next = Math.max(0, Math.min(63, i + offset));
        editor.querySelector(`[data-pixel="${next}"]`).focus();
      }
    });
  }
  $("restore-sample")?.addEventListener("click", () => {
    if (state.source !== null) {
      remember();
      state.pixels = samples[state.source].pixels.slice();
      state.modified = false;
      resetEffects();
      render();
      notify("Original sample and zero effects restored.");
    }
  });
  $("clear")?.addEventListener("click", () => {
    remember();
    state.pixels = Array(64).fill(0);
    state.source = null;
    state.modified = false;
    resetEffects();
    render();
    notify("Canvas cleared.");
  });
  $("undo")?.addEventListener("click", () => {
    if (state.undo.length) {
      const snapshot = state.undo.pop();
      state.pixels = snapshot.pixels;
      state.source = snapshot.source;
      state.modified = snapshot.modified;
      render();
    }
  });
  for (const key of ["dx", "dy", "noise", "contrast"]) $(key)?.addEventListener("input", (e) => {
    state[key] = Number(e.target.value);
    $(key + "-value").textContent = effectValue(key);
    updatePrediction();
  });
  $("reset-effects")?.addEventListener("click", () => {
    resetEffects();
    render();
    notify("Effects reset; source pixels retained.");
  });
  $("noise-seed")?.addEventListener("click", () => {
    state.seed = state.seed + 1 >>> 0;
    updatePrediction();
    notify("Noise seed " + state.seed + ".");
  });
  $("threshold")?.addEventListener("input", (e) => {
    state.threshold = Number(e.target.value);
    $("threshold-value").textContent = pct(state.threshold);
    const s = reviewStats();
    $("review-stat").textContent = `At this threshold, the official test has ${pct(s.coverage)} coverage and ${pct(s.accuracy)} accuracy among accepted predictions. This describes the test set, not your edited input.`;
    $("threshold-note").textContent = thresholdNote() + " Scores remain uncalibrated.";
    updatePrediction();
  });
  for (const key of ["actual", "predicted", "show"]) $(key)?.addEventListener("change", (e) => {
    state[key] = e.target.value;
    state.galleryPage = 0;
    render();
  });
  $("prev")?.addEventListener("click", () => {
    state.galleryPage--;
    render();
  });
  $("next")?.addEventListener("click", () => {
    state.galleryPage++;
    render();
  });
  document.querySelectorAll("[data-confusion]").forEach((el) => el.addEventListener("click", () => {
    const [a, p] = el.dataset.confusion.split(",");
    state.actual = a;
    state.predicted = p;
    state.show = "all";
    state.galleryPage = 0;
    location.hash = "gallery";
  }));
  $("export")?.addEventListener("click", () => exportJSON("inklab-experiment.json", { schema: "inklab-experiment-v1", model: state.model, threshold: state.threshold, basePixels: state.pixels, effects: { dx: state.dx, dy: state.dy, noise: state.noise, contrast: state.contrast, seed: state.seed }, modelInput: currentInput(), prediction: predict(currentInput(), models[state.model]), source: state.source === null ? null : { officialTestId: state.source + 1, originalLabel: samples[state.source].label, edited: state.modified }, notice: "Model scores are uncalibrated. New drawings are outside the evaluated distribution." }));
  $("export-benchmark")?.addEventListener("click", () => exportJSON("inklab-evaluation.json", report));
  $("import")?.addEventListener("click", () => $("import-file").click());
  $("import-file")?.addEventListener("change", async (e) => {
    try {
      const f = e.target.files[0];
      if (!f) return;
      if (f.size > 1e5) throw new Error("Choose a JSON experiment smaller than 100 KB.");
      const input = JSON.parse(await f.text());
      const checked = validateExperiment(input, Object.keys(models));
      remember();
      state.pixels = checked.pixels;
      state.model = checked.model;
      state.threshold = checked.threshold;
      resetEffects();
      Object.assign(state, checked.effects);
      state.source = null;
      state.modified = true;
      render();
      notify("Experiment imported locally.");
    } catch (err) {
      notify("Import could not load: " + err.message);
    }
  });
}
function render() {
  const focused = document.activeElement?.id;
  $("content").innerHTML = ({ play, gallery, bench, about }[state.page] || play)();
  document.querySelectorAll("[data-page]").forEach((el) => {
    const active = el.dataset.page === state.page;
    el.classList.toggle("active", active);
    if (active) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
  document.title = { play: "Playground", gallery: "The mistakes", bench: "Under the hood", about: "About the lab" }[state.page] + " / InkLab";
  bind();
  if (focused) $(focused)?.focus({ preventScroll: true });
}
function route() {
  state.page = ["play", "gallery", "bench", "about"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "play";
  painting = false;
  render();
}
document.addEventListener("pointerup", () => {
  painting = false;
  lastPaint = -1;
});
document.addEventListener("pointercancel", () => {
  painting = false;
  lastPaint = -1;
});
try {
  [models, samples, report] = await Promise.all(["models", "samples", "evaluation"].map(async (key) => {
    const response = await fetch("./data/" + key + ".json");
    if (!response.ok) throw new Error("Could not load " + key);
    return response.json();
  }));
  state.model = report.selected;
  state.threshold = report.reviewThreshold;
  const first = samples.find((s) => s.label === 8);
  state.pixels = first.pixels.slice();
  state.source = first.id;
  window.addEventListener("hashchange", () => {
    // A skip link is an in-page focus action, not a request for the default route.
    if (location.hash === "#main") {
      $("main").focus();
      return;
    }
    route();
    window.scrollTo({ top: 0, behavior: "instant" });
    $("main").focus({ preventScroll: true });
  });
  route();
} catch (err) {
  $("content").innerHTML = '<div class="error" role="alert">The lab could not load. Try reloading or run the prebuilt app through its local HTTP server.</div>';
  console.error(err);
}
