export function validatePixels(pixels) {
  if (!Array.isArray(pixels) || pixels.length !== 64 || pixels.some((v) => !Number.isFinite(v) || v < 0 || v > 16)) throw new RangeError("Expected 64 finite pixel counts from 0 to 16.");
}
export function softmax(logits) {
  const max = Math.max(...logits), e = logits.map((v) => Math.exp(v - max)), total = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / total);
}
export function validateExperiment(input, modelKeys) {
  if (!input || input.schema !== "inklab-experiment-v1") throw new RangeError("This is not an InkLab experiment.");
  validatePixels(input.basePixels);
  if (input.basePixels.some((v) => !Number.isInteger(v))) throw new RangeError("Pixel counts must be integers.");
  if (typeof input.model !== "string" || !modelKeys.includes(input.model) || !Number.isFinite(input.threshold) || input.threshold < 0 || input.threshold > 1) throw new RangeError("Invalid model or review threshold.");
  const source = input.effects ?? {};
  if (typeof source !== "object" || Array.isArray(source)) throw new RangeError("Invalid effects.");
  const effects = { dx: source.dx ?? 0, dy: source.dy ?? 0, noise: source.noise ?? 0, contrast: source.contrast ?? 1, seed: source.seed ?? 42 };
  if (!Number.isInteger(effects.seed) || effects.seed < 0 || effects.seed > 4294967295) throw new RangeError("Noise seed must be an unsigned 32-bit integer.");
  perturb(input.basePixels, effects);
  return { pixels: input.basePixels.slice(), model: input.model, threshold: input.threshold, effects };
}
function dense(input, weights, bias) {
  const out = bias.slice();
  for (let i = 0; i < input.length; i++) for (let j = 0; j < out.length; j++) out[j] += input[i] * weights[i][j];
  return out;
}
export function nearest(pixels, model, count = 3) {
  validatePixels(pixels);
  return model.pixels.map((row, index) => ({ index, label: model.labels[index], distance: row.reduce((s, v, i) => s + (v - pixels[i]) ** 2, 0) })).sort((a, b) => a.distance - b.distance || a.index - b.index).slice(0, count);
}
export function predict(pixels, model) {
  validatePixels(pixels);
  let probs;
  if (model.kind === "neighbors") {
    probs = Array(10).fill(0);
    for (const item of nearest(pixels, model, model.k)) probs[item.label] += 1 / model.k;
  } else if (model.kind === "linear") probs = softmax(dense(pixels.map((v) => v / 16), model.weights, model.bias));
  else if (model.kind === "neural") {
    let a = pixels.map((v) => v / 16);
    model.weights.forEach((w, i) => {
      a = dense(a, w, model.bias[i]);
      if (i < model.weights.length - 1) a = a.map((v) => Math.max(0, v));
    });
    probs = softmax(a);
  } else throw new RangeError("Unknown classifier");
  const ranking = probs.map((score, label) => ({ label, score })).sort((a, b) => b.score - a.score || a.label - b.label);
  return { probs, ranking, label: ranking[0].label, score: ranking[0].score, blank: pixels.reduce((a, b) => a + b, 0) < 8 };
}
export function perturb(pixels, { dx = 0, dy = 0, noise = 0, seed = 42, contrast = 1 } = {}) {
  validatePixels(pixels);
  if (!Number.isInteger(dx) || !Number.isInteger(dy) || Math.abs(dx) > 2 || Math.abs(dy) > 2 || !Number.isFinite(noise) || noise < 0 || noise > 6 || !Number.isFinite(contrast) || contrast < 0.25 || contrast > 1.5) throw new RangeError("Invalid perturbation");
  let rng = seed >>> 0;
  return pixels.map((_, i) => {
    const row = Math.floor(i / 8) - dy, col = i % 8 - dx;
    const value = row >= 0 && row < 8 && col >= 0 && col < 8 ? pixels[row * 8 + col] : 0;
    rng = Math.imul(1664525, rng) + 1013904223 >>> 0;
    const n = (rng / 4294967296 * 2 - 1) * noise;
    return Math.max(0, Math.min(16, Math.floor(value * contrast + n + 0.5)));
  });
}
export function matrixSVG(pixels, { size = 80, color = "#6550bc", background = "#ede9f7", label = "8 by 8 digit image" } = {}) {
  validatePixels(pixels);
  return `<svg viewBox="0 0 80 80" width="${size}" height="${size}" role="img" aria-label="${String(label).replace(/[&<>"']/g, "")}"><rect width="80" height="80" rx="8" fill="${background}"/>${pixels.map((v, i) => v ? `<rect x="${i % 8 * 9 + 4}" y="${Math.floor(i / 8) * 9 + 4}" width="8" height="8" rx="1.5" fill="${color}" opacity="${v / 16}"/>` : "").join("")}</svg>`;
}
