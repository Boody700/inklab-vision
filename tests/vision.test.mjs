import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {predict,nearest,perturb,softmax,validatePixels,validateExperiment} from '../dist/vision.mjs';
const load=async path=>JSON.parse(await readFile(new URL(path,import.meta.url)));
const [models,samples,reference,report]=await Promise.all([load('../dist/data/models.json'),load('../dist/data/samples.json'),load('../reports/parity-reference.json'),load('../reports/evaluation.json')]);
test('experiment imports validate all controls and copy pixels without trusting extra fields',()=>{
  const input={schema:'inklab-experiment-v1',model:'neighbors',threshold:.7,basePixels:samples[0].pixels,effects:{noise:2,seed:7},prediction:{label:99}};
  const p=validateExperiment(input,Object.keys(models));assert.equal(p.effects.dx,0);assert.notEqual(p.pixels,input.basePixels);assert.equal(p.prediction,undefined);
  for(const change of [{model:'__proto__'},{threshold:2},{effects:{seed:'hello'}},{effects:{seed:-1}},{effects:{dx:9}}])assert.throws(()=>validateExperiment({...input,...change},Object.keys(models)));
});
test('all three portable classifiers match Python on every official test image',()=>{
  let worst=0;
  for(const [key,model] of Object.entries(models))for(const s of samples){const p=predict(s.pixels,model);for(let c=0;c<10;c++)worst=Math.max(worst,Math.abs(p.probs[c]-reference[key][s.id][c]));assert.equal(p.label,s.predictions[key]);}
  assert.ok(worst<1e-10,`largest probability difference ${worst}`);
  console.log('5,391 predictions checked; max probability difference:',worst);
});
test('nearest-neighbor distance ties use reference-row index and vote ties use digit order',()=>{
  const p=Array(64).fill(0),m={kind:'neighbors',k:2,pixels:[p,p,p],labels:[4,2,9]};assert.deepEqual(nearest(p,m,2).map(v=>v.index),[0,1]);assert.equal(predict(p,m).label,2);
});
test('blank images are detected separately from model scores',()=>{const p=predict(Array(64).fill(0),models.neighbors);assert.equal(p.blank,true);assert.equal(p.probs.reduce((a,b)=>a+b,0),1);});
test('softmax stays finite for extreme logits and sums to one',()=>{const p=softmax([10000,10001,-10000]);assert.ok(p.every(Number.isFinite));assert.ok(Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-14);});
test('invalid image shape, values and perturbation controls are rejected',()=>{for(const x of [[],Array(64).fill(NaN),Array(64).fill(-1),Array(64).fill(17)])assert.throws(()=>validatePixels(x));assert.throws(()=>perturb(Array(64).fill(0),{dx:3}));assert.throws(()=>perturb(Array(64).fill(0),{contrast:0}));});
test('shifts clip at edges without wraparound; default transform preserves counts',()=>{const p=Array(64).fill(0);p[0]=16;p[7]=8;assert.deepEqual(perturb(p),p);const q=perturb(p,{dx:1});assert.equal(q[1],16);assert.equal(q[8],0);assert.equal(q[0],0);assert.equal(q.reduce((a,b)=>a+b,0),16);});
test('seeded noise is reproducible and does not mutate source pixels',()=>{const p=Array(64).fill(8),a=perturb(p,{noise:4,seed:42}),b=perturb(p,{noise:4,seed:42});assert.deepEqual(a,b);assert.notDeepEqual(a,perturb(p,{noise:4,seed:43}));assert.deepEqual(p,Array(64).fill(8));assert.ok(a.every(v=>Number.isInteger(v)&&v>=0&&v<=16));});
test('all reported model errors reconcile with the inspectable test gallery',()=>{for(const k of Object.keys(models)){const errors=samples.filter(s=>s.label!==s.predictions[k]).length;assert.equal(errors,report.models[k].test.errors);const matrix=report.models[k].test.confusion;assert.equal(matrix.flat().reduce((a,b)=>a+b,0),samples.length);}});
test('browser and Python robustness definitions reconcile on all test images',()=>{const cases=[['Unchanged',{}],['Shift right 1px',{dx:1}],['Shift left 1px',{dx:-1}],['Uniform noise ±2',{noise:2}],['Uniform noise ±4',{noise:4}],['Half contrast',{contrast:.5}]];for(const [label,args] of cases){let correct=0;for(const s of samples)correct+=predict(perturb(s.pixels,{...args,seed:42+s.id}),models[report.selected]).label===s.label?1:0;assert.equal(correct,report.robustness.find(v=>v.label===label).correct);}});
