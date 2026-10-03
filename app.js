'use strict';

const MODEL_SOURCES = [
  'https://justadudewhohacks.github.io/face-api.js/models',
  'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights'
];
const MODEL_TIMEOUT_MS = 60000;
const DETECTION_INTERVAL_MS = 100;
const INPUT_SIZE = 320;
const SCORE_THRESHOLD = 0.5;

const els = {
  video: document.getElementById('video'), overlay: document.getElementById('overlay'),
  start: document.getElementById('startBtn'), stop: document.getElementById('stopBtn'), retry: document.getElementById('retryBtn'),
  modelDot: document.getElementById('modelDot'), modelStatus: document.getElementById('modelStatus'),
  cameraOverlay: document.getElementById('cameraOverlay'), streamStatus: document.getElementById('streamStatus'),
  fps: document.getElementById('fps'), faceCount: document.getElementById('faceCount'), results: document.getElementById('results'), log: document.getElementById('log')
};

let stream = null, running = false, modelsLoaded = false, loadingModels = false, detecting = false;
let lastDetectionAt = 0, frames = 0, lastFpsAt = performance.now();

function log(message) { els.log.textContent = `[${new Date().toLocaleTimeString()}] ${message}`; }
function setModelStatus(text, state = '') { els.modelStatus.textContent = text; els.modelDot.className = `dot ${state}`.trim(); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }

function withTimeout(promise, ms, label) {
  let timer;
  return Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)} seconds.`)), ms); })])
    .finally(() => clearTimeout(timer));
}

async function loadScript(url) {
  if (window.faceapi) return;
  await new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Could not load face-api.js from ${url}`));
    document.head.appendChild(script);
  });
}

async function ensureFaceApi() {
  if (window.faceapi) return;
  const sources = [
    'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js',
    'https://unpkg.com/face-api.js@0.22.2/dist/face-api.min.js'
  ];
  let lastError;
  for (const source of sources) {
    try { await withTimeout(loadScript(source), 20000, 'face-api.js download'); if (window.faceapi) return; }
    catch (e) { lastError = e; }
  }
  throw lastError || new Error('face-api.js could not be loaded.');
}

async function loadModels() {
  if (modelsLoaded) return true;
  if (loadingModels) return false;
  loadingModels = true;
  setModelStatus('Loading models…');
  els.start.disabled = true;
  log('Loading face-api.js and pretrained models…');

  try {
    await ensureFaceApi();
    let lastError;
    for (const base of MODEL_SOURCES) {
      try {
        log(`Trying model source: ${base}`);
        await withTimeout(Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(base),
          faceapi.nets.faceLandmark68Net.loadFromUri(base),
          faceapi.nets.ageGenderNet.loadFromUri(base),
          faceapi.nets.faceExpressionNet.loadFromUri(base)
        ]), MODEL_TIMEOUT_MS, 'Model loading');
        modelsLoaded = true;
        setModelStatus('Models ready', 'ready');
        log('All models loaded successfully.');
        els.start.disabled = false;
        return true;
      } catch (e) {
        lastError = e;
        log(`Model source failed: ${e.message || e}`);
      }
    }
    throw lastError || new Error('No model source was available.');
  } catch (error) {
    setModelStatus('Models unavailable', 'error');
    els.start.disabled = false;
    log(`Model error: ${error.message || error}`);
    return false;
  } finally {
    loadingModels = false;
  }
}

async function startCamera() {
  if (running) return;
  if (!(await loadModels())) return;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Webcam API unavailable. Open this app through localhost or HTTPS, not file://.');
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    els.video.srcObject = stream;
    await els.video.play();
    running = true;
    els.start.disabled = true; els.stop.disabled = false;
    els.cameraOverlay.classList.add('hidden'); els.streamStatus.textContent = 'Live';
    document.querySelector('.live-dot').classList.add('active');
    log('Camera started.');
    resizeCanvas(); requestAnimationFrame(loop);
  } catch (error) {
    setModelStatus(modelsLoaded ? 'Models ready' : 'Models unavailable', modelsLoaded ? 'ready' : 'error');
    log(`Camera error: ${error.message || error}`);
    alert(`Unable to start the webcam.\n\n${error.message || error}\n\nAllow camera access and use http://localhost instead of opening the HTML directly.`);
  }
}

function stopCamera() {
  running = false; detecting = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null; els.video.srcObject = null;
  els.overlay.getContext('2d').clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.start.disabled = !modelsLoaded; els.stop.disabled = true;
  els.cameraOverlay.classList.remove('hidden'); els.streamStatus.textContent = 'Idle';
  document.querySelector('.live-dot').classList.remove('active'); els.faceCount.textContent = '0'; renderResults([]);
}

function resizeCanvas() { if (els.video.videoWidth) { els.overlay.width = els.video.videoWidth; els.overlay.height = els.video.videoHeight; } }
function dominantExpression(expressions) {
  const entries = Object.entries(expressions || {}).sort((a,b) => b[1] - a[1]);
  if (!entries.length) return ['Unknown', 0];
  const labels = {happy:'Happy',sad:'Sad',angry:'Angry',neutral:'Neutral',surprised:'Surprised',fearful:'Fearful',disgusted:'Disgusted'};
  return [labels[entries[0][0]] || entries[0][0], entries[0][1]];
}
function classifyFaceShape(landmarks, box) {
  const jaw = landmarks.getJawOutline(); if (!jaw || jaw.length < 17) return ['Unknown', 0];
  const w = Math.max(1, box.width), h = Math.max(1, box.height);
  const jawW = Math.hypot(jaw[16].x-jaw[0].x, jaw[16].y-jaw[0].y)/w;
  const cheekW = Math.hypot(jaw[14].x-jaw[2].x, jaw[14].y-jaw[2].y)/w;
  const ratio = h/w, taper = cheekW-jawW;
  if (ratio < 1.05 && jawW > .78) return ['Round', .60];
  if (ratio > 1.32 && taper > .08) return ['Oval', .58];
  if (jawW > .82 && ratio < 1.28) return ['Square', .55];
  if (taper > .14) return ['Heart', .52];
  if (taper > .06 && ratio > 1.20) return ['Diamond', .50];
  return ['Oval', .42];
}
function estimateFeatureHints(video, box) {
  const c = document.createElement('canvas'), max = 180; const sx=Math.max(0,Math.floor(box.x)), sy=Math.max(0,Math.floor(box.y));
  const sw=Math.min(video.videoWidth-sx,Math.floor(box.width)), sh=Math.min(video.videoHeight-sy,Math.floor(box.height)); if(sw<=0||sh<=0)return {glasses:'Unavailable',beard:'Unavailable'};
  const scale=Math.min(max/sw,max/sh,1); c.width=Math.max(1,Math.round(sw*scale)); c.height=Math.max(1,Math.round(sh*scale)); const ctx=c.getContext('2d'); ctx.drawImage(video,sx,sy,sw,sh,0,0,c.width,c.height);
  const d=ctx.getImageData(0,0,c.width,c.height).data, lum=(r,g,b)=>.2126*r+.7152*g+.0722*b;
  const avg=(x1,y1,x2,y2)=>{let s=0,n=0;for(let y=Math.floor(y1*c.height);y<y2*c.height;y+=Math.max(2,Math.floor(c.height/20)))for(let x=Math.floor(x1*c.width);x<x2*c.width;x+=Math.max(2,Math.floor(c.width/20))){let i=(y*c.width+x)*4;s+=lum(d[i],d[i+1],d[i+2]);n++;}return n?s/n:0;};
  return { glasses: avg(.18,.34,.82,.54)<70?'Possible':'Not obvious', beard: avg(.20,.52,.80,.72)<72 && avg(.28,.65,.72,.92)<78?'Possible':'Not obvious' };
}
function drawDetections(ds) {
  resizeCanvas(); const ctx=els.overlay.getContext('2d'); ctx.clearRect(0,0,els.overlay.width,els.overlay.height); ctx.save(); ctx.translate(els.overlay.width,0); ctx.scale(-1,1); ctx.lineWidth=3; ctx.font='700 14px system-ui';
  ds.forEach(d=>{const b=d.detection.box;ctx.strokeStyle='#70e1c4';ctx.strokeRect(b.x,b.y,b.width,b.height);const label=`${d.expression[0]} • ${Math.round(d.detection.score*100)}%`;const tw=ctx.measureText(label).width+14;ctx.fillStyle='rgba(7,10,15,.88)';ctx.fillRect(b.x,Math.max(0,b.y-29),tw,26);ctx.fillStyle='#eef4ff';ctx.fillText(label,b.x+7,Math.max(18,b.y-11));});ctx.restore();
}
function renderResults(items) {
  els.faceCount.textContent=String(items.length);
  if(!items.length){els.results.className='results empty';els.results.innerHTML='<div class="empty-state"><div class="empty-icon">⌕</div><strong>No face detected</strong><span>Keep your face visible and well lit.</span></div>';return;}
  els.results.className='results'; els.results.innerHTML=items.map((item,i)=>`<article class="face-card"><div class="face-title"><strong>Face ${i+1}</strong><span class="confidence">DETECTION ${Math.round(item.detection.score*100)}%</span></div><div class="metrics"><div class="metric"><div class="label">Age estimate</div><div class="value">${Math.round(item.age)} years</div></div><div class="metric"><div class="label">Model class</div><div class="value">${escapeHtml(item.gender)}</div></div><div class="metric"><div class="label">Mood</div><div class="value">${escapeHtml(item.expression[0])}</div></div><div class="metric"><div class="label">Face shape</div><div class="value">${escapeHtml(item.faceShape[0])}</div></div></div><div class="feature-list"><div class="feature"><span>Expression confidence</span><span>${Math.round(item.expression[1]*100)}%</span></div><div class="feature"><span>Glasses heuristic</span><span>${escapeHtml(item.features.glasses)}</span></div><div class="feature"><span>Beard heuristic</span><span>${escapeHtml(item.features.beard)}</span></div></div></article>`).join('');
}
async function detect() {
  if(!running||detecting||!els.video.videoWidth)return; const now=performance.now(); if(now-lastDetectionAt<DETECTION_INTERVAL_MS)return; lastDetectionAt=now; detecting=true;
  try { const opts=new faceapi.TinyFaceDetectorOptions({inputSize:INPUT_SIZE,scoreThreshold:SCORE_THRESHOLD}); const ds=await faceapi.detectAllFaces(els.video,opts).withFaceLandmarks().withAgeAndGender().withFaceExpressions(); const enriched=ds.map(d=>({ ...d, expression:dominantExpression(d.expressions), faceShape:classifyFaceShape(d.landmarks,d.detection.box), features:estimateFeatureHints(els.video,d.detection.box), gender:d.gender==='male'?'Male':d.gender==='female'?'Female':'Uncertain'})); drawDetections(enriched);renderResults(enriched);frames++;const t=performance.now();if(t-lastFpsAt>=1000){els.fps.textContent=`${frames} FPS`;frames=0;lastFpsAt=t;}}
  catch(e){log(`Detection error: ${e.message||e}`);} finally{detecting=false;}
}
function loop(){if(!running)return;detect().finally(()=>requestAnimationFrame(loop));}

els.start.addEventListener('click',startCamera); els.stop.addEventListener('click',stopCamera); els.retry.addEventListener('click',()=>loadModels()); window.addEventListener('resize',resizeCanvas); window.addEventListener('beforeunload',stopCamera);
setModelStatus('Ready to load'); log('Click Start Camera. Models will load only when needed.');
