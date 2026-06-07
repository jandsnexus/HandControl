// =====================================================================
//  Anatomie Explorer — gemeinsame Engine
//  Eine Datei für alle Viewer. Jede Organseite ruft nur mountViewer() auf.
// =====================================================================
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12";

// ---------------------------------------------------------------------
//  1) One-Euro-Filter  — glättet die zittrigen Handdaten (Präzision!)
// ---------------------------------------------------------------------
class LowPass {
  constructor() { this.s = null; }
  filter(x, a) { this.s = (this.s === null) ? x : a * x + (1 - a) * this.s; return this.s; }
}
class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.02, dCutoff = 1.0) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.xPrev = null; this.tPrev = null;
    this.xF = new LowPass(); this.dxF = new LowPass();
  }
  alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(x, t) {
    if (this.tPrev === null) { this.tPrev = t; this.xPrev = x; return x; }
    let dt = (t - this.tPrev) / 1000; if (dt <= 0) dt = 1 / 60;
    const dx = (x - this.xPrev) / dt;
    const edx = this.dxF.filter(dx, this.alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const fx = this.xF.filter(x, this.alpha(cutoff, dt));
    this.xPrev = x; this.tPrev = t;
    return fx;
  }
}

// ---------------------------------------------------------------------
//  2) Hilfsfunktionen für die prozeduralen Modelle
// ---------------------------------------------------------------------
function tagPart(obj, partId, label, info) {
  obj.traverse(o => { o.userData.part = { partId, label, info }; });
  obj.userData.part = { partId, label, info };
  return obj;
}
function wrinkle(geo, amp, freq) {
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const n = Math.sin(x * freq) * Math.cos(y * freq) * Math.sin(z * freq);
    const f = 1 + amp * n;
    pos.setXYZ(i, x * f, y * f, z * f);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
}

// ---------------------------------------------------------------------
//  3) Modell-Bauer (alles aus Code, lädt sofort, keine Downloads nötig)
// ---------------------------------------------------------------------
function buildDNA() {
  const g = new THREE.Group();
  const N = 26, step = 0.34, radius = 1.15, turn = 0.42;
  const backbone = new THREE.Group();
  const rungs = new THREE.Group();

  const sphGeo = new THREE.SphereGeometry(0.14, 16, 16);
  const matA = new THREE.MeshStandardMaterial({ color: 0x2dd4bf, metalness: 0.2, roughness: 0.4 });
  const matB = new THREE.MeshStandardMaterial({ color: 0x7c9cff, metalness: 0.2, roughness: 0.4 });
  const baseColors = [0xf5a524, 0xef4f6b, 0x9d7bff, 0x49c96d];

  for (let i = 0; i < N; i++) {
    const a = i * turn;
    const y = i * step - (N * step) / 2;
    const ax = Math.cos(a) * radius, az = Math.sin(a) * radius;
    const bx = Math.cos(a + Math.PI) * radius, bz = Math.sin(a + Math.PI) * radius;

    const s1 = new THREE.Mesh(sphGeo, matA); s1.position.set(ax, y, az); backbone.add(s1);
    const s2 = new THREE.Mesh(sphGeo, matB); s2.position.set(bx, y, bz); backbone.add(s2);

    const dx = bx - ax, dz = bz - az;
    const len = Math.sqrt(dx * dx + dz * dz);
    const rung = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.05, len, 8),
      new THREE.MeshStandardMaterial({ color: baseColors[i % baseColors.length], roughness: 0.5 })
    );
    rung.position.set((ax + bx) / 2, y, (az + bz) / 2);
    rung.rotation.z = Math.PI / 2;
    rung.rotation.y = -Math.atan2(dz, dx);
    rungs.add(rung);
  }
  tagPart(backbone, "backbone", "Zucker-Phosphat-Rückgrat",
    "Die beiden äußeren Stränge der Doppelhelix. Sie bestehen aus abwechselnd Zucker (Desoxyribose) und Phosphat und geben der DNA ihre Stabilität und Form.");
  tagPart(rungs, "bases", "Basenpaare",
    "Die Sprossen der Leiter. Jede Sprosse ist ein Paar aus zwei Basen (A-T oder G-C). Ihre Reihenfolge speichert die gesamte Erbinformation.");
  g.add(backbone); g.add(rungs);
  return { group: g, axis: "y", zoom: 7, spin: 0.004 };
}

function buildBrainReal() {
  const g = new THREE.Group();
  const mkMat = (c) => new THREE.MeshStandardMaterial({ color: c, metalness: 0.0, roughness: 0.8 });

  function hemisphere(sign) {
    const geo = new THREE.IcosahedronGeometry(1.15, 5);
    wrinkle(geo, 0.10, 9);
    const m = new THREE.Mesh(geo, mkMat(0xe8a0a8));
    m.scale.set(0.9, 0.85, 1.1);
    m.position.set(sign * 0.62, 0.25, 0);
    return m;
  }
  const left = hemisphere(-1);
  const right = hemisphere(1);
  tagPart(left, "hemL", "Linke Großhirnhälfte",
    "Steuert vor allem die rechte Körperseite. Bei den meisten Menschen sitzen hier Sprache und logisches Denken.");
  tagPart(right, "hemR", "Rechte Großhirnhälfte",
    "Steuert vor allem die linke Körperseite. Stark beteiligt an räumlichem Denken, Kreativität und Musik.");

  const cgeo = new THREE.IcosahedronGeometry(0.6, 4);
  wrinkle(cgeo, 0.06, 16);
  const cere = new THREE.Mesh(cgeo, mkMat(0xd98a93));
  cere.scale.set(1.25, 0.7, 0.9);
  cere.position.set(0, -0.85, -0.85);
  tagPart(cere, "cere", "Kleinhirn (Cerebellum)",
    "Koordiniert Bewegungen, Gleichgewicht und Feinmotorik. Klein, aber enthält mehr als die Hälfte aller Nervenzellen des Gehirns.");

  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22, 0.32, 1.0, 20),
    mkMat(0xc77a82)
  );
  stem.position.set(0, -1.25, -0.35);
  stem.rotation.x = 0.35;
  tagPart(stem, "stem", "Hirnstamm",
    "Verbindet Gehirn und Rückenmark. Steuert lebenswichtige Funktionen wie Atmung, Herzschlag und Schlaf-Wach-Rhythmus.");

  g.add(left, right, cere, stem);
  return { group: g, axis: "y", zoom: 6, spin: 0.003 };
}

function buildHeart() {
  const g = new THREE.Group();
  const mkMat = (c) => new THREE.MeshStandardMaterial({ color: c, metalness: 0.1, roughness: 0.55 });

  const bodyGeo = new THREE.SphereGeometry(1.1, 32, 32);
  const body = new THREE.Mesh(bodyGeo, mkMat(0xc0303a));
  body.scale.set(1.0, 1.15, 0.9);
  body.position.y = -0.2;
  tagPart(body, "ventricles", "Herzkammern (Ventrikel)",
    "Die beiden unteren, muskelstärksten Räume. Die linke Kammer pumpt das Blut in den ganzen Körper, die rechte in die Lunge.");

  const atriaL = new THREE.Mesh(new THREE.SphereGeometry(0.55, 24, 24), mkMat(0xd6515a));
  atriaL.position.set(-0.5, 0.75, 0);
  const atriaR = new THREE.Mesh(new THREE.SphereGeometry(0.55, 24, 24), mkMat(0xd6515a));
  atriaR.position.set(0.5, 0.75, 0);
  const atria = new THREE.Group(); atria.add(atriaL, atriaR);
  tagPart(atria, "atria", "Vorhöfe (Atrien)",
    "Die beiden oberen Räume. Sie nehmen das zurückströmende Blut auf und geben es an die Kammern weiter.");

  const aorta = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.32, 1.3, 20), mkMat(0xb83b54));
  aorta.position.set(-0.2, 1.4, 0);
  aorta.rotation.z = 0.25;
  const aortaArch = new THREE.Mesh(new THREE.TorusGeometry(0.35, 0.16, 16, 24, Math.PI), mkMat(0xb83b54));
  aortaArch.position.set(0.1, 1.9, 0);
  aortaArch.rotation.z = -0.4;
  const vessels = new THREE.Group(); vessels.add(aorta, aortaArch);
  tagPart(vessels, "aorta", "Aorta (Hauptschlagader)",
    "Die größte Arterie des Körpers. Sie führt das sauerstoffreiche Blut aus der linken Kammer in den Kreislauf.");

  g.add(body, atria, vessels);
  return { group: g, axis: "y", zoom: 7, spin: 0.004 };
}

function buildLungs() {
  const g = new THREE.Group();
  const mkMat = (c) => new THREE.MeshStandardMaterial({ color: c, metalness: 0.0, roughness: 0.7 });

  function lung(sign) {
    const grp = new THREE.Group();
    const main = new THREE.Mesh(new THREE.SphereGeometry(0.85, 28, 28), mkMat(0xe0959c));
    main.scale.set(0.75, 1.4, 0.8);
    main.position.set(sign * 0.95, -0.2, 0);
    const lobe = new THREE.Mesh(new THREE.SphereGeometry(0.55, 24, 24), mkMat(0xe0959c));
    lobe.scale.set(0.8, 1.0, 0.8);
    lobe.position.set(sign * 0.95, 0.85, 0);
    grp.add(main, lobe);
    return grp;
  }
  const lL = lung(-1), lR = lung(1);
  tagPart(lL, "lungL", "Linker Lungenflügel",
    "Etwas kleiner als der rechte, weil das Herz Platz braucht. Besteht aus zwei Lappen.");
  tagPart(lR, "lungR", "Rechter Lungenflügel",
    "Der größere Flügel mit drei Lappen. Hier findet ein Großteil des Gasaustauschs statt.");

  const trachea = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 1.2, 18), mkMat(0xcf7d86));
  trachea.position.set(0, 1.2, 0);
  const bronchL = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 0.8, 14), mkMat(0xcf7d86));
  bronchL.position.set(-0.45, 0.55, 0); bronchL.rotation.z = 0.7;
  const bronchR = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 0.8, 14), mkMat(0xcf7d86));
  bronchR.position.set(0.45, 0.55, 0); bronchR.rotation.z = -0.7;
  const airways = new THREE.Group(); airways.add(trachea, bronchL, bronchR);
  tagPart(airways, "trachea", "Luftröhre & Bronchien",
    "Die Luftröhre teilt sich in zwei Hauptbronchien, die in die Lungenflügel führen und sich dort immer feiner verzweigen.");

  g.add(lL, lR, airways);
  return { group: g, axis: "y", zoom: 7, spin: 0.0035 };
}

function buildCrystal() {
  const g = new THREE.Group();
  const geo = new THREE.IcosahedronGeometry(1.6, 1);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color: 0x2dd4bf, metalness: 0.3, roughness: 0.35, flatShading: true
  }));
  tagPart(mesh, "crystal", "Testobjekt",
    "Ein einfaches geometrisches Objekt zum Ausprobieren der Steuerung.");
  const wire = new THREE.LineSegments(
    new THREE.WireframeGeometry(geo),
    new THREE.LineBasicMaterial({ color: 0xaef7ec, transparent: true, opacity: 0.25 })
  );
  g.add(mesh, wire);
  return { group: g, axis: "y", zoom: 6, spin: 0.0035 };
}

const PRESETS = {
  dna:     { title: "DNA Doppelhelix", subtitle: "Rückgrat · Basenpaare", build: buildDNA },
  brain:   { title: "Menschliches Gehirn", subtitle: "Großhirn · Kleinhirn · Hirnstamm", build: buildBrainReal },
  heart:   { title: "Menschliches Herz", subtitle: "Kammern · Vorhöfe · Aorta", build: buildHeart },
  lungs:   { title: "Menschliche Lunge", subtitle: "Lungenflügel · Atemwege", build: buildLungs },
  crystal: { title: "Testobjekt", subtitle: "Phase 1 Prototyp", build: buildCrystal }
};

// ---------------------------------------------------------------------
//  4) CSS  (wird in die Seite eingefügt)
// ---------------------------------------------------------------------
const CSS = `
:root{
  --bg:#070b0f; --bg2:#0d141b; --accent:#2dd4bf; --accent-dim:#155e57;
  --text:#e6f1ef; --text-dim:#6b8a86; --warn:#f5a524; --line:rgba(45,212,191,.18);
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace}
#ae-scene{position:fixed;inset:0;z-index:1}
body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(circle at 50% 45%,rgba(45,212,191,.10),transparent 55%),radial-gradient(circle at 80% 90%,rgba(45,212,191,.05),transparent 50%)}
.ae-hud{position:fixed;z-index:10;pointer-events:none}
#ae-top{top:0;left:0;right:0;display:flex;align-items:center;justify-content:space-between;padding:16px 22px}
.ae-brand{display:flex;align-items:center;gap:12px}
.ae-back{pointer-events:auto;text-decoration:none;color:var(--text-dim);font-size:18px;border:1px solid var(--line);border-radius:8px;width:34px;height:34px;display:flex;align-items:center;justify-content:center;transition:.2s}
.ae-back:hover{color:var(--accent);border-color:var(--accent)}
.ae-dot{width:9px;height:9px;border-radius:50%;background:var(--text-dim);transition:.3s}
.ae-dot.live{background:var(--accent);box-shadow:0 0 12px 1px var(--accent);animation:aepulse 2s infinite}
@keyframes aepulse{0%,100%{opacity:1}50%{opacity:.4}}
.ae-brand h1{font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase}
.ae-brand span{font-size:11px;color:var(--text-dim);letter-spacing:1px}
#ae-status{font-size:11px;color:var(--text-dim);letter-spacing:1px;text-align:right;line-height:1.6}
#ae-status b{color:var(--accent);font-weight:500}
#ae-legend{bottom:20px;left:22px;font-size:11px;color:var(--text-dim);line-height:1.9;letter-spacing:.5px}
#ae-legend .k{color:var(--text)} #ae-legend .sep{color:var(--accent-dim)}
#ae-cam{bottom:20px;right:22px;width:200px;height:150px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--bg2)}
#ae-cam.hidden{display:none}
#ae-video,#ae-overlay{position:absolute;inset:0;width:100%;height:100%;transform:scaleX(-1)}
#ae-video{object-fit:cover}
#ae-camlabel{position:absolute;top:6px;left:8px;z-index:2;font-size:9px;letter-spacing:1px;color:var(--accent);text-shadow:0 0 6px var(--bg)}
#ae-tools{top:64px;right:22px;display:flex;flex-direction:column;gap:8px;align-items:flex-end}
.ae-btn{pointer-events:auto;font-family:inherit;font-size:11px;letter-spacing:1px;color:var(--text);background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer;display:flex;align-items:center;gap:6px;transition:.2s}
.ae-btn:hover{border-color:var(--accent);color:var(--accent)}
.ae-btn.active{border-color:var(--accent);color:var(--accent)}
#ae-panel{position:fixed;z-index:12;top:50%;right:22px;transform:translateY(-50%);width:280px;background:var(--bg2);border:1px solid var(--line);border-radius:12px;padding:18px;opacity:0;pointer-events:none;transition:opacity .25s}
#ae-panel.show{opacity:1;pointer-events:auto}
#ae-panel .ph{display:flex;align-items:center;gap:8px;margin-bottom:10px}
#ae-panel .ph .pdot{width:9px;height:9px;border-radius:50%;background:var(--warn)}
#ae-panel h3{font-size:15px;font-weight:700}
#ae-panel .pinfo{font-size:12px;color:var(--text-dim);line-height:1.65;margin-bottom:14px}
#ae-panel label{font-size:10px;letter-spacing:1px;color:var(--accent);text-transform:uppercase}
#ae-panel textarea{width:100%;margin-top:6px;background:var(--bg);border:1px solid var(--line);border-radius:8px;color:var(--text);font-family:inherit;font-size:12px;padding:8px;resize:vertical;min-height:64px}
#ae-panel textarea:focus{outline:none;border-color:var(--accent)}
#ae-pclose{position:absolute;top:12px;right:14px;cursor:pointer;color:var(--text-dim);background:none;border:none;font-size:18px;font-family:inherit}
#ae-pclose:hover{color:var(--text)}
#ae-saved{font-size:10px;color:var(--accent);margin-top:6px;height:12px;letter-spacing:1px}
#ae-gate{position:fixed;inset:0;z-index:30;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;text-align:center;padding:24px;background:radial-gradient(circle at 50% 40%,rgba(13,20,27,.6),var(--bg));backdrop-filter:blur(2px)}
#ae-gate.gone{opacity:0;pointer-events:none;transition:opacity .6s}
#ae-gate h2{font-size:22px;font-weight:700;letter-spacing:.5px}
#ae-gate p{font-size:13px;color:var(--text-dim);max-width:420px;line-height:1.7}
#ae-start{pointer-events:auto;font-family:inherit;font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--bg);background:var(--accent);border:none;border-radius:8px;padding:14px 34px;cursor:pointer;box-shadow:0 0 30px rgba(45,212,191,.4);transition:.2s}
#ae-start:hover{transform:translateY(-2px);box-shadow:0 0 44px rgba(45,212,191,.6)}
#ae-start:disabled{opacity:.5;cursor:wait}
#ae-skip{pointer-events:auto;background:none;border:none;cursor:pointer;color:var(--text-dim);font-family:inherit;font-size:11px;letter-spacing:1px;text-decoration:underline;text-underline-offset:4px}
#ae-skip:hover{color:var(--text)}
#ae-err{color:var(--warn);font-size:12px;min-height:16px;max-width:420px}
@media(max-width:640px){#ae-cam{width:130px;height:98px}#ae-panel{width:calc(100% - 44px);top:auto;bottom:180px;transform:none}}
`;

// ---------------------------------------------------------------------
//  5) Haupt-Funktion
// ---------------------------------------------------------------------
export function mountViewer(config) {
  const preset = PRESETS[config.preset] || PRESETS.crystal;

  // --- CSS einfügen ---
  const styleEl = document.createElement("style");
  styleEl.textContent = CSS;
  document.head.appendChild(styleEl);

  // --- DOM aufbauen ---
  document.body.innerHTML = `
  <div id="ae-scene"></div>
  <div id="ae-top" class="ae-hud">
    <div class="ae-brand">
      <a class="ae-back" href="index.html" title="Zur Übersicht">&#8592;</a>
      <div class="ae-dot" id="ae-dot"></div>
      <div><h1>${preset.title}</h1><span>${preset.subtitle}</span></div>
    </div>
    <div id="ae-status">Kamera: <b id="ae-camstate">aus</b><br>Hände: <b id="ae-handstate">0</b></div>
  </div>
  <div id="ae-tools" class="ae-hud">
    <button class="ae-btn active" id="ae-rotbtn">&#8635; Auto-Drehung</button>
    <button class="ae-btn" id="ae-resetbtn">&#8634; Ansicht zurück</button>
  </div>
  <div id="ae-legend" class="ae-hud">
    <div><span class="k">Pinch</span> (Daumen+Zeige) <span class="sep">&#8594;</span> drehen</div>
    <div><span class="k">2 Hände</span> <span class="sep">&#8594;</span> zoomen</div>
    <div><span class="k">Klick</span> auf ein Teil <span class="sep">&#8594;</span> Info</div>
  </div>
  <div id="ae-cam" class="ae-hud hidden">
    <span id="ae-camlabel">LIVE</span>
    <video id="ae-video" autoplay playsinline muted></video>
    <canvas id="ae-overlay"></canvas>
  </div>
  <div id="ae-panel">
    <button id="ae-pclose">&times;</button>
    <div class="ph"><span class="pdot"></span><h3 id="ae-ptitle"></h3></div>
    <p class="pinfo" id="ae-pinfo"></p>
    <label>Deine Notiz</label>
    <textarea id="ae-pnote" placeholder="z.B. für die Klausur merken..."></textarea>
    <div id="ae-saved"></div>
  </div>
  <div id="ae-gate">
    <h2>${preset.title}</h2>
    <p>Diese Seite öffnet deine Kamera und erkennt deine Hand. Pinch zum Drehen, zwei Hände zum Zoomen, Klick auf ein Teil für Infos. Es wird nichts aufgenommen — alles läuft live im Browser.</p>
    <button id="ae-start">Kamera starten</button>
    <button id="ae-skip">Ohne Kamera ansehen (nur Maus)</button>
    <div id="ae-err"></div>
  </div>`;

  // --- Three.js Szene ---
  const sceneEl = document.getElementById("ae-scene");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  sceneEl.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);

  const built = preset.build();
  const group = built.group;
  scene.add(group);
  camera.position.set(0, 0, built.zoom);

  scene.add(new THREE.AmbientLight(0x88ffee, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(4, 5, 6); scene.add(key);
  const rim = new THREE.PointLight(0x2dd4bf, 2, 40); rim.position.set(-6, -3, 4); scene.add(rim);

  // --- gemeinsamer Eingabe-Zustand ---
  let targetRotX = 0, targetRotY = 0, targetZoom = built.zoom;
  const ZOOM_MIN = 3, ZOOM_MAX = 14;
  let autoRotate = true, interacting = false;

  // --- Maus-Fallback ---
  let dragging = false, lastMX = 0, lastMY = 0, moved = 0, downX = 0, downY = 0;
  renderer.domElement.addEventListener("pointerdown", e => {
    dragging = true; moved = 0; lastMX = downX = e.clientX; lastMY = downY = e.clientY;
  });
  window.addEventListener("pointerup", e => {
    if (dragging && moved < 6) tryPick(e.clientX, e.clientY); // Klick statt Drag
    dragging = false;
  });
  window.addEventListener("pointermove", e => {
    if (!dragging) return;
    moved += Math.abs(e.clientX - lastMX) + Math.abs(e.clientY - lastMY);
    interacting = true;
    targetRotY += (e.clientX - lastMX) * 0.006;
    targetRotX += (e.clientY - lastMY) * 0.006;
    targetRotX = Math.max(-1.4, Math.min(1.4, targetRotX));
    lastMX = e.clientX; lastMY = e.clientY;
  });
  renderer.domElement.addEventListener("wheel", e => {
    e.preventDefault(); interacting = true;
    targetZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, targetZoom + e.deltaY * 0.005));
  }, { passive: false });

  // --- Buttons ---
  const rotBtn = document.getElementById("ae-rotbtn");
  rotBtn.addEventListener("click", () => {
    autoRotate = !autoRotate;
    rotBtn.classList.toggle("active", autoRotate);
  });
  document.getElementById("ae-resetbtn").addEventListener("click", () => {
    targetRotX = 0; targetRotY = 0; targetZoom = built.zoom;
  });

  // --- Anklicken (Raycasting) + Info-Panel ---
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const panel = document.getElementById("ae-panel");
  const pTitle = document.getElementById("ae-ptitle");
  const pInfo = document.getElementById("ae-pinfo");
  const pNote = document.getElementById("ae-pnote");
  const pSaved = document.getElementById("ae-saved");
  let currentPartId = null;

  function tryPick(clientX, clientY) {
    mouse.x = (clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(group.children, true);
    for (const h of hits) {
      let o = h.object;
      while (o && !(o.userData && o.userData.part)) o = o.parent;
      if (o && o.userData.part) { openPanel(o.userData.part); return; }
    }
  }
  function noteKey(id) { return "anote:" + config.preset + ":" + id; }
  function openPanel(part) {
    currentPartId = part.partId;
    pTitle.textContent = part.label;
    pInfo.textContent = part.info;
    let saved = "";
    try { saved = localStorage.getItem(noteKey(part.partId)) || ""; } catch (e) {}
    pNote.value = saved;
    pSaved.textContent = "";
    panel.classList.add("show");
  }
  pNote.addEventListener("input", () => {
    if (!currentPartId) return;
    try {
      localStorage.setItem(noteKey(currentPartId), pNote.value);
      pSaved.textContent = "gespeichert";
    } catch (e) { pSaved.textContent = "speichern nicht möglich"; }
  });
  document.getElementById("ae-pclose").addEventListener("click", () => panel.classList.remove("show"));

  // --- Render-Schleife (sanftes Nachführen) ---
  function animate() {
    requestAnimationFrame(animate);
    if (autoRotate && !interacting) targetRotY += built.spin;
    group.rotation.x += (targetRotX - group.rotation.x) * 0.12;
    group.rotation.y += (targetRotY - group.rotation.y) * 0.12;
    camera.position.z += (targetZoom - camera.position.z) * 0.12;
    interacting = false;
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // --- Hand-Tracking ---
  const video = document.getElementById("ae-video");
  const overlay = document.getElementById("ae-overlay");
  const octx = overlay.getContext("2d");
  const camState = document.getElementById("ae-camstate");
  const handState = document.getElementById("ae-handstate");
  const dot = document.getElementById("ae-dot");
  const errEl = document.getElementById("ae-err");

  let handLandmarker = null, lastVideoTime = -1;
  let pinching = false, lastPX = 0, lastPY = 0;
  let twoStartDist = null, twoStartZoom = null;
  const fPX = new OneEuro(), fPY = new OneEuro(), fDist = new OneEuro(1.0, 0.01);

  const CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],
    [10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  const dist2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  async function startCamera() {
    errEl.textContent = "";
    try {
      const fileset = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm");
      handLandmarker = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO", numHands: 2
      });
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" } });
      video.srcObject = stream;
      await video.play();
      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
      document.getElementById("ae-cam").classList.remove("hidden");
      camState.textContent = "an";
      dot.classList.add("live");
      detectLoop();
    } catch (e) {
      console.error(e);
      errEl.textContent = "Kamera/Modell-Fehler: " + (e.message || e);
    }
  }

  function detectLoop() {
    if (handLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      handleHands(handLandmarker.detectForVideo(video, performance.now()));
    }
    requestAnimationFrame(detectLoop);
  }

  function handleHands(res) {
    const hands = res.landmarks || [];
    handState.textContent = hands.length;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    octx.strokeStyle = "rgba(45,212,191,.7)"; octx.fillStyle = "#aef7ec"; octx.lineWidth = 2;
    for (const lm of hands) {
      for (const [a, b] of CONN) {
        octx.beginPath();
        octx.moveTo(lm[a].x * overlay.width, lm[a].y * overlay.height);
        octx.lineTo(lm[b].x * overlay.width, lm[b].y * overlay.height);
        octx.stroke();
      }
      for (const p of lm) { octx.beginPath(); octx.arc(p.x * overlay.width, p.y * overlay.height, 3, 0, 6.28); octx.fill(); }
    }
    const now = performance.now();

    if (hands.length >= 2) {
      pinching = false;
      const d = fDist.filter(dist2(hands[0][0], hands[1][0]), now);
      if (twoStartDist === null) { twoStartDist = d; twoStartZoom = targetZoom; }
      interacting = true;
      targetZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, twoStartZoom * (twoStartDist / Math.max(d, 1e-4))));
      return;
    }
    twoStartDist = null;

    if (hands.length === 1) {
      const lm = hands[0];
      const handSize = Math.max(dist2(lm[0], lm[9]), 1e-4);
      const ratio = dist2(lm[4], lm[8]) / handSize;
      const px = fPX.filter(1 - (lm[4].x + lm[8].x) / 2, now);
      const py = fPY.filter((lm[4].y + lm[8].y) / 2, now);
      // Hysterese: greifen unter 0.40, loslassen erst über 0.55
      if (!pinching && ratio < 0.40) { pinching = true; lastPX = px; lastPY = py; }
      else if (pinching && ratio > 0.55) { pinching = false; }
      if (pinching) {
        interacting = true;
        targetRotY += (px - lastPX) * 5.0;
        targetRotX += (py - lastPY) * 5.0;
        targetRotX = Math.max(-1.4, Math.min(1.4, targetRotX));
        lastPX = px; lastPY = py;
      }
      return;
    }
    pinching = false;
  }

  // --- Start-Gate ---
  const gate = document.getElementById("ae-gate");
  const startBtn = document.getElementById("ae-start");
  document.getElementById("ae-skip").addEventListener("click", () => gate.classList.add("gone"));
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true; startBtn.textContent = "lädt…";
    await startCamera();
    if (!errEl.textContent) gate.classList.add("gone");
    startBtn.disabled = false; startBtn.textContent = "Kamera starten";
  });
}
