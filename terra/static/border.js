// Border watch: the whole UAE land border in 3D, the sensor line along it, and simulated incidents replayed
// from the detection chain. Click an incident (map marker, list or log) to fly in to its 3D replay.
// World frame: kilometres, X = east, Z = -north, Y = height (km); one group scaled in Y by the relief slider.
import * as THREE from "three";
import { MapControls } from "three/addons/controls/MapControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";

const $ = (id) => document.getElementById(id);
const LAT0 = 24.2, LON0 = 54.0;
const TIERS = {
  advisory: { tag: "DRONE HEARD", lvl: 1 }, border_watch: { tag: "NEAR THE BORDER", lvl: 2 }, on_course: { tag: "HEADING FOR A POST", lvl: 2 },
  approval: { tag: "NEEDS APPROVAL", lvl: 2 }, border_cross: { tag: "CROSSED THE BORDER", lvl: 3 }, take_cover: { tag: "TAKE COVER ORDERED", lvl: 3 },
  evacuate: { tag: "EVACUATE", lvl: 3 }, all_clear: { tag: "ALL CLEAR", lvl: 0 },
};
const CSS = getComputedStyle(document.documentElement);
const tierColor = (t) => CSS.getPropertyValue(`--t-${t}`).trim() || "#9fb7c9";
const CITIES = [
  ["ABU DHABI", 24.4539, 54.3773], ["DUBAI", 25.2048, 55.2708], ["SHARJAH", 25.3463, 55.4209], ["AL AIN", 24.2075, 55.7447],
  ["HATTA", 24.7967, 56.1167], ["RAS AL KHAIMAH", 25.7895, 55.9432], ["FUJAIRAH", 25.1288, 56.3265],
  ["MADINAT ZAYED", 23.6833, 53.7], ["LIWA", 23.14, 53.79], ["RUWAIS", 24.11, 52.73], ["GHUWAIFAT", 24.12, 51.62],
];
const FOREIGN = [["SAUDI ARABIA", 22.45, 52.6], ["OMAN", 23.55, 56.45], ["MUSANDAM (OM)", 26.12, 56.3]];

// ---------------------------------------------------------------- geometry helpers
function mpd(lat) {                       // metres per degree (lat, lon), WGS84 series as terra/dem.py
  const p = (lat * Math.PI) / 180;
  return [111132.92 - 559.82 * Math.cos(2 * p) + 1.175 * Math.cos(4 * p), 111412.84 * Math.cos(p) - 93.5 * Math.cos(3 * p)];
}
function proj(lon, lat) { const [a, b] = mpd(lat); return [((lon - LON0) * b) / 1000, (-(lat - LAT0) * a) / 1000]; }
function unproj(X, Z) {
  let lat = LAT0 - (Z * 1000) / mpd(LAT0)[0];
  lat = LAT0 - (Z * 1000) / mpd(lat)[0];
  return [LON0 + (X * 1000) / mpd(lat)[1], lat];
}
function local2ll(inc, x, y) { const [a, b] = mpd(inc.lat); return [inc.lon + x / b, inc.lat + y / a]; }
function L2W(inc, x, y, z) { const [lon, lat] = local2ll(inc, x, y); const [X, Z] = proj(lon, lat); return new THREE.Vector3(X, z / 1000, Z); }
function dms(v, pos, neg) {
  const s = v >= 0 ? pos : neg; v = Math.abs(v);
  const d = Math.floor(v), m = Math.floor((v - d) * 60), sec = Math.round(((v - d) * 60 - m) * 60);
  return `${String(d).padStart(pos === "E" ? 3 : 2, "0")}°${String(m).padStart(2, "0")}′${String(sec).padStart(2, "0")}″${s}`;
}
const fmtT = (s) => { s = Math.max(0, Math.floor(s)); return [s / 3600, (s % 3600) / 60, s % 60].map((v) => String(Math.floor(v)).padStart(2, "0")).join(":"); };

// ---------------------------------------------------------------- state
const S = {
  theatre: null, border: null, incidents: [], events: [], now: 0, len: 600, playing: false, speed: 15,
  exag: 3, focus: null, tiles: {}, logShown: 0, lastNow: -1, banner: null, sel: null,
};

// ---------------------------------------------------------------- three.js
const view = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
view.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
view.appendChild(labels.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x040706);
const camera = new THREE.PerspectiveCamera(42, 1, 0.005, 6000);
const controls = new MapControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.maxPolarAngle = 1.32; controls.minDistance = 0.25; controls.maxDistance = 1600; controls.zoomToCursor = true;
const world = new THREE.Group();
world.scale.y = S.exag;
scene.add(world);
const G = { theatre: new THREE.Group(), overlay: new THREE.Group(), local: new THREE.Group(), dyn: new THREE.Group(), build: new THREE.Group() };
Object.values(G).forEach((g) => world.add(g));
const fatMats = [];

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h); labels.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
  fatMats.forEach((m) => m.resolution.set(w, h));
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------- terrain shader (contours, lat/lon grid, coast)
const terrainVS = `
  attribute vec2 ll;
  varying float vH; varying vec2 vLL; varying vec3 vW;
  #include <common>
  #include <logdepthbuf_pars_vertex>
  void main() {
    vH = position.y * 1000.0; vLL = ll;
    vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
    #include <logdepthbuf_vertex>
  }`;
const terrainFS = `
  uniform float uMinor, uMajor, uGrid, uBright, uLand, uHoleOn; uniform vec4 uHole;
  varying float vH; varying vec2 vLL; varying vec3 vW;
  #include <logdepthbuf_pars_fragment>
  float iso(float v, float w) { float d = abs(fract(v - 0.5) - 0.5) / max(fwidth(v), 1e-6); return 1.0 - min(d / w, 1.0); }
  void main() {
    #include <logdepthbuf_fragment>
    if (uHoleOn > 0.5 && vLL.x > uHole.x && vLL.x < uHole.z && vLL.y > uHole.y && vLL.y < uHole.w) discard;
    vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
    float sh = clamp(dot(n, normalize(vec3(-0.55, 0.7, -0.45))), 0.0, 1.0);
    vec3 phos = vec3(0.22, 1.0, 0.53);
    vec3 col;
    if (vH <= 0.5) {
      col = vec3(0.010, 0.030, 0.045);
    } else {
      vec3 land = mix(vec3(0.040, 0.066, 0.050), vec3(0.150, 0.205, 0.140), smoothstep(0.0, 1600.0, vH));
      col = land * (0.32 + 1.0 * sh) * uLand;
      float c = max(iso(vH / uMinor, 1.0) * 0.16, iso(vH / uMajor, 1.3) * 0.42);
      col = mix(col, phos, c * uBright);
    }
    float coast = 1.0 - smoothstep(0.0, fwidth(vH) * 1.5 + 1e-4, abs(vH - 0.5));
    col = mix(col, vec3(0.24, 0.90, 0.82), coast * 0.75);
    if (uGrid > 0.5) {
      float g = max(max(iso(vLL.x * 4.0, 1.0), iso(vLL.y * 4.0, 1.0)) * 0.10, max(iso(vLL.x, 1.2), iso(vLL.y, 1.2)) * 0.30);
      col += phos * g * 0.45;
    }
    gl_FragColor = vec4(col, 1.0);
  }`;
function terrainMaterial(minor, major, bright, hole, landGain = 1) {
  return new THREE.ShaderMaterial({
    vertexShader: terrainVS, fragmentShader: terrainFS,
    uniforms: { uMinor: { value: minor }, uMajor: { value: major }, uGrid: { value: 1 }, uBright: { value: bright }, uLand: { value: landGain },
                uHoleOn: { value: 0 }, uHole: { value: new THREE.Vector4() } },
    polygonOffset: !hole, polygonOffsetFactor: hole ? 0 : -2, polygonOffsetUnits: hole ? 0 : -2,
  });
}
function gridMesh(rows, cols, vertex, mat) {
  const pos = new Float32Array(rows * cols * 3), ll = new Float32Array(rows * cols * 2);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const i = r * cols + c, v = vertex(r, c);
    pos.set([v[0], v[1], v[2]], i * 3); ll.set([v[3], v[4]], i * 2);
  }
  const idx = new Uint32Array((rows - 1) * (cols - 1) * 6);
  let k = 0;
  for (let r = 0; r < rows - 1; r++) for (let c = 0; c < cols - 1; c++) {
    const a = r * cols + c, b = a + 1, d = a + cols, e = d + 1;
    idx.set([a, d, b, b, d, e], k); k += 6;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("ll", new THREE.BufferAttribute(ll, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeBoundingSphere();
  return new THREE.Mesh(g, mat);
}
const b64f32 = (s) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); };

// theatre height sampler (m)
function hT(lon, lat) {
  const T = S.theatre; if (!T) return 0;
  const [lo0, , , la1] = T.bounds;
  const c = (lon - lo0) / T.step - 0.5, r = (la1 - lat) / T.step - 0.5;
  const c0 = Math.max(0, Math.min(T.cols - 2, Math.floor(c))), r0 = Math.max(0, Math.min(T.rows - 2, Math.floor(r)));
  const fc = Math.min(Math.max(c - c0, 0), 1), fr = Math.min(Math.max(r - r0, 0), 1), H = T.H, n = T.cols;
  return (H[r0 * n + c0] * (1 - fc) + H[r0 * n + c0 + 1] * fc) * (1 - fr) + (H[(r0 + 1) * n + c0] * (1 - fc) + H[(r0 + 1) * n + c0 + 1] * fc) * fr;
}
// local tile sampler (m), x/y metres from the incident origin
function hL(tile, x, y) {
  const ix = (x + tile.half_m) / tile.res_m, iy = (y + tile.half_m) / tile.res_m, n = tile.n;
  const c0 = Math.max(0, Math.min(n - 2, Math.floor(ix))), r0 = Math.max(0, Math.min(n - 2, Math.floor(iy)));
  const fc = Math.min(Math.max(ix - c0, 0), 1), fr = Math.min(Math.max(iy - r0, 0), 1), H = tile.H;
  return (H[r0 * n + c0] * (1 - fc) + H[r0 * n + c0 + 1] * fc) * (1 - fr) + (H[(r0 + 1) * n + c0] * (1 - fc) + H[(r0 + 1) * n + c0 + 1] * fc) * fr;
}
function groundW(X, Z) {                    // ground height (m) under a world point, local tile first
  const [lon, lat] = unproj(X, Z);
  const f = S.focus, tile = f && S.tiles[f.id];
  if (tile) {
    const [a, b] = mpd(f.lat), x = (lon - f.lon) * b, y = (lat - f.lat) * a;
    if (Math.abs(x) < tile.half_m && Math.abs(y) < tile.half_m) return hL(tile, x, y);
  }
  return hT(lon, lat);
}

// ---------------------------------------------------------------- theatre
let theatreMat, clusterMat;
function buildTheatre() {
  const T = S.theatre;
  const [lo0, , , la1] = T.bounds;
  theatreMat = terrainMaterial(100, 500, 0.9, true);
  const mesh = gridMesh(T.rows, T.cols, (r, c) => {
    const lat = la1 - (r + 0.5) * T.step, lon = lo0 + (c + 0.5) * T.step, [X, Z] = proj(lon, lat);
    return [X, T.H[r * T.cols + c] / 1000, Z, lon, lat];
  }, theatreMat);
  G.theatre.add(mesh);
  for (const [n, lat, lon] of CITIES) addLabel(G.overlay, n, "lbl city", lon, lat, hT(lon, lat) + 40);
  for (const [n, lat, lon] of FOREIGN) addLabel(G.overlay, n, "lbl foreign", lon, lat, hT(lon, lat) + 40);
}
function addLabel(group, text, cls, lon, lat, h) {
  const el = document.createElement("div"); el.className = cls; el.textContent = text;
  const o = new CSS2DObject(el); const [X, Z] = proj(lon, lat); o.position.set(X, h / 1000, Z); group.add(o); return o;
}
function fatLine(pts, color, width, opacity, depthTest) {
  const g = new LineGeometry(); g.setPositions(pts);
  const m = new LineMaterial({ color, linewidth: width, transparent: opacity < 1, opacity, depthTest, depthWrite: false });
  m.resolution.set(window.innerWidth, window.innerHeight); fatMats.push(m);
  const l = new Line2(g, m); l.computeLineDistances(); l.renderOrder = 7; return l;
}
const secLabels = [];
function buildBorder() {
  const B = S.border;
  for (const s of B.sectors) {
    const pts = [];
    // densify every ~1 km so the line follows the relief
    for (let i = 0; i < s.line.length - 1; i++) {
      const [a0, a1] = s.line[i], [b0, b1] = s.line[i + 1];
      const [ax, az] = proj(a0, a1), [bx, bz] = proj(b0, b1), n = Math.max(1, Math.ceil(Math.hypot(bx - ax, bz - az)));
      for (let j = 0; j < n; j++) { const f = j / n, lon = a0 + (b0 - a0) * f, lat = a1 + (b1 - a1) * f, [X, Z] = proj(lon, lat); pts.push(X, (hT(lon, lat) + 60) / 1000, Z); }
    }
    const [l0, l1] = s.line[s.line.length - 1], [X, Z] = proj(l0, l1); pts.push(X, (hT(l0, l1) + 60) / 1000, Z);
    s.lines = [fatLine(pts, 0xffb020, 7, 0.16, false), fatLine(pts, 0xffb020, 1.8, 0.95, false)];
    s.lines.forEach((l) => G.overlay.add(l));
    const m = s.line[Math.floor(s.line.length / 2)];
    secLabels.push(addLabel(G.overlay, s.id, "lbl sec", m[0], m[1], hT(m[0], m[1]) + 120));
  }
  // the sensor line: every deployed cluster as a point, hidden inside a focused incident's tile
  const n = B.clusters.length, pos = new Float32Array(n * 3), ll = new Float32Array(n * 2);
  B.clusters.forEach(([lon, lat], i) => { const [X, Z] = proj(lon, lat); pos.set([X, (hT(lon, lat) + 20) / 1000, Z], i * 3); ll.set([lon, lat], i * 2); });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("ll", new THREE.BufferAttribute(ll, 2));
  clusterMat = new THREE.ShaderMaterial({
    transparent: true, depthTest: false, depthWrite: false,
    uniforms: { uHoleOn: { value: 0 }, uHole: { value: new THREE.Vector4() }, uSize: { value: 2.0 } },
    vertexShader: `attribute vec2 ll; varying vec2 vLL; uniform float uSize;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main(){ vLL = ll; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); gl_PointSize = uSize;
      #include <logdepthbuf_vertex>
      }`,
    fragmentShader: `uniform float uHoleOn; uniform vec4 uHole; varying vec2 vLL;
      #include <logdepthbuf_pars_fragment>
      void main(){
      #include <logdepthbuf_fragment>
      if (uHoleOn > 0.5 && vLL.x > uHole.x && vLL.x < uHole.z && vLL.y > uHole.y && vLL.y < uHole.w) discard;
      gl_FragColor = vec4(0.24, 0.90, 0.82, 0.55); }`,
  });
  const pts = new THREE.Points(g, clusterMat); pts.renderOrder = 4; pts.frustumCulled = false;   // under the border line
  G.overlay.add(pts);
  S.clusterPoints = pts;
}

// ---------------------------------------------------------------- incidents and the watch timeline
function buildTimeline() {
  S.events = [];
  for (const inc of S.incidents) {
    inc.firstT = inc.alerts.length ? inc.alerts[0].t : null;
    inc.alerts.forEach((a) => S.events.push({ T: inc.t0 + a.t, inc, ...a }));
    // marker on the theatre map
    const el = document.createElement("div"); el.className = "mk"; el.innerHTML = `<div class="ring"></div><div class="t"></div>`;
    el.addEventListener("click", () => focus(inc));
    const [X, Z] = proj(inc.lon, inc.lat);
    inc.marker = new CSS2DObject(el); inc.marker.position.set(X, (hT(inc.lon, inc.lat) + 300) / 1000, Z); inc.marker.visible = false;
    G.overlay.add(inc.marker);
  }
  S.events.sort((a, b) => a.T - b.T);
  S.len = Math.ceil(Math.max(...S.incidents.map((i) => i.t0 + i.duration)) + 45);
  $("len").textContent = `WATCH ${fmtT(S.len)}`;
  const ticks = $("ticks");
  ticks.innerHTML = "";
  for (const inc of S.incidents) {
    const worst = worstTier(inc, Infinity);
    const d = document.createElement("div"); d.className = "tick";
    d.style.left = `${(100 * inc.t0) / S.len}%`; d.style.width = `${(100 * inc.duration) / S.len}%`;
    d.style.setProperty("--c", worst ? tierColor(worst) : "#3f5c4c");
    d.innerHTML = `<span>${inc.id}</span>`; ticks.appendChild(d);
  }
}
function worstTier(inc, upTo) {                 // highest-level tier sent by the incident up to its local time upTo
  let best = null;
  for (const a of inc.alerts) { if (a.t > upTo) break; if (TIERS[a.tier] && a.tier !== "all_clear" && (!best || TIERS[a.tier].lvl > TIERS[best].lvl)) best = a.tier; }
  return best;
}
function incidentState(inc, T) {
  const tl = T - inc.t0;
  if (tl < 0) return { phase: "pending" };
  if (tl > inc.duration + 30) return { phase: "closed", worst: worstTier(inc, Infinity) };
  const worst = inc.firstT !== null && tl >= inc.firstT ? worstTier(inc, tl) : null;
  return { phase: "active", worst, tl };
}

// ---------------------------------------------------------------- panels
function buildPanels() {
  const B = S.border;
  const st = $("stats");
  st.innerHTML = [[`${Math.round(B.length_km).toLocaleString()} KM`, "LAND BORDER"], [B.sectors.length, "SECTORS"],
    [B.clusters.length.toLocaleString(), "MIC CLUSTERS"], [`${B.spacing_m} M`, "SPACING, 2 ROWS"], ["0", "ACTIVE TRACKS"]]
    .map(([b, s], i) => `<div><b ${i === 4 ? 'id="nTracks"' : ""}>${b}</b><span>${s}</span></div>`).join("");
  $("secCount").textContent = `${B.sectors.length}`;
  const box = $("sectors"); box.innerHTML = "";
  let grp = null;
  for (const s of B.sectors) {
    if (s.group !== grp) {
      grp = s.group;
      const h = document.createElement("div"); h.className = "grp"; h.style.gridColumn = "1 / 4";
      h.textContent = (B.groups[grp] || grp).toUpperCase(); box.appendChild(h);
    }
    const d = document.createElement("button"); d.type = "button"; d.className = "sec"; d.id = `sec-${s.id}`;
    d.title = `${s.id}: ${s.length_km} km, ${s.clusters} clusters`;
    d.innerHTML = `<i></i>${s.id}`;
    d.addEventListener("click", () => flySector(s));
    box.appendChild(d);
  }
  const ib = $("incidents"); ib.innerHTML = "";
  for (const inc of S.incidents) {
    const d = document.createElement("button"); d.type = "button"; d.className = "inc"; d.id = `inc-${inc.id}`;
    d.innerHTML = `<span class="id">${inc.id}</span><span class="st">PENDING</span><span class="nm">${inc.sector} · ${inc.name}</span><span class="what"></span>`;
    d.addEventListener("click", () => { focus(inc); if (S.now < inc.t0 || S.now > inc.t0 + inc.duration) setNow(inc.t0 + Math.max(0, (inc.firstT ?? 0) - 5)); });
    ib.appendChild(d);
  }
}
function updatePanels() {
  const T = S.now;
  $("clock").textContent = `${fmtT(6 * 3600 + T)} GST`;
  $("watch").textContent = `WATCH T+${fmtT(T)}`;
  $("head").style.left = `${(100 * T) / S.len}%`;
  $("scrub").value = String(Math.round((1000 * T) / S.len));
  let con = 0;
  const secLvl = {};
  for (const inc of S.incidents) {
    const s = incidentState(inc, T), el = $(`inc-${inc.id}`), lvl = s.worst ? TIERS[s.worst].lvl : 0;
    if (s.phase === "active" && s.worst) { con = Math.max(con, lvl); secLvl[inc.sector] = Math.max(secLvl[inc.sector] || 0, lvl); }
    el.classList.toggle("live", s.phase === "active" && lvl > 0 && lvl < 3);
    el.classList.toggle("hot", s.phase === "active" && lvl >= 3);
    el.classList.toggle("sel", S.focus === inc);
    el.querySelector(".st").textContent = s.phase === "pending" ? `STARTS IN ${fmtT(inc.t0 - T).slice(3)}` : s.phase === "closed" ? "ENDED"
      : s.worst ? TIERS[s.worst].tag : "NOTHING HEARD YET";
    el.querySelector(".what").textContent = s.phase === "closed" ? inc.outcome : "";
    el.classList.toggle("done", s.phase === "closed");
    // map marker
    const show = s.phase === "active" && s.worst;
    inc.marker.visible = !!show && S.focus !== inc;
    if (show) {
      inc.marker.element.style.setProperty("--c", tierColor(s.worst));
      inc.marker.element.querySelector(".t").textContent = `${inc.id} · ${TIERS[s.worst].tag}`;
    }
  }
  for (const s of S.border.sectors) {
    const el = $(`sec-${s.id}`), l = secLvl[s.id] || 0;
    el.classList.toggle("watch", l > 0 && l < 3); el.classList.toggle("alert", l >= 3);
  }
  const c = $("con");
  c.className = `con ${con >= 3 ? "alert" : con > 0 ? "watch" : ""}`;
  $("conText").textContent = con >= 3 ? "ALERT" : con > 0 ? "WATCH" : "NOMINAL";
  updateLog();
}
function updateLog() {
  const T = S.now, log = $("log");
  if (T < S.lastNow) { log.innerHTML = ""; S.logShown = 0; }
  const shown = S.events.filter((e) => e.T <= T);
  for (let i = S.logShown; i < shown.length; i++) {
    const e = shown[i], d = document.createElement("div");
    const fresh = S.playing && S.lastNow >= 0 && e.T > S.lastNow && T >= S.lastNow;   // reached by playing forward, not by a jump back
    d.className = `ev${fresh ? " new" : ""}`;
    d.style.setProperty("--c", tierColor(e.tier));
    d.innerHTML = `<span class="tm">${fmtT(6 * 3600 + e.T)}</span><span class="tg">${TIERS[e.tier]?.tag || e.tier.toUpperCase()} · ${e.inc.sector}</span><span class="tx"></span>`;
    d.querySelector(".tx").textContent = e.text;
    d.addEventListener("click", () => focus(e.inc));
    log.prepend(d);
    if (fresh && ["border_cross", "take_cover", "border_watch", "evacuate"].includes(e.tier)) showBanner(e);
  }
  S.logShown = shown.length; S.lastNow = T;
  $("logCount").textContent = shown.length ? `${shown.length}` : "";
  if (!shown.length && !log.children.length) log.innerHTML = `<div class="empty">NO EVENTS. SENSOR LINE NOMINAL.</div>`;
  else log.querySelector(".empty")?.remove();
}
let bannerTimer = null;
function showBanner(e) {
  const b = $("banner");
  b.className = `banner${TIERS[e.tier].lvl < 3 ? " watch" : ""}`;
  $("bannerText").textContent = `${TIERS[e.tier].tag} · ${e.inc.sector} · ${e.inc.name.toUpperCase()}`;
  S.banner = e; b.hidden = false;
  clearTimeout(bannerTimer); bannerTimer = setTimeout(() => { b.hidden = true; }, 9000);
}
$("bannerGo").addEventListener("click", () => { if (S.banner) focus(S.banner.inc); $("banner").hidden = true; });

// ---------------------------------------------------------------- camera moves
let fly = null;
function flyTo(target, distance, polar = 0.85, azimuth = -0.5, dur = 1.6) {
  const off = new THREE.Vector3(Math.sin(polar) * Math.sin(azimuth), Math.cos(polar), Math.sin(polar) * Math.cos(azimuth)).multiplyScalar(distance);
  fly = { t: 0, dur, p0: camera.position.clone(), q0: controls.target.clone(), p1: target.clone().add(off), q1: target.clone() };
}
function stepFly(dt) {
  if (!fly) return;
  fly.t += dt; const u = Math.min(fly.t / fly.dur, 1), e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
  camera.position.lerpVectors(fly.p0, fly.p1, e); controls.target.lerpVectors(fly.q0, fly.q1, e);
  if (u >= 1) fly = null;
}
function home() {
  const [X, Z] = proj(54.25, 24.15);
  flyTo(new THREE.Vector3(X, 0, Z), 640, 0.62, -0.08, 1.8);
}
function flySector(s) {
  const m = s.line[Math.floor(s.line.length / 2)], [X, Z] = proj(m[0], m[1]);
  if (S.focus) leaveFocus(false);
  flyTo(new THREE.Vector3(X, (hT(m[0], m[1]) * S.exag) / 1000, Z), 55, 0.8, -0.4);
}

// ---------------------------------------------------------------- focus on an incident: local 3D replay
const L = { clusters: [], clusterLabels: [], bearings: null, tracks: new Map(), truth: null, truthPath: null };
async function loadTile(inc) {
  if (S.tiles[inc.id]) return S.tiles[inc.id];
  const r = await fetch(`/api/terrain?lat=${inc.lat}&lon=${inc.lon}&half_m=${inc.half_m}`);
  const t = await r.json();
  t.H = b64f32(t.h); delete t.h; delete t.texture;
  S.tiles[inc.id] = t; return t;
}
async function focus(inc, auto = false) {
  S.autoFocused = auto;
  if (S.focus === inc) { flyLocal(inc); return; }
  if (S.focus) leaveFocus(false);
  S.focus = inc; S.sel = null;
  $("crumbText").textContent = `THEATRE ▸ ${inc.sector} ▸ ${inc.id} · ${inc.name.toUpperCase()}`;
  $("back").hidden = false;
  const tile = await loadTile(inc);
  if (S.focus !== inc) return;
  showTile(inc, tile);
  buildIncidentObjects(inc, tile);
  applyLayers();
  flyLocal(inc);
  setAsk(inc); zetTabs();
  if (!$("zet").hidden) zetRun("first_heard");
  if (!inc.built) loadSiteData(inc, tile);
}

// the detailed terrain tile around a local origin, cut into the theatre map, with the border draped on it
function showTile(inc, tile) {
  // hole in the theatre mesh and the cluster points where the detailed tile sits
  const [lo0, la0] = local2ll(inc, -tile.half_m, -tile.half_m), [lo1, la1] = local2ll(inc, tile.half_m, tile.half_m);
  for (const m of [theatreMat, clusterMat]) { m.uniforms.uHole.value.set(lo0, la0, lo1, la1); m.uniforms.uHoleOn.value = 1; }
  for (const s of S.border.sectors) {           // the coarse border line would float over the detailed tile
    const near = s.line.some(([lon, lat]) => lon > lo0 - 0.01 && lon < lo1 + 0.01 && lat > la0 - 0.01 && lat < la1 + 0.01);
    if (near) s.lines.forEach((l) => { l.visible = false; });
  }
  const n = tile.n, mat = terrainMaterial(20, 100, 0.8, false, 2.6);   // brighter up close: relief is the point here
  G.local.add(gridMesh(n, n, (r, c) => {
    const x = -tile.half_m + c * tile.res_m, y = -tile.half_m + r * tile.res_m, [lon, lat] = local2ll(inc, x, y), [X, Z] = proj(lon, lat);
    return [X, tile.H[r * n + c] / 1000, Z, lon, lat];
  }, mat));
  // border inside the tile, draped
  for (const b of inc.border) {
    const pts = [];
    for (let i = 0; i < b.xy.length - 1; i++) {
      const [ax, ay] = b.xy[i], [bx, by] = b.xy[i + 1], k = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 40));
      for (let j = 0; j <= k; j++) { const x = ax + ((bx - ax) * j) / k, y = ay + ((by - ay) * j) / k; if (Math.abs(x) > tile.half_m || Math.abs(y) > tile.half_m) continue; const p = L2W(inc, x, y, hL(tile, x, y) + 6); pts.push(p.x, p.y, p.z); }
    }
    if (pts.length >= 6) { G.local.add(fatLine(pts, 0xffb020, 10, 0.18, true)); G.local.add(fatLine(pts, 0xffb020, 2.4, 1, true)); }
  }
}

// posts, clusters, bearings and the simulated truth of one incident or built scenario
function buildIncidentObjects(inc, tile) {
  // posts (protected sites)
  for (const s of inc.sites) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.92, 1, 48), new THREE.MeshBasicMaterial({ color: 0x9fb7c9, transparent: true, opacity: 0.7, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.scale.setScalar(s.radius_m / 1000);
    const p = L2W(inc, s.x, s.y, hL(tile, s.x, s.y) + 3); ring.position.copy(p); G.local.add(ring);
    const [lon, lat] = local2ll(inc, s.x, s.y); addLabel(G.local, s.name.toUpperCase(), "lbl post", lon, lat, hL(tile, s.x, s.y) + 40);
  }
  // clusters
  L.clusters = inc.clusters.map((c, k) => {
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.018), new THREE.MeshBasicMaterial({ color: 0x3ce6d0 }));
    m.position.copy(L2W(inc, c.x, c.y, c.z + 6)); G.local.add(m);
    const el = document.createElement("div"); el.className = "lbl clu"; el.textContent = `C${String(k + 1).padStart(2, "0")}`;
    const o = new CSS2DObject(el); o.position.copy(L2W(inc, c.x, c.y, c.z + 45)); G.local.add(o);
    return { mesh: m, label: o, p: m.position.clone() };
  });
  const bg = new THREE.BufferGeometry(); bg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(inc.clusters.length * 6), 3));
  L.bearings = new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: 0xffb020, transparent: true, opacity: 0.55, depthWrite: false }));
  L.bearings.frustumCulled = false; G.dyn.add(L.bearings);
  // simulated truth, one path and one marker per drone (hidden unless SIM TRUTH)
  L.truthPaths = []; L.truths = [];
  for (const d of inc.truth) {
    const tp = []; for (const p of d.p) { const w = L2W(inc, p[0], p[1], p[2]); tp.push(w.x, w.y, w.z); }
    if (tp.length >= 6) {
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(tp, 3));
      const line = new THREE.Line(g, new THREE.LineDashedMaterial({ color: 0xf2f2ea, dashSize: 0.05, gapSize: 0.04, transparent: true, opacity: 0.5 }));
      line.computeLineDistances(); G.local.add(line); L.truthPaths.push(line);
    }
    const te = document.createElement("div"); te.className = "lbl truth";
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), new THREE.MeshBasicMaterial({ color: 0xf2f2ea }));
    const label = new CSS2DObject(te); mesh.add(label); label.position.set(0, 0.03, 0);
    G.dyn.add(mesh); L.truths.push({ mesh, label });
  }
}

// ---------------------------------------------------------------- site data from Zetaris (shelters, units, posts)
async function loadSiteData(inc, tile) {
  const src = $("siteSrc");
  src.hidden = false; src.className = "src"; src.textContent = "SITE DATA · ZETARIS …";
  let d;
  try { d = await (await fetch(`/api/sitedata/${inc.id}`)).json(); } catch (e) { d = { source: `unavailable: ${e.message}`, sites: [], shelters: [], units: [] }; }
  if (S.focus !== inc) return;
  const z = d.source === "zetaris";
  src.className = `src${z ? "" : " off"}`;
  src.textContent = z ? `SITE DATA · ZETARIS · ${d.sites.length} POSTS · ${d.shelters.length} SHELTERS · ${d.units.length} UNITS`
                      : `SITE DATA · ${String(d.source).toUpperCase()}`;
  const inTile = (o) => Math.abs(o.x) < tile.half_m && Math.abs(o.y) < tile.half_m;
  const known = new Set(inc.sites.map((p) => p.name));
  for (const p of d.sites.filter((q) => inTile(q) && !known.has(q.name))) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.92, 1, 48), new THREE.MeshBasicMaterial({ color: 0x9fb7c9, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.scale.setScalar(p.radius_m / 1000); ring.position.copy(L2W(inc, p.x, p.y, hL(tile, p.x, p.y) + 3)); G.local.add(ring);
    const [lon, lat] = local2ll(inc, p.x, p.y); addLabel(G.local, p.name.toUpperCase(), "lbl post", lon, lat, hL(tile, p.x, p.y) + 40);
  }
  for (const sh of d.shelters.filter(inTile)) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.03), new THREE.MeshBasicMaterial({ color: 0x63a4ff }));
    m.position.copy(L2W(inc, sh.x, sh.y, hL(tile, sh.x, sh.y) + 15)); G.local.add(m);
    const [lon, lat] = local2ll(inc, sh.x, sh.y); addLabel(G.local, `${sh.name.toUpperCase()} · CAP ${sh.capacity}`, "lbl shel", lon, lat, hL(tile, sh.x, sh.y) + 55);
  }
  for (const u of d.units.filter(inTile)) {
    const m = new THREE.Mesh(new THREE.ConeGeometry(0.018, 0.04, 3), new THREE.MeshBasicMaterial({ color: 0x39ff88 }));
    m.position.copy(L2W(inc, u.x + 25, u.y + 25, hL(tile, u.x, u.y) + 20)); G.local.add(m);
    const [lon, lat] = local2ll(inc, u.x + 25, u.y + 25); addLabel(G.local, `${u.name.toUpperCase()} (${u.size})`, "lbl unit", lon, lat, hL(tile, u.x, u.y) + 70);
  }
}

// ---------------------------------------------------------------- ask Nemotron about the focused incident
function setAsk(inc) {
  $("askQ").disabled = !inc; $("askGo").disabled = !inc;
  const post = inc && inc.sites[0] ? inc.sites[0].name : "the nearest post";      // name it: "the post" is ambiguous
  const qs = ["Which units are near the drone, and how far away are they?",
    `Where is the nearest shelter for ${post}, and how long is the walk?`, `How long until the drone reaches ${post}?`];
  document.querySelectorAll("#chips button").forEach((b, i) => { b.dataset.q = qs[i]; });
  $("askQ").placeholder = inc ? `Ask about ${inc.id} at the current replay time` : "Open an incident, then ask about it";
  if (!inc) $("answer").innerHTML = "";
}
async function askNow(question) {
  const inc = S.focus; if (!inc || !question.trim()) return;
  const tl = Math.max(0, Math.min(inc.duration, S.now - inc.t0));
  const box = $("answer");
  box.innerHTML = `<div class="q"></div><div class="a wait">ASKING… (the model may take up to a minute)</div>`;
  box.querySelector(".q").textContent = `${inc.id} @ ${fmtT(6 * 3600 + inc.t0 + tl)} · ${question}`;
  $("askGo").disabled = true;
  let r;
  try { r = await (await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: inc.id, t: tl, question }) })).json(); }
  catch (e) { r = { error: e.message }; }
  $("askGo").disabled = !S.focus;
  const a = box.querySelector(".a"); a.classList.remove("wait");
  if (r.error) { a.textContent = `Could not ask: ${r.error}`; return; }
  a.textContent = r.answer || "(no answer)";
  const tools = (r.tools || []).map((x) => x.tool).join(", ") || "none";
  box.insertAdjacentHTML("beforeend", `<div class="meta">MODEL <b></b> · VIA <b></b> · SITE DATA <b></b> · LOOKUPS <b></b></div>`);
  const bs = box.querySelectorAll(".meta b");
  const via = { "meterless-nightshift": "METERLESS NIGHTSHIFT", "local-ollama": "THIS PC (OLLAMA)", "self-hosted": "SELF-HOSTED" };
  bs[0].textContent = r.fallback ? "NONE (model busy: system facts)" : String(r.model).toUpperCase();
  bs[1].textContent = via[r.route] || String(r.route || "?").toUpperCase();
  bs[2].textContent = String(r.site_source).toUpperCase(); bs[3].textContent = tools.toUpperCase();
  if (r.unverified && r.unverified.length) box.insertAdjacentHTML("beforeend", `<div class="warn">Not found in the data, check: ${r.unverified.map((x) => String(x).replace(/[<>&]/g, "")).join(", ")}</div>`);
}
$("askForm").addEventListener("submit", (e) => { e.preventDefault(); askNow($("askQ").value); });
document.querySelectorAll("#chips button").forEach((b) => b.addEventListener("click", () => { if (!S.focus) return; $("askQ").value = b.dataset.q; askNow(b.dataset.q); }));
function flyLocal(inc) {
  const tile = S.tiles[inc.id]; if (!tile) return;
  const c = L2W(inc, 0, 0, hL(tile, 0, 0)); c.y *= S.exag;
  flyTo(c, (inc.half_m * 2.1) / 1000, 0.95, -0.35, 2.0);
}
function clearGroup(g) {
  for (const o of [...g.children]) {
    o.traverse((x) => { if (x.isCSS2DObject) x.element.remove(); x.geometry?.dispose(); if (x.material) { const i = fatMats.indexOf(x.material); if (i >= 0) fatMats.splice(i, 1); x.material.dispose?.(); } });
    g.remove(o);
  }
}
function leaveFocus(goHome = true) {
  S.focus = null; S.sel = null;
  clearGroup(G.local); clearGroup(G.dyn); L.tracks.clear(); L.clusters = [];
  theatreMat.uniforms.uHoleOn.value = 0; clusterMat.uniforms.uHoleOn.value = 0;
  S.border.sectors.forEach((s) => s.lines.forEach((l) => { l.visible = true; }));
  $("crumbText").textContent = "THEATRE · UAE"; $("back").hidden = true; $("card").hidden = true;
  $("siteSrc").hidden = true; setAsk(null); zetTabs(); $("caption").hidden = true;
  if (!$("zet").hidden) zetRun("overview");
  if (goHome) home();
}
$("back").addEventListener("click", () => { if (S.mode === "scenario") backToBuild(); else leaveFocus(true); });

// ---------------------------------------------------------------- local replay, every animation frame
function trackObj(id) {
  if (L.tracks.has(id)) return L.tracks.get(id);
  const g = new THREE.Group();
  const red = 0xff3b3b;
  const mk = new THREE.Mesh(new THREE.OctahedronGeometry(0.03), new THREE.MeshBasicMaterial({ color: red }));
  const ell = new THREE.Mesh(new THREE.RingGeometry(0.94, 1, 48), new THREE.MeshBasicMaterial({ color: red, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false }));
  ell.rotation.x = -Math.PI / 2;
  const mkLine = (c, o) => { const ge = new THREE.BufferGeometry(); ge.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3)); const l = new THREE.Line(ge, new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: o })); l.frustumCulled = false; return l; };
  const drop = mkLine(red, 0.45), vel = mkLine(0xff8a7a, 0.9);
  const tg = new THREE.BufferGeometry(); tg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(600), 3));
  const trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: red, transparent: true, opacity: 0.55 })); trail.frustumCulled = false;
  const el = document.createElement("div"); el.className = "lbl trk"; el.style.pointerEvents = "auto"; el.style.cursor = "pointer";
  const lab = new CSS2DObject(el); lab.center.set(-0.08, 1.3);
  el.addEventListener("click", () => { S.sel = id; });
  g.add(mk, ell, drop, vel, trail, lab); G.dyn.add(g);
  const o = { g, mk, ell, drop, vel, trail, lab };
  L.tracks.set(id, o); return o;
}
function setLine(line, a, b) { const p = line.geometry.attributes.position; p.setXYZ(0, a.x, a.y, a.z); p.setXYZ(1, b.x, b.y, b.z); p.needsUpdate = true; }
function signedBorder(inc, x, y) {               // signed distance (m) to the border, + inside the UAE (UAE on the left)
  let best = null;
  for (const b of inc.border) for (let i = 0; i < b.xy.length - 1; i++) {
    const [ax, ay] = b.xy[i], [bx, by] = b.xy[i + 1], dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy; if (!L2) continue;
    const u = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / L2)), qx = ax + u * dx, qy = ay + u * dy, r = Math.hypot(x - qx, y - qy);
    if (!best || r < best.r) best = { r, s: Math.sign(dx * (y - ay) - dy * (x - ax)) || 1 };
  }
  return best ? best.r * best.s : null;
}
function updateLocal() {
  const inc = S.focus, tile = inc && S.tiles[inc.id];
  if (!tile || !L.bearings) return 0;
  const tl = S.now - inc.t0, live = tl >= 0 && tl <= inc.duration;
  const fi = Math.max(0, Math.min(inc.frames.length - 1, Math.floor(tl * inc.replay_hz)));
  const fr = live ? inc.frames[fi] : null;
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 160);
  const size = (p) => Math.min(4, Math.max(0.4, camera.position.distanceTo(p) / 6));   // about the same size on screen at any range
  // clusters: green while their CNN says drone
  L.clusters.forEach((c, k) => {
    const on = fr && fr.cnn[k] === "1";
    c.mesh.material.color.setHex(on ? 0x39ff88 : 0x2a9c8e); c.mesh.scale.setScalar(size(c.p) * (on ? 1.4 + 0.5 * pulse : 1));
    c.label.element.style.color = on ? "#39ff88" : "";
  });
  // bearings, drawn level (flat clusters measure elevation poorly)
  const bp = L.bearings.geometry.attributes.position; let nb = 0;
  if (fr && $("lyrBearings").checked) for (const [k, ux, uy] of fr.bearings) {
    const c = L.clusters[k]; if (!c) continue; const h = Math.hypot(ux, uy) || 1;
    const end = L2W(inc, inc.clusters[k].x + (2500 * ux) / h, inc.clusters[k].y + (2500 * uy) / h, inc.clusters[k].z + 6);
    bp.setXYZ(nb * 2, c.p.x, c.p.y, c.p.z); bp.setXYZ(nb * 2 + 1, end.x, end.y, end.z); nb++;
  }
  L.bearings.geometry.setDrawRange(0, nb * 2); bp.needsUpdate = true;
  // tracks
  const seen = new Set();
  for (const tr of fr ? fr.tracks : []) {
    seen.add(tr.id);
    const o = trackObj(tr.id), p = L2W(inc, tr.x, tr.y, tr.z), gnd = hL(tile, tr.x, tr.y);
    o.mk.position.copy(p); o.mk.rotation.y += 0.04; o.mk.scale.setScalar(size(p));
    o.ell.position.set(p.x, L2W(inc, tr.x, tr.y, gnd + 4).y, p.z); o.ell.scale.set((2 * tr.sx) / 1000, (2 * tr.sy) / 1000, 1);
    setLine(o.drop, p, L2W(inc, tr.x, tr.y, gnd));
    setLine(o.vel, p, L2W(inc, tr.x + 30 * tr.vx, tr.y + 30 * tr.vy, tr.z + 30 * tr.vz));
    // trail: this track's positions over the last 40 s
    const tp = o.trail.geometry.attributes.position; let n = 0;
    for (let j = fi; j >= Math.max(0, fi - 40 * inc.replay_hz) && n < 200; j--) {
      const q = inc.frames[j].tracks.find((r) => r.id === tr.id); if (!q) break;
      const w = L2W(inc, q.x, q.y, q.z); tp.setXYZ(n++, w.x, w.y, w.z);
    }
    o.trail.geometry.setDrawRange(0, n); tp.needsUpdate = true;
    o.lab.position.copy(p);
    const spd = Math.hypot(tr.vx, tr.vy), hdg = ((Math.atan2(tr.vx, tr.vy) * 180) / Math.PI + 360) % 360;
    o.lab.element.textContent = `TRK ${String(tr.id).padStart(2, "0")} · ${Math.round(tr.z - gnd)} M AGL · ${spd.toFixed(0)} M/S ${String(Math.round(hdg)).padStart(3, "0")}°`;
    o.g.visible = true; o.lab.visible = true;
  }
  for (const [id, o] of L.tracks) if (!seen.has(id)) { o.g.visible = false; o.lab.visible = false; }   // CSS2D ignores the parent's flag
  // simulated truth
  const showTruth = $("lyrTruth").checked && live;
  (L.truthPaths || []).forEach((l) => { l.visible = $("lyrTruth").checked; });
  (L.truths || []).forEach(({ mesh, label }, i) => {
    mesh.visible = showTruth; label.visible = showTruth;
    const d = inc.truth[i];
    if (!showTruth || !d) return;
    const j = Math.max(0, Math.min(d.t.length - 1, Math.round(tl * inc.replay_hz)));
    mesh.position.copy(L2W(inc, ...d.p[j])); mesh.scale.setScalar(size(mesh.position));
    label.element.textContent = `SIM TRUTH${inc.truth.length > 1 ? ` ${i + 1}` : ""} · ${d.cls.toUpperCase()} · ${Math.round(d.agl[j])} M AGL`;
  });
  updateCaption(inc, fr, tl);
  updateCard(inc, fr, tile, tl);
  return fr ? fr.tracks.length : 0;
}
function updateCard(inc, fr, tile, tl) {
  const card = $("card");
  const trs = fr ? fr.tracks : [];
  if (!trs.length) { card.hidden = true; return; }
  const tr = trs.find((t) => t.id === S.sel) || trs.reduce((a, b) => (b.n > a.n ? b : a));
  const gnd = hL(tile, tr.x, tr.y), [lon, lat] = local2ll(inc, tr.x, tr.y);
  const spd = Math.hypot(tr.vx, tr.vy), hdg = ((Math.atan2(tr.vx, tr.vy) * 180) / Math.PI + 360) % 360;
  const sd = signedBorder(inc, tr.x, tr.y);
  const crossed = inc.alerts.some((a) => a.tier === "border_cross" && a.t <= tl && a.text.includes(`track ${tr.id} `));
  let border = "—";
  if (sd !== null) border = sd >= 0 ? `${Math.round(sd)} M INSIDE UAE` : `${Math.round(-sd)} M OUT (${inc.neighbour === "SAU" ? "SAUDI" : "OMANI"} SIDE)`;
  const status = crossed ? "CROSSED" : sd !== null && sd < 0 ? "BEYOND BORDER" : "INSIDE";
  card.hidden = false;
  card.innerHTML = `<h3>TRACK ${String(tr.id).padStart(2, "0")} · ${status}</h3><dl>
    <dt>TYPE</dt><dd>UAS · ACOUSTIC (CNN)</dd>
    <dt>POSITION</dt><dd>${dms(lat, "N", "S")} ${dms(lon, "E", "W")}</dd>
    <dt>SECTOR</dt><dd>${inc.sector}</dd>
    <dt>BORDER</dt><dd>${border}</dd>
    <dt>HEIGHT</dt><dd>${Math.round(tr.z - gnd)} M AGL ± ${Math.round(tr.sz)}</dd>
    <dt>SPEED / HDG</dt><dd>${spd.toFixed(1)} M/S · ${String(Math.round(hdg)).padStart(3, "0")}°</dd>
    <dt>POS ERROR (1σ)</dt><dd>± ${Math.round(Math.max(tr.sx, tr.sy))} M</dd>
    <dt>UPDATES</dt><dd>${tr.n}</dd></dl>
    <div class="note">Height is coarse: flat ground clusters (D-38).</div>`;
}

// ---------------------------------------------------------------- after-action summary (uses the simulated truth)
function outcome(inc) {
  const d = inc.truth[0];
  if (!d) return "";
  const sdAt = (t) => { const j = Math.max(0, Math.min(d.t.length - 1, Math.round(t * inc.replay_hz))); return signedBorder(inc, d.p[j][0], d.p[j][1]); };
  let crossT = null;
  for (let j = 1; j < d.t.length; j++) {
    const a = signedBorder(inc, d.p[j - 1][0], d.p[j - 1][1]), b = signedBorder(inc, d.p[j][0], d.p[j][1]);
    if (a !== null && b !== null && a < 0 && b >= 0) { crossT = d.t[j]; break; }
  }
  const fh = inc.frames.findIndex((f) => f.cnn.includes("1")), ft = inc.frames.findIndex((f) => f.tracks.length);
  const heardT = fh >= 0 ? inc.frames[fh].t : null, trackT = ft >= 0 ? inc.frames[ft].t : null;
  const km = (m) => (m >= 1000 ? `${(m / 1000).toFixed(1)} KM` : `${Math.round(m)} M`);
  if (heardT === null) return "NOT HEARD";
  const sd = sdAt(heardT);
  const heard = sd < 0 ? `HEARD ${km(-sd)} BEFORE THE BORDER` : `HEARD ONLY ${km(sd)} INSIDE THE UAE`;
  let track = "NEVER TRACKED";
  if (trackT !== null) track = crossT === null ? "TRACKED · DID NOT CROSS" : trackT < crossT ? `TRACKED ${Math.round(crossT - trackT)} S BEFORE IT CROSSED` : "TRACKED ONLY AFTER IT CROSSED";
  return `${heard} · ${track}`;
}

// ---------------------------------------------------------------- auto-follow
function autoFollow() {
  if (!$("follow").checked || S.mode !== "watch") return;
  const f = S.focus;
  if (f && S.autoFocused && S.now > f.t0 + f.duration + 6) { S.autoFocused = false; leaveFocus(true); return; }
  if (f || !S.playing) return;
  const inc = S.incidents.find((i) => i.firstT !== null && S.now >= i.t0 + i.firstT && S.now <= i.t0 + i.duration);
  if (inc) focus(inc, true);
}
function closeHelp() { $("help").hidden = true; try { localStorage.setItem("bw-help-seen", "1"); } catch { /* private mode */ } if (!S.playing) $("play").click(); }
$("helpGo").addEventListener("click", closeHelp);
$("helpX").addEventListener("click", closeHelp);
$("helpBtn").addEventListener("click", () => { $("help").hidden = false; });

// ---------------------------------------------------------------- Zetaris drawer: questions across the databases
const SQL_KW = /\b(SELECT|FROM|JOIN|LEFT|WHERE|GROUP BY|ORDER BY|UNION ALL|ON|AND|LIMIT)\b/g;
function prettySql(sql) { return sql.replace(SQL_KW, "\n$1").replace(/^\n/, ""); }
function zetTabs() {
  document.querySelectorAll(".ztabs button").forEach((b) => {
    b.disabled = b.hasAttribute("data-inc") && (!S.focus || !!S.focus.built);   // built tests are not in the stores
    b.title = b.disabled ? "Open an incident first" : "";
  });
}
async function zetRun(q) {
  const body = $("zetBody");
  document.querySelectorAll(".ztabs button").forEach((b) => b.classList.toggle("on", b.dataset.q === q));
  const id = S.focus ? S.focus.id : "";
  body.innerHTML = `<div class="wait">QUERYING THROUGH ZETARIS…</div>`;
  let r;
  try { r = await (await fetch(`/api/federated?q=${encodeURIComponent(q)}&id=${encodeURIComponent(id)}`)).json(); }
  catch (e) { r = { error: e.message }; }
  if (r.error) { body.innerHTML = `<div class="err"></div>`; body.firstChild.textContent = r.error; return; }
  const esc = (v) => String(v ?? "—").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  const num = (v) => typeof v === "number";
  body.innerHTML = `<p class="about">${esc(r.about)}</p>
    <div class="srcs">JOINED: ${r.sources.map((x) => `<b>${esc(x)}</b>`).join(" + ")}</div>
    <table><thead><tr>${r.columns.map((c) => `<th>${esc(c.replace(/_/g, " ").toUpperCase())}</th>`).join("")}</tr></thead>
    <tbody>${r.rows.length ? r.rows.map((row) => `<tr>${row.map((v) => `<td class="${num(v) ? "num" : ""}">${esc(v)}</td>`).join("")}</tr>`).join("")
      : `<tr><td colspan="${r.columns.length}">No rows.</td></tr>`}</tbody></table>
    <details><summary>THE ZETARIS QUERY</summary><pre></pre></details>`;
  body.querySelector("pre").textContent = prettySql(r.sql);
}
$("zetBtn").addEventListener("click", () => {
  const z = $("zet"); z.hidden = !z.hidden; zetTabs();
  if (!z.hidden) zetRun(S.focus ? "first_heard" : "overview");
});
$("zetX").addEventListener("click", () => { $("zet").hidden = true; });
document.querySelectorAll(".ztabs button").forEach((b) => b.addEventListener("click", () => { if (!b.disabled) zetRun(b.dataset.q); }));

// ---------------------------------------------------------------- controls
function setNow(T) { S.now = Math.max(0, Math.min(S.len, T)); }
$("play").addEventListener("click", () => { S.playing = !S.playing; $("play").textContent = S.playing ? "❚❚" : "▶"; if (S.now >= S.len) setNow(0); });
document.querySelectorAll(".speeds button").forEach((b) => b.addEventListener("click", () => {
  S.speed = Number(b.dataset.s); document.querySelectorAll(".speeds button").forEach((x) => x.classList.toggle("on", x === b));
}));
$("scrub").addEventListener("input", (e) => setNow((Number(e.target.value) / 1000) * S.len));
$("next").addEventListener("click", () => {
  const e = S.events.find((x) => x.T > S.now + 1 && ["border_watch", "border_cross", "take_cover"].includes(x.tier));
  if (!e) return;
  setNow(e.T - 6); focus(e.inc);
  if (!S.playing) $("play").click();
});
function applyLayers() {
  S.clusterPoints.visible = $("lyrClusters").checked;
  L.clusters.forEach((c) => { c.mesh.visible = $("lyrClusters").checked; c.label.visible = $("lyrClusters").checked; });
  const g = $("lyrGrid").checked ? 1 : 0;
  theatreMat.uniforms.uGrid.value = g;
  G.local.children.forEach((m) => { if (m.material?.uniforms?.uGrid) m.material.uniforms.uGrid.value = g; });
}
["lyrClusters", "lyrGrid", "lyrTruth", "lyrBearings"].forEach((id) => $(id).addEventListener("change", applyLayers));
$("exag").addEventListener("input", (e) => {
  const v = Number(e.target.value), k = v / S.exag;
  S.exag = v; world.scale.y = v; $("exagV").textContent = `${v}×`;
  const off = camera.position.clone().sub(controls.target);
  controls.target.y *= k; camera.position.copy(controls.target).add(off);
  if (S.mode === "build") redrawBuild();                                   // domes stay true spheres
  else if (S.mode === "scenario" && $("lyrRanges").checked) redrawBuildOverlayFaint();
});
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;   // typing a question
  if (e.code === "Space") { e.preventDefault(); $("play").click(); }
  if (e.key === "Escape") {
    if (B.tool) setTool(null);
    else if (S.mode === "scenario") backToBuild();
    else if (S.focus) leaveFocus(true);
  }
  if (S.mode === "build" && B.tool === "draw" && e.key === "Enter") finishDrone();
  if (S.mode === "build" && B.tool === "draw" && e.key === "Backspace") { e.preventDefault(); B.cur.pop(); redrawBuild(); }
  if (e.key === "n" || e.key === "N") $("next").click();
});

// cursor readout: march the view ray down to the ground
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let lastMove = 0;
function groundPoint(clientX, clientY) {          // where the view ray under the cursor meets the ground
  ndc.set((clientX / window.innerWidth) * 2 - 1, -(clientY / window.innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const o = ray.ray.origin, d = ray.ray.direction, far = camera.position.distanceTo(controls.target) * 6 + 50;
  let prev = 0;
  for (let i = 1; i <= 220; i++) {
    const s = far * Math.pow(i / 220, 2), x = o.x + d.x * s, y = o.y + d.y * s, z = o.z + d.z * s;
    if (y <= (groundW(x, z) / 1000) * S.exag) {
      const sm = (prev + s) / 2, X = o.x + d.x * sm, Z = o.z + d.z * sm, [lon, lat] = unproj(X, Z);
      return { X, Z, lon, lat, h: groundW(X, Z) };
    }
    prev = s;
  }
  return null;
}
renderer.domElement.addEventListener("pointermove", (ev) => {
  const t = performance.now(); if (t - lastMove < 60) return; lastMove = t;
  const g = groundPoint(ev.clientX, ev.clientY);
  $("readout").textContent = g ? `${dms(g.lat, "N", "S")}  ${dms(g.lon, "E", "W")}  ·  ELEV ${Math.round(g.h)} M` : "--";
  if (typeof buildHover === "function") buildHover(g);
});

// ---------------------------------------------------------------- BUILD A TEST: place microphones and drones, run, replay
// Local frame: metres east (x) and north (y) of the protected site (the origin), as the server's scenarios.
const B = { origin: null, tile: null, pinc: null, clusters: [], drones: [], cur: [], cls: "mid", agl: 60, speed: 14,
            ambient: "desert", tool: null, ranges: null, job: null, estTimer: null, down: null };
const MAX_REACH_M = 5400;
const PRESETS = [
  { id: "liwa", label: "Desert border post · Liwa", sub: "quadcopter from the Saudi side, quiet desert",
    origin: [22.8205, 53.5935], ambient: "desert", ring: [700, 6], centre: true,
    drones: [{ cls: "mid", agl: 60, speed: 14, waypoints: [[-400, -2600], [-150, -900], [0, 0]] }] },
  { id: "adh", label: "City site · Abu Dhabi", sub: "large petrol drone over town noise",
    origin: [24.47, 54.37], ambient: "town", ring: [300, 6], centre: true,
    drones: [{ cls: "ts_mr", agl: 90, speed: 16, waypoints: [[-1600, 900], [-500, 300], [0, 0]] }] },
  { id: "hatta", label: "Mountain border · Hatta", sub: "FPV through the Hajar, hills block sound",
    origin: [24.7905, 56.1455], ambient: "desert", ring: [600, 6], centre: true,
    drones: [{ cls: "fpv", agl: 40, speed: 22, waypoints: [[1500, -1900], [600, -700], [0, 0]] }] },
];

function initBuild() {
  fetch("/api/build/ranges").then((r) => r.json()).then((d) => { B.ranges = d; fillBuildControls(); }).catch(() => {});
  G.build.renderOrder = 3;
  world.add(G.build);
  const pb = $("bPresets");
  PRESETS.forEach((p) => {
    const b = document.createElement("button"); b.type = "button";
    b.innerHTML = `${p.label}<small class="dim"> · ${p.sub}</small>`;
    b.addEventListener("click", () => loadPreset(p)); pb.appendChild(b);
  });
  $("tabWatch").addEventListener("click", () => setMode("watch"));
  $("tabBuild").addEventListener("click", () => setMode("build"));
  $("bPick").addEventListener("click", () => setTool(B.tool === "pick" ? null : "pick"));
  $("bMic").addEventListener("click", () => setTool(B.tool === "mic" ? null : "mic"));
  $("bMicClear").addEventListener("click", () => { B.clusters = []; redrawBuild(); });
  $("bDraw").addEventListener("click", () => { if (B.tool === "draw") finishDrone(); else { B.cur = []; setTool("draw"); } });
  $("bDroneDone").addEventListener("click", finishDrone);
  $("bRun").addEventListener("click", runBuild);
  $("bAmb").addEventListener("change", (e) => { B.ambient = e.target.value; redrawBuild(); });
  $("bAgl").addEventListener("input", (e) => { B.agl = Number(e.target.value); $("bAglV").textContent = `${B.agl} m`; });
  $("bSpd").addEventListener("input", (e) => { B.speed = Number(e.target.value); $("bSpdV").textContent = `${B.speed} m/s`; });
  $("lyrRanges").addEventListener("change", () => {                // only matters on a test replay; editing always shows them
    if (S.mode !== "scenario") return;
    if ($("lyrRanges").checked) redrawBuildOverlayFaint(); else clearGroup(G.build);
  });
  // a click (not a drag) on the map acts with the current tool
  renderer.domElement.addEventListener("pointerdown", (ev) => { B.down = [ev.clientX, ev.clientY, ev.button]; });
  renderer.domElement.addEventListener("pointerup", (ev) => {
    if (!B.down || ev.button !== 0 || B.down[2] !== 0) return;
    if (Math.hypot(ev.clientX - B.down[0], ev.clientY - B.down[1]) > 5) return;
    if (S.mode === "build" && B.tool) buildClick(groundPoint(ev.clientX, ev.clientY));
  });
  renderer.domElement.addEventListener("dblclick", () => { if (S.mode === "build" && B.tool === "draw") finishDrone(); });
  refreshTests();
}

function fillBuildControls() {
  const sel = $("bAmb"); sel.innerHTML = "";
  for (const [k, v] of Object.entries(B.ranges.ambients)) { const o = document.createElement("option"); o.value = k; o.textContent = v; sel.appendChild(o); }
  sel.value = B.ambient;
  const box = $("bTypes"); box.innerHTML = "";
  for (const [k, v] of Object.entries(B.ranges.classes)) {
    const b = document.createElement("button"); b.type = "button"; b.dataset.cls = k;
    b.innerHTML = `${v.replace(/ \(.*\)/, "")}<small>${v.match(/\((.*)\)/)?.[1] || ""}</small>`;
    b.addEventListener("click", () => { B.cls = k; setDroneDefaults(k); redrawBuild(); });
    box.appendChild(b);
  }
  redrawBuild();
}
function setDroneDefaults(cls) {
  const d = { fpv: [40, 22], mid: [60, 14], ts_mr: [90, 16], ts_fw: [150, 28] }[cls] || [60, 14];
  B.agl = d[0]; B.speed = d[1];
  $("bAgl").value = B.agl; $("bSpd").value = B.speed; $("bAglV").textContent = `${B.agl} m`; $("bSpdV").textContent = `${B.speed} m/s`;
}
const rangeM = (cls = B.cls) => (B.ranges ? B.ranges.ranges[B.ambient][cls] : 800);

function setMode(m) {
  if (S.mode === m) return;
  setTool(null);
  if (S.playing) $("play").click();
  if (S.focus) leaveFocus(false);
  $("tabWatch").classList.toggle("on", m === "watch"); $("tabBuild").classList.toggle("on", m !== "watch");
  $("watchPane").hidden = m !== "watch"; $("buildPane").hidden = m === "watch";
  $("lyrRangesWrap").hidden = m === "watch";
  $("back").textContent = "◂ THEATRE"; $("caption").hidden = true;
  if (m === "watch") {
    clearBuildView();
    S.mode = "watch";
    setIncidents(S.watchIncidents); setNow(0);
    home();
  } else {
    S.mode = "build";
    setIncidents([]);
    if (B.origin) showBuildArea(); else { setTool("pick"); home(); }
  }
}

function setIncidents(list) {
  for (const inc of S.incidents) { if (inc.marker) { inc.marker.element.remove(); G.overlay.remove(inc.marker); inc.marker = null; } }
  S.incidents = list; S.events = []; S.lastNow = -1; S.logShown = 0; $("log").innerHTML = "";
  if (list.length) { buildTimeline(); buildPanels(); }
  else { S.len = 60; $("ticks").innerHTML = ""; $("len").textContent = "BUILD MODE"; }
}

function setTool(t) {
  B.tool = t;
  $("bPick").classList.toggle("on", t === "pick"); $("bMic").classList.toggle("on", t === "mic");
  $("bDraw").classList.toggle("on", t === "draw"); $("bDroneDone").hidden = t !== "draw";
  $("bDraw").textContent = t === "draw" ? "✎ Drawing… (double-click to finish)" : "✎ Draw drone path";
  const msg = { pick: "CLICK THE MAP WHERE THE SITE YOU WANT TO PROTECT IS",
                mic: "CLICK TO PLACE A MICROPHONE CLUSTER · CLICK ONE TO REMOVE IT · ESC WHEN DONE",
                draw: "CLICK THE DRONE'S PATH, START FIRST · DOUBLE-CLICK OR ENTER TO FINISH · BACKSPACE UNDOES" }[t];
  $("instr").hidden = !msg; if (msg) $("instr").textContent = msg;
  if (t !== "draw" && B.cur.length) { B.cur = []; redrawBuild(); }
  stepMarks();
}

function toLocal(g) {                                   // ground point -> metres from the protected site
  const [a, b] = mpd(B.origin[0]);
  return [(g.lon - B.origin[1]) * b, (g.lat - B.origin[0]) * a];
}

async function buildClick(g) {
  if (!g) return;
  if (B.tool === "pick") { B.origin = [g.lat, g.lon]; B.clusters = []; B.drones = []; setTool("mic"); await showBuildArea(); return; }
  if (!B.origin) return;
  const [x, y] = toLocal(g);
  if (Math.hypot(x, y) > MAX_REACH_M) { flashInstr("TOO FAR FROM THE SITE: KEEP WITHIN 5 KM"); return; }
  if (B.tool === "mic") {
    const near = B.clusters.findIndex(([cx, cy]) => Math.hypot(cx - x, cy - y) < Math.max(40, camera.position.distanceTo(controls.target) * 6));
    if (near >= 0) B.clusters.splice(near, 1);
    else if (B.clusters.length >= 24) { flashInstr("AT MOST 24 MICROPHONE CLUSTERS"); return; }
    else B.clusters.push([Math.round(x), Math.round(y)]);
  } else if (B.tool === "draw") {
    const last = B.cur[B.cur.length - 1];          // a double-click to finish also sends two clicks on the same spot
    if (!last || Math.hypot(last[0] - x, last[1] - y) > 15) B.cur.push([Math.round(x), Math.round(y)]);
  }
  redrawBuild();
}
function flashInstr(text) {
  const el = $("instr"), old = el.textContent, was = el.hidden;
  el.hidden = false; el.textContent = text; el.style.borderColor = "var(--red)";
  setTimeout(() => { el.style.borderColor = ""; el.textContent = old; el.hidden = was && !B.tool; }, 1800);
}
function finishDrone() {
  if (B.cur.length >= 2) {
    if (B.drones.length >= 6) flashInstr("AT MOST 6 DRONES");
    else B.drones.push({ cls: B.cls, agl: B.agl, speed: B.speed, waypoints: B.cur.slice() });
  }
  B.cur = []; setTool(null); redrawBuild();
}

async function showBuildArea() {
  clearBuildView();
  const [lat, lon] = B.origin;
  B.pinc = { id: `build_${lat.toFixed(4)}_${lon.toFixed(4)}`, lat, lon, half_m: 6000, border: localBorder(lat, lon, 6000), built: true };
  $("bPlaceText").innerHTML = `Site at <b>${dms(lat, "N", "S")} ${dms(lon, "E", "W")}</b>. Click <b>Pick on map</b> to move it.`;
  B.tile = await loadTile(B.pinc);
  if (S.mode !== "build" || !B.pinc || B.pinc.lat !== lat || B.pinc.lon !== lon) return;
  showTile(B.pinc, B.tile);
  redrawBuild();
  const c = L2W(B.pinc, 0, 0, hL(B.tile, 0, 0)); c.y *= S.exag;
  flyTo(c, 5.5, 0.9, -0.35, 1.8);
}
function localBorder(lat, lon, half) {                 // border lines near the site, in metres from it
  const [a, b] = mpd(lat), out = [];
  for (const s of S.border.sectors) {
    const xy = s.line.map(([lo, la]) => [(lo - lon) * b, (la - lat) * a]);
    if (xy.some(([x, y]) => Math.abs(x) < half * 1.5 && Math.abs(y) < half * 1.5)) out.push({ sector: s.id, xy });
  }
  return out;
}
function clearBuildView() {
  clearGroup(G.build); clearGroup(G.local);
  theatreMat.uniforms.uHoleOn.value = 0; clusterMat.uniforms.uHoleOn.value = 0;
  S.border.sectors.forEach((s) => s.lines.forEach((l) => { l.visible = true; }));
}

function loadPreset(p) {
  B.origin = p.origin.slice(); B.ambient = p.ambient; $("bAmb").value = p.ambient;
  B.clusters = [];
  const [r, n] = p.ring;
  for (let k = 0; k < n; k++) { const a = (2 * Math.PI * k) / n; B.clusters.push([Math.round(r * Math.sin(a)), Math.round(r * Math.cos(a))]); }
  if (p.centre) B.clusters.push([0, 0]);
  B.drones = p.drones.map((d) => ({ ...d, waypoints: d.waypoints.map((w) => w.slice()) }));
  B.cls = p.drones[0].cls; setDroneDefaults(B.cls);
  setTool(null);
  showBuildArea();
}

// draw: the site, microphones with their listening domes, gaps between neighbours, drone paths
const DOME_GEO = new THREE.SphereGeometry(1, 48, 16, 0, 2 * Math.PI, 0, Math.PI / 2);          // unit upper half-sphere
const DOME_WIRE = new THREE.WireframeGeometry(new THREE.SphereGeometry(1, 16, 5, 0, 2 * Math.PI, 0, Math.PI / 2));
function groundAt(x, y) { return B.tile && Math.abs(x) < B.tile.half_m && Math.abs(y) < B.tile.half_m ? hL(B.tile, x, y) : hT(...local2ll(B.pinc, x, y)); }
function redrawBuild() {
  document.querySelectorAll("#bTypes button").forEach((b) => b.classList.toggle("on", b.dataset.cls === B.cls));
  clearGroup(G.build);
  if (!B.pinc) { stepMarks(); updateEstimate(); return; }
  const inc = B.pinc, R = rangeM();
  const W = (x, y, dz = 0) => L2W(inc, x, y, groundAt(x, y) + dz);
  // protected site
  const site = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 64), new THREE.MeshBasicMaterial({ color: 0x9fb7c9, side: THREE.DoubleSide, transparent: true, opacity: 0.85, depthTest: false }));
  site.rotation.x = -Math.PI / 2; site.scale.setScalar(0.15); site.position.copy(W(0, 0, 4)); site.renderOrder = 6; G.build.add(site);
  addLabel(G.build, "PROTECTED SITE", "lbl site", ...local2ll(inc, 0, 0), groundAt(0, 0) + 60);
  // listening domes: a transparent half-sphere per cluster, radius = how far it hears this drone type.
  // The world is stretched in height by the relief factor, so each dome is squashed back by it to stay a
  // true sphere: a drone path at its real height is inside a dome exactly when that cluster can hear it.
  const rk = R / 1000;
  for (const [x, y] of B.clusters) {
    const base = W(x, y, 0);
    const dome = new THREE.Mesh(DOME_GEO, new THREE.MeshBasicMaterial({ color: 0x3ce6d0, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide }));
    dome.scale.set(rk, rk / S.exag, rk); dome.position.copy(base); dome.renderOrder = 2; G.build.add(dome);
    const wire = new THREE.LineSegments(DOME_WIRE, new THREE.LineBasicMaterial({ color: 0x3ce6d0, transparent: true, opacity: 0.22, depthWrite: false }));
    wire.scale.copy(dome.scale); wire.position.copy(base); wire.renderOrder = 3; G.build.add(wire);
    // ground footprint, faint: overlapping footprints read darker where 2 or more clusters hear
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1, 72), new THREE.MeshBasicMaterial({ color: 0x3ce6d0, transparent: true, opacity: 0.05, depthTest: false, depthWrite: false, side: THREE.DoubleSide }));
    disc.rotation.x = -Math.PI / 2; disc.scale.setScalar(rk); disc.position.copy(W(x, y, 2)); disc.renderOrder = 2; G.build.add(disc);
    const rim = new THREE.Mesh(new THREE.RingGeometry(0.985, 1, 96), new THREE.MeshBasicMaterial({ color: 0x3ce6d0, transparent: true, opacity: 0.55, depthTest: false, depthWrite: false, side: THREE.DoubleSide }));
    rim.rotation.x = -Math.PI / 2; rim.scale.setScalar(R / 1000); rim.position.copy(W(x, y, 2)); rim.renderOrder = 3; G.build.add(rim);
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.03), new THREE.MeshBasicMaterial({ color: 0x3ce6d0, depthTest: false }));
    m.position.copy(W(x, y, 10)); m.renderOrder = 7; G.build.add(m);
  }
  B.clusters.forEach(([x, y], i) => addLabel(G.build, `M${i + 1}`, "lbl mic", ...local2ll(inc, x, y), groundAt(x, y) + 70));
  // gaps: each cluster to its two nearest neighbours
  const seen = new Set(); let tooWide = 0;
  B.clusters.forEach(([x, y], i) => {
    B.clusters.map(([u, v], j) => [j, Math.hypot(u - x, v - y)]).filter(([j]) => j !== i).sort((a, b) => a[1] - b[1]).slice(0, 2).forEach(([j, d]) => {
      const key = i < j ? `${i}-${j}` : `${j}-${i}`; if (seen.has(key)) return; seen.add(key);
      const [u, v] = B.clusters[j], cls = d <= R ? "ok" : d <= 2 * R ? "warn" : "bad";
      if (cls !== "ok") tooWide++;
      const col = { ok: 0x39ff88, warn: 0xffb020, bad: 0xff3b3b }[cls];
      G.build.add(fatLine([...W(x, y, 6).toArray(), ...W(u, v, 6).toArray()], col, 2, 0.9, false));
      addLabel(G.build, `${d >= 1000 ? (d / 1000).toFixed(1) + " km" : Math.round(d) + " m"}`, `lbl gap ${cls}`, ...local2ll(inc, (x + u) / 2, (y + v) / 2), groundAt((x + u) / 2, (y + v) / 2) + 30);
    });
  });
  // drone paths (finished, and the one being drawn)
  const drawPath = (wps, d, label, dashed) => {
    const pts = [];
    for (let i = 0; i < wps.length - 1; i++) {
      const [ax, ay] = wps[i], [bx, by] = wps[i + 1], k = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 60));
      for (let j = 0; j <= k; j++) { const x = ax + ((bx - ax) * j) / k, y = ay + ((by - ay) * j) / k; pts.push(...W(x, y, d.agl).toArray()); }
    }
    if (pts.length >= 6) G.build.add(fatLine(pts, dashed ? 0xffd24a : 0xff5a36, dashed ? 2 : 3, 0.95, false));
    wps.forEach(([x, y]) => { const s = new THREE.Mesh(new THREE.SphereGeometry(0.015, 10, 8), new THREE.MeshBasicMaterial({ color: 0xff8a7a, depthTest: false })); s.position.copy(W(x, y, d.agl)); s.renderOrder = 8; G.build.add(s); });
    if (label && wps.length) addLabel(G.build, label, "lbl drone", ...local2ll(inc, ...wps[0]), groundAt(...wps[0]) + d.agl + 80);
  };
  const names = B.ranges ? B.ranges.classes : {};
  B.drones.forEach((d, i) => drawPath(d.waypoints, d, `D${i + 1} · ${(names[d.cls] || d.cls).toUpperCase()} · START`, false));
  if (B.cur.length) drawPath(B.cur, { agl: B.agl }, null, true);
  // panel text
  const cname = (names[B.cls] || B.cls).toLowerCase();
  let t = `Each dome shows how far (in every direction, including up) a cluster can hear a <b>${cname}</b> in <b>${(B.ranges?.ambients[B.ambient] || B.ambient).toLowerCase()}</b>: about <b>${R >= 1000 ? (R / 1000).toFixed(1) + " km" : R + " m"}</b>. `;
  t += `To fix a drone's position, two clusters must hear it, so keep neighbours <b>no more than ${R >= 1000 ? (R / 1000).toFixed(1) + " km" : R + " m"} apart</b> (green lines). `;
  if (B.clusters.length) t += tooWide ? `<span class="warn">${tooWide} gap(s) are too wide (amber: only one cluster hears the middle; red: nobody does).</span>` : "All gaps are close enough.";
  $("bMicText").innerHTML = `${B.clusters.length} placed. ${t}`;
  const list = $("bDrones"); list.innerHTML = "";
  B.drones.forEach((d, i) => {
    const len = d.waypoints.slice(1).reduce((s, w, k) => s + Math.hypot(w[0] - d.waypoints[k][0], w[1] - d.waypoints[k][1]), 0);
    const row = document.createElement("div"); row.className = "it";
    row.innerHTML = `<span>D${i + 1} · ${names[d.cls] || d.cls} · ${(len / 1000).toFixed(1)} km · ${d.agl} m · ${d.speed} m/s</span><button type="button" aria-label="Remove">✕</button>`;
    row.querySelector("button").addEventListener("click", () => { B.drones.splice(i, 1); redrawBuild(); });
    list.appendChild(row);
  });
  stepMarks(); updateEstimate();
}
function stepMarks() {
  const done = [!!B.origin, B.clusters.length >= 2, B.drones.length >= 1];
  ["bs1", "bs2", "bs3"].forEach((id, i) => { $(id).classList.toggle("done", done[i]); $(id).classList.toggle("now", !done[i] && done.slice(0, i).every(Boolean)); });
  $("bs4").classList.toggle("now", done.every(Boolean));
  if (S.mode === "build") document.querySelector("#buildPane .bstep.now")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
function buildHover() { /* reserved: hover hints */ }

function buildRequest() {
  return { origin: B.origin, clusters: B.clusters, drones: B.drones, ambient: B.ambient, seed: 7,
           sites: [{ name: "Protected site", x: 0, y: 0, radius_m: 150 }],
           name: `Test · ${PRESETS.find((p) => p.origin[0] === B.origin?.[0] && p.origin[1] === B.origin?.[1])?.label || `${B.origin?.[0].toFixed(3)}N ${B.origin?.[1].toFixed(3)}E`}` };
}
function updateEstimate() {
  clearTimeout(B.estTimer);
  const ready = B.origin && B.clusters.length >= 2 && B.drones.length >= 1;
  $("bRun").disabled = !ready || !!B.job;
  if (!ready) { $("bEst").textContent = "Place at least 2 microphones and 1 drone first."; return; }
  B.estTimer = setTimeout(async () => {
    try {
      const r = await (await fetch("/api/build/estimate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildRequest()) })).json();
      if (r.error) { $("bEst").innerHTML = `<span class="warn">${r.error}</span>`; $("bRun").disabled = true; return; }
      const m = r.estimate_s / 60;
      $("bEst").innerHTML = `Estimated time: <b>${m < 1.5 ? "about 1 minute" : `about ${Math.round(m)} minutes`}</b> (${B.clusters.length} clusters × ${B.drones.length} drone${B.drones.length > 1 ? "s" : ""}).` +
        (r.near_border ? " Near the border: crossing alerts are on." : "") +
        (m > 8 ? ` <span class="warn">Long run: fewer clusters, fewer drones or shorter paths are faster.</span>` : "");
    } catch { $("bEst").textContent = "Could not reach the server."; }
  }, 250);
}

async function runBuild() {
  if (B.job) return;
  setTool(null);
  let r;
  try { r = await (await fetch("/api/build/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildRequest()) })).json(); }
  catch (e) { r = { error: e.message }; }
  if (r.error) { $("bEst").innerHTML = `<span class="bad">${r.error}</span>`; return; }
  B.job = r.id; $("bRun").disabled = true;
  $("runcard").hidden = false; $("runBar").style.width = "2%"; $("runStage").textContent = "Starting";
  const poll = setInterval(async () => {
    let st;
    try { st = await (await fetch(`/api/build/status/${B.job}`)).json(); } catch { return; }
    const det = st.detail.match(/\((\d+)\/(\d+)\)/);
    $("runStage").textContent = st.stage + (det && st.stage.startsWith("Simulating") ? ` (${det[1]} of ${det[2]})` : "");
    $("runBar").style.width = `${Math.round(100 * st.fraction)}%`;
    $("runTime").textContent = `${Math.floor(st.elapsed_s / 60)}:${String(st.elapsed_s % 60).padStart(2, "0")} elapsed`;
    if (st.status === "done" || st.status === "error") {
      clearInterval(poll);
      const id = B.job; B.job = null; $("runcard").hidden = true; updateEstimate();
      if (st.status === "error") { $("bEst").innerHTML = `<span class="bad">The run failed: ${st.error}</span>`; return; }
      refreshTests();
      const inc = await (await fetch(`/api/incident/${id}`)).json();
      playScenario(inc);
    }
  }, 1000);
}

async function refreshTests() {
  let rows = [];
  try { rows = await (await fetch("/api/build/list")).json(); } catch { /* server not reachable */ }
  const box = $("bList"); box.innerHTML = rows.length ? "" : `<div class="empty">No tests yet.</div>`;
  for (const r of rows) {
    const d = document.createElement("div"); d.className = "it";
    d.innerHTML = `<span>${r.name} · ${r.clusters} mics · ${r.drones} drone${r.drones > 1 ? "s" : ""}</span>`;
    d.addEventListener("click", async () => { const inc = await (await fetch(`/api/incident/${r.id}`)).json(); playScenario(inc); });
    box.appendChild(d);
  }
}

function playScenario(inc) {
  setTool(null);
  clearBuildView();
  inc.outcome = outcome(inc);
  S.mode = "scenario";
  setIncidents([inc]);
  $("watchPane").hidden = true; $("buildPane").hidden = false;
  $("lyrRangesWrap").hidden = false;
  focus(inc).then(() => {
    $("crumbText").textContent = `TEST REPLAY · ${inc.name.toUpperCase()}`;
    $("back").textContent = "◂ EDIT TEST";
    // keep the listening circles over the replay, faintly, when RANGES is on
    if (B.pinc && $("lyrRanges").checked) redrawBuildOverlayFaint();
  });
  setNow(Math.max(0, (inc.firstT ?? 0) - 8));
  if (!S.playing) $("play").click();
}
function redrawBuildOverlayFaint() {
  redrawBuild();
  G.build.traverse((o) => { if (o.material && o.material.opacity !== undefined) o.material.opacity *= 0.5; });
  G.build.children.filter((o) => o.isCSS2DObject && (o.element.classList.contains("drone") || o.element.classList.contains("site"))).forEach((o) => { o.visible = false; });
}
function backToBuild() {
  if (S.playing) $("play").click();
  leaveFocus(false);
  $("back").textContent = "◂ THEATRE";
  S.mode = "build"; setIncidents([]);
  if (B.origin) showBuildArea(); else home();
}

// plain-language caption for whatever incident or test is on screen
function updateCaption(inc, fr, tl) {
  const el = $("caption");
  if (!fr) { el.hidden = !(S.mode === "scenario"); if (!el.hidden) el.innerHTML = tl < 0 ? "Starting…" : "The test has ended. Press <b>◂ EDIT TEST</b> to change it, or replay with the timeline."; return; }
  const heard = [...fr.cnn].reduce((n, c) => n + (c === "1"), 0), trs = fr.tracks;
  const recent = inc.alerts.filter((a) => a.to === "operators" && a.t <= tl && tl - a.t < 6).pop();
  let html;
  if (recent && TIERS[recent.tier]?.lvl >= 2) html = `<span class="bad">${TIERS[recent.tier].tag}</span> · ${recent.text.split(":")[0]}`;
  else if (trs.length) {
    const t0 = trs.reduce((a, b) => (b.n > a.n ? b : a)), site = inc.sites[0], sp = Math.hypot(t0.vx, t0.vy);
    const where = site ? (() => { const d = Math.hypot(t0.x - site.x, t0.y - site.y);
      return ` · drone <b>${d >= 1000 ? (d / 1000).toFixed(1) + " km" : Math.round(d) + " m"}</b> from ${site.name.toLowerCase().startsWith("protected") ? "the site" : site.name}`; })() : "";
    html = `<b>Position fixed</b> from ${heard} microphone${heard === 1 ? "" : "s"}${where} at ${sp.toFixed(0)} m/s` + (trs.length > 1 ? ` · <span class="warn">${trs.length} tracks</span>` : "");
  } else if (heard >= 2) html = `<b>${heard} microphones hear a drone</b> · combining their directions…`;
  else if (heard === 1) html = `<b>1 microphone hears a drone</b> · a second one is needed to fix its position`;
  else html = "Listening · no microphone hears a drone yet";
  el.hidden = false; el.innerHTML = html;
}

// ---------------------------------------------------------------- main loop
let tPrev = performance.now(), tUi = 0;
function loop(t) {
  const dt = Math.min((t - tPrev) / 1000, 0.1); tPrev = t;
  if (S.playing) { setNow(S.now + dt * S.speed); if (S.now >= S.len) { S.playing = false; $("play").textContent = "▶"; } }
  stepFly(dt); controls.update();
  if (t - tUi > 100) autoFollow();
  const dist = camera.position.distanceTo(controls.target);
  secLabels.forEach((l) => { l.visible = dist < 260; });
  const nTr = updateLocal();
  if (t - tUi > 100) {
    tUi = t; updatePanels();
    const active = S.incidents.reduce((n, inc) => { const tl = S.now - inc.t0; if (tl < 0 || tl > inc.duration) return n; const f = inc.frames[Math.min(inc.frames.length - 1, Math.floor(tl * inc.replay_hz))]; return n + (f ? f.tracks.length : 0); }, 0);
    const el = $("nTracks"); if (el) el.textContent = String(S.focus ? Math.max(active, nTr) : active);
  }
  renderer.render(scene, camera); labels.render(scene, camera);
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- start
async function start() {
  resize();
  const [theatre, border, list] = await Promise.all([fetch("/api/theatre").then((r) => r.json()), fetch("/api/border").then((r) => r.json()),
    fetch("/api/incidents").then((r) => r.json())]);
  theatre.H = b64f32(theatre.h); delete theatre.h;
  S.theatre = theatre; S.border = border;
  S.incidents = await Promise.all(list.map((i) => fetch(`/api/incident/${i.id}`).then((r) => r.json())));
  S.incidents.forEach((inc) => { inc.outcome = outcome(inc); });
  S.mode = "watch"; S.watchIncidents = S.incidents;
  buildTheatre(); buildBorder(); buildTimeline(); buildPanels(); applyLayers();
  initBuild();
  const [X, Z] = proj(54.25, 24.15);
  camera.position.set(X, 900, Z + 900); controls.target.set(X, 0, Z);
  home();
  $("loading").hidden = true;
  let seen = false;
  try { seen = localStorage.getItem("bw-help-seen") === "1"; } catch { /* private mode */ }
  $("help").hidden = seen;
  if (seen) $("play").click();                  // first visit: start once the legend is closed
  updatePanels();
  requestAnimationFrame(loop);
}
window.__bw = { S, L, B, camera, controls };       // for headless checks (tests/ui)
start().catch((e) => { $("loading").textContent = `FAILED TO LOAD: ${e.message}`; console.error(e); });
