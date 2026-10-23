// UAE Acoustic Range: area picker, 3D placement on terrain, run, playback.
// World axes: X = east (m), Z = -north (m), Y = (height - hmin) * exaggeration.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const $ = (id) => document.getElementById(id);
const TIER = { advisory: "Advisory", on_course: "On course", take_cover: "Take cover", evacuate: "Evacuate", approval: "Approval", all_clear: "All clear" };

// ---------------------------------------------------------------- state
const S = {
  overview: null, pick: null, area: null,          // area: {lat, lon, half_m, res_m, n, H (Float32Array), min, max, name}
  tool: "orbit", exag: 1.5,
  clusters: [], sites: [], flights: [], current: null,   // flights: {cls, agl, speed, alt_mode, waypoints: [[x,y]]}
  result: null, t: 0, playing: false, speed: 4, last: null,
};

// ---------------------------------------------------------------- three.js setup
const view = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
view.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.position = "absolute";
labels.domElement.style.inset = "0";
labels.domElement.style.pointerEvents = "none";
view.appendChild(labels.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1412);
const camera = new THREE.PerspectiveCamera(45, 1, 5, 200000);
camera.position.set(0, 4000, 6000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
scene.add(new THREE.AmbientLight(0xffffff, 1.0));

const groups = { terrain: new THREE.Group(), placed: new THREE.Group(), play: new THREE.Group() };
Object.values(groups).forEach((g) => scene.add(g));

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  labels.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);

// ---------------------------------------------------------------- terrain helpers
function heightAt(x, y) {
  const A = S.area;
  if (!A) return 0;
  const fx = (x + A.half_m) / A.res_m, fy = (y + A.half_m) / A.res_m;
  const ix = Math.max(0, Math.min(A.n - 2, Math.floor(fx))), iy = Math.max(0, Math.min(A.n - 2, Math.floor(fy)));
  const tx = Math.max(0, Math.min(1, fx - ix)), ty = Math.max(0, Math.min(1, fy - iy));
  const H = A.H, n = A.n;
  const h00 = H[iy * n + ix], h10 = H[iy * n + ix + 1], h01 = H[(iy + 1) * n + ix], h11 = H[(iy + 1) * n + ix + 1];
  return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty;
}
const W = (x, y, z) => new THREE.Vector3(x, (z - S.area.min) * S.exag, -y);   // local metres -> world
const fromWorld = (v) => [v.x, -v.z];
const scaleM = () => (S.area ? S.area.half_m / 60 : 50);                      // marker size

function label(text, cls) {
  const d = document.createElement("div");
  d.className = "label3d " + (cls || "");
  d.textContent = text;
  return new CSS2DObject(d);
}

function clearGroup(g) {
  while (g.children.length) {
    const c = g.children.pop();
    c.traverse?.((o) => { if (o.element) o.element.remove(); o.geometry?.dispose?.(); });
  }
}

function buildTerrain() {
  clearGroup(groups.terrain);
  const A = S.area, n = A.n, size = 2 * A.half_m;
  const geo = new THREE.PlaneGeometry(size, size, n - 1, n - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const h = A.H[(n - 1 - r) * n + c];                     // geometry rows run north to south
    pos.setY(r * n + c, (h - A.min) * S.exag);
  }
  geo.computeVertexNormals();
  const tex = new THREE.TextureLoader().load(A.texture);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex }));
  mesh.name = "terrain";
  groups.terrain.add(mesh);
  // 1 km grid ticks along the south edge for scale
  const pts = [];
  for (let x = -Math.floor(A.half_m / 1000) * 1000; x <= A.half_m; x += 1000) {
    pts.push(W(x, -A.half_m, heightAt(x, -A.half_m)), W(x, -A.half_m - scaleM() * 3, heightAt(x, -A.half_m)));
  }
  groups.terrain.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x8fa097 })));
  const n0 = label("N ↑ · 1 km ticks", "");
  n0.position.copy(W(A.half_m * 0.85, A.half_m * 0.95, A.max));
  groups.terrain.add(n0);
}

function frameCamera(top, box) {
  // box: [x0, y0, x1, y1] in local metres to frame (default: the whole area)
  const A = S.area;
  const [x0, y0, x1, y1] = box || [-A.half_m, -A.half_m, A.half_m, A.half_m];
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, r = Math.max(x1 - x0, y1 - y0, 200) / 2;
  const g = box ? heightAt(cx, cy) : (A.max + A.min) / 2;
  controls.target.copy(W(cx, cy, g));
  if (top) camera.position.copy(W(cx, cy - 1, g + r * 2.6));
  else camera.position.copy(W(cx, cy - r * 1.9, g + r * 1.3));
  controls.update();
}

function focusPlaced() {
  const xs = [], ys = [];
  S.clusters.forEach(([x, y]) => { xs.push(x); ys.push(y); });
  S.sites.forEach((s) => { xs.push(s.x - s.radius_m, s.x + s.radius_m); ys.push(s.y - s.radius_m, s.y + s.radius_m); });
  S.flights.forEach((f) => f.waypoints.forEach(([x, y]) => { xs.push(x); ys.push(y); }));
  if (!xs.length || !S.area) return;
  const pad = 150;
  frameCamera(false, [Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) + pad, Math.max(...ys) + pad]);
}

// ---------------------------------------------------------------- overview map
async function loadOverview() {
  const r = await fetch("/api/overview");
  S.overview = await r.json();
  const img = new Image();
  img.onload = () => { S.overview.img = img; drawOverview(); };
  img.src = S.overview.png;
  for (const p of S.overview.presets) $("preset").add(new Option(p.name, JSON.stringify(p)));
  for (const [k, v] of Object.entries(S.overview.drone_classes)) $("cls").add(new Option(v, k));
  $("cls").value = "mid";
  const amb = { desert: "Quiet desert", desert_wind: "Desert, windy", town: "Town", highway: "Near a highway" };
  for (const k of S.overview.ambient) $("ambient").add(new Option(amb[k] || k, k));
  loadRuns();
  S.examples = await (await fetch("/api/examples")).json();
  for (const x of S.examples) $("example").add(new Option(x.title || x.area_name, x.id));
}

$("example").addEventListener("change", async (e) => {
  const x = S.examples.find((v) => v.id === e.target.value);
  if (!x) return;
  await loadScenario(x);
  status("Example loaded. Press Run simulation.");
});

async function loadScenario(s) {
  // load the area, then put the scenario's clusters, sites, flights and settings in place
  if (!(await loadArea(s.origin[0], s.origin[1], s.half_m, s.area_name || "Scenario"))) return;
  S.clusters = s.clusters.map((c) => [...c]);
  S.sites = (s.sites || []).map((v) => ({ ...v }));
  S.flights = s.drones.map((d) => ({ ...d, waypoints: d.waypoints.map((w) => [...w]) }));
  S.current = null;
  $("terrainOn").checked = s.terrain !== false;
  if (s.mount) $("mount").value = s.mount;
  if (s.ambient) $("ambient").value = s.ambient;
  if (s.temp_c !== undefined) $("temp").value = s.temp_c;
  if (s.rh !== undefined) $("rh").value = s.rh;
  if (s.seed !== undefined) $("seed").value = s.seed;
  redrawPlaced();
  focusPlaced();
}

function drawOverview() {
  const cv = $("overview"), ctx = cv.getContext("2d"), O = S.overview;
  if (!O?.img) return;
  const [lo0, la0, lo1, la1] = O.bounds;
  cv.height = Math.round(cv.width * (la1 - la0) / ((lo1 - lo0) * Math.cos(24.5 * Math.PI / 180)));
  ctx.drawImage(O.img, 0, 0, cv.width, cv.height);
  const toPx = (lat, lon) => [(lon - lo0) / (lo1 - lo0) * cv.width, (la1 - lat) / (la1 - la0) * cv.height];
  const sel = S.pick || S.area;
  if (sel) {
    const half = +$("areaSize").value;
    const dlat = half / 110700, dlon = half / (111320 * Math.cos(sel.lat * Math.PI / 180));
    const [x0, y0] = toPx(sel.lat + dlat, sel.lon - dlon), [x1, y1] = toPx(sel.lat - dlat, sel.lon + dlon);
    ctx.strokeStyle = "#e0a040"; ctx.lineWidth = 2;
    ctx.strokeRect(x0 - 2, y0 - 2, Math.max(x1 - x0, 1) + 4, Math.max(y1 - y0, 1) + 4);
    ctx.beginPath(); ctx.arc((x0 + x1) / 2, (y0 + y1) / 2, 7, 0, 2 * Math.PI); ctx.stroke();
  }
}

$("overview").addEventListener("click", (e) => {
  const cv = e.currentTarget, r = cv.getBoundingClientRect(), [lo0, la0, lo1, la1] = S.overview.bounds;
  const lon = lo0 + (e.clientX - r.left) / r.width * (lo1 - lo0);
  const lat = la1 - (e.clientY - r.top) / r.height * (la1 - la0);
  S.pick = { lat: +lat.toFixed(4), lon: +lon.toFixed(4), name: `${lat.toFixed(3)}° N, ${lon.toFixed(3)}° E` };
  $("pick").textContent = `Selected ${S.pick.name}. Press Load area.`;
  drawOverview();
});
$("preset").addEventListener("change", (e) => {
  if (!e.target.value) return;
  const p = JSON.parse(e.target.value);
  S.pick = { lat: p.lat, lon: p.lon, name: p.name };
  $("areaSize").value = String(p.half_m);
  $("pick").textContent = `Selected ${p.name}. Press Load area.`;
  drawOverview();
});
$("areaSize").addEventListener("change", drawOverview);

async function loadArea(lat, lon, half, name, keepPlaced = false) {
  status(`Loading terrain for ${name}…`);
  const r = await fetch(`/api/terrain?lat=${lat}&lon=${lon}&half_m=${half}`);
  const d = await r.json();
  if (!r.ok) { status(d.error || "Could not load terrain"); return false; }
  const bytes = Uint8Array.from(atob(d.h), (c) => c.charCodeAt(0));
  S.area = { ...d, H: new Float32Array(bytes.buffer), name };
  if (!keepPlaced) { S.clusters = []; S.sites = []; S.flights = []; S.current = null; clearResult(); }
  $("areaName").textContent = `${name} · ${Math.round(2 * half / 1000)} km · ground ${Math.round(d.min)}–${Math.round(d.max)} m`;
  buildTerrain();
  frameCamera(false);
  redrawPlaced();
  drawOverview();
  status("Place mic clusters, a protected site and a drone flight, then run.");
  return true;
}
$("loadArea").addEventListener("click", () => {
  if (!S.pick) { status("Click the map or choose a preset first."); return; }
  loadArea(S.pick.lat, S.pick.lon, +$("areaSize").value, S.pick.name);
});

// ---------------------------------------------------------------- placement
document.querySelectorAll("[data-tool]").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
function setTool(t) {
  S.tool = t;
  document.querySelectorAll("[data-tool]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.tool === t)));
  controls.enabled = true;
  if (t === "path" && !S.current) newFlight();
  const msg = { orbit: "Drag to rotate, scroll to zoom, right-drag to pan.", cluster: "Click the ground to place a mic cluster.",
    ring: "Click the ground: 4 clusters in a square plus one in the middle.", site: "Click the ground to place a protected site.",
    path: "Click waypoints for the drone. New flight starts another." };
  status(msg[t]);
}
function newFlight() {
  S.current = { cls: $("cls").value, agl: +$("agl").value, speed: +$("speed").value, alt_mode: $("altMode").value, waypoints: [] };
  S.flights.push(S.current);
  redrawPlaced();
}
$("newFlight").addEventListener("click", () => { newFlight(); setTool("path"); });
$("clearAll").addEventListener("click", () => { S.clusters = []; S.sites = []; S.flights = []; S.current = null; clearResult(); redrawPlaced(); });
for (const id of ["cls", "agl", "speed", "altMode"]) $(id).addEventListener("change", () => {
  if (S.current) Object.assign(S.current, { cls: $("cls").value, agl: +$("agl").value, speed: +$("speed").value, alt_mode: $("altMode").value });
  redrawPlaced();
});

const ray = new THREE.Raycaster();
let down = null;
renderer.domElement.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5 || S.tool === "orbit" || !S.area) return;
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), camera);
  const hit = ray.intersectObject(groups.terrain.getObjectByName("terrain"))[0];
  if (!hit) return;
  const [x, y] = fromWorld(hit.point).map((v) => Math.round(v));
  if (S.tool === "cluster") S.clusters.push([x, y]);
  else if (S.tool === "ring") {
    const s = +$("spacing").value / 2;
    S.clusters.push([x - s, y - s], [x + s, y - s], [x + s, y + s], [x - s, y + s], [x, y]);
  } else if (S.tool === "site") S.sites.push({ name: `Site ${S.sites.length + 1}`, x, y, radius_m: +$("siteR").value });
  else if (S.tool === "path") { if (!S.current) newFlight(); S.current.waypoints.push([x, y]); }
  clearResult();
  redrawPlaced();
});

function flightZ(f, x, y) {
  return f.alt_mode === "fixed" && f.waypoints.length
    ? Math.max(heightAt(...f.waypoints[0]) + f.agl, heightAt(x, y) + 5) : heightAt(x, y) + f.agl;
}

function redrawPlaced() {
  clearGroup(groups.placed);
  if (!S.area) return;
  const s = scaleM();
  S.clusters.forEach(([x, y], k) => {
    const g = heightAt(x, y);
    const m = new THREE.Mesh(new THREE.ConeGeometry(s * 0.8, s * 2, 3), new THREE.MeshBasicMaterial({ color: 0xc9d2cc }));
    m.position.copy(W(x, y, g)).add(new THREE.Vector3(0, s, 0));
    m.name = `cluster${k}`;
    groups.placed.add(m);
    const l = label(`C${k + 1}`);
    l.position.copy(W(x, y, g)).add(new THREE.Vector3(0, s * 3, 0));
    groups.placed.add(l);
  });
  for (const site of S.sites) {
    const pts = [];
    for (let i = 0; i <= 64; i++) {
      const a = (i / 64) * 2 * Math.PI, x = site.x + site.radius_m * Math.cos(a), y = site.y + site.radius_m * Math.sin(a);
      pts.push(W(x, y, heightAt(x, y) + 2));
    }
    groups.placed.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x9fe0bd })));
    const l = label(site.name, "site");
    l.position.copy(W(site.x, site.y, heightAt(site.x, site.y))).add(new THREE.Vector3(0, s * 2, 0));
    groups.placed.add(l);
  }
  S.flights.forEach((f, i) => {
    if (!f.waypoints.length) return;
    const dense = [];
    for (let j = 1; j < f.waypoints.length; j++) {
      const [x0, y0] = f.waypoints[j - 1], [x1, y1] = f.waypoints[j];
      const n = Math.max(2, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (S.area.res_m * 2)));
      for (let q = 0; q <= n; q++) { const x = x0 + (x1 - x0) * q / n, y = y0 + (y1 - y0) * q / n; dense.push(W(x, y, flightZ(f, x, y))); }
    }
    if (dense.length) groups.placed.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(dense),
      new THREE.LineDashedMaterial({ color: 0xf2f2ea, dashSize: s, gapSize: s * 0.6 })).computeLineDistances());
    f.waypoints.forEach(([x, y], j) => {
      const top = W(x, y, flightZ(f, x, y)), bot = W(x, y, heightAt(x, y));
      const m = new THREE.Mesh(new THREE.SphereGeometry(s * 0.45, 12, 8), new THREE.MeshBasicMaterial({ color: j === 0 ? 0xe0a040 : 0xf2f2ea }));
      m.position.copy(top);
      groups.placed.add(m);
      groups.placed.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([top, bot]), new THREE.LineBasicMaterial({ color: 0x8fa097 })));
      if (j === 0) { const l = label(`Flight ${i + 1}`, "drone"); l.position.copy(top).add(new THREE.Vector3(0, s * 1.5, 0)); groups.placed.add(l); }
    });
  });
  renderItems();
  $("run").disabled = !(S.area && S.clusters.length >= 2 && S.flights.some((f) => f.waypoints.length >= 2));
}

function renderItems() {
  const ul = $("items");
  ul.innerHTML = "";
  const add = (what, where, del) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="what"></span><span class="where"></span><button title="Remove">×</button>`;
    li.querySelector(".what").textContent = what;
    li.querySelector(".where").textContent = where;
    li.querySelector("button").onclick = () => { del(); clearResult(); redrawPlaced(); };
    ul.appendChild(li);
  };
  S.clusters.forEach(([x, y], k) => add(`C${k + 1}`, `${x} E, ${y} N · ground ${Math.round(heightAt(x, y))} m`, () => S.clusters.splice(k, 1)));
  S.sites.forEach((s, k) => add(s.name, `${s.x} E, ${s.y} N · r ${s.radius_m} m`, () => S.sites.splice(k, 1)));
  S.flights.forEach((f, k) => add(`Flight ${k + 1}`, `${S.overview.drone_classes[f.cls]} · ${f.agl} m · ${f.speed} m/s · ${f.waypoints.length} pts`,
    () => { S.flights.splice(k, 1); if (S.current === f) S.current = null; }));
}

// ---------------------------------------------------------------- run
function scenarioJSON() {
  return {
    origin: [S.area.lat, S.area.lon], half_m: S.area.half_m, area_name: S.area.name,
    terrain: $("terrainOn").checked, mount: $("mount").value,
    clusters: S.clusters, sites: S.sites,
    drones: S.flights.filter((f) => f.waypoints.length >= 2).map((f) => ({ ...f })),
    ambient: $("ambient").value, temp_c: +$("temp").value, rh: +$("rh").value, seed: +$("seed").value,
  };
}
$("run").addEventListener("click", async () => {
  $("run").disabled = true;
  $("log").innerHTML = "";
  const r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenarioJSON()) });
  const d = await r.json();
  if (!r.ok) { logLine(d.error, true); $("run").disabled = false; return; }
  status(`Run ${d.id} started`);
  poll(d.id);
});
function logLine(text, err) {
  const li = document.createElement("li");
  li.textContent = text;
  if (err) li.className = "err";
  $("log").appendChild(li);
  $("log").scrollTop = 1e9;
}
async function poll(id) {
  let seen = 0;
  for (;;) {
    await new Promise((res) => setTimeout(res, 1500));
    const r = await fetch(`/api/job/${id}`);
    const d = await r.json();
    if (d.progress) { for (const [t, m] of d.progress.slice(seen)) logLine(`${t.toFixed(0).padStart(4)} s  ${m}`); seen = d.progress.length; }
    if (d.status === "error") { logLine(d.error, true); status("Run failed"); $("run").disabled = false; return; }
    if (d.summary) { d.id = id; showResult(d); $("run").disabled = false; loadRuns(); return; }
  }
}
async function loadRuns() {
  const r = await fetch("/api/runs");
  const rows = await r.json();
  const sel = $("runs");
  sel.length = 1;
  for (const x of rows) sel.add(new Option(`${x.id} · ${x.area || "area"} · ${x.clusters} clusters${x.terrain ? "" : " · flat"}`, x.id));
}
$("runs").addEventListener("change", async (e) => {
  if (!e.target.value) return;
  const d = await (await fetch(`/api/job/${e.target.value}`)).json();
  await loadScenario(d.scenario);
  d.id = e.target.value;
  showResult(d);
  focusPlaced();
});

// ---------------------------------------------------------------- results and playback
function clearResult() {
  S.result = null; S.playing = false;
  clearGroup(groups.play);
  $("bottom").hidden = true;
  resize();
}

function showResult(R) {
  S.result = R;
  clearGroup(groups.play);
  const s = scaleM();
  const P = (R.play = {});
  P.truth = R.drones.map((d) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(s * 0.7, 16, 10), new THREE.MeshBasicMaterial({ color: 0xf2f2ea }));
    const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xf2f2ea, transparent: true, opacity: 0.5 }));
    const l = label("", "drone");
    groups.play.add(m, trail, l);
    return { m, trail, l, d };
  });
  P.tracks = new THREE.Group();
  groups.play.add(P.tracks);
  P.rays = R.clusters.map(() => { const l = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xe0a040 })); groups.play.add(l); return l; });
  P.los = R.clusters.map(() => { const l = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x57c48a, transparent: true, opacity: 0.6 })); groups.play.add(l); return l; });
  // results panels
  const sm = R.summary, f = (v, u = " s") => (v === null || v === undefined ? "—" : `${v}${u}`);
  $("summary").innerHTML = `<dl>
    <dt>Warning before site</dt><dd class="big">${sm.warning_s === null ? (sm.drone_enters_site_s === null ? "drone never enters a site" : "no take-cover alert") : sm.warning_s + " s"}</dd>
    <dt>Drone first audible</dt><dd>${f(sm.first_audible_s)}</dd>
    <dt>CNN first hears it</dt><dd>${f(sm.first_cnn_s)}</dd>
    <dt>First position</dt><dd>${f(sm.first_track_s)}</dd>
    <dt>First alert</dt><dd>${f(sm.first_alert_s)}</dd>
    <dt>Take cover sent</dt><dd>${f(sm.first_take_cover_s)}</dd>
    <dt>Drone enters site</dt><dd>${f(sm.drone_enters_site_s)}</dd>
    <dt>Direction error</dt><dd>${f(sm.median_direction_error_deg, "° median")}</dd>
    <dt>Position error</dt><dd>${f(sm.median_track_error_m, " m median")}</dd>
    <dt>CNN frames</dt><dd>${sm.cnn_frames_vs_audible.cnn_on_audible} of ${sm.cnn_frames_vs_audible.audible} audible · ${sm.cnn_frames_vs_audible.cnn_on_inaudible} when inaudible</dd>
    <dt>Terrain</dt><dd>${R.scenario.terrain === false ? "off (flat ground)" : "on"} · run ${R.timing ? R.timing.total_s + " s" : ""}</dd></dl>`;
  $("clusterTable").innerHTML = `<tr><th>Cluster</th><th>Ground</th><th>Audible</th><th>CNN</th><th>Blocked</th><th>Listen</th></tr>` +
    sm.per_cluster.map((c) => `<tr><td>C${c.k + 1}</td><td>${c.ground_m} m</td><td>${Math.round(100 * c.audible_frac)}%</td><td>${Math.round(100 * c.cnn_frac)}%</td><td>${Math.round(100 * c.blocked_frac)}%</td><td><button data-k="${c.k}">▶</button></td></tr>`).join("");
  $("clusterTable").querySelectorAll("button[data-k]").forEach((b) => b.addEventListener("click", () => listen(+b.dataset.k)));
  const ticks = $("ticks");
  ticks.innerHTML = "";
  for (const a of R.alerts) if (a.to === "operators") {
    const i = document.createElement("i");
    i.style.left = `${(100 * a.t) / R.duration}%`;
    i.style.background = `var(--t-${a.tier}, #888)`;
    i.title = `${a.t} s ${TIER[a.tier] || a.tier}`;
    ticks.appendChild(i);
  }
  $("scrub").max = R.duration;
  $("bottom").hidden = false;
  resize();
  S.t = 0; S.shown = -1;
  render(0);
  status(`Run ${R.id || ""} ready: press Play`);
}

let audio = null;
function listen(k) {
  if (audio) audio.pause();
  audio = new Audio(`/api/job/${S.result.id}/audio/${k}`);
  audio.currentTime = S.t;
  audio.play().catch(() => {});
  status(`Playing cluster C${k + 1} from ${S.t.toFixed(1)} s (16 kHz reference mic)`);
}

function interpPath(d, t) {
  const T = d.t;
  if (t <= T[0]) return d.p[0];
  if (t >= T[T.length - 1]) return d.p[d.p.length - 1];
  let lo = 0, hi = T.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
  const a = d.p[lo], b = d.p[hi], u = (t - T[lo]) / (T[hi] - T[lo]);
  return [0, 1, 2].map((i) => a[i] + u * (b[i] - a[i]));
}

function render(t) {
  const R = S.result;
  if (!R) return;
  const P = R.play, s = scaleM();
  const i = Math.max(0, Math.min(R.t.length - 1, Math.floor(t * R.frame_hz)));
  $("clock").textContent = `${t.toFixed(1)} s`;
  $("scrub").value = t;
  P.truth.forEach((o, di) => {
    const p = interpPath(o.d, t);
    o.m.position.copy(W(p[0], p[1], p[2]));
    const pts = [];
    for (let j = 0; j < o.d.t.length && o.d.t[j] <= t; j += 2) pts.push(W(o.d.p[j][0], o.d.p[j][1], o.d.p[j][2]));
    o.trail.geometry.dispose();
    o.trail.geometry = new THREE.BufferGeometry().setFromPoints(pts.length > 1 ? pts : [o.m.position, o.m.position]);
    o.l.position.copy(o.m.position).add(new THREE.Vector3(0, s * 1.6, 0));
    o.l.element.textContent = `drone ${di + 1} · ${Math.round(p[2] - heightAt(p[0], p[1]))} m up`;
  });
  const truth0 = R.drones.length ? interpPath(R.drones[0], t) : null;
  R.clusters.forEach((c, k) => {
    const hear = R.p_cnn[k][i] >= 0.5;
    const mesh = groups.placed.getObjectByName(`cluster${k}`);
    if (mesh) mesh.material.color.set(hear ? 0xe0a040 : 0xc9d2cc);
    const base = W(c.x, c.y, c.z).add(new THREE.Vector3(0, s * 1.2, 0));
    // flat clusters measure the compass bearing; their elevation is weak (design D-30), so draw bearings level
    const u = R.u[k][i], used = R.used[k][i] && $("showRays").checked;
    const h = Math.hypot(u[0], u[1]) || 1, len = S.area.half_m * 0.5;
    P.rays[k].visible = !!used;
    if (used) {
      P.rays[k].geometry.dispose();
      const end = W(c.x + (u[0] / h) * len, c.y + (u[1] / h) * len, c.z).add(new THREE.Vector3(0, s * 1.2, 0));
      P.rays[k].geometry = new THREE.BufferGeometry().setFromPoints([base, end]);
    }
    const showLos = $("showLos").checked && truth0 && R.screen[k].length;
    P.los[k].visible = !!showLos;
    if (showLos) {
      const blocked = R.screen[k][0].blocked[i];
      P.los[k].material.color.set(blocked ? 0xe0566a : 0x57c48a);
      P.los[k].geometry.dispose();
      P.los[k].geometry = new THREE.BufferGeometry().setFromPoints([W(c.x, c.y, c.z), W(truth0[0], truth0[1], truth0[2])]);
    }
  });
  clearGroup(P.tracks);
  for (const tr of R.frames[i].tracks) {
    const pos = W(tr.x, tr.y, tr.z);
    const m = new THREE.Mesh(new THREE.SphereGeometry(s * 0.6, 14, 10), new THREE.MeshBasicMaterial({ color: 0x63a4ff }));
    m.position.copy(pos);
    const e = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 12), new THREE.MeshBasicMaterial({ color: 0x63a4ff, transparent: true, opacity: 0.18, depthWrite: false }));
    e.scale.set(2 * tr.sx, 2 * Math.min(tr.sz, 300) * S.exag, 2 * tr.sy);
    e.position.copy(pos);
    const v = new THREE.Line(new THREE.BufferGeometry().setFromPoints([pos, W(tr.x + tr.vx * 8, tr.y + tr.vy * 8, tr.z + tr.vz * 8)]), new THREE.LineBasicMaterial({ color: 0x63a4ff }));
    const l = label(`track ${tr.id}`, "track");
    l.position.copy(pos).add(new THREE.Vector3(0, -s * 1.8, 0));
    P.tracks.add(m, e, v, l);
  }
  const tr0 = R.frames[i].tracks[0];
  const nh = R.p_cnn.filter((row) => row[i] >= 0.5).length;
  $("nowstat").textContent = `${nh}/${R.clusters.length} clusters hear it` + (tr0 && truth0 ? ` · track error ${Math.round(Math.hypot(tr0.x - truth0[0], tr0.y - truth0[1]))} m` : "");
  renderFeed(t);
}

function renderFeed(t) {
  const R = S.result, vis = R.alerts.filter((a) => a.t <= t);
  if (vis.length === S.shown) return;
  S.shown = vis.length;
  const ol = $("feed");
  ol.innerHTML = vis.length ? "" : `<li class="empty">No alerts yet. They appear here at the second they were sent.</li>`;
  for (const a of vis.slice().reverse()) {
    const li = document.createElement("li");
    li.style.setProperty("--c", `var(--t-${a.tier}, #888)`);
    li.innerHTML = `<div class="m"><b>${a.t.toFixed(1)} s</b><span class="tier">${TIER[a.tier] || a.tier}</span><span></span></div><div class="x"></div>`;
    li.querySelector(".m span:last-child").textContent = `→ ${a.to}`;
    li.querySelector(".x").textContent = a.text;
    ol.appendChild(li);
  }
}

$("play").addEventListener("click", () => {
  if (!S.result) return;
  if (S.t >= S.result.duration) { S.t = 0; S.shown = -1; }
  S.playing = !S.playing; S.last = null;
  $("play").textContent = S.playing ? "Pause" : "Play";
});
document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
  S.speed = +b.dataset.speed;
  document.querySelectorAll("[data-speed]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
}));
$("scrub").addEventListener("input", (e) => { S.t = +e.target.value; S.shown = -1; render(S.t); });
$("exag").addEventListener("input", (e) => { S.exag = +e.target.value; if (S.area) { buildTerrain(); redrawPlaced(); if (S.result) showResult(S.result); } });
$("topView").addEventListener("click", () => S.area && frameCamera(true));
$("tiltView").addEventListener("click", () => S.area && frameCamera(false));
$("focus").addEventListener("click", focusPlaced);
for (const id of ["showLos", "showRays"]) $(id).addEventListener("change", () => render(S.t));

function status(msg) { $("status").textContent = msg; }

function loop(now) {
  if (S.playing && S.result) {
    if (S.last !== null) S.t = Math.min(S.result.duration, S.t + ((now - S.last) / 1000) * S.speed);
    S.last = now;
    render(S.t);
    if (S.t >= S.result.duration) { S.playing = false; $("play").textContent = "Replay"; }
  }
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
  requestAnimationFrame(loop);
}
resize();
loadOverview();
requestAnimationFrame(loop);
