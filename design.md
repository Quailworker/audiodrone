# Acoustic Drone Detection and Tracking: Design

- **Status:** draft v0.6, 2026-10-23.
  - v0.6 compares the v0.5 results with the theory (5.11): what matched, three deviations (model relies on > 1 kHz; real range shorter than estimated, G ≈ 2–7 dB; log vs mel draw in sim), and what is still untested. Updates 8.3, F3, section 15, open questions and the Al-Emadi licence caveat (18).
  - v0.5 adds a failure analysis of model B (5.10): real-sound false alarms (critical), codec and high-pass sensitivity, moving engines, masking, real detection range ~150 m.
  - v0.5 records the first trained models and how to replicate them: the 6,400-scene sim_v1 dataset, the frame-label rule actually used (`snr_lines2`, 5.8.6), the training recipe, results for CNN14 + log-frequency (B), CNN14 + stock mel (A) and a small CNN trained from scratch (C), and the fixes found by code review (5.9). Full scale changed to 120 dB SPL (14.6). Replication steps are in README.md; AGENTS.md is the test checklist.
  - v0.4 records the CNN build decisions: the real CNN14 checkpoint (32 kHz), render-then-convert, sim-only training with Joseph's own field recordings as the test, detection-only scope, drone classes incl. two-stroke (4.4), the scenario simulator (14.6), the field test protocol (14.7) and flat ground clusters as an alternative to the tetrahedron (7.8). New decisions D-24 to D-31.
  - v0.2 added CNN input representation and training data (sections 5.5–5.6, D-19 to D-21).
  - v0.3 adds the pretrained CNN14 model (section 5.7, D-22 and D-23) and a known-flaws review (section 16).
- **Owner:** Joseph
- **Scope:** everything discussed so far about detecting drones by sound, estimating their position, height and velocity, and the microphone hardware.

---

## 0. How to read this document

This document records **where each idea came from**, so that if something turns out to be wrong you can trace it back and fix it at the source.

**Origin tags**

| Tag | Meaning |
|---|---|
| **[J]** | Joseph's idea, question or decision. These are the backbone of the design. |
| **[J←C]** | Claude proposed it and Joseph adopted it. |
| **[C]** | Claude's extension, filling in detail under one of Joseph's ideas. |
| **[C revises J]** | Claude changed part of one of Joseph's ideas. The reason is given, so it can be checked and reverted if the reason is wrong. |

**Evidence tags**

| Tag | Meaning |
|---|---|
| *(derived)* | Follows from the stated equations and assumptions. Check the assumptions. |
| *(simulated)* | Tested in a simulation written during the discussion. Best-case: real data will be worse. |
| *(estimate)* | A rough engineering number, not measured. |
| *(unverified)* | A product spec, price or literature claim that has not been checked against a primary source. |
| *(open)* | Not decided yet. |

Each decision has an ID (**D-nn**) and lists the decisions it **depends on**. If a decision is found to be wrong, everything that depends on it should be rechecked.

---

## 1. Goal [J]

Detect drones by their sound with a network of microphones, and estimate for each drone:

- whether a drone is present, and when it is heard;
- what kind of drone it is;
- where it is, **including its height**;
- its velocity;
- whether it is moving towards or away from a microphone or a location.

---

## 2. System architecture

### 2.1 Architecture as proposed by Joseph [J]

Paraphrased from Joseph's description, with typos corrected:

> Initial detection is done by an array of **3 microphones in one cluster**. Clusters are placed in assorted areas so that **each cluster's range slightly overlaps** with another cluster. The microphones convert the sound they hear into the **time-frequency domain**, store it as data, and **transfer it to a server** that runs a **CNN**. The CNN looks for drone frequencies, identifies whether it is a drone, and is **trained on what kind of drone makes what frequency**. Whether the drone is moving towards or away is determined by **finding the gradient of the frequency** at a single frequency value, by turning that frequency point into a **functional**. **Acoustic triangulation** is then used to determine whether the drone is moving away from or towards the microphone.

Later decisions by Joseph:

- Use a **4th microphone on top**, making each cluster a triangular pyramid (tetrahedron), so the height of the drone can be found. (D-07)
- Use a **Kalman filter** to estimate and predict the drone's state, including height. (D-12)
- (v0.4) Start the CNN from **CNN14 as shipped by NeMo**, fine-tune on **rendered drone audio converted to a log-frequency spectrogram**. (D-22)
- (v0.4) **Train on simulation only**; test on his own field recordings of a drone flying over a mic. (D-25)
- (v0.4) First CNN does **detection only**. (D-26)
- (v0.4) Drone classes: 5–7" FPV, 7–12" electric, ~10"+ two-stroke. (D-27)
- (v0.4) Generate data by **rendering whole scenarios**: mics at fixed positions, a drone flying over. (D-29)
- (v0.4) Prefer microphones **on one plane / on the ground surface** rather than a raised tetrahedron, if height can still be found. (D-30)

### 2.2 Current architecture (after the revisions below)

```
CLUSTER (one per site)                                     [J] clusters; [J←C] 4 mics
  4 MEMS mics, regular tetrahedron, 20 cm edge          (under review: flat ground cluster, D-30)
  audio → STFT (time-frequency) [J]
        → whitening + harmonic comb trigger [C]
          on trigger:
            → 3D direction to the drone (GCC-PHAT on 6 mic pairs) [C revises J: done on cluster]
            → frequency track f(t) of the strongest harmonic (time-frequency functional) [J]
            → drone-only loudness from the comb bins [C]
            → report: GPS timestamp, direction, f(t), loudness, compressed spectrogram clip

SERVER                                                     [J] server + CNN
  CNN/CRNN: is it a drone, and when [J]; what type [J, revised by C]
  match detections across clusters (time + f0) [C]
  Kalman filter (EKF): position incl. height, 3D velocity, f0 [J]
  towards/away: from the tracked position over time [J triangulation; C revises J frequency gradient]
  outputs: track, height, velocity, heading, towards/away, time to arrival
```

---

## 3. Decision register

| ID | Decision | Origin | Evidence | Depends on | Status |
|---|---|---|---|---|---|
| D-01 | Detect drones in the frequency domain by finding **evenly spaced spikes** (harmonics) | [J] | derived | — | Adopted |
| D-02 | Use a **CNN** to find **when** a drone is heard (sound event detection) | [J] | open: architecture not trained yet | D-01 | Adopted |
| D-03 | Feed explicit harmonic-comb features into the CNN alongside the spectrogram (CRNN) | [C] | open | D-01, D-02 | Proposed |
| D-04 | Identify drone **type** with the CNN | [J] | open | D-02 | Adopted, revised by D-05 |
| D-05 | Type must not rely on frequency alone; use whine/BPF ratio, rotor count, harmonic shape; needs real recordings | [C revises J] | derived | D-04 | Proposed |
| D-06 | Microphones grouped in **clusters**, spread over the area | [J] | — | — | Adopted |
| D-07 | Each cluster is a **4-mic regular tetrahedron** (originally 3 mics) | [J←C] | derived | D-06 | **Under review (D-30)** |
| D-08 | Tetrahedron edge **20 cm**; direction finding uses frequencies **< 860 Hz** | [C] | derived | D-07 | Proposed |
| D-09 | Cluster ranges must **overlap substantially** (spacing ≈ range), not slightly | [C revises J] | derived | D-06 | Proposed |
| D-10 | **Direction finding runs on the cluster**; server receives directions, not only spectrograms | [C revises J] | derived | D-06, D-07 | Proposed |
| D-11 | Use a **time-frequency functional** to get one frequency's amplitude and exact frequency over time | [J] | simulated | D-01 | Adopted |
| D-12 | Use a **Kalman filter** (EKF) to estimate and predict position, height and velocity | [J] | simulated | D-07, D-10 | **Adopted** |
| D-13 | Kalman state includes the drone's pitch f₀, with **fast-changing f₀** allowed | [C] | simulated | D-12 | Proposed |
| D-14 | Height comes from **3D triangulation of tetrahedral directions inside the Kalman filter** | [J] + [C] | simulated | D-07, D-12 | **Adopted** |
| D-15 | Towards/away is decided from the **tracked position**, with the frequency gradient as a supporting cue | [J] triangulation; [C revises J] | derived | D-11, D-12 | Proposed |
| D-16 | Velocity from **Doppler at several clusters at once** (inside the Kalman filter) | [C] | simulated | D-12 | Proposed |
| D-17 | Train with **simulated drone audio**, test on real recordings | [J] simulate; [C] split | simulated | D-02 | Adopted; detailed by D-25 |
| D-18 | Which drone types are in scope (affects whether lawnmower/motorbike sounds are "not drone") | [J] | — | D-04, D-17 | **Resolved by D-27** (airframe of two-strokes still open) |
| D-19 | CNN input is a **converted view**: a **log-frequency spectrogram with loudness in dB** | [J] converted view; [C] log axis | derived; not yet tested on a CNN | D-02 | **Adopted** |
| D-20 | Feed several views: log spectrogram, harmonic-salience map, per-frame features, optionally beamformed audio | [J] converted view; [C] specific views | open | D-03, D-19 | Proposed |
| D-21 | Training data: widely varied simulated drones on real backgrounds, then fine-tune and test on real recordings; type identification from real per-model recordings | [J] simulated; [C] mix and order | open | D-05, D-17 | **Revised by D-25** (no real fine-tuning for now) |
| D-22 | Start from **pretrained CNN14 as shipped by NeMo Curator: `Cnn14_DecisionLevelMax` (32 kHz, frame-level)**, fine-tuned on simulated (rendered) audio converted to a log-frequency spectrogram. v0.3's "frame-level 16 kHz" checkpoint does not exist | [J]; checkpoint [C] | checkpoint verified; not trained yet | D-02, D-19, D-21 | **Adopted (revised v0.4)** |
| D-23 | CNN14 front end: start with stock log-mel as a baseline, then a mel scale with a lower bend point (log-spaced from ~50 Hz), longer window (n_fft 4096 at 32 kHz; 5.8.4) | [C] | derived; not tested | D-19, D-22 | Proposed |
| D-24 | **Render time-domain audio, then apply the same conversion as for real mic audio**; never synthesise spectrograms directly | [J←C] | derived | D-19, D-22 | **Adopted** |
| D-25 | **Train on simulated data only; test on Joseph's own field recordings** (never trained on), plus a harder held-out sim test set | [J]; held-out design [C] | open | D-17, D-21 | **Adopted** |
| D-26 | First CNN does **detection only** (no type identification) | [J] | — | D-02, D-04 | **Adopted** |
| D-27 | Drone classes: **5–7" FPV, 7–12" electric, ~10"+ two-stroke petrol** | [J]; sound models [C] | estimate | D-18 | **Adopted** |
| D-28 | Handling two-stroke drones vs ground engines: proposed 3 outputs, electric drone / engine / nothing | [C] | open | D-26, D-27 | **Open** (labels kept for all options) |
| D-29 | Data comes from a **scenario simulator**: mics at fixed positions, sources flying trajectories, propagation physics per mic, long clips covering whole passes | [J] idea; [C] physics | derived; not validated | D-17, D-24 | **Adopted** |
| D-30 | Mic clusters **flat, on the ground surface**; height from overlapping clusters + Doppler in the EKF (instead of a tetrahedron) | [J] idea; [C] analysis | derived; to be simulated | D-07, D-09, D-12, D-14 | Proposed |
| D-31 | Field test protocol: drone passes at several distances with a flight log, plus no-drone and confuser recordings | [J] test; [C] protocol | — | D-25 | Proposed |

---

## 4. How drones sound [C]

This physics is used by almost every later section. If it is wrong, recheck sections 5, 6, 11, 12 and 15.

### 4.1 Sound sources

1. **Blade-passing tone.** Each blade passes the listener once per revolution:
   ```
   f_shaft = RPM / 60
   BPF     = n_blades × f_shaft
   example: 2 blades, 6000 RPM → f_shaft = 100 Hz → BPF = 200 Hz
   ```
   Harmonics appear at 2·BPF, 3·BPF and so on, with falling amplitude (≈ 1/k¹·² in the simulator). *(derived; the amplitude law is a modelling choice)*
2. **Motor whine** at the electrical frequency: `f_motor = pole_pairs × f_shaft` (a 14-pole motor has 7 pole pairs, so 7 × 100 = 700 Hz). *(derived)*
3. **Broadband aerodynamic noise**, modulated at the BPF. *(modelling choice)*
4. **Several rotors** at slightly different speeds (±3% in the simulator) cause **beating**. *(modelling choice)*

### 4.2 Drone sounds are not universal

| Factor | Effect | Evidence |
|---|---|---|
| Propeller diameter D | At a similar tip speed v_tip: `BPF = n_blades · v_tip / (π·D)`, so **bigger drones sound lower** | derived |
| Weight / payload | Hover thrust: `N·k·ω² = m·g → ω ∝ √m`. +20% mass gives about +10% pitch | derived |
| Motor pole count | Changes the whine frequency | derived |
| Flight state | Hover: tight comb. Forward flight: front and rear rotors split. Climb/brake: the whole comb slides | derived (qualitative) |
| Fixed-wing / piston drones | Single comb; piston engines sound like mopeds or lawnmowers | unverified |

Approximate BPF by drone class *(estimate)*:

| Class | Prop D | RPM | Blades | BPF |
|---|---|---|---|---|
| Heavy-lift / agricultural | ~1.0 m | ~1,800 | 2 | ~60 Hz |
| Consumer camera drone | ~0.24 m | ~6,500 | 2 | ~217 Hz |
| 5-inch FPV racer | ~0.13 m | 15,000–30,000 | 3 | 750 Hz and up |

**Range to cover: about 40 Hz to 2 kHz** for the fundamental.

### 4.3 A property that does not change with RPM [C]

```
f_whine / BPF = (pole_pairs × f_shaft) / (n_blades × f_shaft) = pole_pairs / n_blades
```

RPM cancels out, so this ratio is a fingerprint of the drone's design and a candidate feature for type identification (D-05). *(derived)*

### 4.4 Drone classes in scope (D-27) [J], sound models [C]

**Joseph's decision [J]:** cover electric drones from **5-inch to 7-inch** (FPV), **7-inch to 12-inch**, and from about **10 inch upward, two-stroke petrol engines**.

| Class | Power | Props | RPM hover → full | Main tone (BPF) | Other content |
|---|---|---|---|---|---|
| 5–7" FPV | Electric, 4 rotors | 3-blade (some 2) | ~12,000 → 30,000 | ~600 → 1,500 Hz | Motor whine 2–5 kHz |
| 7–12" | Electric, 4–8 rotors | 2–3 blade | ~5,000 → 14,000 | ~170 → 700 Hz | Whine 0.5–2 kHz |
| ~10"+ two-stroke | Petrol, single-cylinder | 2-blade, direct drive | ~2,000 idle → 9,000 | ~70 → 300 Hz | Exhaust pulse train, very rich in harmonics |

All numbers *(estimate)*. Overall: fundamentals ~70 Hz to ~1.5 kHz, useful content up to ~5 kHz, which fits a 32 kHz sample rate (Nyquist 16 kHz) and CNN14's 50 Hz–14 kHz band.

**Prop size and RPM.** At a roughly constant tip speed (tip Mach number), RPM scales inversely with diameter *(derived)*:
```
v_tip = π · D · RPM/60     →     RPM ≈ 60 · v_tip / (π · D)
D = 5" (0.127 m), v_tip ≈ 150 m/s  →  RPM ≈ 22,600  →  3-blade BPF ≈ 1,130 Hz
D = 12" (0.305 m), v_tip ≈ 120 m/s →  RPM ≈ 7,500   →  2-blade BPF ≈ 250 Hz
```

**Electric multirotor model** [C]: each rotor i is its own harmonic comb at BPF_i = n_b · RPM_i/60, plus motor whine at p · RPM_i/60 (p = pole pairs). Rotors differ by a few percent, so at harmonic k two rotors **beat** at
```
f_beat,k = k · |BPF_i − BPF_j|
example: BPF 200 Hz, rotors 3% apart → 6 Hz beat at k = 1, 60 Hz beat at k = 10
```
The beating rate grows with harmonic number. This is a multirotor fingerprint (also the cause of flaw F2).

**Two-stroke model** [C] *(derived; parameters estimated)*. A single-cylinder two-stroke fires **once per revolution**:
```
f_shaft = RPM/60,   f_fire = f_shaft,   BPF = n_b · f_shaft   (direct drive, so all lines are harmonics of f_shaft)
```
The exhaust is a train of pressure pulses g(t) repeated every T = 1/f_fire:
```
p(t) = Σₙ g(t − nT)   →   P(f) = f_fire · Σₖ G(k·f_fire) · δ(f − k·f_fire)
```
A short pulse (width τ_p) has a nearly flat |G(f)| up to about 1/τ_p, so **many harmonics are strong**: the "rasp". Cycle-to-cycle timing jitter σ_T makes high harmonics incoherent:
```
coherent amplitude of harmonic k  ∝  exp(−½ · (2π · k · f_fire · σ_T)²)
```
Low harmonics stay as sharp lines and high harmonics smear into noise. A 2-blade prop puts its BPF on harmonic 2, reinforcing even harmonics.

**Electric vs two-stroke**

| | Electric multirotor | Two-stroke |
|---|---|---|
| Source | 4–8 rotors at slightly different RPM | One engine |
| Comb | Several close combs that beat | One very strong, very regular comb |
| Character | Smooth hum and whine | Harsh rasp (chainsaw, moped) |
| Throttle | Each rotor changes separately (that is how it steers) | Whole comb slides together |

**Confusion with ground engines (D-28, open).** Chainsaws, leaf blowers, mopeds and lawnmowers use the same kind of engine, so on one mic they can sound almost the same as a two-stroke drone. Options:
1. Label ground two-strokes "not drone" and let the CNN look for subtle cues (prop tone over the engine, smooth airborne throttle vs stop-start revving). Risk: missing real two-stroke drones.
2. **Proposed:** three outputs, **electric drone / engine / nothing**. "Engine" means "could be a drone"; the array decides later (an engine up in the air is a drone).
3. Leave ground engines out, so every engine is a drone. Simple, but false alarms near gardens and roads.

The simulator labels every source separately, so any option can be chosen at training time. Open: whether the two-stroke drones are **fixed-wing** (fast, large Doppler, never hover) or **multirotor**; both are rendered until decided.

---

## 5. Detection [J]

### 5.1 Joseph's idea

> Work in the frequency domain, find the drone's frequency from spikes at regular intervals, and use a CNN to identify the times when the drone is heard. [J]

### 5.2 Harmonic comb (evenly spaced spikes) [C detail under J]

**Harmonic sum.** For each candidate fundamental f₀:
```
H(f₀) = Σₖ₌₁ᴷ P(k·f₀)      P = whitened power spectrum, K ≈ 10
```
Why it helps *(derived)*: at the true f₀, the drone's energy adds up K-fold. Random noise in K unrelated bins also adds K-fold on average, but its fluctuation only grows about √K-fold. The drone's peak therefore stands about √K times further above the noise fluctuations than a single harmonic would.

**Whitening.** Divide each frequency bin by its running median over a few seconds, so steady background (traffic rumble, wind) flattens out and tones stand out.

**Frequency resolution.** To resolve harmonics Δ Hz apart with about 3 bins per gap:
```
sr/N ≤ Δ/3  →  N ≥ 3·sr/Δ
Δ = 50 Hz, sr = 16 kHz  →  N ≥ 960  →  N = 1024 (64 ms window)
```
Use a **linear** frequency axis from 0 to 4 kHz, not mel: mel bands widen at high frequencies and blur the upper harmonics. *(derived)*

**Known problems**
- **Octave errors:** H(f₀/2) also scores well. Penalise candidates with empty odd harmonics.
- **Confusers also make combs:** lawnmowers (~25 Hz spacing), motorbikes, insects (150–600 Hz, overlapping drones). The comb alone is not enough, so a classifier is needed (D-03).

### 5.3 CNN for when the drone is heard [J], architecture [C]

Proposed architecture (sound event detection), *(open: not trained yet)*:

```
log-frequency spectrogram (dB, PCEN; see 5.5) → CNN branch ─┐
harmonic-salience map (see 5.5)               → CNN branch ─┼→ per-frame vector → BiGRU → P(drone) per frame
per-frame comb features (best f₀, peak strength, comb-ness ratio, whine/BPF ratio, beating rate) ─┘
→ median filter → start an event above 0.6, end below 0.4, drop events < 0.5 s
→ events: [start, end, confidence, f₀ track]
```

- **PCEN** (per-channel energy normalisation) acts as an automatic gain per frequency band; it suits faint, distant sounds in noise.
- **Labels:** frame-level, from 20 s simulated scenes in which drones enter and leave. Frames where the drone is very faint are marked "ignore."
- **Metrics:** event-based and segment-based F1 (the `sed_eval` library), broken down by SNR, background and drone type.

### 5.4 Drone type [J], revised [C revises J]

- **Joseph's idea:** train the CNN on what kind of drone makes what frequency.
- **Revision:** frequency alone does not identify type, because one model's BPF changes with payload, RPM and flight state, and different models overlap (section 4.2). Use instead:
  - the whine/BPF ratio (section 4.3);
  - rotor count, from the beating;
  - harmonic shape.
- **Requirement:** **real recordings of each drone model**. Synthetic data can only teach broad classes.
- **If this revision is wrong:** for a narrow, fixed set of drones at known loads, frequency alone might be enough. Revert to Joseph's version if real data shows clean frequency separation between the target models.

### 5.5 CNN input: a converted view of the frequency data (D-19, D-20)

**Joseph's idea [J]** (paraphrased):

> Instead of training the CNN only on the raw frequency domain of a simulated drone type, take the drone's frequency-domain data and **convert it into another form that brings out the drone's frequency structure further**.

Joseph's summary of the purpose [J]:

> Convert it so there are **more consistent factors and fewer inconsistent factors**, so the CNN can identify the patterns more easily.

**Main conversion: log frequency axis [C detail under J]** *(derived)*

1. A drone makes harmonics at f₀, 2f₀, 3f₀, ...
2. On a **linear** axis (evenly spaced in Hz), the gap between harmonics equals f₀. A faster drone **stretches** the pattern:
   ```
   f₀ = 100 Hz: 100, 200, 300, 400  → gaps 100, 100, 100 Hz
   f₀ = 200 Hz: 200, 400, 600, 800  → gaps 200, 200, 200 Hz
   ```
3. On a **log** axis (evenly spaced in octaves), position = log₂(f / f_ref). Because log(k·f₀) = log f₀ + log k, a faster drone only **shifts** the pattern; its shape does not change:
   ```
   f₀ = 100 Hz → 0, 1, 1.58, 2 octaves   → gaps 1, 0.58, 0.42
   f₀ = 200 Hz → 1, 2, 2.58, 3 octaves   → gaps 1, 0.58, 0.42
   ```
4. A CNN's filters slide over the whole image, so a pattern learned in one place is recognised anywhere: shifting is handled natively, stretching is not.

**Result:** the drone's pattern keeps **one shape at every RPM** (pitch-invariant shape). The CNN learns one shape and should generalise across RPM, payload and drone size with less training data. This is the "more consistent factors" in Joseph's summary: the pitch variation, which does not matter for "is this a drone", is turned into a shift the CNN already ignores.

**Clarifications [C]**

- **The pitch information is kept.** It moves into the pattern's vertical position, so the CNN can still use it for type and f₀.
- **Only the pitch variation is removed.** Noise, harmonic strengths, rotor beating and distance still vary and must be covered by training data and augmentation (5.6).
- **Trade-off:** a log axis squeezes high harmonics together. The exact comb measurement (5.2) therefore stays on a **linear** axis. The two serve different jobs.

**Decibels are a separate conversion** [J asked whether "converting to decibels" describes this; C: no]:

| | Decibels | Log frequency |
|---|---|---|
| Applied to | Loudness (height / colour) | The frequency axis |
| Formula | 20·log₁₀(amplitude) | log₂(f / f_ref) |
| Effect | Faint and loud sounds both visible | Pitch change becomes a shift instead of a stretch |
| Moves the harmonics? | No | Yes |

Both are used. Correct name for the CNN input: **a log-frequency spectrogram with loudness in decibels** (constant-Q transform in dB), with PCEN normalisation.

**Other converted views [C]** (D-20, *open: not tested*)

| View | What it shows | Shape |
|---|---|---|
| Log-frequency spectrogram (dB, PCEN) | The whole sound, pitch-invariant in shape | frequency × time image |
| Harmonic-salience map | H(f₀) per frame: a comb of K harmonics collapses into **one bright line** at the BPF; noise stays faint | f₀ × time image |
| Per-frame features | Comb strength, best f₀, whine/BPF ratio (4.3), rotor beating rate, spectral flatness | numbers per frame |
| Beamformed source audio (optional) | Each tetrahedron steers towards the drone before the views are made: about +6 dB SNR with 4 mics | (input to the views above) |

The spectrogram and the salience map have different vertical axes (frequency vs f₀), so they go through **separate CNN branches** joined per frame, not stacked as image channels.

**Evidence status:** the shift property is derived and is shown qualitatively in the "Drone Harmonics in 3D" page (appendix B). **That it actually improves CNN accuracy here has not been tested.** See open question 9.

### 5.6 Training data (D-21)

**Joseph's question [J]:** should the CNN be tuned to the frequency domain of one simulated drone type, or trained on a converted view? Answered in 5.5 (converted view). The data plan [C]:

| Data | Role | Reason |
|---|---|---|
| Simulated drones, **widely varied**: heavy-lift to FPV, 2–3 blades, 4–8 rotors, pole pairs, hover / forward flight / climb, RPM wander | Bulk training for detection | Varying everything forces the CNN to learn the general structure (harmonic comb + rotor beating), not one simulated drone |
| Real backgrounds and confusers (ESC-50, UrbanSound8K, own recordings) mixed with the drones | Bulk training | Synthetic backgrounds are too clean; real lawnmowers and motorbikes are the false alarms that matter |
| Real drone recordings (public datasets, own GPS-logged flights) | Fine-tuning and **all testing** | Simulation is never perfect; only real recordings show whether it works |
| Real recordings of each drone model | Type classifier | Type cannot be learned from simulation (D-05) |

**Training order**
1. Pre-train detection on simulated drones over real backgrounds.
2. Fine-tune on real drone recordings.
3. Test only on real recordings the model has not seen.
4. Add type identification as a second output once per-model recordings exist.

**Augmentation:** random SNR and gain, small pitch shifts (a vertical shift on the log axis), time and frequency masking (SpecAugment), varied microphone response and distance.

**Labels**
- Frame-level drone / no-drone, with event start and end times (detection).
- Drone type per event (classification).
- Simulation settings (RPM, distance, SNR, background) so failures can be analysed.

**Pitfall to avoid [C]:** training on one simulated drone type teaches the CNN the simulator, not drones.

### 5.7 Pretrained model: CNN14 (D-22, D-23)

**Joseph's idea [J]** (paraphrased):

> Use **CNN14** as the starting model and **fine-tune** it on rendered (simulated) audio that has been converted to the frequency domain and then to a **log-frequency spectrogram**.

**Where CNN14 comes from [C]** *(checked against documentation; see section 18)*
- CNN14 is from **PANNs** (Kong et al.), pretrained on AudioSet (about 2 million clips, 527 classes). The code is MIT-licensed.
- It is **not a core NeMo model**. **NeMo Curator** includes an adapter that runs a PANNs CNN14 checkpoint. It ships only the **decision-level (frame-level) variants**, which output predictions at 100 frames per second.
- **Frame-level output is required** to find *when* a drone is heard (D-02). The base CNN14 gives one answer per clip.
- A **16 kHz checkpoint** (`Cnn14_16k`) exists, with settings: 64 mel bands, 50–8000 Hz, window 512, hop 160. This matches the simulator's 16 kHz audio.
- About 80M parameters, so it runs on the server, not on the cluster. A smaller variant (E-PANNs) exists if needed.

**Why it fits [C]**
- Pretraining on AudioSet means it already responds to engines, buzzing and mechanical sounds.
- Fine-tuning then needs far less drone data than training from scratch.
- Reported fine-tuning results include 0.960 accuracy on MSoS (PANNs paper) and ~95.75% on ESC-50 (community repo) *(unverified)*.

**Conflict with the log-frequency plan (D-19) [C]** *(derived)*

1. **Mel is not logarithmic at low frequencies.** CNN14 was pretrained on a log-mel spectrogram:
   ```
   m(f) = 2595 · log₁₀(1 + f/700)
   f ≪ 700 Hz:  log₁₀(1 + x) ≈ x / ln 10   →  m ≈ 1.61·f      (linear)
   f ≫ 700 Hz:  m ≈ 2595 · log₁₀(f/700)                     (logarithmic)
   ```
   Below ~700 Hz the mel scale is nearly linear, which is where drone fundamentals sit (50–500 Hz). So CNN14's stock input does **not** give the pitch-invariant shape of 5.5 in the drone's main range.
2. **Frequency resolution is too coarse.** A 512-sample window at 16 kHz gives FFT bins 16000/512 = 31 Hz wide. Harmonics 60 Hz apart (heavy-lift drone) get only about 2 bins per gap.

**Front-end options (D-23) [C]**

CNN14 takes **raw audio** and computes its spectrogram in its first layer. Converting to a log-frequency spectrogram therefore means **changing that internal layer**, not preparing images separately.

| Option | Change | Pretraining kept | Pitch-invariant over drone range |
|---|---|---|---|
| **A. Stock log-mel (baseline)** | None; fine-tune as-is | Full | Only above ~700 Hz |
| **B. Mel with a lower bend point** | In m(f), replace 700 with ~50 Hz so the scale is log-spaced from ~50 Hz; keep 64 bands; window 2048 samples (7.8 Hz bins), hop 160 | Most (same input shape) | Yes |
| **C. True log-frequency (CQT)** | Replace the front layer with a constant-Q transform, possibly more bands | Partial (early layers must re-adapt) | Yes |

**Plan:** A as the baseline, then B. Compare on **real** recordings (open question 9). Move to C only if B falls short.

**Practical notes [C]**
- **Changing the band count requires replacing `bn0`.** In the official `models.py`, `bn0 = BatchNorm2d(64)` is applied after transposing the spectrogram so each of the 64 mel bands is normalised separately *(verified against the official code)*. Even with 64 bands kept (option B), its statistics should be re-learned, because each band now covers different frequencies. The convolutional layers and global pooling accept any input height.
- **Fine-tuning order:**
  1. Replace the 527-class output with drone / no-drone (type later).
  2. Train the new head with the rest frozen.
  3. Unfreeze deeper layers with a small learning rate.
- **Extra views** (harmonic-salience map, comb features, whine/BPF ratio; D-20) cannot pass through CNN14. Join them with CNN14's frame-level output just before the final layers.
- **Rendered audio alone is not enough.** Mix simulated drones into real backgrounds, and test only on real recordings (D-21).
- **Licence:** PANNs code is MIT. Check the terms for the pretrained weights and AudioSet before any commercial use *(unverified)*.

### 5.8 CNN build plan, v0.4 (D-22 revised, D-24 to D-26)

**Joseph's pipeline [J]** (confirmed 2026-10-22):

```
simulated drone → mic signal → frequency-domain graph → conversion (log frequency, dB) → CNN14 → detection
```

The idea carried over from the drone tracker: take the raw data, convert it into a form where every drone looks the same (pitch changes become shifts, D-19), and train the CNN on that.

#### 5.8.1 Render audio first, then convert (D-24) [J←C]

| Option | Generated | Problem |
|---|---|---|
| Draw spectrograms directly | Clean spikes at f₀, 2f₀, 3f₀ | Too perfect: no window smearing, rotor beating, Doppler bends or mixed-in noise. The CNN would learn "perfect spikes" and miss real ones. |
| **Render the waveform, then run the same conversion as real audio** | What the mic would record | Simulated and real pipelines are identical from the mic onward. CNN14 also takes raw audio as input. |

#### 5.8.2 The checkpoint (D-22, revised) [C] *(verified against the PANNs README, models.py and the NeMo Curator docs, 2026-10-22)*

- v0.3 said "frame-level 16 kHz variant". **That checkpoint does not exist.** The 16 kHz CNN14 (`Cnn14_16k_mAP=0.438`) is clip-level only.
- NeMo Curator's CNN14 adapter defaults to **`Cnn14_DecisionLevelMax_mAP=0.385.pth`** (Zenodo record 3987831), which is the starting model.
- Settings: **32 kHz**, window 1024, hop 320, 64 mel bands, fmin 50 Hz, fmax 14,000 Hz, 527 AudioSet classes.
- **The simulator therefore renders at 32 kHz.**
- Loaded with the official PANNs `models.py` (MIT), the same code the NeMo adapter vendors. Installing all of NeMo Curator is not needed.

**Frame rates inside CNN14** *(derived from models.py)*:
```
STFT hop 320 at 32 kHz            → 100 frames/s at the input
6 conv blocks, time pooled ×2 in 5 → ÷32 → 3.125 frames/s inside the network
framewise output interpolated ×32 → back to 100 frames/s
clip output = max over time        ("DecisionLevelMax")
```
So "when" has an effective resolution of about 0.32 s, which is fine for detection.

#### 5.8.3 What mel is, and why its bend point matters [C]

The mel scale spaces frequencies the way human hearing judges pitch:
```
m(f) = 2595 · log₁₀(1 + f/700)
check: m(1000) = 2595 · log₁₀(2.43) ≈ 1000 mel
```

| f | mel |
|---|---|
| 100 Hz | 150 |
| 200 Hz | 283 |
| 1,000 Hz | 1,000 |
| 4,000 Hz | 2,146 |
| 8,000 Hz | 2,840 |

Generalise the bend point f_b (700 in standard mel):
```
m_b(f) = A · log₁₀(1 + f/f_b)
dm_b/df = A / (ln 10 · (f_b + f))
f ≪ f_b:  dm/df ≈ constant           → linear: harmonic pattern STRETCHES with RPM
f ≫ f_b:  dm/df ∝ 1/f, i.e. ∝ d(log f) → logarithmic: pattern SHIFTS with RPM
```
Standard mel is linear below ~700 Hz, which is where most drone fundamentals sit. Moving the bend to f_b ≈ 50 Hz (option B, D-23) makes the whole drone range logarithmic.

#### 5.8.4 Resolution needed for a log-frequency front end [C] *(derived)*

N bands spaced logarithmically from f_min to f_max:
```
band ratio      r = (f_max/f_min)^(1/N)
band width      Δf(f) ≈ f · (r − 1)
quality factor  Q = 1/(r − 1)
50 → 14,000 Hz, N = 64:  r = 280^(1/64) ≈ 1.092,  Q ≈ 10.9,  Δf(50 Hz) ≈ 4.6 Hz
```

| Front end | Bin / window | Lowest bands | Pretraining kept |
|---|---|---|---|
| A. Stock log-mel (baseline) | 31.25 Hz bins, 32 ms | Linear below 700 Hz | Full |
| B. Mel bent at 50 Hz, 64 bands, n_fft 4096, hop 320 | 7.8 Hz bins, 128 ms | Bottom ~6 bands share bins (interpolated) | Most (same 64 × T shape; relearn bn0) |
| B′. Same with n_fft 8192 | 3.9 Hz bins, 256 ms | Fully resolved | Most |
| C. Constant-Q transform, window T_k = Q/f_k | 0.22 s at 50 Hz, 11 ms at 1 kHz | Fully resolved, sharp in time at high f | Partial |

FFT resolution: bin = sr/n_fft, so resolving 4.6 Hz needs n_fft ≥ 32,000/4.6 ≈ 6,960 → 8,192. Drone pitch and Doppler change over seconds, so 128–256 ms windows are acceptable for detection. **Plan:** A as the baseline, then B; B′ or C only if B falls short on the lowest classes (heavy two-strokes near 70 Hz).

#### 5.8.5 Scope and outputs (D-26) [J]

- **Now: detection only.** Is a drone present in each frame? Type identification comes later.
- Output heads, per D-28: proposed **electric drone / engine / nothing** (per frame, sigmoid per class). **Implemented (v0.5):** three sigmoid heads, `drone_electric`, `drone_engine`, `ground_engine`, so every D-28 option can still be read off: option 2 ("any engine") is max(drone_engine, ground_engine).
- Fine-tuning recipe (unchanged from 5.7): replace the 527-class layer, relearn `bn0`, train the head frozen, then unfreeze deeper blocks at a low learning rate.
- **Crops:** clips are long (full passes, 14.6). Train on random 10 s crops with their frame labels; run inference on the whole clip, since the network is fully convolutional in time.

#### 5.8.6 Frame labels from the simulator [C]

Each frame is labelled from the exact simulated SNR at the mic:
```
SNR_drone(frame) = 10 · log₁₀( E_drone / E_background )   over 50 Hz – 8 kHz, per 10 ms frame
label = 1        if SNR_drone ≥ θ_hi
label = ignore   if θ_lo ≤ SNR_drone < θ_hi       (barely audible: not scored, no gradient)
label = 0        if SNR_drone < θ_lo
starting values θ_hi = −10 dB, θ_lo = −20 dB   (estimate; narrowband drone tones are detectable well below 0 dB broadband SNR)
```

**Rule actually used (v0.5, label version `snr_lines2`).** Broadband SNR proved a poor label: a drone with clear tones can sit at −20 dB broadband, while a single-bin rule ("best bin ≥ 3 dB") marked far, inaudible drones as positive whenever one bin poked up, and the pilot model's electric-drone recall stalled at ~22 %. The label now asks for **at least two separate tone regions** standing out, which is what makes a drone recognisable (a harmonic comb, 5.2):

```
For each source category c (drone_electric, drone_engine, ground_engine) against everything else at mic 0:
  P_c(f,t), P_rest(f,t)          STFT power, n_fft 1024, hop 320 (10 ms), 32 kHz, bins in 50 Hz – 8 kHz
  L(f,t) = 10·log₁₀( U[P_c] / U[P_rest] )      U = 3-bin × 5-frame box average  (local SNR, dB)
  group bins in fours (4 × 31.25 Hz = 125 Hz), take the max L in each group  → G(g,t)
  repeat 4×:  pick g* = argmax_g G(g,t); record G(g*,t); set G(g*−1..g*+1, t) = −∞   (non-max suppression,
              so one tone straddling a group edge counts once)
  → lines_c(t) = the 4 strongest separate regions, sorted: lines[0] ≥ lines[1] ≥ …
  positive  if lines[1] ≥ 6 dB          (the 2nd-strongest separate region is ≥ 6 dB above everything else)
  negative  if lines[0] < 0 dB          (not even one region stands out)
  ignore    otherwise                   (masked out of the loss and of all metrics)
```
Also stored per scene for relabelling without re-rendering: broadband and single-bin SNR, 48 log bands (40 Hz – 14 kHz, 25 frames/s) per source, and per drone the range r, elevation, ṙ, emitted and received BPF. Code: `sim/scene.py: separate_regions, frame_labels`, thresholds in `cnn/data.py: POS_DB, NEG_DB`.

#### 5.8.7 Train and test protocol (D-25) [J], held-out design [C]

**Joseph's decision [J]:** train on simulated data only (sim gave the best results in the drone tracker). Real testing uses **Joseph's own field recordings**: a mic set up, a drone flying over, checking whether it is picked up. These recordings are **never** used for training. This revises D-21 (fine-tuning on real recordings is deferred).

| Split | Contents | Purpose |
|---|---|---|
| train | Simulated, randomised over all parameters | Learning |
| val | Same distribution, different seeds | Early stopping, thresholds |
| test-sim-held-out | **RPM beyond the training range**, rotor counts not seen in training, unseen background types, lower SNR | Tests whether the log conversion generalises. A sim test on the training distribution only proves the CNN learned the simulator. |
| test-real | Joseph's field recordings (14.7) | The real answer |

**Metrics**
- Frame and event F1 (`sed_eval`).
- **Detection range**: P(detect) against distance.
- **False alarms per hour** on no-drone recordings.
- Detection latency: how early in a pass the drone is first detected.

### 5.9 First trained models (v0.5) [C] *(simulated test data only)*

Everything here is reproducible from the repo: README.md gives the commands, `results/` holds the logs and metrics, and the trained weights are the GitHub release `models-v0.5`.

#### 5.9.1 Dataset `sim_v1`

Rendered by the scenario simulator (14.6), mic 0 only, 32 kHz, 24-bit FLAC, full scale 120 dB SPL. Rendering is deterministic per seed: on one platform the files are bit-identical; between Linux and Windows the samples differ by at most 1 LSB (−150 dB), and the labels agree on every frame (checked). `results/sim_v1_manifest.csv` lists every scene's length, level, sources and label counts; `scripts/manifest.py verify` checks a re-render against it.

| Split | Seeds | Scenes | Hours | Clip length min/median/max | Scenes with electric / engine drone / ground engine / no drone |
|---|---|---|---|---|---|
| train | `p` 0–3999 and `a` 1,000,000–1,001,999 where seed mod 20 ∉ {0,1} | 5,400 | 112.5 | 20 / 73 / 120 s | 2,552 / 1,710 / 927 / 1,328 |
| val | same ranges, seed mod 20 = 0 | 300 | 6.2 | 20 / 71 / 120 s | 126 / 116 / 47 / 74 |
| test_sim | same ranges, seed mod 20 = 1 | 300 | 6.2 | 20 / 76 / 120 s | 132 / 104 / 38 / 82 |
| test_heldout | `ah` 0–399, rendered with `--heldout` (seed + 10,000,000) | 400 | 8.2 | 20 / 71 / 120 s | 191 / 120 / 67 / 101 |

The held-out set uses conditions never seen in training: RPM above each class's training range, 5- and 7-rotor drones, rain as the background and louder backgrounds. About 10 % of frames are positive for drone_electric. Rendering takes about 4 s of CPU per scene (the whole set took about 2 h across two PCs with 22 workers).

#### 5.9.2 Training recipe

| | B: CNN14 + log-frequency | A: CNN14 + stock mel | C: small CNN from scratch |
|---|---|---|---|
| Front end (fixed, not trained) | bent mel, n_fft 4096, 64 bands, 50 Hz – 14 kHz (5.8.3) | CNN14's own: mel, n_fft 1024, 64 bands, 50 Hz – 14 kHz | same as B |
| Network | `Cnn14_DecisionLevelMax` (PANNs), 527-class layer replaced by 3 | same | 4 conv blocks (32/64/128/256) + BiGRU 128, 1.47 M parameters |
| Start | PANNs AudioSet weights; `bn0` reset (bands moved) | PANNs AudioSet weights | random |
| Stage 1 (1,500 steps, AdamW lr 1e-3) | train `bn0`, `fc1`, head only | same | same |
| Stage 2 (15,000 steps, AdamW, cosine) | all except the front end, lr 1e-4 | same | all, lr 1e-3 |
| Batch | 16 random 10 s crops (60 % centred on a positive frame) | same | same |

Common to all: masked binary cross-entropy per frame and class with pos_weight (2, 3, 3), ignore frames masked out; weight decay 1e-4; gradient clipping at 5; bf16 autocast; SpecAugment inside the model. Waveform augmentation: gain −25 to +15 dB, random polarity, random first-order shelf EQ (70 % of crops; models an unknown mic response). The model is evaluated on val every 1,000 steps and the best mean F1 is kept (best step: B 15,000, A 16,500, C 15,000). Training is not seeded, so a retrain lands within about ±0.01 F1 of these numbers, not exactly on them.

Compute: B ran 1 h 58 min on an RTX 3080 capped at 240 W (0.43 s/step; the n_fft 4096 front end is the slow part), A 34 min on the same card (0.12 s/step), C 33 min on an RTX 4060 Ti.

#### 5.9.3 Results: frame F1 (ignore frames excluded), threshold 0.5

| Model | test_sim electric | engine | ground engine | test_heldout electric | engine | ground engine |
|---|---|---|---|---|---|---|
| B: CNN14 + log-f | **0.924** | **0.930** | 0.921 | 0.896 | **0.903** | **0.873** |
| A: CNN14 + mel | 0.922 | 0.912 | **0.926** | **0.913** | 0.898 | 0.850 |
| C: small, scratch | 0.883 | 0.834 | 0.890 | 0.875 | 0.820 | 0.845 |

**Detection probability vs distance (B, test_sim; A and C are within a few points):**

| Distance (m) | 0–25 | 25–50 | 50–100 | 100–200 | 200–400 | 400–800 | 800–1600 |
|---|---|---|---|---|---|---|---|
| Electric drone | 0.76 | 0.80 | 0.63 | 0.35 | 0.21 | 0.10 | 0.02 |
| Two-stroke drone | 1.00 | 0.85 | 0.86 | 0.65 | 0.57 | 0.40 | 0.22 |

**False alarms per hour on drone-free scenes (test_sim, 1.73 h; events ≥ 0.5 s):**

| Model | drone_electric | drone_engine | ground_engine |
|---|---|---|---|
| B | 1.7 | 27.8 | 11.0 |
| A | 1.7 | 27.8 | 4.0 |
| C | 0.6 | 11.0 | 0.6 |

#### 5.9.4 Findings

1. **Starting from CNN14 helps:** +0.04 to +0.09 F1 over the same task trained from scratch, and the gap holds on held-out conditions. This supports D-22.
2. **Log-frequency vs mel is a draw on simulated data** (≤ 0.02 F1 either way). The D-19 question is not settled by sim; it needs real recordings, especially at RPMs outside the training range.
3. **Generalisation:** held-out RPM, rotor counts and backgrounds cost only about 0.02–0.05 F1.
4. **Close-range electric drones are only detected 76 % of the time (0–25 m).** Suspected cause: the range metric counts every frame, including frames the label marks ignore or negative (e.g. a louder confuser on top of the drone). To check before believing it (open question 22).
5. **Engine-drone false alarms (~28/h) come from ground engines** (chainsaws, blowers, mopeds): the D-28 problem, as predicted. Electric-drone false alarms are low (0.5–2/h).
6. Electric drones fade past ~100 m, two-strokes are still heard half the time at 200–400 m. Plausible physically, but simulator-only (F1 in 16.1).

#### 5.9.5 Fixes from code review, before these runs

| Problem | Effect | Fix |
|---|---|---|
| Stage 2 also unfroze the STFT and filter bank | the front end drifted from what real audio gets | stage 2 trains everything except `*extractor*` |
| Fractional delay read before the start of the source buffer | a fake tone for far aircraft | out-of-buffer reads are silence |
| One tone straddling a 4-bin group edge counted as two regions | false positives in labels | non-max suppression (5.8.6) |
| A second, shorter drone stopped dead at the end of its path | an unphysical stationary source | it continues at its final velocity |
| Crops not aligned to the 320-sample hop | labels off by up to one frame | crop start rounded down to a hop |
| False alarms counted correctly detected chainsaws | inflated false-alarm rate | only frames labelled negative count |
| Range credit in 2-drone scenes | detections of one drone credited to the other | credited to the nearer drone of that class |
| Full scale 110 dB SPL | clipping on close sources | 120 dB SPL |

### 5.10 Failure analysis of model B (v0.5) [C]

Model B (CNN14 + log-frequency) was tested five ways. Code: `analysis/` (`dump`, `frames`, `slices`, `robust`, `sweeps`, `real`). A is shown where it differs.

| Test | Data |
|---|---|
| Slices of every error | 700 test scenes (test_sim + test_heldout), 1.3 M frames, joined with every simulated parameter |
| Recording-chain robustness | test_sim with 21 perturbations (gain, clipping, filters, codecs, noise, wind, hum) |
| Controlled sweeps | 868 new scenes: range per class at 40 and 55 dBA, tip speed 50–210 m/s, each confuser and background alone |
| **Real drones** | Fraunhofer IDMT outdoor flights (CC BY 4.0): DJI Phantom 4, DJI Mavic, Align, 163 × 30 s, GPS distance |
| **Real non-drone sounds** | ESC-50 (CC BY-NC, test only): 2,000 real 5 s clips in 50 classes, at 3 playback levels |

#### 5.10.1 Ranked weaknesses

| # | Weakness | Evidence | Severity |
|---|---|---|---|
| W1 | **Open-set false alarms on real sounds.** drone_electric fires (≥ 0.5 s event) on **76–82 % of real ESC-50 clips** that contain no drone: crows, babies, laughter, sirens, horns, toilets, wind, footsteps. Independent of level (0, −20, −40 dB). Removing any one octave band does not stop it; keeping only > 1 kHz makes 98 % fire | ESC-50 test; occlusion test on 40 clips | **Critical**: the model would alarm constantly in the field |
| W2 | **Lossy compression and small-mic high-pass** | electric F1 0.92 → 0.77 (MP3 low bitrate), → 0.72 (high-pass 400 Hz, FP ×9); Opus low bitrate 0.88 | High: phone or cheap recorders; record the field test uncompressed |
| W3 | **Moving engines read as engine drones** | with no engine drone present, drone_engine fires on 14 % of light-aircraft frames, 14 % moped, 7 % mower, 4.5 % helicopter. Chainsaws are fine (0.2 %), absorbed by ground_engine | High for two-stroke detection (D-28) |
| W4 | **Masking by loud backgrounds and other tonal sources** | recall 92 % overall but 56 % at 65–75 dBA background, 77 % when the loudest other source is > 20 dB above the drone in band, ~54 % when an aircraft or ground engine dominates | Medium: partly physics |
| W5 | **Real detection range ~150 m** (rural, DJI-class drones) | real flights: 93 % ≤ 25 m, 90 % 25–50, 77 % 50–100, 49 % 100–150, 3.5 % 150–200, 0 % beyond | Medium: sets cluster spacing (8.2) |
| W6 | **Big slow electric drones look like engines** | 7–12" at 4,000–7,500 RPM: 65–91 % detected, up to 62 % of frames called drone_engine | Medium |
| W7 | **Two drones at once** | electric recall 80.5 % with two drones vs 92.6 % with one (beating combs) | Medium |
| W8 | **Over-confident probabilities** | at p 0.5–0.6 only ~30 % of frames are positive; best F1 threshold ≈ 0.7 (drone heads), 0.9 (ground_engine) | Low: free fix |
| W9 | Very close range (< 10 m) | 54.5 % recall on 220 frames / 7 scenes; the inspected case is a revving chainsaw masking the drone | Low: small sample |

#### 5.10.2 What works

- **Real drones are detected**, with no real data in training: ≥ 98 % "some drone" within 50 m on all three real models; fires 0 % when the same drone is > 200 m away, so it is not firing on the site's ambience.
- **B beats A on real audio** (Phantom 4 static 0.94 vs 0.77; 10–25 m 0.93 vs 0.80): the first real evidence for D-19.
- **RPM generalisation (FPV):** 100 % from 6,500 to 25,700 RPM (training ~10–17k).
- **Simulated confusers and backgrounds alone** barely trigger it (worst: insects 6.7 %, wind at 80 dBA 2.6 %); chainsaws and mopeds go to ground_engine as designed.
- **Robust** to gain −30 dB, clipping, 48 kHz resampling, 5 ms shift, high-bitrate MP3, ±6 dB/decade tilt, 16 kHz recorders.
- The 76 % "electric detection at 0–25 m" in 5.9 was mostly the metric: only 83 % of those frames are labelled audible; recall on audible frames is 90 % (0–25 m), 100 % (25–50 m).

#### 5.10.3 Causes and fixes

| Weakness | Most likely cause | Fix (in order of value) |
|---|---|---|
| W1 | The simulator's negatives contain no voices, animals, alarms, water, clicks or real ambience; the electric head learned "structured sound above 1 kHz that is not a known negative" | Add a large, diverse set of real non-drone sounds as negatives (needs a decision: D-25 / open question 21). Options with clean licences: FSD50K (CC BY subset), AudioSet-derived clips, own recordings. Or synthesise many more negative families. Re-test on ESC-50 |
| W2 | Training never saw codec artefacts or strong high-pass | Augment: random MP3/Opus round trips, high-pass 50–500 Hz, low-pass 4–16 kHz |
| W3 | Moving engines on the ground and in the air share the drone's Doppler and pulse structure | More moving-engine negatives (mopeds, aircraft, mowers) in drone-free scenes; decide D-28 (option 2, "any engine" = max of two heads, removes most of this) |
| W4, W7 | Few training scenes with a loud tonal interferer on top of a drone, or two drones | Oversample these mixtures |
| W5 | Physics plus real conditions; may improve with W1/W2 fixes | Measure on Joseph's flights; size cluster spacing from the real curve |
| W6 | Few low-RPM large electric drones in training | Widen the "mid" class to lower tip speeds and larger props |
| W8 | pos_weight (2, 3, 3) shifts probabilities up | Tune thresholds on val, or temperature-scale |

Follow-up real tests (Svanström et al. ground-mic clips, IDMT by drone and range, ESC-50 by category) and the upgrade plan: `results/real_world/REAL_TESTS.md`, `UPGRADES.md`.

Not finished: the 4,000-scene extra sim test set (rendered, in `data/sim_eval`) was only 6 % predicted when the run was stopped for low memory; the 700-scene results above stand on their own.

### 5.11 Theory vs results (v0.6) [C]

A check of the v0.5 results (5.9, 5.10, `results/real_world/REAL_TESTS.md`) against what the theory in this document predicted. **Scope:** these results are **single-microphone detection only** (channel 0). Nothing here tests direction finding, height, the Kalman filter, Doppler velocity or the frequency-tracking functional (5.11.4).

#### 5.11.1 Predictions that held

| Prediction (where) | Result | Verdict |
|---|---|---|
| The harmonic comb is the drone signature (4, D-01) | Real DJI drones detected with sim-only training (≥ 98 % within 50 m); broadband sounds barely trigger it (rain 2 %), tonal ones do | Consistent |
| The comb alone is not enough; confusers also make combs (5.2, F5) | drone_electric fires on 76–82 % of ESC-50 clips (crows, horns, sirens, voices) (W1) | Strongly confirmed; now the main problem |
| Simulated negatives will not transfer (5.6) | REAL_TESTS: "the physics transfers; the negatives don't" | Consistent |
| Motors share the drone's comb; the tracker should reject stationary sources (F5) | Camera-mount servo whine flagged on 96–100 % of frames | Consistent: correct physics, needs the motion check downstream |
| Pretraining helps (D-22) | CNN14 +0.04 to +0.09 F1 over scratch; real drone frames 91 % (B) vs 72 % (C) | Consistent |
| Big props → low BPF and dense harmonics; frequency alone cannot give type (4.2, D-05) | Big slow electric drones called "engine" (W6) | Consistent |
| Several similar drones are hard to separate (F6) | Recall 92.6 % → 80.5 % with two drones (W7) | Consistent |
| Background type matters more than the SNR number (14.4) | Recall 56 % under 65–75 dBA backgrounds; ~54 % when an aircraft or engine dominates (W4) | Consistent |
| The closest drone is the loudest, so the easiest | Sim showed 76 % at 0–25 m, but that was the metric: 90 % on audible frames (5.10.2); real flights 99 % | Consistent after correction |
| Mics need low-frequency response (7.5) | High-pass at 400 Hz: F1 0.92 → 0.72, false positives ×9 (W2) | Consistent |
| Every +6 dB of source level doubles the range (8.3) | Smaller Mavic falls off sooner (29 % at 50–100 m); larger Align reaches further (80 % at 100–150 m) | Consistent in scaling |

#### 5.11.2 Deviation 1: the model listens mostly above 1 kHz

- **Theory emphasis:** the blade-passing fundamental, 50–500 Hz (4.1, 5.2).
- **Result:** with only content above 1 kHz kept, drone_electric still fires on 98 % of clips (W1 occlusion test). The electric head relies mainly on upper harmonics and motor whine.
- **What this explains** *(derived from sections already in this document)*:
  1. **Why log vs mel is a draw in sim (5.9.4, finding 2).** Mel is linear below ~700 Hz and logarithmic above (5.8.3):
     ```
     m(f) ≈ 1.61·f  (f ≪ 700 Hz),    m(f) ≈ 2595·log₁₀(f/700)  (f ≫ 700 Hz)
     ```
     Where the model looks (> 1 kHz), both front ends are already logarithmic. The log-frequency advantage of D-19 was predicted for the low band, which the model uses little. On real audio B still beat A (hovering Phantom 0.94 vs 0.77; drone-vs-background AUC 0.84 vs 0.77), so D-19 is **partly** supported.
  2. **Why electric drones fade sooner than two-strokes.** Air absorption grows with frequency (ISO 9613-1). A model relying on > 1 kHz loses the drone sooner than the low, loud engine tones: sim electric fades past ~100 m, two-strokes are heard half the time at 200–400 m.
  3. **Why everyday sounds trigger it.** Most everyday tonal sounds (voices, birds, horns, sirens) sit at 0.3–4 kHz, where the electric head listens.
- **Open:** whether pushing the model to use the low band (e.g. real tonal negatives above 1 kHz) would recover range in noise and widen the B-vs-A gap (open question 26).

#### 5.11.3 Deviation 2: real detection range is shorter than estimated

- **Estimate (8.3):** `r_max = 10^((L_src − L_bg + G)/20)`. Quiet rural (40 dBA), 80 dB drone: 320 m (G = 10 dB) to 1,000 m (G = 20 dB).
- **Result:** Phantom 4, rural site: 49 % at 100–150 m, 3.5 % at 150–200 m, so r ≈ 125 m.
- **Back-calculation** *(derived)*:
  ```
  20·log₁₀(125) ≈ 42 dB   →   L_src − L_bg + G ≈ 42
  L_src = 80, L_bg ≈ 40–45   →   G ≈ 2–7 dB
  ```
- **Meaning:** the single-mic CNN detects the drone only a few dB below the background, not 10–20 dB below as assumed. Alternatively the Phantom is quieter than 80 dB at 1 m (open question 27). Either way, **plan cluster spacing from ~100 m, not 300 m** (8.2, F3).

#### 5.11.4 Not yet tested

| Theory | Decisions | How it could be tested |
|---|---|---|
| Tetrahedral direction finding, height | D-07, D-08, D-14 | IDMT recordings are 8-channel array data with GPS: real drone direction finding without new recordings (open question 25) |
| Triangulation, towards/away | D-15 | Needs two or more clusters (field test) |
| Multi-cluster Doppler velocity, Kalman filter | D-12, D-13, D-16 | Needs per-rotor simulation first (F2), then the field test |
| Frequency-tracking functional | D-11 | Can be run on IDMT or Svanström drone clips now |
| Direction accuracy assumption (σ_u ≈ 0.02) | F4 | Same as the first row |

#### 5.11.5 Summary

The detection results fit the theory: the predicted risks (tonal false alarms, type confusion, masking) all appeared. The useful new information is in the deviations. The model relies on content above 1 kHz, and the real range implies a much smaller detection margin (G ≈ 2–7 dB). These change the range model and cluster spacing, and partly explain why log vs mel was a draw.

---

## 6. The time-frequency functional [J]

### 6.1 Joseph's idea

> Use the concept of functionals to find a function for a certain frequency in the frequency domain, which also includes the time domain. [J]
>
> Find the gradient based on a frequency value. [J]

### 6.2 Formulation [C detail under J]

A linear functional compares the signal with a test function (an inner product):
```
⟨x, g⟩ = ∫ x(τ) · g*(τ) dτ
```

- **Plain Fourier transform:** g = e^{i2πf₀τ}. It integrates over all time, so time information is lost.
- **Time-localised version** (STFT / Gabor):
  ```
  g_{t,f₀}(τ) = w(τ − t) · e^{i2πf₀τ}
  X(t, f₀) = ∫ x(τ) · w(τ − t) · e^{−i2πf₀τ} dτ = |X|·e^{iφ}
  ```
  Fix f₀ and vary t, and you get a function of time: amplitude A(t) = |X| and phase φ(t).

### 6.3 Gradient over time gives the exact frequency [C detail under J]

For a component at f near f₀, multiplying by e^{−i2πf₀τ} and low-pass filtering leaves `(A/2)·e^{i(2π(f−f₀)τ + θ)}`, so:
```
f(t) = f₀ + (1/2π) · ∂φ/∂t
```
This is complex demodulation. **Simulated result:** median frequency error **0.2 Hz** and amplitude error about 4% on a Doppler-shifted 1 kHz tone at about 0–3 dB broadband SNR. A spectrogram bin is 7.8 Hz. *(simulated)*

### 6.4 Gradient over frequency finds the peak [C detail under J]

With a Gaussian window of width σₜ, the peak in frequency is also Gaussian, with σ_f = 1/(2πσₜ):
```
|X(f)| ≈ A₀·exp(−½((f − f_p)/σ_f)²)
→ (∂|X|/∂f)/|X| = −(f − f_p)/σ_f²
→ f_p = f + σ_f² · (∂|X|/∂f)/|X|
```
For a single clean tone this step lands on the peak in one move. It is demonstrated in the interactive "Drone Tone Probe" page (appendix B). *(derived; demonstrated, not benchmarked)*

### 6.5 Trade-off

A window of length T separates frequencies about 1/T apart (Gabor limit σₜ·σ_f ≥ 1/(4π)):
- **Long T:** precise frequency, but the tone only stays inside the band for part of a Doppler sweep.
- **Short T:** follows the whole sweep, but is noisier.
*(derived)*

### 6.6 What the functional is used for

- **f(t):** the Doppler curve, used for velocity (D-16), the Doppler cross-check on height, and as a supporting towards/away cue.
- **A(t) summed over the comb bins:** the drone-only loudness, used for the towards/away loudness trend.

---

## 7. Microphone array [J←C]

### 7.1 History

1. **[J]** 3 microphones per cluster.
2. **[C]** A flat triangle measures azimuth well but gives elevation without its sign, and poorly near the horizon. Proposed raising a 4th microphone.
3. **[J]** Adopted: a 4th microphone on top, forming a triangular pyramid.

### 7.2 Why 3 coplanar microphones are not enough for height *(derived)*

A flat triangle measures only the horizontal part of the direction unit vector: `|u_horizontal| = cos(el)`. So:
```
σ_el ≈ σ_u / sin(el)
```
This is poor near the horizon, which is where distant drones are, and it cannot tell above from below.

### 7.3 Tetrahedron *(derived)*

The top microphone's timing difference against the base contains `h_m·u_z / c`, so it **measures u_z directly**, sign included:
```
el = asin(u_z),    σ_el ≈ σ_u / cos(el)      (accurate near the horizon)
```

**Use a *regular* tetrahedron.** Direction accuracy depends on Σ pᵢpᵢᵀ (positions measured from the array centre). With vertices (1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1):
```
diagonal = 4,   off-diagonal = 1 − 1 − 1 + 1 = 0   →   Σ pᵢpᵢᵀ = 4·I
```
The result is the same in every direction, so accuracy is equal whichever way the drone is.

### 7.4 Sizing (D-08) [C]

```
top mic height = a·√(2/3) ≈ 0.816·a
no false directions:  every pair spacing ≤ λ/2  →  f_max = c / (2a)
accuracy:             σ_u ≈ c·σ_τ / a
```

| Option | Edge a | Top mic height | Direction-finding band |
|---|---|---|---|
| Compact | 8 cm | 6.5 cm | up to ~2 kHz |
| **Chosen: wide** | **20 cm** | **16.3 cm** | **up to ~860 Hz** (about 2.5× better accuracy than compact) |

**Check:** with σ_τ ≈ 10 µs and a = 0.2 m, σ_u ≈ 343 × 10⁻⁵ / 0.2 ≈ 0.017. *(estimate: σ_τ is assumed, not measured)*

### 7.5 Microphone requirements [C]

| Requirement | Reason |
|---|---|
| Response from ~40 Hz to ~8 kHz | Large-drone BPF near 50 Hz; whine and harmonics to a few kHz |
| Omnidirectional | Directionality comes from the array |
| 4 **sample-synchronised** channels | Direction finding needs the timing between mics |
| Low self-noise (only matters in quiet places) | Self-noise = 94 − SNR spec. 33 dBA in a 40 dBA background adds only about 0.8 dB *(derived)* |
| Windshield on every mic | Wind is the largest outdoor noise source and is strongest at drone frequencies |

Not suitable: parabolic dishes and shotgun mics. A dish only helps when its diameter exceeds λ, and λ = 1.7 m at 200 Hz. *(derived)*

### 7.6 Candidate parts *(unverified: check datasheets, prices, stock)*

| Use | Part |
|---|---|
| Array mics | TDM MEMS (e.g. ICS-52000), or I²S MEMS (INMP441, ICS-43434, SPH0645, IM69D130; SNR ~61–69 dB) |
| 4-channel alternative | 4-input USB audio interface with small measurement capsules |
| Cluster computer | Raspberry Pi 4/5 |
| Timing and position | GPS module |
| Single reference mic for testing | miniDSP UMIK-1 or Dayton UMM-6 (calibrated USB measurement mics) |
| Off-the-shelf arrays (planar, so prototyping only) | ReSpeaker USB Mic Array v2.0 (use 6-channel raw firmware), miniDSP UMA-8 / UMA-16 |

### 7.7 Mounting and calibration [C]

- Pole-mounted 2–3 m above the ground, with a thin open frame. **(v0.4: questioned; a pole puts ground-echo notches into the drone band. See 7.8.)**
- Measure the built mic positions to about 1 mm.
- **Level each array carefully.** A 1° tilt adds 1° to every elevation, about **1.9 m of height error at 100 m** *(derived)*. Calibrate with a drone flight that logs its own GPS position.
- Survey the clusters' relative heights once (laser level or RTK GPS). Consumer GPS altitude is off by several metres.

### 7.8 Flat clusters on the ground instead of a tetrahedron (D-30) [J], analysis [C]

**Joseph's question [J]:** since velocity can be estimated, is a tetrahedron needed? He prefers to keep the microphones on one plane, or at least on the ground surface, even if not all at exactly the same level.

**Answer [C]:** no, the tetrahedron is not strictly necessary. Flat clusters on the ground can give height. Their cost is poor elevation accuracy for drones far away and low in the sky; overlapping clusters and Doppler inside the Kalman filter recover most of that. **Status: proposed. The scenario simulator (14.6) will compare the layouts before anything is built.**

**What a flat cluster measures** *(derived)*. A 4-mic square of half-side d at the ground, vertices (±d, ±d, 0):
```
Σ pᵢpᵢᵀ = diag(4d², 4d², 0)    → isotropic in the horizontal plane; measures u_x, u_y only
u_z = +√(1 − u_x² − u_y²)         sign is known: the drone cannot be below the ground
```
The two v0.3 objections to flat arrays, re-checked for ground-level arrays:

| Objection | Still a problem on the ground? |
|---|---|
| Cannot tell above from below | **No.** Below is physically impossible. The ground echo comes from the mirror direction, which has the same \|elevation\|, so it gives the same height. |
| Poor accuracy near the horizon | **Yes.** This is the real cost. |

**Elevation accuracy** (in units of the direction error σ_u):
```
flat:         el = acos(|u_h|)  →  σ_el ≈ σ_u / sin(el)
tetrahedron:  el = asin(u_z)    →  σ_el ≈ σ_u / cos(el)
equal at el = 45°
```

| Elevation | Flat | Tetrahedron |
|---|---|---|
| 10° | 5.8 σ_u | 1.0 σ_u |
| 30° | 2.0 σ_u | 1.15 σ_u |
| 45° | 1.4 σ_u | 1.4 σ_u |
| 70° | 1.06 σ_u | 2.9 σ_u |

**Flat clusters beat a tetrahedron above 45°.**

**Height error from one cluster** (h = ρ·tan el):
```
σ_h ≈ ρ · sec²(el) · σ_el
example ρ = 100 m, h = 30 m (el = 16.7°), σ_u = 0.02:
  flat:         σ_el = 0.02/0.287 = 0.070 rad  →  σ_h ≈ 100 × 1.09 × 0.070 ≈ 7.6 m
  tetrahedron:  σ_el = 0.02/0.958 = 0.021 rad  →  σ_h ≈ 2.2 m
```

**How a flat network gets height**
1. **Overlap.** A drone far from one cluster (low angle) is close to another (steep angle). The EKF weights each measurement by its own accuracy, so the steep view dominates.
   ```
   hexagonal grid, spacing s: farthest point from any cluster = s/√3
   want el ≥ 45° there, i.e. ρ ≤ h:   s ≤ √3 · h_min ≈ 1.73 · h_min
   ```

   | Lowest altitude h_min | Spacing s | Clusters per km² ≈ 10⁶ / ((√3/2)·s²) |
   |---|---|---|
   | 10 m | 17 m | ~4,000 |
   | 30 m | 52 m | ~430 |
   | 60 m | 104 m | ~110 |

   Strict 45° coverage is expensive for low drones (see F3). In practice the EKF needs good height from a steep view only part of the time, so looser spacing may be enough; the simulator will tell.
2. **Doppler and velocity [J].** Azimuths from several clusters give the ground track (ρ, ρ̇). The Doppler shift depends on ṙ, and r includes height:
   ```
   r = √(ρ² + h²),   ṙ = (ρ·ρ̇ + h·ḣ)/r
   level flight (ḣ = 0):   h² = (ρ·ρ̇ / ṙ)² − ρ²
   ```
   Inside the EKF this enters automatically through ∂f/∂p (13.2). In the v0.3 simulation, Doppler improved position ~3× and velocity ~10×.
3. **Optional: widely spaced single mics with GPS-synced clocks (TDOA).**
   ```
   τ_ij = (|p − m_i| − |p − m_j|)/c,     ∂τ_ij/∂z = (u_i,z − u_j,z)/c,   u_i,z = h/r_i
   ```
   Height sensitivity is large when the drone is above or inside the network and small far outside it. Needs about microsecond sync (GPS timing pulse) *(unverified for chosen modules)*.

**Surface ("boundary") mounting helps** *(derived)*. A mic at height h_m hears the direct sound plus a ground echo with extra path
```
Δ = 2 · h_m · sin(el)
first cancellation notch at  f₁ = c / (2Δ) = c / (4 · h_m · sin el)
h_m = 2 m (pole), el = 30°:   Δ = 2 m    → f₁ ≈ 86 Hz   (inside the drone band)
h_m = 1 cm (surface), el = 30°: Δ = 1 cm → f₁ ≈ 17 kHz  (above the band)
```
On the surface the echo arrives in phase, so the pressure doubles (**+6 dB**) with no notches in the drone band. Wind is also weaker at ground level. This **revises 7.7** (pole mounting at 2–3 m), which puts comb-filter notches into the band. Uneven ground is fine: survey each mic once, and height differences between clusters add a little vertical aperture.

**Layouts to compare in simulation**

| Layout | Height near a cluster | Height far away | Cost and complexity |
|---|---|---|---|
| Flat ground clusters (4 mics, square, 20–30 cm) | Good (better than tetrahedron above 45°) | Poor alone; helped by overlap + Doppler | Simplest to install |
| Tetrahedron (D-07) | Good | Good | Needs a raised 4th mic, levelling |
| Distributed single mics + GPS sync | Very good inside the network | Poor outside | Microsecond timing |

D-07 (tetrahedron) is now **under review** pending this comparison.

---

## 8. Cluster layout and range

### 8.1 Joseph's idea

> Place clusters in assorted areas so that one cluster's range slightly overlaps another's. [J]

### 8.2 Revision: overlap must be substantial [C revises J] *(derived)*

Triangulation needs **at least two clusters hearing the drone at once**. The hardest point to cover twice is right next to a cluster, whose nearest neighbour is s away. So:
```
spacing s ≤ detection range R      (not ≈ 2R)
```
Aim for **3 or more clusters** hearing every point. In the Kalman simulation (section 13), 3 clusters gave sub-metre error and 2 gave a few metres.

**If this revision is wrong:** only if height and position are not needed in the gaps between clusters. Single-cluster detection still works with slight overlap.

### 8.3 Detection range model [C] *(estimate)*

```
r_max = 10^((L_src − L_bg + G) / 20)
L_src ≈ 80 dB at 1 m (consumer quad, assumed)
G = detection margin from narrowband processing, assumed 10–20 dB
```

| Environment | L_bg | G = 10 dB | G = 20 dB |
|---|---|---|---|
| Quiet rural | 40 dBA | ~320 m | ~1,000 m |
| Suburban | 50 dBA | ~100 m | ~320 m |
| Urban street | 65 dBA | ~18 m | ~56 m |

- Every +6 dB doubles the range.
- An M-mic array adds about 10·log₁₀(M) dB: 4 mics ≈ +6 dB, about 2× range.
- **G should be measured** from the detector's detection-vs-SNR curve, not assumed.
- Wind, temperature layers and buildings can shrink range a lot beyond ~100 m.

**Update from real data (v0.6, 5.11.3):** the real Phantom 4 range at a rural site was ~100–150 m, not the 320–1,000 m in the table. Back-calculated G ≈ 2–7 dB for the single-mic CNN (or a quieter source). The table above is **optimistic**. Use ~100 m for cluster spacing until Joseph's own flights measure it.

---

## 9. What runs on the cluster vs the server

### 9.1 Joseph's idea

> Convert sound to the time-frequency domain, store it as data, transfer it to a server that runs the CNN. [J]

### 9.2 Revisions [C revises J]

1. **Direction finding must run on the cluster** *(derived)*. A magnitude spectrogram discards phase, and direction finding needs the inter-mic timing that lives in the phase.
2. **Bandwidth:** a full-precision spectrogram is larger than the raw audio *(derived)*:
   ```
   raw audio:   3 ch × 16,000 × 16 bit                ≈ 0.77 Mbit/s
   spectrogram: 3 ch × 256 bins × 62.5 fr/s × 32 bit  ≈ 1.5 Mbit/s
   ```
   So the cluster runs a cheap comb trigger and sends only a compact report when it fires.

Joseph's core idea is kept: time-frequency processing, data sent to a server, and the CNN on the server.

**If this revision is wrong:** if raw multichannel audio (with phase) is streamed to the server, direction finding could move to the server. That needs about 1 Mbit/s per cluster and good time sync.

### 9.3 Cluster report contents [C]

- GPS timestamp
- 3D direction (az, el) and its quality
- f(t) of the strongest harmonic
- drone-only loudness
- 8-bit spectrogram clip

### 9.4 Time sync [C]

Clusters send directions, so timestamps only need to agree to **tens of milliseconds** (NTP or GPS) *(derived)*. Measuring arrival-time differences *between* clusters would need microsecond sync, so that approach is avoided.

### 9.5 Matching detections across clusters [C]

The same drone has nearly the same f₀ at every cluster (within a few % of Doppler). Match detections by **time + f₀** before triangulating. *(derived)*

---

## 10. Height [J]

### 10.1 History

1. **[J]** Can relative motion equations give the drone's height?
2. **[C]** Not from azimuth alone (10.2). Motion helps only if a measurement depends on height.
3. **[J]** A 4th microphone on top (tetrahedron) gives the height?
4. **[C]** One tetrahedron gives the **elevation angle**; height needs a distance too, from a second cluster or a range estimate.
5. **[J]** Decision: tetrahedral arrays + Kalman filter to estimate (and predict) height.

### 10.2 Why azimuth alone cannot give height *(derived)*

```
θ = atan2(y − y_c, x − x_c)
```
h does not appear, so no number of clusters and no motion model can recover height from azimuths alone.

### 10.3 Chosen method: 3D triangulation inside the Kalman filter (D-14)

Each cluster's direction is a line in 3D, and the drone is where the lines meet. Equivalently, from one cluster:
```
h = z_cluster + ρ · tan(el)
σ_h ≈ ρ · sec²(el) · σ_el
example: ρ = 100 m, el = 16.7°, σ_el ≈ 0.02  →  σ_h ≈ 2.2 m (one cluster)
```
*(derived)*. In the Kalman filter this happens automatically every update. **Simulated: about 0.3 m height error with 3–5 clusters hearing** (section 13). *(simulated, best case)*

### 10.4 Fallbacks and cross-checks [C]

| Method | Formula | Use | Weakness |
|---|---|---|---|
| One cluster: elevation + range | h = r·sin(el) | Only one cluster hears | Needs a range (Doppler pass or loudness) |
| Doppler + ground track (level flight) | `r = √(ρ² + h²)`, `ṙ = ρρ̇/r` → `h² = (ρρ̇/ṙ)² − ρ²`; or `h = √(d² − ρ_min²)` | Cross-check on close passes | σ_h ≈ d·σ_d/h: poor when h is small compared with the pass distance |
| Loudness difference between clusters | `q = 10^((L_A − L_B)/10)`, `h² = (ρ_B² − q·ρ_A²)/(q − 1)` | Rough extra input | ±15–25 m; rotor directivity and gain errors |
| Planar elevation (3-mic) | `cos(el) = |u_horizontal|` | Superseded by the tetrahedron | Poor near the horizon |

All formulas *(derived)*; accuracy numbers *(estimate)*.

### 10.5 Height reference [C]

Triangulation gives height **relative to the clusters**. Height above ground under the drone = z − terrain(x, y), which needs a terrain map if the site is not flat.

### 10.6 Ground reflection [C]

Sound reflected from the ground looks like a mirror source below the horizon. The tetrahedron can tell up from down, so keep the peak above the horizon. Pole mounting helps separate the reflection in time. *(derived, qualitative)*

---

## 11. Distance from audio [J question]

| Method | Idea | Result | Evidence |
|---|---|---|---|
| Loudness | `r = 10^((L_src − L)/20)` | Unknown L_src: 6 dB of uncertainty means ×2 in distance, so distance **bands**, not metres | derived |
| Spectral tilt | Air absorbs high frequencies faster | Small effect below ~100 m; extra input only | estimate |
| Doppler fly-by (Doppler TMA) | Fit the S-shaped pitch curve | Speed and closest distance without knowing loudness | derived |
| Arrays | Triangulation / TDOA multilateration | Reliable position | derived |

**Doppler TMA derivation** (straight line at constant speed v, closest distance d at time t₀):
```
r(t) = √(d² + v²(t − t₀)²),   ṙ = v²(t − t₀)/r,   f(t) = f₀(1 − ṙ/c)
let T = d/v and x = (t − t₀)/T:   f = f₀·[1 − (v/c)·x/√(1 + x²)]
step height ±f₀v/c  →  v;   width T = d/v  →  d = v·T
```

- **Example:** v = 11.9 m/s, d = 22.6 m, 10th harmonic at 2831 Hz. Then T ≈ 1.9 s, step ±98 Hz, centre slope ≈ 52 Hz/s. Needs about **6T ≈ 11 s** of track to see the flat ends of the curve; a 3 s clip is too short. *(derived)*
- **Limits:** no information while hovering; RPM changes look like Doppler; assumes a straight path; gives distance but not direction.
- Literature: speed and range are known to be hard to separate in single-sensor Doppler TMA at low SNR or small Doppler (section 18).

**Dataset note** [C]: the simulated dataset peak-normalises clips and draws SNR independently of distance, which removes the loudness cue. To learn distance, use calibrated levels: source level + 1/r + fixed background, with no peak normalisation. *(derived)*

---

## 12. Towards or away

### 12.1 Joseph's idea

> Determine whether the drone is moving towards or away by finding the gradient of the frequency at a single frequency value, using a functional; then use acoustic triangulation to determine whether the drone is moving away or towards the microphone. [J]

### 12.2 Findings [C revises J, first half] *(derived)*

For a straight pass:
```
r̈ = v²d²/r³ > 0  →  ṙ increases throughout  →  f = f₀(1 − ṙ/c) always falls
```

- **The pitch falls even while the drone is approaching**, so a falling frequency does not mean moving away.
- Also:
  - the true f₀ is unknown;
  - RPM changes shift the pitch exactly like Doppler;
  - a drone flying straight at the mic has constant pitch until it passes.
- **What the frequency gradient can show:**
  ```
  ḟ = −f₀v²d²/(c·r³)
  ```
  |ḟ| is largest at the closest point. A fall that is speeding up means before the closest point; one that is slowing down means after. This is a weak cue, because it is noisy and RPM changes look the same.
- **Better single-cluster cue:** the trend of the drone-only loudness, measured in the comb bins:
  ```
  dL/dt ≈ −8.7·ṙ/r dB/s
  ```
  Rising means approaching.

### 12.3 Decision (D-15)

- **Primary [J, triangulation]:** the tracked position over time, from the Kalman filter, gives ṙ to each cluster or to any **protected location**, plus heading and time to arrival.
- **Supporting [J, functional]:** the frequency track (Doppler consistency) and the loudness trend. These are also the fallback where only one cluster hears the drone.

**If the revision is wrong:** the frequency gradient would only work alone if f₀ were known and RPM constant, which is not the case for multirotors.

---

## 13. Velocity and the Kalman filter [J]

### 13.1 Velocity from frequency [J question], multi-cluster solution [C]

- **One cluster, one moment:** gives only the radial velocity ṙ = c(f₀ − f)/f, and only if f₀ is known, which it is not.
- **Several clusters at the same emission time** *(derived)*: each cluster sees the same f₀ from a different direction uᵢ. Exact Doppler, rearranged:
  ```
  fᵢ = f₀ · c/(c + uᵢ·v)   →   fᵢ = f₀ − (fᵢ/c)·uᵢ·v
  ```
  This is linear in the unknowns (f₀, vₓ, v_y, v_z). Solve by least squares.
- **RPM changes cancel**, because f₀ is one of the unknowns. Each cluster's track must be aligned to the emission time (shift by rᵢ/c).

Snapshot test: drone at 30 m height, v = (9, −6, 1.5) m/s, 2 kHz harmonic, 0.2 Hz frequency error *(simulated)*:

| Setup | Direction error | Velocity RMS error (x, y, z) |
|---|---|---|
| 4 clusters, on the ground | 0.02 | 9, 5, 191 m/s (unusable) |
| 5 clusters, on the ground | 0.02 | 0.14, 0.16, 2.2 m/s |
| 5 clusters, on the ground | 0.005 | 0.04, 0.05, 0.64 m/s |
| 5 clusters, one on a 25 m mast | 0.02 | 0.13, 0.14, 0.73 m/s |

Takeaways:
- 4 clusters is the theoretical minimum but very fragile (condition number ~1,200 vs 46 with 5).
- Vertical velocity is the weakest axis when all clusters are on the ground.
- Direction errors matter more than frequency errors.

### 13.2 Kalman filter (D-12) [J], design [C]

**State:** `s = [x, y, z, vₓ, v_y, v_z, f₀]`

**Predict** (constant velocity plus random acceleration; f₀ random walk):
```
s⁻ = F·s,   P⁻ = F·P·Fᵀ + Q
Q per axis = q_acc·[[Δt³/3, Δt²/2], [Δt²/2, Δt]],   Q_f0 = q_f0·Δt
```

**Update** for each cluster that hears the drone:
```
measurements:  u = (p − c)/r,   f = f₀·c_s/(c_s + u·v)
y = z − h(s⁻),   S = H·P⁻·Hᵀ + R,   K = P⁻·Hᵀ·S⁻¹
s = s⁻ + K·y,    P = (I − K·H)·P⁻·(I − K·H)ᵀ + K·R·Kᵀ   (Joseph form)
```

**Jacobians** *(derived)*:
```
∂u/∂p = (I − u·uᵀ)/r
∂f/∂v = −f₀·c_s/(c_s + ṙ)² · u
∂f/∂f₀ = c_s/(c_s + ṙ)
∂f/∂p = −f₀·c_s/(c_s + ṙ)² · (I − u·uᵀ)·v/r
```

**Simulation results** *(simulated, mean of 5 runs, best case)*: 60 s loop over a 100 × 100 m field of 5 clusters, ~10 m/s, 5 s hover, 15 s climbing turn, RPM wander plus a 2% jump at the start of the climb.

| Setup | Position RMS (x, y, z) | Velocity RMS (x, y, z) | Hover speed reported (true 0) |
|---|---|---|---|
| 5 clusters, directions only | 0.5, 0.5, 0.6 m | 1.7, 1.1, 0.9 m/s | 1.37 m/s |
| **5 clusters, directions + Doppler** | **0.15, 0.12, 0.32 m** | **0.09, 0.10, 0.59 m/s** | **0.22 m/s** |
| 3 clusters, directions + Doppler | 0.3, 0.6, 0.7 m | 0.7, 1.0, 1.1 m/s | 0.83 m/s |
| 2 clusters, directions + Doppler | 2.4, 2.7, 4.4 m | 1.7, 1.7, 1.7 m/s | 1.22 m/s |

**Findings**
- Doppler improves velocity about 10× and position about 3×.
- The filter works with fewer clusters than the snapshot method, because it carries its estimate between detections.
- **q_f0 must be high (D-13).** With q_f0 = 4 Hz²/s, the RPM jump at climb onset was blamed on vertical motion (v_z error 1.48 m/s). With q_f0 = 100, it dropped to 0.59 m/s.
- **The raised mast cluster did not help in the filter** (it did in the snapshot test). The cause is not confirmed.

**Tuning**

| Parameter | Meaning | Value used |
|---|---|---|
| q_acc | How hard the drone can manoeuvre | 6 (m/s²)²/s |
| q_f0 | How fast the RPM can change | 100 Hz²/s |
| σ_u | Direction error per cluster | 0.02 (assumed; measure) |
| σ_f | Frequency tracking error | 0.3 Hz (assumed; measure) |

**Prediction** [J]: `h(t + Δt) = h + v_z·Δt`, with uncertainty growing via P. Trust a few seconds ahead more than longer horizons.

**Not modelled yet**
- Sound travel delay: 0.3 s at 100 m is about 3 m at 10 m/s. Fix: evaluate h(s) at p − v·r/c.
- Several drones and false detections (needs data association, e.g. global nearest neighbour).
- Echoes.
- Several rotors overlapping in the tracked harmonic.
- Filter initialisation, which should come from a first two-cluster triangulation.

---

## 14. Simulation and training data

### 14.1 Joseph's idea

> Simulate drone noise for a neural network. [J]

### 14.2 Simulator [C] *(modelling choices; not validated against real recordings)*

**Source**
- BPF harmonics (2–3 blades, 3,000–10,000 RPM)
- motor whine (7 pole pairs)
- BPF-modulated broadband noise
- 4/6/8 rotors at ±3% RPM, with ±2% slow drift

**Propagation**
- 1/r spreading
- heuristic air-absorption low-pass
- Doppler via a time-varying delay (relative delay only)

**Mixing**
- **Backgrounds** (synthetic): wind, traffic, rain, birds; real WAVs via `--bg-dir`
- **Confusers** (synthetic): lawnmower, motorbike, insect
- SNR −15 to +15 dB

### 14.3 Generated sample *(simulated)*

- 400 clips, 3 s, 16 kHz mono
- 203 drone, 106 confuser, 91 background only
- split 316 / 36 / 48
- `labels.csv`: label, split, source, background, confuser, snr_db, rpm, n_blades, n_rotors, bpf_hz, path, closest_m, speed_mps, closest_t_s, seed

### 14.4 Findings from the examples

- Background type matters more than the SNR number. A −15 dB drone was still visible under rain (high-frequency hiss) but nearly invisible under traffic (low-frequency rumble). Evaluate per background.
- Doppler at drone speeds is only a few percent, so the bend in a spectrogram is subtle.

### 14.5 Known gaps

- Clip-level labels only; sound event detection needs frame-level scenes (D-02).
- BPF range should widen to ~40 Hz–2 kHz with drone profiles (heavy-lift through FPV).
- Pole pairs are fixed; flight states (forward-flight rotor split, climb) are not modelled.
- Peak normalisation removes the distance cue (section 11).
- Synthetic backgrounds and confusers are placeholders.
- **D-18 (open):** if piston-engine drones are in scope, the lawnmower/motorbike "not drone" labels teach the wrong thing.

### 14.6 Scenario simulator (D-29) [J], physics [C]

**Joseph's idea [J]:** render whole scenarios. Put microphones at fixed positions, fly a drone over them, and record what each microphone hears, like the Blender scenes used for the drone tracker.

The v0.3 simulator (14.2) made isolated 3 s clips. Its code was lost with the session folder (Appendix A), so it is rebuilt as a scenario simulator in `sim/`.

**Scene contents**
- Microphones at chosen 3D positions, including height above ground (surface or pole).
- Any number of **sources**, each with a trajectory p(t), a sound generator and a label category: drone (electric / two-stroke), ground engine, vehicle, insect, fan, generator, bird, rain, wind, and so on.
- Sources do not have to fly: a chainsaw is a source with a fixed position, a motorbike is a source moving along the ground.

**Emission time** *(derived)*. Sound reaching mic m at time t left the source at τ, where
```
t = τ + |p(τ) − m| / c
```
Solved by fixed-point iteration τ ← t − |p(τ) − m|/c. This contracts with factor |v|/c (< 0.15 even at 50 m/s), so 3–4 iterations reach sub-microsecond accuracy.

**Received signal**
```
x_m(t) = D(θ, τ) · s(τ) / r(τ)                      direct path
       + R_g · D(θ′, τ′) · s(τ′) / r′(τ′)            ground reflection (image source at (x, y, −z))
       + Σ backgrounds + wind_m(t) + self-noise_m(t)
then air absorption applied per path as a time-varying filter
```
- **Doppler appears automatically.** dτ/dt = 1/(1 + ṙ/c), so a tone emitted at f₀ is received at f₀/(1 + ṙ/c), the same model as the EKF (13.2). Nothing Doppler-specific is coded.
- **Spherical spreading** 1/r, in pascals, with source level L_src at 1 m.
- **Fractional delay:** s(τ) is rendered on a uniform source-time grid and read at the non-integer τ by 4-point cubic (Lagrange) interpolation.
- **Air absorption:** α(f) from ISO 9613-1 (temperature and humidity randomised) applied per STFT frame as gain 10^(−α(f)·r/20). Distant drones lose their high harmonics first.
- **Ground:** reflection factor R_g randomised from soft (grass, ~0.5) to hard (asphalt, ~1.0). Mic height randomised, mostly surface (1–5 cm, D-30) with some poles (1–3 m), so both cases are covered.
- **Directivity:** rotors are louder in some directions. Modelled as D(θ) = 1 + β·cos²θ with θ measured from the rotor axis, β randomised (estimate).

**Diffuse backgrounds** (wind in trees, traffic rumble, rain) are rendered as about 8 plane waves from random directions near the horizon, each carrying independent noise with the same spectrum. This gives the correct correlation between nearby mics automatically. For a spherically isotropic field the coherence between two mics d apart is
```
γ(f) = sin(k·d) / (k·d),     k = 2πf/c
```
which matters for direction finding later. Wind noise at the mic is independent per mic (turbulence at the windshield).

**Calibrated levels.** Audio is stored in pascals with a fixed full scale (120 dB SPL; v0.4 said 110, raised because close sources clipped), 24-bit, so **distance is still audible in loudness** (fixing the 14.5 / section 11 problem). Gain is randomised only at training time.

**Clip length [J]:** long enough to hear the **entire pass**. A pass starts and ends where the drone is far enough to be barely audible:
```
duration = path length / speed,   path from −R_far to +R_far along the track
R_far ≈ 250–600 m depending on class loudness  →  typical clips 20–120 s
```
Hover, orbit and loiter scenarios are capped at 120 s.

**Scenario families**

| Family | Description |
|---|---|
| Flyover | Straight pass, closest horizontal distance 0–300 m, class-dependent altitude and speed |
| Approach–hover–depart | Electric and two-stroke multirotors only |
| Orbit / loiter | Circle at a random radius around a point |
| Random waypoints | Turns, climbs, descents; RPM follows thrust (climb → RPM up) |
| Fixed-wing pass | Two-stroke fixed-wing: 20–45 m/s, never hovers |
| No drone | Backgrounds and confusers only (about 25% of scenes) |

**Physics of the drone flight → RPM** *(derived, simplified)*. Hover thrust T = m·g split over N rotors, thrust ∝ ω², so ω_hover ∝ √(m·g/N). Climbing at acceleration a scales thrust by (g + a)/g, so RPM scales by √((g + a)/g). Forward flight tilts the drone: front and rear rotors split by a few percent.

**Labels per 10 ms frame** (stored next to each clip): per-source SNR at each mic (5.8.6), distance, elevation, radial velocity ṙ, emitted f₀ and received f₀, and the source category. Any of the D-28 labelling options can be built from these.

**Randomised every scene ("domain randomisation")**: class, size, blade count, rotor count, RPM and wander, harmonic amplitude shape, whine pole pairs, source level, directivity, trajectory, speed, altitude, ground type, mic height, temperature and humidity, background mix and level (35–70 dBA), confusers, wind.

**Held out for test-sim-held-out (5.8.7):** RPM above the training range for each class, 7- and 5-rotor configurations, one background family, and SNRs below the training minimum.

### 14.7 Field test protocol (D-31) [J], details [C]

**Joseph's plan [J]:** set up the microphone, fly a drone over, and check whether it is picked up.

| Recording | Tells us |
|---|---|
| Drone flying over | Detections and misses |
| Same place, **no drone** | False alarms. Without this, a detector that always says "drone" would pass. |
| No drone, with a car, mower, chainsaw or fan nearby | Confusion with similar sounds |

- **Ground truth:** the drone's flight log (GPS and time), or spoken "drone up / drone down" markers on the recording.
- **Distances:** passes at roughly 10, 30, 60 and 100 m, to get a detection-range curve (detects up to X m). That number sets the cluster spacing.
- **Microphone:** one mic is enough for detection. WAV, ≥ 32 kHz, uncompressed, windshield, the same gain for every session. If possible, also record a calibration tone or note the mic's sensitivity, so levels can be compared with the simulator.
- **Fair test:** the simulator's class ranges must contain the test drone's prop size, blade count and RPM, but the simulator is not tuned to that drone.
- The recordings are never used for training (D-25).

---

## 15. Known errors and corrections during the design

Recorded so that similar mistakes are easier to spot.

| Where | Error | Correction |
|---|---|---|
| Simulator `propagate` | Full travel delay made distant clips start with up to ~1 s of silence | Only the change in delay is applied |
| Multi-cluster velocity, first attempt | Sign error (ṙ = +u·v) and the linear Doppler approximation (~1.7 Hz error) | Exact Doppler rearranged into a linear form |
| Kalman test, first flight | Flight left the coverage area (median 1 cluster hearing), so results were meaningless | Flight redesigned as a loop over the cluster field |
| Kalman tuning | q_f0 too low; RPM jump read as vertical motion | q_f0 = 100 |
| Mast cluster | Helped in the snapshot test, not in the filter | Left optional; cause unconfirmed |
| Example captions | Three captions claimed more than the spectrograms showed | Re-picked examples, rewrote captions |
| Design v0.3, D-22 | Called for a "frame-level 16 kHz" CNN14; no such checkpoint exists (16 kHz is clip-level only) | Use NeMo's `Cnn14_DecisionLevelMax` at 32 kHz; render at 32 kHz |
| Appendix A prototype code | Kept only in a temporary session folder, which was later deleted | Rebuilt in the repo (`sim/`); anything worth keeping now goes into git |
| "Noise sound database from a sponsor" | Searched the event page, Gmail (student account) and Drive; nothing found | Unresolved; public datasets listed in section 18 |
| Range estimate (8.3) | Assumed a detection margin G of 10–20 dB; real flights imply G ≈ 2–7 dB, so the estimated range was 3–8× too long | Corrected in 8.3 and 5.11.3; spacing planned from ~100 m |
| Dataset suggestion (section 18, early discussion) | Recommended the Al-Emadi DroneAudioDataset without checking its licence; it forbids security and surveillance use | Caveat added in section 18; REAL_TESTS.md had already excluded it |
---

## 16. Known flaws and risks [C]

A review of the whole plan, in order of severity. Each item names the decisions it affects and whether the flaw is in Joseph's core idea or in Claude's extensions.

**Summary:** Joseph's core ideas hold up:
- clusters with overlapping coverage,
- time-frequency processing with a server CNN,
- triangulation for position and towards/away,
- tetrahedron + Kalman filter for height,
- the converted view for the CNN.

Most critical flaws are in Claude's extensions: the single-f₀ Doppler model, the assumed direction accuracy, and numbers that come only from simulation.

### 16.1 Critical: could change the design

| # | Flaw | Why it matters | Affects | Origin | Fix |
|---|---|---|---|---|---|
| F1 | **No validation on real audio.** All accuracy figures come from Claude's simulator, tested against itself (clean comb, omnidirectional source, no echoes, no wind) | Sub-metre position, 0.1 m/s velocity, 0.3 m height and 0.2 Hz tracking are best-case and may be far off | D-12 to D-16 | [C] | Record a real drone with a GPS log using one or two prototype clusters; measure actual errors first. v0.4: Joseph's field recordings are now the planned test (D-25, 14.7) |
| F2 | **A multirotor has no single f₀.** Each rotor has its own BPF, a few % apart, changing constantly (that is how it steers) | Doppler differences between clusters are also only a few % (up to ~2v/c ≈ 7% at 12 m/s). If clusters lock onto different rotors' lines, the error is as large as the signal. The Kalman simulation used one clean tone | D-13, D-16, 9.5 | [C] | Model rotor lines separately, or track each harmonic's line-group centre; re-run simulations with rotor spread |
| F3 | **Coverage needs many clusters.** Each covers a hexagon of (√3/2)·s². 1 km² needs ~115 clusters at s = 100 m, ~460 at s = 50 m. Urban range ~20–50 m | Cost and maintenance (power, network, weatherproofing, levelling, surveying). The range itself rests on assumed numbers (80 dB at 1 m, G = 10–20 dB) | D-06, D-09 | Range numbers [C]; layout [J] | Measure real range first; scope to protected sites rather than open-area coverage. **v0.6:** real Phantom 4 range ~100–150 m (5.11.3), so plan s ≈ 100 m (~115 clusters/km²) |
| F4 | **Direction accuracy is assumed** (σ_u = 0.02, σ_τ = 10 µs) | Tonal signals give repeating GCC-PHAT peaks (ambiguous direction). At 60 Hz, λ = 5.7 m is ~30× the array size. FPV BPF (≥ 750 Hz) is near the 860 Hz alias limit | D-08, D-14 | [C] | Test direction finding on real drone audio with the real array; use the broadband component too; consider a second smaller array or a larger aperture with more mics. v0.4: flat ground clusters (D-30) are worse than a tetrahedron below 45° elevation |

### 16.2 Significant: needs design work

| # | Flaw | Affects | Origin | Fix |
|---|---|---|---|---|
| F5 | Missing false-alarm sources: fans and air conditioners (real blade-passing tones), generators and transformers (50 Hz mains-hum comb), helicopters, idling cars | D-02, D-17 | [C] (confuser list) | Reject tracks that are stationary and near the ground; add these sounds to training data |
| F6 | Several drones of the same model have nearly equal f₀, so time + f₀ matching fails | 9.5, D-12 | [C] | Multi-target tracking: per-track state, assignment by geometry; simulate it |
| F7 | Kalman gaps: no sound-delay model (~3 m at 100 m and 10 m/s); no outlier rejection; initialisation assumed close; vertical velocity weakest (±0.6 m/s best case) | D-12, D-13 | [C] | Emission-time model; Mahalanobis gating; initialise from a two-cluster triangulation |
| F8 | Echoes from buildings give false directions; simulations had none | D-14 | [C] | Gating (F7), require agreement between clusters; expect worse urban accuracy |
| F9 | Wind noise sits in the drone's main band (50–500 Hz) and is worse on pole mounts; ignored in range estimates | D-09, 7.7 | [C] | Good windshields; measure range in windy conditions |
| F10 | **Recording law and privacy.** Mics also capture conversations. In Victoria, check the Surveillance Devices Act 1999 *(unverified, get advice)* | 9.3 | — | Clusters already send features, not raw audio; keep spectrogram clips short and limited to drone frequencies, or drop them |

### 16.3 Minor

- **Unknown drone types:** the type classifier should be able to answer "drone, type unknown" rather than force a known class (D-04, D-05).
- **High altitude:** quiet drones above ~100–150 m may be inaudible. Consider pairing with RF or radar for serious use.
- **Dataset gaps:** see 14.5.
- **D-19 untested:** see open question 9.
- **No audio buffer:** a short ring buffer (e.g. 30 s) on each cluster would allow re-analysis on request.

---

## 17. Open questions and next steps

1. **D-18:** which drone types must be detected (consumer/FPV multirotors only, or also fixed-wing and piston)?
2. Pick and test the 4-channel synchronous audio hardware for the tetrahedron.
3. Measure real σ_u, σ_f and detection range G with a built cluster and a GPS-logged drone flight.
4. Add sound-delay correction, initialisation and multi-drone data association to the Kalman filter.
5. ~~Build the frame-labelled scene generator and train/evaluate the CRNN (D-02, D-03).~~ Done v0.5 (14.6, 5.9).
6. Widen the simulator: drone profiles, flight states, calibrated levels for distance.
7. Collect real recordings per drone model for type identification (D-05).
8. Decide whether a mast cluster is worthwhile, using real data.
9. Test D-19: train the same CNN on linear vs log-frequency input and compare accuracy on real recordings, especially at RPMs not seen in training.
10. ~~Source of real test data~~ Resolved: real testing uses Joseph's own flyover recordings (D-25, 14.7) and the public datasets in `results/real_world/REAL_TESTS.md`.
11. Re-run the velocity and Kalman simulations with per-rotor lines (F2).
12. Add multi-target tracking, emission-time correction and gating to the Kalman filter (F6, F7).
13. Fine-tune CNN14 with front ends A and B and compare on real recordings (D-22, D-23). v0.5: trained and compared on sim (5.9, roughly a draw); real recordings still needed.
14. ~~Confirm CNN14's `bn0` layout~~ Done: per-band BatchNorm2d(64), confirmed in the official code.
15. Get advice on recording-law requirements before any public deployment (F10).
16. **D-28:** choose how two-stroke drones are separated from chainsaws and other ground engines (proposed: electric drone / engine / nothing).
17. Are the two-stroke drones fixed-wing or multirotor? (Both are rendered until decided.)
18. Which drone and which microphone for the field test (14.7)? Check that the drone sits inside the simulated class ranges.
19. Compare flat ground clusters, tetrahedra and distributed GPS-synced mics in the scenario simulator: height error vs distance and cluster spacing (D-30).
20. Tune the frame-label thresholds (v0.5 rule: 2nd tone region ≥ 6 dB positive, 1st < 0 dB negative, 5.8.6) against what a person can hear in the rendered clips.
21. Decide whether real recorded backgrounds (not drones) may be mixed into training, if the purely synthetic backgrounds turn out too clean.
22. Explain the 76 % electric-drone detection at 0–25 m (5.9.4): metric or model?
23. Decide how to score drone_engine against ground engines (≈ 28 false engine-drone alarms per hour on drone-free scenes, mostly chainsaws and blowers; D-28).
24. Run B and A on Joseph's flyover recordings (14.7) with `cnn/infer.py`.
25. Test direction finding (GCC-PHAT) on the IDMT 8-channel array recordings against their GPS positions: the first real check of F4 and D-14 (5.11.4).
26. Does the electric head's reliance on > 1 kHz (5.11.2) cost range or noise robustness? Test band-limited inputs and real tonal negatives above 1 kHz.
27. Measure real drone source levels (L_src at 1 m) from the IDMT recordings and GPS distances, to separate "quiet source" from "small G" in 5.11.3.

---

## 18. References

*(unverified unless stated: only abstracts or search results were read)*

**Doppler TMA**
- B. G. Ferguson, ASA 130th Meeting (1995): single-sensor blade-rate Doppler for propeller aircraft speed and CPA range. https://auditory.org/asamtgs/asa95stl/4pSP/4pSP10.html
- B. G. Ferguson et al., ASA 124th Meeting (1992): turboprop over a 15-microphone line array. https://auditory.org/asamtgs/asa92nwo/1pEA/1pEA10.html
- Single-hydrophone tonal motion-parameter estimation, Sensors (2023): speed/range coupling at small Doppler or low SNR. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10422508/
- Xu & Yang, matched Wigner transform for single-sensor motion parameters. https://doi.org/10.1631/jzus.A1300263
- Doppler-only TMA in high-duty-cycle sonar (Cramér-Rao bound, ML efficiency). https://hal-univ-tln.archives-ouvertes.fr/hal-03661347

**Search terms:** Doppler TMA, CPA estimation, LOFAR, GCC-PHAT, SRP-PHAT, MUSIC, TDOA multilateration, acoustic UAV detection and localisation, ISO 9613-1 (atmospheric absorption).

**Datasets** (not checked in this session): ESC-50, UrbanSound8K, FSD50K, DEMAND, MS-SNSD (backgrounds); DREGON (drone audio). Real tests actually used: see `results/real_world/REAL_TESTS.md` (Fraunhofer IDMT, Svanström et al. 2021, ESC-50).

**Do not use:** DroneAudioDataset (Al-Emadi et al.). Its licence forbids security and surveillance use, which is this project's purpose (found in REAL_TESTS.md; v0.6).

**Pretrained audio models**
- Kong et al., PANNs: Large-Scale Pretrained Audio Neural Networks. https://arxiv.org/pdf/1912.10211
- NeMo Curator PANNs adapter (decision-level CNN14, 100 fps, MIT-licensed vendored code). https://docs.nvidia.com/nemo/curator/nemo-curator/nemo_curator/models/audio/sed/panns.md
- NeMo Curator CNN14. https://docs.nvidia.com/nemo/curator/nemo-curator/nemo_curator/models/audio/sed/cnn14
- audioset_tagging_cnn README (`Cnn14_16k` checkpoint and settings). https://github.com/zanilzanzan/audioset_tagging_cnn/blob/master/README.md
- Community CNN14 fine-tuning results. https://github.com/sithu31296/audio-tagging
- CNN14 fine-tuned for engine sounds (example). https://huggingface.co/joakes90/engine_sound_cassifier
- E-PANNs (smaller variant). https://arxiv.org/pdf/2305.18665

**Tools:** `librosa.pcen`, `sed_eval`, `scipy.optimize.curve_fit`.

---

## Appendix A: prototype code

**Lost (checked 2026-10-22): the session folder no longer exists.** The simulator is rebuilt in the repo under `sim/` (14.6); the CNN code is in `cnn/` (5.8, 5.9). See README.md for the repo layout and replication steps. Original location, for the record:
`/tmp/claude-1000/-home-quail/13037ded-e6b5-4d0b-a991-c21a656b52b4/scratchpad/`

| File | Contents |
|---|---|
| `drone_sim.py` | Drone source model, propagation (distance, absorption, Doppler), SNR mixing |
| `make_dataset.py` | Labelled WAV dataset generator (backgrounds, confusers, `--bg-dir`) |
| `build_examples.py` | Example page builder (spectrograms + audio) |
| `track_freq.py` | Time-frequency functional / complex demodulation tracker and test |
| `doppler_velocity.py` | Multi-cluster Doppler velocity solver and test |
| `ekf_tracker.py` | `DroneEKF` (directions + Doppler) and the simulated flight comparison |

## Appendix B: interactive pages (private claude.ai artifacts)

- Simulated drone dataset examples: https://claude.ai/artifact/Bre8zHcDAjabxAoTeqN81H
- Drone Tone Probe (functional, phase gradient, frequency gradient): https://claude.ai/artifact/6V61Eg3Zjk9og3vJtqEwhd
- Drone Harmonics in 3D (linear vs log frequency axis, same sound): https://claude.ai/artifact/B3UgztorC2HZ4JJhAVKgS9
