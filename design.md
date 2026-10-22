# Acoustic Drone Detection and Tracking: Design

- **Status:** draft v0.3, 2026-10-22.
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

### 2.2 Current architecture (after the revisions below)

```
CLUSTER (one per site)                                     [J] clusters; [J←C] 4 mics
  4 MEMS mics, regular tetrahedron, 20 cm edge
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
| D-07 | Each cluster is a **4-mic regular tetrahedron** (originally 3 mics) | [J←C] | derived | D-06 | **Adopted** |
| D-08 | Tetrahedron edge **20 cm**; direction finding uses frequencies **< 860 Hz** | [C] | derived | D-07 | Proposed |
| D-09 | Cluster ranges must **overlap substantially** (spacing ≈ range), not slightly | [C revises J] | derived | D-06 | Proposed |
| D-10 | **Direction finding runs on the cluster**; server receives directions, not only spectrograms | [C revises J] | derived | D-06, D-07 | Proposed |
| D-11 | Use a **time-frequency functional** to get one frequency's amplitude and exact frequency over time | [J] | simulated | D-01 | Adopted |
| D-12 | Use a **Kalman filter** (EKF) to estimate and predict position, height and velocity | [J] | simulated | D-07, D-10 | **Adopted** |
| D-13 | Kalman state includes the drone's pitch f₀, with **fast-changing f₀** allowed | [C] | simulated | D-12 | Proposed |
| D-14 | Height comes from **3D triangulation of tetrahedral directions inside the Kalman filter** | [J] + [C] | simulated | D-07, D-12 | **Adopted** |
| D-15 | Towards/away is decided from the **tracked position**, with the frequency gradient as a supporting cue | [J] triangulation; [C revises J] | derived | D-11, D-12 | Proposed |
| D-16 | Velocity from **Doppler at several clusters at once** (inside the Kalman filter) | [C] | simulated | D-12 | Proposed |
| D-17 | Train with **simulated drone audio**, test on real recordings | [J] simulate; [C] split | simulated | D-02 | Adopted |
| D-18 | Which drone types are in scope (affects whether lawnmower/motorbike sounds are "not drone") | — | open | D-04, D-17 | **Open** |
| D-19 | CNN input is a **converted view**: a **log-frequency spectrogram with loudness in dB** | [J] converted view; [C] log axis | derived; not yet tested on a CNN | D-02 | **Adopted** |
| D-20 | Feed several views: log spectrogram, harmonic-salience map, per-frame features, optionally beamformed audio | [J] converted view; [C] specific views | open | D-03, D-19 | Proposed |
| D-21 | Training data: widely varied simulated drones on real backgrounds, then fine-tune and test on real recordings; type identification from real per-model recordings | [J] simulated; [C] mix and order | open | D-05, D-17 | Proposed |
| D-22 | Start from **pretrained CNN14** (PANNs, AudioSet; frame-level 16 kHz variant), fine-tuned on simulated (rendered) audio converted to a log-frequency spectrogram | [J] | open: not trained yet | D-02, D-19, D-21 | **Adopted** |
| D-23 | CNN14 front end: start with stock log-mel as a baseline, then a mel scale with a lower bend point (log-spaced from ~50 Hz), longer window | [C] | derived; not tested | D-19, D-22 | Proposed |

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
- **Changing the band count** probably requires replacing the first normalisation layer (`bn0`, which appears to normalise each mel band separately). This was seen only in a derived copy of the code, so check the official `models.py` or the checkpoint's layer names *(unverified)*. The convolutional layers and global pooling accept any input height.
- **Fine-tuning order:**
  1. Replace the 527-class output with drone / no-drone (type later).
  2. Train the new head with the rest frozen.
  3. Unfreeze deeper layers with a small learning rate.
- **Extra views** (harmonic-salience map, comb features, whine/BPF ratio; D-20) cannot pass through CNN14. Join them with CNN14's frame-level output just before the final layers.
- **Rendered audio alone is not enough.** Mix simulated drones into real backgrounds, and test only on real recordings (D-21).
- **Licence:** PANNs code is MIT. Check the terms for the pretrained weights and AudioSet before any commercial use *(unverified)*.

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

- Pole-mounted 2–3 m above the ground, with a thin open frame.
- Measure the built mic positions to about 1 mm.
- **Level each array carefully.** A 1° tilt adds 1° to every elevation, about **1.9 m of height error at 100 m** *(derived)*. Calibrate with a drone flight that logs its own GPS position.
- Survey the clusters' relative heights once (laser level or RTK GPS). Consumer GPS altitude is off by several metres.

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
| "Noise sound database from a sponsor" | Searched the event page, Gmail (student account) and Drive; nothing found | Unresolved; public datasets listed in section 18 |

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
| F1 | **No validation on real audio.** All accuracy figures come from Claude's simulator, tested against itself (clean comb, omnidirectional source, no echoes, no wind) | Sub-metre position, 0.1 m/s velocity, 0.3 m height and 0.2 Hz tracking are best-case and may be far off | D-12 to D-16 | [C] | Record a real drone with a GPS log using one or two prototype clusters; measure actual errors first |
| F2 | **A multirotor has no single f₀.** Each rotor has its own BPF, a few % apart, changing constantly (that is how it steers) | Doppler differences between clusters are also only a few % (up to ~2v/c ≈ 7% at 12 m/s). If clusters lock onto different rotors' lines, the error is as large as the signal. The Kalman simulation used one clean tone | D-13, D-16, 9.5 | [C] | Model rotor lines separately, or track each harmonic's line-group centre; re-run simulations with rotor spread |
| F3 | **Coverage needs many clusters.** Each covers a hexagon of (√3/2)·s². 1 km² needs ~115 clusters at s = 100 m, ~460 at s = 50 m. Urban range ~20–50 m | Cost and maintenance (power, network, weatherproofing, levelling, surveying). The range itself rests on assumed numbers (80 dB at 1 m, G = 10–20 dB) | D-06, D-09 | Range numbers [C]; layout [J] | Measure real range first; scope to protected sites rather than open-area coverage |
| F4 | **Direction accuracy is assumed** (σ_u = 0.02, σ_τ = 10 µs) | Tonal signals give repeating GCC-PHAT peaks (ambiguous direction). At 60 Hz, λ = 5.7 m is ~30× the array size. FPV BPF (≥ 750 Hz) is near the 860 Hz alias limit | D-08, D-14 | [C] | Test direction finding on real drone audio with the real array; use the broadband component too; consider a second smaller array or a larger aperture with more mics |

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
5. Build the frame-labelled scene generator and train/evaluate the CRNN (D-02, D-03).
6. Widen the simulator: drone profiles, flight states, calibrated levels for distance.
7. Collect real recordings per drone model for type identification (D-05).
8. Decide whether a mast cluster is worthwhile, using real data.
9. Test D-19: train the same CNN on linear vs log-frequency input and compare accuracy on real recordings, especially at RPMs not seen in training.
10. Check whether `~/drone18_field_test` and `~/field_recorder` contain real drone recordings usable for fine-tuning and testing (D-21). Not yet opened.
11. Re-run the velocity and Kalman simulations with per-rotor lines (F2).
12. Add multi-target tracking, emission-time correction and gating to the Kalman filter (F6, F7).
13. Fine-tune CNN14 with front ends A and B and compare on real recordings (D-22, D-23).
14. Confirm CNN14's `bn0` layout in the official PANNs code before changing the band count (D-23).
15. Get advice on recording-law requirements before any public deployment (F10).

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

**Datasets** (not checked in this session): ESC-50, UrbanSound8K, FSD50K, DEMAND, MS-SNSD (backgrounds); DroneAudioDataset (Al-Emadi et al.), DREGON (drone audio).

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

Written during the discussion in a temporary session folder; **copy these somewhere permanent if you want to keep them**:
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
