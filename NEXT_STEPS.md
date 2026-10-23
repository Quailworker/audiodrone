# Next steps (as of design v0.8.1, 2026-10-23)

What to build next, in order. Details and reasons are in `design.md`; this page is the to-do list.

## How to start the next task on the Quail desktop

On the desktop, in a terminal:

```bash
cd ~/audiodrone          # wherever the repo is cloned
git pull
claude "Read tasks/phase1_localisation.md and carry it out, steps 0 to 6. Follow AGENTS.md. Write your report to tasks/phase1_localisation_REPORT.md and push your work."
```

Run it in a permission mode that doesn't stop for every approval (auto or accept-edits) so it can work unattended. `gh` must be logged in to push.

Remote start from the laptop needs SSH over Tailscale, and the desktop uses Tailscale SSH **check mode**: each connection must be approved in a browser (login.tailscale.com) before it works.

## 1. Phase 1: localisation in simulation: **done** (design v0.9)

**Done 2026-10-23.** Results: design.md 13.5, `results/loc_v1/`, `tasks/phase1_localisation_REPORT.md`.
- Tetrahedra beat flat clusters.
- The ground echo needs a reflection-aware beamformer.
- Steering on the drone's comb lines is the biggest gain (2–5×).
- Real azimuth accuracy is about 8.5°.
- Doppler (step 3b) is not built yet.

Original plan, kept for reference:

**Instructions:** [`tasks/phase1_localisation.md`](tasks/phase1_localisation.md)

Why first: height, speed and position feed everything downstream (the likelihood rating D-32, the danger zone D-35, the alerts D-34). Direction accuracy (F4) and the rotor-spread problem (F2) are the biggest untested risks.

- Multi-cluster scenes (tetrahedron vs flat, pole vs ground, 100 m vs 150 m spacing) with ground truth
- Direction finding per cluster (SRP-PHAT, 100–860 Hz)
- Kalman tracker (directions; Doppler optional)
- Scoring: direction, height and velocity error vs distance and number of clusters
- Real check: direction finding on the 8-channel IDMT recordings vs GPS (open question 25)
- Write-up: design.md v0.9, section 13.5

## 2. Phase 2: decide and alert (fixed logic, no language model)

Uses the Phase 1 tracks. Instructions to be written after Phase 1 reports.

- Likelihood rating (D-32, design 5.12): CNN probability × height × speed × persistence; tune in simulation
- Danger zone (D-35, design 13.3): closest approach, arrival time, uncertainty corridor
- Alert core: tiers advisory / take cover / evacuate (design 13.4); first alert within ~1 s

## 3. Phase 3: the people-facing layer

- Nemotron agent with tools (D-34, design 13.4); numbers only from tool results
- Zetaris: personnel, unit and shelter locations, cluster reports, tracks (MCP)
- Meterless Nightshift: serve Nemotron locally on site (confirm it can)
- Alert delivery (phone, SMS, sirens, ATAK) and a map view

## In parallel (independent of the phases)

- **Detector retrain** (UPGRADES.md priorities 2–6): recall gate, recording-chain augmentation, recall-first thresholds, hard mixtures, wider mid class. One retrain on Anna's 3080. No real negatives (D-25).
- **Hardware (D-37, design 9.6):** each cluster is an **ESP32-S3 (8 MB PSRAM) with 4 TDM MEMS mics on one I²S bus**, sending packets; no Raspberry Pi. Build one on the bench first: check the 4 channels are sample-synchronised, stream raw audio over Wi-Fi (mode A) for the prototype, then add the comb trigger and buffered sending (mode B). About $110–235 per cluster (open questions 35, 36).
- **Field test** (design 14.7): own flyovers, uncompressed WAV, GPS flight log, plus drone-free recordings at the same site. Check recording law first (F10).

## Decisions needed from Joseph

| Open question | Decision |
|---|---|
| 30 | Which Nemotron model, and whether Meterless Nightshift can serve it |
| 31 | Where personnel, unit and shelter locations come from, and who may access them |
| 32 | Alert delivery channels, and who may approve an evacuation |
| 33 | Likely approach routes, expected drone speeds and required warning time (sets the tripwire ring, D-36) |
| 34 | Danger radius, look-ahead limit and widening factor for the alert rule |

## Backups

- Tag `backup-2026-10-23-v0.8`: snapshot of the repo at design v0.8
- Release `models-v0.5`: trained models B, A, C
