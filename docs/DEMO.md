# Demo script: Border watch (rendered, about 3 minutes)

A simulated demo, said as such: every incident was rendered on real UAE terrain and run through the real detection chain (CNN, frequency tracker, beamformer, tracker, alerts). Nothing is staged by hand. The screen's header says "SIMULATED INCIDENTS, NOT A LIVE FEED".

## 10 minutes before

1. `bash scripts/demo_check.sh --start`. Every line should be `OK`, ending with `READY`.
   - If Nightshift fails, the Ask box still answers with Nemotron Mini on this PC and shows "THIS PC (OLLAMA)". It's fine to say so.
   - If Zetaris fails: `cd ~/zetaris/zetaris-platform && docker compose up -d`, wait 2 minutes, then check again.
2. On the GPU PC: Nightshift open. Nothing heavy running is good, but it isn't needed: the NVIDIA provider is hosted.
3. Open `http://127.0.0.1:8790` in Chrome, press F11 (full screen) and **Ctrl+Shift+R** (fresh page). Close the legend. Leave SIM TRUTH off, AUTO-FOLLOW on and speed at 15×.
4. Press **Space** (pause), then drag the timeline to the start. Close other apps and notifications.
5. Try one Ask question once (any incident), so the first live answer isn't the slow one.

## The run

| Time | Do | Say (keep it short) |
|---|---|---|
| 0:00 | Whole border on screen, paused | "Small drones are hard for radar and cameras, but they are loud, and their sound is a set of evenly spaced tones. We detect and track them by ear." |
| 0:15 | Point at the header numbers and the amber line | "This is the UAE's 1,024 km land border. Every cyan dot is a cluster of 4 microphones on the ground, two rows, 3,395 in all. Everything you'll see is simulated, but run through our real detection system." |
| 0:35 | Click **LIWA-1** in the left list, press **Space** (play) | "A two-stroke drone comes in from the Saudi side." |
| 0:45 | Clusters flash green, amber bearing lines appear | "Each green cluster's neural network hears a drone. The amber lines are the direction each one hears it from." |
| 0:55 | Red diamond and ring; the track card on the left | "Where the directions cross is the drone: position, height and speed, and the ring is our uncertainty. It was heard about a kilometre before the border." |
| 1:10 | "NEAR THE BORDER", then "CROSSED THE BORDER" (banner, log) | "The alert rules are fixed code, not AI: near the border, then crossed, once per intruder." |
| 1:25 | **ZETARIS** button → **ALL SECTORS** | "Each sector's sensors keep their own database, the tracker keeps another, operations another. **Zetaris** joins them in place: this is one query over seven databases." |
| 1:40 | **FIRST TO HEAR**, open **THE ZETARIS QUERY** | "Which clusters heard it first, and from what bearing: the sector store joined with the tracker's store. Here's the actual query." |
| 1:55 | Close the drawer. Ask: **Units near it?** | "Operators ask in plain English. **Nemotron**, through **Meterless Nightshift**, answers from the Zetaris data." Point at "VIA METERLESS NIGHTSHIFT · SITE DATA ZETARIS". |
| 2:15 | Click **HATTA-1**, play for a few seconds, then point at its after-action line | "Honest limit: a quiet electric drone in the Hajar mountains was only heard at the border. The hills block the sound. That's our next problem: where to put the sensors." |
| 2:35 | **◂ THEATRE** (or Esc) | "Next: fewer split tracks, a real field test with our own drone, and running Nemotron fully on site. Thanks." |

**Optional live segment, "build a test" (about 3 minutes; start the run while you talk):**
1. **BUILD A TEST**, then **Desert border post · Liwa**. Say: "Each circle is how far one microphone cluster can hear this drone in quiet desert, about 860 m. Two clusters must hear it to fix its position, so the lines between them must stay green."
2. Click **▶ RUN SIMULATION** (about 2 minutes). While it runs: "It's simulating what each of the 7 microphones hears on the real terrain, then running our neural network and tracker on that sound."
3. When the replay starts, follow the captions. "1 microphone hears a drone" becomes "Position fixed from 2 microphones" and then the crossing alert.
4. Optionally, switch background noise to **Town**: the circles shrink to 170 m. "In a city you'd need clusters every couple of hundred metres. That's our next problem."

If time is short, skip HATTA-1. If there's spare time, show **SIM TRUTH** on Liwa: the white dashed line is where the drone really flew, and the red track follows it.

## Likely questions

| Question | Answer |
|---|---|
| Is this real? | The flights are simulated. The sound is rendered on real elevation data. Detection, tracking and alerts are our real code. On real recordings, the detector catches real drones out to about 100 m. |
| Why sound, not radar? | Small drones have a tiny radar signature and fly low. Microphones are cheap, passive (they emit nothing) and see behind obstacles a camera can't. |
| How do you know where it is? | Each 4-mic cluster measures a direction. Two or more directions cross at the drone. A Kalman filter keeps the estimate and its uncertainty. |
| What does each sponsor tool add? | **Zetaris** joins separate databases (each sector, the tracker, operations) in place, without copying. **Meterless Nightshift** routes the language model. **Nemotron** turns the system's numbers into answers for operators. |
| Does the AI decide the alerts? | No. Alerts come from fixed, tested rules. Nemotron only words answers, and every number it gives is checked against the data. |
| Why not put everything in one database? | Real deployments have separate systems per sector and per organisation. Zetaris queries them where they are, and that's the point. |
| What doesn't work yet? | One drone sometimes becomes several tracks. Quiet drones in mountains or town noise are heard late. On real audio the detector has false alarms on everyday sounds. No field test yet. |
| How much would it cost? | The design uses ESP32-based clusters (cheap MEMS mics). Full border coverage at this spacing is 3,395 clusters. Spacing and placement are a planning choice. |
| Why train only on simulation? | Labelled real drone audio is scarce. Simulation gives unlimited exact labels, and we test on real recordings to check it isn't just learning the simulator. |
| Who did what? | (Answer honestly.) The concept and decisions are yours. Claude wrote most of the software. |

## If something goes wrong live

| Problem | What to do |
|---|---|
| The page is blank | Press Ctrl+Shift+R. If it's still blank, check the internet (three.js loads from a CDN). |
| The Ask box is slow | It answers with system facts after 45 s at worst. Carry on talking. |
| The ZETARIS drawer errors | Skip it and describe it: "seven databases, one query". |
| You get lost in 3D | Press Esc (back to the whole border), then click the incident again. |
