@echo off
rem Phase 1 renders on Windows PC (RTX 3080), run as a scheduled task at below-normal priority.
rem Only the first line ran from here; the rest of the Windows PC's work was moved to pc2_chain_loc.ps1
rem (flat_ground_100 there, tetra_pole_150 on the Linux desktop).
cd /d "C:\work\audiodrone"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
.venv\Scripts\python.exe -m loc.render --config tetra_ground_100 --count 300 --workers 8 --out data\loc_v1 > data\render_loc_3.out 2>&1
