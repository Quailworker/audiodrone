@echo off
cd /d "C:\Users\pc\Joseph AI\audiodrone"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
.venv\Scripts\python.exe -m sim.render --out data\sim_v1 --start 1000000 --count 2000 --workers 10 --prefix a --relabel > data\render_a.out 2>&1
.venv\Scripts\python.exe -m sim.render --out data\sim_v1 --start 0 --count 400 --workers 10 --heldout --prefix a --relabel > data\render_ah.out 2>&1
