@echo off
cd /d "C:\work\audiodrone"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
.venv\Scripts\python.exe -m sim.render --out data\sim_v1 --start 2134 --count 1866 --workers 9 --prefix p --relabel > data\render_p2.out 2>&1
