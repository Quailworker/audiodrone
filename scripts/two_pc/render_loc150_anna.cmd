@echo off
rem tetra_pole_150 split: the Linux desktop renders drone scenes 0-199, Anna 200-299 (then their DOA here).
cd /d "C:\Users\pc\Joseph AI\audiodrone"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
.venv\Scripts\python.exe -m loc.render --config tetra_pole_150 --skip 200 --count 100 --workers 6 --out data\loc_v1 > data\render_loc_4.out 2>&1
.venv\Scripts\python.exe -m loc.run_doa --config tetra_pole_150 --data data\loc_v1 > data\doa_loc_4.out 2>&1
