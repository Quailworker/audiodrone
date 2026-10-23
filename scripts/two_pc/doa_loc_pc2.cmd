@echo off
rem Direction finding for tetra_ground_100 on the Windows PC's GPU (main methods only). flat_ground_100 ran from
rem pc2_chain_loc.ps1; tetra_pole_150 ran on the Linux desktop.
cd /d "C:\work\audiodrone"
.venv\Scripts\python.exe -m loc.run_doa --config tetra_ground_100 --data data\loc_v1 --wait-for 300 > data\doa_loc_3.out 2>&1
