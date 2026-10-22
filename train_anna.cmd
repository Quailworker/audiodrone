@echo off
rem Anna's PC (RTX 3080): the two CNN14 runs, B (log-frequency) then A (stock mel), each followed by evaluation.
cd /d "C:\Users\pc\Joseph AI\audiodrone"
set OMP_NUM_THREADS=2
nvidia-smi -pl 240
del runs\ANNA_DONE 2>nul
.venv\Scripts\python.exe -m cnn.train --data data\sim_v1 --out runs\B_logf --frontend logf --arch cnn14 --batch 16 --workers 3 > runs\B_logf.out 2>&1
.venv\Scripts\python.exe -m cnn.evaluate --data data\sim_v1 --split test_sim test_heldout --model runs\B_logf\best.pt --frontend logf --arch cnn14 > runs\B_logf.eval.out 2>&1
.venv\Scripts\python.exe -m cnn.train --data data\sim_v1 --out runs\A_mel --frontend mel --arch cnn14 --batch 16 --workers 3 > runs\A_mel.out 2>&1
.venv\Scripts\python.exe -m cnn.evaluate --data data\sim_v1 --split test_sim test_heldout --model runs\A_mel\best.pt --frontend mel --arch cnn14 > runs\A_mel.eval.out 2>&1
echo done > runs\ANNA_DONE
