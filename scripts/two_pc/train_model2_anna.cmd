@echo off
rem Anna's PC (RTX 3080): model 2 = CNN14 log-f plus the two frequency-tracker channels (frontend logf_comb),
rem same data, steps and batch as model 1 (B_logf), then evaluation on the same test splits.
rem Copied to the audiodrone root on Anna's PC and run there as a hidden scheduled task at below-normal priority, 3 dataloader workers, GPU capped at 240 W.
cd /d "C:\Users\pc\Joseph AI\audiodrone"
set OMP_NUM_THREADS=2
nvidia-smi -pl 240
del runs\M2_DONE 2>nul
.venv\Scripts\python.exe -m cnn.train --data data\sim_v1 --out runs\M2_logf_comb --frontend logf_comb --arch cnn14 --batch 16 --workers 3 > runs\M2_logf_comb.out 2>&1
.venv\Scripts\python.exe -m cnn.evaluate --data data\sim_v1 --split test_sim test_heldout --model runs\M2_logf_comb\best.pt --frontend logf_comb --arch cnn14 > runs\M2_logf_comb.eval.out 2>&1
echo done > runs\M2_DONE
