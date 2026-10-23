@echo off
rem Nemotron 3.5 Lightning 30B-A3B on Nightshift's bundled llama.cpp server (OpenAI-compatible API),
rem on Anna's PC. Localhost only, API key required; the Linux desktop reaches it through an SSH tunnel.
rem MoE expert weights stay in RAM (--cpu-moe) so only a few GB of VRAM are used on this shared PC.
set M=C:\Users\pc\Joseph AI\models\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
set K=C:\Users\pc\Joseph AI\llama-server\api.key
set L=C:\Users\pc\Joseph AI\llama-server\llama-server.log
"C:\Program Files\Nightshift\llama\llama-server.exe" -m "%M%" -a nemotron-3.5-lightning --host 127.0.0.1 --port 18080 --api-key-file "%K%" -ngl 99 --cpu-moe -c 16384 -fa auto --jinja --log-file "%L%" > "C:\Users\pc\Joseph AI\llama-server\stdout.log" 2>&1
