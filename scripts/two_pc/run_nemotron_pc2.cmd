@echo off
rem Nemotron 3.5 Lightning 30B-A3B on Nightshift's bundled llama.cpp server (OpenAI-compatible API),
rem on the Windows PC. Localhost only, API key required; the Linux desktop reaches it through an SSH tunnel.
rem MoE expert weights stay in RAM (--cpu-moe) so only a few GB of VRAM are used on this shared PC.
set M=C:\work\models\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
set K=C:\work\llama-server\api.key
set L=C:\work\llama-server\llama-server.log
"C:\Program Files\Nightshift\llama\llama-server.exe" -m "%M%" -a nemotron-3.5-lightning --host 127.0.0.1 --port 18080 --api-key-file "%K%" -ngl 99 --cpu-moe -c 16384 -fa auto --jinja --log-file "%L%" > "C:\work\llama-server\stdout.log" 2>&1
