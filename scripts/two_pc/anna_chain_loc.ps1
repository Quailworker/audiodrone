# After the running tetra_ground_100 render and DOA finish: render flat_ground_100 and run its DOA
# (variants on 100 scenes, main methods on the rest, then the oracle comb experiment on 100).
$root = "C:\Users\pc\Joseph AI\audiodrone"
Set-Location $root
$env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
function Wait-For($pattern) {
  while (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like $pattern }) { Start-Sleep 30 }
}
Wait-For "*loc.render --config tetra_ground_100*"
$doa = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m loc.run_doa --config flat_ground_100 --data data\loc_v1 --variants --limit 100 --wait-for 100" -RedirectStandardOutput "data\doa_loc_2a.out" -RedirectStandardError "data\doa_loc_2a.err" -PassThru -NoNewWindow
& .venv\Scripts\python.exe -m loc.render --config flat_ground_100 --count 300 --workers 8 --out data\loc_v1 *> data\render_loc_2.out
$doa.WaitForExit()
Wait-For "*loc.run_doa --config tetra_ground_100*"
& .venv\Scripts\python.exe -m loc.run_doa --config flat_ground_100 --data data\loc_v1 --wait-for 300 *> data\doa_loc_2b.out
& .venv\Scripts\python.exe -m loc.run_doa --config flat_ground_100 --data data\loc_v1 --comb --limit 100 *> data\doa_loc_2c.out
"CHAIN DONE" | Out-File data\anna_chain.done
