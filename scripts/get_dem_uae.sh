#!/usr/bin/env bash
# Download the Copernicus GLO-30 and GLO-90 DSM tiles covering the UAE (N22-N26, E051-E056) for terra/.
# Public data on AWS (no account). Tiles over open sea do not exist (404) and are skipped.
# Usage: DEM_DIR=/path/to/dem scripts/get_dem_uae.sh     (default: data/dem in the repo, the same default as terra/dem.py)
set -u
D=${DEM_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/dem}
mkdir -p "$D/glo30" "$D/glo90"
for res in 30 90; do
  c=$([ $res = 30 ] && echo 10 || echo 30)
  for lat in 22 23 24 25 26; do for lon in 51 52 53 54 55 56; do
    n=Copernicus_DSM_COG_${c}_N${lat}_00_E0${lon}_00_DEM; f="$D/glo$res/$n.tif"
    [ -s "$f" ] && continue
    code=$(curl -s -o "$f.part" -w "%{http_code}" "https://copernicus-dem-${res}m.s3.amazonaws.com/$n/$n.tif")
    if [ "$code" = 200 ]; then mv "$f.part" "$f"; echo "glo$res N$lat E$lon ok"; else rm -f "$f.part"; echo "glo$res N$lat E$lon $code"; fi
  done; done
done
