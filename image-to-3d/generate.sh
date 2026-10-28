#!/bin/zsh
# Image -> textured 3D model with TRELLIS.2 on Apple Silicon, plus a game-ready low-poly.
# Usage: ./generate.sh image.png [fast|best] [tris] [size_m]
#   fast = 512 pipeline (~2 min), best = 1024 cascade (~13 min)
#   tris = low-poly budget (default 10000); size_m = largest dimension in metres (default 0 = keep)
# Output: outputs/<name>-<mode>.glb (high-poly), game/<name>-<mode>/ (low-poly FBX/GLB + HDRP maps),
#         renders/<name>-<mode>{,_low}_grid.png previews.
set -e
cd "${0:A:h}"
img="${1:A}"; mode="${2:-fast}"; tris="${3:-10000}"; size="${4:-0}"
name="$(basename "${img%.*}")-$mode"
[[ $mode == best ]] && pipe=1024_cascade || pipe=512
.venv/bin/python generate.py "$img" --output "outputs/$name" --pipeline-type $pipe --texture-size 2048
blender -b -P lowpoly.py -- "outputs/$name.glb" "game/$name" --tris "$tris" --tex 2048 --size "$size" 2>&1 | grep "\[lowpoly"
blender -b -P render_views.py -- "outputs/$name.glb" "renders/$name" 2>&1 | grep GRID
blender -b -P render_views.py -- "game/$name/${name}_low.glb" "renders/${name}_low" 2>&1 | grep GRID
