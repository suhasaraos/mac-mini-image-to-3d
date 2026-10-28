"""Helper run outside Blender (needs xatlas): in.npz (verts, faces) -> out.npz (uvs per face corner).

python xatlas_uv.py in.npz out.npz [resolution]
"""
import sys
import numpy as np
import xatlas

src, dst = sys.argv[1], sys.argv[2]
res = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
d = np.load(src)
atlas = xatlas.Atlas()
atlas.add_mesh(d["verts"].astype(np.float32), d["faces"].astype(np.uint32))
co = xatlas.ChartOptions()
co.max_iterations = 4
co.max_cost = 16.0                 # allow more stretch per chart -> fewer, bigger charts (default 2)
co.normal_deviation_weight = 1.0   # default 2: tolerate curvier charts
co.roundness_weight = 0.05
po = xatlas.PackOptions()
po.resolution = res
po.padding = max(4, res // 256)   # texels between charts so mip-maps don't bleed
po.bilinear = True
po.blockAlign = True
atlas.generate(co, po)
vmap, idx, uvs = atlas[0]
# Same face order as the input: corner k of face f gets uvs[idx[f, k]].
np.savez(dst, corner_uvs=uvs[idx].astype(np.float32), charts=np.int32(atlas.chart_count))
print(f"xatlas: {atlas.chart_count} charts, {float(np.atleast_1d(atlas.utilization)[0]) * 100:.0f}% texture used")
