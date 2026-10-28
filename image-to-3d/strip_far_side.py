"""Remove metal parts (e.g. buckles) that TRELLIS mirrored onto the hidden side of an object.

blender -b -P strip_far_side.py -- in.glb out.glb [--axis y] [--side +] [--margin 0.12] [--metal 0.5]

A single photo only shows one side, so TRELLIS guesses the other side and often mirrors
details onto it. This deletes faces whose metallic value is above --metal and whose centre
lies on the hidden half (--side of the --axis, beyond --margin x the object's depth from the
centre). The shell under the deleted parts already exists, so no holes appear.
"""
import sys, argparse
import bpy, bmesh
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--axis", default="y", choices="xyz")
ap.add_argument("--side", default="+", choices="+-")
ap.add_argument("--margin", type=float, default=0.12)
ap.add_argument("--metal", type=float, default=0.5)
ap.add_argument("--bright", type=float, default=0.3, help="metal parts must also be at least this bright")
ap.add_argument("--keep-below", type=float, default=0.15, help="never touch the bottom fraction (the sole)")
ap.add_argument("--keep-small", type=float, default=0.08, help="keep metal parts whose size is below this fraction of the object (rivets)")
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.src)
obj = next(o for o in bpy.context.scene.objects if o.type == "MESH")
me = obj.data

# Find the metallic texture: Principled "Metallic" <- Separate Color <- image (glTF stores it in B).
mat = me.materials[0]
bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
link = bsdf.inputs["Metallic"].links[0]
node, chan = link.from_node, link.from_socket.name
while node.type != "TEX_IMAGE":
    node = node.inputs[0].links[0].from_node
img = node.image
w, h = img.size
bc_node = bsdf.inputs["Base Color"].links[0].from_node
bc = np.empty(w * h * 4, dtype=np.float32); bc_node.image.pixels.foreach_get(bc)
bc = bc.reshape(bc_node.image.size[1], bc_node.image.size[0], 4)
px = np.empty(w * h * 4, dtype=np.float32); img.pixels.foreach_get(px)
px = px.reshape(h, w, 4)
ci = {"Red": 0, "Green": 1, "Blue": 2, "R": 0, "G": 1, "B": 2}.get(chan, 2)

ax = "xyz".index(a.axis)
mw = obj.matrix_world
bm = bmesh.new(); bm.from_mesh(me)
uv = bm.loops.layers.uv.active
cos = np.array([(mw @ v.co)[ax] for v in bm.verts])
lo, hi = cos.min(), cos.max(); mid = (lo + hi) / 2; depth = hi - lo
cut = mid + (a.margin * depth if a.side == "+" else -a.margin * depth)

zs = np.array([(mw @ v.co)[2] for v in bm.verts]); zmin = zs.min(); zcut = zmin + a.keep_below * (zs.max() - zmin)
def on_far_side(f):
    p = mw @ f.calc_center_median()
    return (p[ax] >= cut if a.side == "+" else p[ax] <= cut) and p[2] > zcut

def islands(faces):
    faces = set(faces); out = []
    while faces:
        st = [faces.pop()]; isl = []
        while st:
            g = st.pop(); isl.append(g)
            for e in g.edges:
                for k in e.link_faces:
                    if k in faces: faces.remove(k); st.append(k)
        out.append(isl)
    return out

doomed = []
for f in bm.faces:
    if not on_far_side(f):
        continue
    u = sum(l[uv].uv.x for l in f.loops) / len(f.loops)
    v = sum(l[uv].uv.y for l in f.loops) / len(f.loops)
    yy, xx = min(h - 1, int(v * h)) % h, min(w - 1, int(u * w)) % w
    m = px[yy, xx, ci]
    lum = bc[min(bc.shape[0] - 1, int(v * bc.shape[0])), min(bc.shape[1] - 1, int(u * bc.shape[1])), :3].mean()
    # TRELLIS often marks black plastic as metal too; real metal parts here are bright.
    if m > a.metal and lum > a.bright:
        doomed.append(f)
# Group metal faces into physical parts by touching voxels (the mesh itself is loose pieces,
# so mesh connectivity says nothing), then keep parts smaller than --keep-small (rivets).
size = max(obj.dimensions); vox = size / 80
cells = {}
for f in doomed:
    p = mw @ f.calc_center_median()
    cells.setdefault((int(p[0] // vox), int(p[1] // vox), int(p[2] // vox)), []).append(f)
seen, parts = set(), []
for c0 in cells:
    if c0 in seen: continue
    st, part = [c0], []
    seen.add(c0)
    while st:
        c = st.pop(); part.append(c)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy, c[2] + dz)
                    if n in cells and n not in seen:
                        seen.add(n); st.append(n)
    parts.append(part)
keep, nkept = set(), 0
for part in parts:
    ext = max(max(c[i] for c in part) - min(c[i] for c in part) + 1 for i in range(3)) * vox
    if ext < a.keep_small * size:
        nkept += 1
        for c in part: keep.update(cells[c])
doomed = [f for f in doomed if f not in keep]
bmesh.ops.delete(bm, geom=doomed, context="FACES")
print(f"[strip] {len(parts)} metal parts on that side, kept {nkept} small ones (rivets)")
loose = [v for v in bm.verts if not v.link_faces]
bmesh.ops.delete(bm, geom=loose, context="VERTS")
bm.to_mesh(me); bm.free()
print(f"[strip] removed {len(doomed)} metal faces on the {a.side}{a.axis} side "
      f"(cut at {cut:.3f}, object spans {lo:.3f}..{hi:.3f})")

bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=a.dst, use_selection=True, export_format="GLB")
