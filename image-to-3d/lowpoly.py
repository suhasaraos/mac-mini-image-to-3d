"""High-poly TRELLIS GLB -> game-ready low-poly with baked maps (Unity HDRP layout).

blender -b -P lowpoly.py -- in.glb out_dir [--tris 10000] [--tex 2048] [--size 0]

Writes to out_dir:
  <name>_low.fbx / <name>_low.glb   low-poly mesh (glb has the maps embedded, for preview)
  <name>_BaseColor.png              sRGB albedo
  <name>_Normal.png                 tangent-space normal, OpenGL (Y+), what Unity expects
  <name>_MaskMap.png                HDRP mask: R metallic, G AO, B detail mask (0), A smoothness
  <name>_Roughness.png, _Metallic.png, _AO.png  (individual maps, for other engines)
--size: if > 0, scale so the largest dimension is this many metres (e.g. 0.32 for a boot).
"""
import sys, math, os, argparse, time
import warnings; warnings.filterwarnings("ignore")
import bpy, bmesh, mathutils
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("glb"); ap.add_argument("out")
ap.add_argument("--tris", type=int, default=10000)
ap.add_argument("--tex", type=int, default=2048)
ap.add_argument("--size", type=float, default=0.0)
ap.add_argument("--extrude", type=float, default=0.006, help="bake ray start offset outside the low mesh, as a fraction of size")
ap.add_argument("--cull", action="store_true", help="experimental: drop inward-facing faces from the bake source")
ap.add_argument("--transfer", default="nearest", choices=["nearest", "ray"], help="how colour/metal/roughness are taken from the high mesh")
ap.add_argument("--smooth-side", default="", help="e.g. +y: flatten normals on that half (where mirrored parts were removed)")
ap.add_argument("--despeck", action="store_true", help="remove bright specks on dark surfaces")
ap.add_argument("--ray", type=float, default=0.04, help="bake ray reach, as a fraction of the object size")
ap.add_argument("--voxels", type=int, default=350, help="remesh resolution: voxels across the largest dimension")
ap.add_argument("--recreate", nargs="*", default=[],
                choices=["decimate-only", "no-solidify", "smart-uv", "diffuse-color", "no-normal-flip"],
                help="turn a fix back off, to reproduce an earlier failure (for the video)")
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)
name = os.path.splitext(os.path.basename(a.glb))[0]
T0 = time.time()
def log(*m): print(f"[lowpoly {time.time() - T0:6.1f}s]", *m, flush=True)

# ---------- 1. Import and clean the high-poly ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.glb)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
high = bpy.context.view_layer.objects.active
high.name = "high"
# Drop glTF parent empties but keep the world transform, then apply it.
if high.parent:
    mw = high.matrix_world.copy(); high.parent = None; high.matrix_world = mw
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

dims = high.dimensions
if a.size > 0:
    s = a.size / max(dims)
    high.scale = (s, s, s)
    bpy.ops.object.transform_apply(scale=True)
size = max(high.dimensions)

bm = bmesh.new(); bm.from_mesh(high.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=size * 1e-5)
loose = [v for v in bm.verts if not v.link_faces]
bmesh.ops.delete(bm, geom=loose, context="VERTS")
bm.to_mesh(high.data); bm.free()
high_tris = sum(len(p.vertices) - 2 for p in high.data.polygons)
log(f"high: {high_tris} tris, size {size:.3f}")

# ---------- 1b. Bake source: drop inward-facing twin faces ----------
# TRELLIS shells are double-sided sheets: an outward face with the real colour and an inward
# twin (often grey, the inside colour) at the same spot. Renders cull back faces so this is
# invisible, but the bake hits whichever twin comes first, giving grey patches. Keep a face
# only if some ray in its front hemisphere escapes the object.
def cull_inward(obj):
    from mathutils.bvhtree import BVHTree
    bm = bmesh.new(); bm.from_mesh(obj.data); bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    golden = math.pi * (3 - math.sqrt(5)); n = 32; dirs = []
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n; r = math.sqrt(1 - y * y); t = golden * i
        dirs.append(mathutils.Vector((math.cos(t) * r, y, math.sin(t) * r)))
    eps = size * 2e-4; drop = []
    for f in bm.faces:
        c = f.calc_center_median(); nrm = f.normal
        o = c + nrm * eps
        if not any(d.dot(nrm) > 0.05 and bvh.ray_cast(o, d)[0] is None for d in dirs):
            drop.append(f)
    bmesh.ops.delete(bm, geom=drop, context="FACES")
    bm.to_mesh(obj.data); bm.free()
    return len(drop)

# ---------- 2. Low-poly copy: two-pass collapse decimation ----------
low = high.copy(); low.data = high.data.copy(); low.name = "low"
bpy.context.scene.collection.objects.link(low)
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = low; low.select_set(True)
# TRELLIS meshes are thousands of open pieces, which collapse-decimation can't merge across.
# A voxel remesh rebuilds one watertight shell first; fine detail comes back via the normal bake.
# The voxel remesh needs closed volumes, so first give every open sheet a little thickness.
vox = size / a.voxels
R = set(a.recreate)
if R: log("RECREATING earlier failure(s):", ", ".join(sorted(R)))
if "decimate-only" in R:  # attempt 1: plain Decimate on the triangle soup
    def decimate_to(target):
        cur = sum(len(p.vertices) - 2 for p in low.data.polygons)
        m = low.modifiers.new("dec", "DECIMATE")
        m.decimate_type = "COLLAPSE"; m.ratio = target / cur; m.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=m.name)
    decimate_to(a.tris * 4); decimate_to(a.tris)
    log(f"decimate only: asked for {a.tris} tris, got {sum(len(p.vertices) - 2 for p in low.data.polygons)}")
m = low.modifiers.new("thick", "SOLIDIFY")
m.thickness = vox * 2.0; m.offset = 0.0; m.use_even_offset = False; m.use_quality_normals = False
if "no-solidify" in R or "decimate-only" in R:
    low.modifiers.remove(m)
else:
    bpy.ops.object.modifier_apply(modifier=m.name)
m = low.modifiers.new("remesh", "REMESH")
m.mode = "VOXEL"; m.voxel_size = vox; m.adaptivity = 0.0; m.use_smooth_shade = True
if "decimate-only" in R:
    low.modifiers.remove(m)
else:
    bpy.ops.object.modifier_apply(modifier=m.name)
# Drop tiny floating bits the remesh kept (stray debris from generation).
bm = bmesh.new(); bm.from_mesh(low.data); bm.faces.ensure_lookup_table()
seen, islands = set(), []
for f in bm.faces:
    if f.index in seen: continue
    isl, st = [], [f]
    while st:
        g = st.pop()
        if g.index in seen: continue
        seen.add(g.index); isl.append(g)
        st.extend(h for e in g.edges for h in e.link_faces if h.index not in seen)
    islands.append(isl)
total = len(bm.faces)
junk = [f for isl in islands if len(isl) < total * 0.002 for f in isl] if "decimate-only" not in R else []
bmesh.ops.delete(bm, geom=junk, context="FACES")
bm.to_mesh(bm_mesh := low.data); bm.free()
log(f"remesh: {total} faces, {len(islands)} islands, removed {len(junk)} debris faces")

def decimate_to(target):
    cur = sum(len(p.vertices) - 2 for p in low.data.polygons)
    if cur <= target: return
    m = low.modifiers.new("dec", "DECIMATE")
    m.decimate_type = "COLLAPSE"; m.ratio = target / cur; m.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=m.name)

if "decimate-only" not in R:
  decimate_to(a.tris * 6)

# Delete faces nobody can see: the thickening above leaves inner walls inside every sheet.
# A face is hidden if rays in every direction from it hit the mesh again.
from mathutils.bvhtree import BVHTree
bm = bmesh.new(); bm.from_mesh(low.data); bm.faces.ensure_lookup_table()
bvh = BVHTree.FromBMesh(bm)
n_dirs = 48
golden = math.pi * (3 - math.sqrt(5))
dirs = []
for i in range(n_dirs):  # evenly spread directions on a sphere
    y = 1 - 2 * (i + 0.5) / n_dirs; r = math.sqrt(1 - y * y); t = golden * i
    dirs.append(mathutils.Vector((math.cos(t) * r, y, math.sin(t) * r)))
eps = size * 1e-4
hidden = []
for f in bm.faces:
    c = f.calc_center_median(); nrm = f.normal
    seen = False
    for d in dirs:
        o = c + d * eps + (nrm if d.dot(nrm) > 0 else -nrm) * eps
        if bvh.ray_cast(o, d)[0] is None:
            seen = True; break
    if not seen and "decimate-only" not in R: hidden.append(f)
bmesh.ops.delete(bm, geom=hidden, context="FACES")
loose = [v for v in bm.verts if not v.link_faces]
bmesh.ops.delete(bm, geom=loose, context="VERTS")
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)   # all outward, so bake rays go the right way
bm.to_mesh(low.data); bm.free()
log(f"removed {len(hidden)} hidden faces")

decimate_to(a.tris * 2)
decimate_to(a.tris)

# Remove crumbs left after hidden-face removal (pieces of a few triangles).
bm = bmesh.new(); bm.from_mesh(low.data); bm.faces.ensure_lookup_table()
seen, crumbs = set(), []
for f in bm.faces:
    if f.index in seen: continue
    isl, st = [], [f]
    while st:
        g = st.pop()
        if g.index in seen: continue
        seen.add(g.index); isl.append(g)
        st.extend(h for e in g.edges for h in e.link_faces if h.index not in seen)
    if len(isl) <= 20 and "decimate-only" not in R: crumbs += isl
bmesh.ops.delete(bm, geom=crumbs, context="FACES")
bm.to_mesh(low.data); bm.free()
log(f"removed {len(crumbs)} crumb faces")
low_tris = sum(len(p.vertices) - 2 for p in low.data.polygons)
log(f"low: {low_tris} tris")

# Smooth shading with hard edges above 60 degrees (the normal map handles the rest).
bpy.ops.object.shade_smooth()
try:
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(60))
except Exception:
    pass

if a.cull:
    log(f"bake source: dropped {cull_inward(high)} inward-facing faces")

# ---------- 3. Fresh UVs ----------
while low.data.uv_layers: low.data.uv_layers.remove(low.data.uv_layers[0])
uvl = low.data.uv_layers.new(name="UVMap")
# xatlas makes far fewer, larger charts than Smart UV Project on a decimated mesh.
import subprocess, tempfile
XPY = os.environ.get("XATLAS_PYTHON", "python3")   # any Python with xatlas + numpy (pip install xatlas numpy)
tmp = tempfile.mkdtemp()
me = low.data
verts = np.array([v.co[:] for v in me.vertices], dtype=np.float32)
faces = np.array([p.vertices[:] for p in me.polygons], dtype=np.int64)
np.savez(os.path.join(tmp, "in.npz"), verts=verts, faces=faces)
r = subprocess.run([XPY, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "xatlas_uv.py"),
                    os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz"), str(a.tex)],
                   capture_output=True, text=True)
if "smart-uv" in R:
    r.returncode = 1; r.stderr = "(recreating the original Smart UV Project unwrap)"
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.pack_islands(margin=0.004, rotate=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    log("smart UV project:", len({tuple(l.uv) for l in uvl.data}), "unique UV points")
elif r.returncode == 0:
    log(r.stdout.strip())
    cu = np.load(os.path.join(tmp, "out.npz"))["corner_uvs"]       # (faces, 3, 2), same face order
    uvl.data.foreach_set("uv", cu.reshape(-1))
else:
    log("xatlas failed, falling back to Smart UV Project:", r.stderr[-300:])
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(75), island_margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")

# ---------- 4. Bake setup ----------
scene = bpy.context.scene
scene.render.engine = "CYCLES"
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"; prefs.get_devices()
    for d in prefs.devices: d.use = True
    scene.cycles.device = "GPU"
except Exception as e:
    log("GPU setup failed, using CPU:", e)
scene.cycles.samples = 16
bake = scene.render.bake
bake.use_selected_to_active = True
bake.use_cage = False
bake.cage_extrusion = size * a.extrude      # rays start this far outside the low mesh...
bake.max_ray_distance = size * a.ray     # ...and look this far for the high surface
bake.margin = max(8, a.tex // 128)

low.data.materials.clear()
bmat = bpy.data.materials.new("bake"); bmat.use_nodes = True
low.data.materials.append(bmat)
nodes = bmat.node_tree.nodes
img_node = nodes.new("ShaderNodeTexImage"); nodes.active = img_node

def new_image(tag, color):
    im = bpy.data.images.new(f"{name}_{tag}", a.tex, a.tex, alpha=False, float_buffer=False)
    im.colorspace_settings.name = "sRGB" if color else "Non-Color"
    return im

def do_bake(tag, btype, color=False, samples=16, **kw):
    im = new_image(tag, color)
    img_node.image = im
    scene.cycles.samples = samples
    bpy.ops.object.select_all(action="DESELECT")
    high.select_set(True); low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.bake(type=btype, use_selected_to_active=True, cage_extrusion=bake.cage_extrusion,
                        max_ray_distance=bake.max_ray_distance, margin=bake.margin, **kw)
    log("baked", tag)
    return im

maps = {}
maps["Normal"] = do_bake("Normal", "NORMAL", normal_space="TANGENT")
# Some TRELLIS sheets are wound backwards, so the bake records their normal pointing INTO the
# surface (tangent-space z < 0). Lit, those spots act like mirrors and show as grey patches.
# A real surface normal can't face inward, so flip those back out.
_n = np.empty(a.tex * a.tex * 4, dtype=np.float32); maps["Normal"].pixels.foreach_get(_n); _n = _n.reshape(-1, 4)
_v = _n[:, :3] * 2 - 1
_inward = (_v[:, 2] < 0) if "no-normal-flip" not in R else np.zeros(len(_v), bool)
_v[_inward] *= -1
_n[:, :3] = (_v + 1) / 2
maps["Normal"].pixels.foreach_set(_n.ravel())
log(f"normal map: flipped {_inward.mean() * 100:.1f}% inward-facing texels")
maps["AO"] = do_bake("AO", "AO", samples=64)

# Roughness and metallic: route each Principled input into an emission shader on the high mesh, bake EMIT.
def route_to_emission(input_name):
    for mat in high.data.materials:
        if not mat or not mat.use_nodes: continue
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if not bsdf or not out: continue
        em = nt.nodes.get("_bake_emit") or nt.nodes.new("ShaderNodeEmission")
        em.name = "_bake_emit"
        for l in list(em.inputs["Color"].links): nt.links.remove(l)
        src = bsdf.inputs[input_name]
        if src.is_linked:
            nt.links.new(src.links[0].from_socket, em.inputs["Color"])
        else:
            v = src.default_value
            em.inputs["Color"].default_value = tuple(v) if hasattr(v, "__len__") else (v, v, v, 1)
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])

# Base colour via emission too: Cycles' diffuse-colour pass leaves metal black (diffuse is 0 on metal).
if "diffuse-color" in R:
    maps["BaseColor"] = do_bake("BaseColor", "DIFFUSE", color=True, pass_filter={"COLOR"})
else:
    route_to_emission("Base Color"); maps["BaseColor"] = do_bake("BaseColor", "EMIT", color=True)
route_to_emission("Roughness"); maps["Roughness"] = do_bake("Roughness", "EMIT")
route_to_emission("Metallic");  maps["Metallic"] = do_bake("Metallic", "EMIT")

# Colour, metal and roughness by nearest-surface transfer: for every texel of the low mesh, take
# the closest point on the high mesh and copy its texture values there. Ray baking can land on
# the wrong layer of TRELLIS's loose, overlapping sheets and leave grey patches; nearest-point
# lookup has no ray direction to get wrong. The ray-baked values stay only in the margin texels.
if a.transfer == "nearest" and "diffuse-color" not in R:
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import barycentric_transform
    pos_im = bpy.data.images.new(f"{name}_pos", a.tex, a.tex, alpha=True, float_buffer=True)
    pos_im.pixels.foreach_set(np.full(a.tex * a.tex * 4, -1e6, dtype=np.float32))
    img_node.image = pos_im
    bpy.ops.object.select_all(action="DESELECT"); low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.bake(type="POSITION", use_selected_to_active=False, margin=0)
    P = np.empty(a.tex * a.tex * 4, dtype=np.float32); pos_im.pixels.foreach_get(P); P = P.reshape(-1, 4)
    covered = np.nonzero((P[:, 0] > -1e5) & (np.abs(P[:, :3]).sum(1) > 0))[0]
    # The POSITION pass comes back uniformly scaled on this Blender build; map it onto the
    # low mesh's real bounding box so lookups land on the right spot.
    bb = np.array([low.matrix_world @ mathutils.Vector(c) for c in low.bound_box])
    pmin, pmax = P[covered, :3].min(0), P[covered, :3].max(0)
    P[covered, :3] = (P[covered, :3] - pmin) / (pmax - pmin) * (bb.max(0) - bb.min(0)) + bb.min(0)
    log(f"position pass rescaled by {((pmax - pmin) / (bb.max(0) - bb.min(0))).round(2)}")

    hbm = bmesh.new(); hbm.from_mesh(high.data)
    bmesh.ops.triangulate(hbm, faces=hbm.faces[:])
    hbm.faces.ensure_lookup_table()
    huv = hbm.loops.layers.uv.active
    tri_co = [[l.vert.co.copy() for l in f.loops] for f in hbm.faces]
    tri_uv = [[l[huv].uv.to_3d() for l in f.loops] for f in hbm.faces]
    hbvh = BVHTree.FromBMesh(hbm)

    hm = high.data.materials[0].node_tree
    hb = next(n for n in hm.nodes if n.type == "BSDF_PRINCIPLED")
    def src_image(sock):
        n = hb.inputs[sock].links[0].from_node
        while n.type != "TEX_IMAGE": n = n.inputs[0].links[0].from_node
        im = n.image; w, h = im.size
        arr = np.empty(w * h * 4, dtype=np.float32); im.pixels.foreach_get(arr)
        return arr.reshape(h, w, 4)
    src_bc, src_mr = src_image("Base Color"), src_image("Metallic")
    sh, sw = src_bc.shape[:2]; mh, mw_ = src_mr.shape[:2]

    out_bc = np.empty(a.tex * a.tex * 4, dtype=np.float32); maps["BaseColor"].pixels.foreach_get(out_bc); out_bc = out_bc.reshape(-1, 4)
    out_m = np.empty_like(out_bc); maps["Metallic"].pixels.foreach_get(out_m.ravel()); out_m = out_m.reshape(-1, 4)
    out_r = np.empty_like(out_bc); maps["Roughness"].pixels.foreach_get(out_r.ravel()); out_r = out_r.reshape(-1, 4)
    V = mathutils.Vector
    for k in covered:
        p = V(P[k, :3])
        loc, _, fi, _ = hbvh.find_nearest(p)
        if fi is None: continue
        c, t = tri_co[fi], tri_uv[fi]
        uv = barycentric_transform(loc, c[0], c[1], c[2], t[0], t[1], t[2])
        u, v = uv.x % 1.0, uv.y % 1.0
        out_bc[k, :3] = src_bc[min(sh - 1, int(v * sh)), min(sw - 1, int(u * sw)), :3]
        mr = src_mr[min(mh - 1, int(v * mh)), min(mw_ - 1, int(u * mw_))]
        out_m[k, :3] = mr[2]; out_r[k, :3] = mr[1]
    maps["BaseColor"].pixels.foreach_set(out_bc.ravel())
    maps["Metallic"].pixels.foreach_set(out_m.ravel())
    maps["Roughness"].pixels.foreach_set(out_r.ravel())
    hbm.free()
    log(f"nearest-surface transfer: {len(covered)} texels")

# ---------- 5. Save maps + pack the HDRP mask map ----------
# TRELLIS often predicts dark plastic as metal (a black metal shell looks fine on its own but
# turns into flat grey patches once lit). Only bright surfaces are allowed to stay metallic.
def _px(im):
    arr = np.empty(a.tex * a.tex * 4, dtype=np.float32); im.pixels.foreach_get(arr)
    return arr.reshape(a.tex, a.tex, 4)
_bcpx = _px(maps["BaseColor"]); _bc = _bcpx[..., :3].mean(-1)
_mt = _px(maps["Metallic"])
_gate = np.clip((_bc - 0.35) / 0.15, 0, 1)        # only bright surfaces may stay metal
_fake = _mt[..., 0] * (1 - _gate)                  # how much "dark metal" each texel had
# A grey surface marked as metal renders nearly black, so keep that look as dark plastic.
_bcpx[..., :3] *= (1 - 0.8 * _fake)[..., None]
maps["BaseColor"].pixels.foreach_set(_bcpx.ravel())
_mt[..., :3] *= _gate[..., None]
maps["Metallic"].pixels.foreach_set(_mt.ravel())
log(f"metal limited to bright areas: {(_gate < 1).mean() * 100:.0f}% of texels capped")

def _box(img, r):
    """Box blur (2r+1)^2 of an HxW or HxWxC array, via summed-area tables."""
    pad = np.pad(img, [(r + 1, r)] * 2 + [(0, 0)] * (img.ndim - 2), mode="edge").astype(np.float64)
    c = pad.cumsum(0).cumsum(1)
    k = 2 * r + 1
    out = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (out / (k * k)).astype(np.float32)

if a.despeck:
    _bc = _px(maps["BaseColor"]); lum = _bc[..., :3].mean(-1)
    local = _box(lum, 3)
    speck = (lum - local > 0.05) & (local < 0.3)
    smooth = _box(_bc[..., :3], 6)
    _bc[speck, :3] = smooth[speck]
    maps["BaseColor"].pixels.foreach_set(_bc.ravel())
    log(f"despeck: replaced {speck.mean() * 100:.1f}% of texels")

if a.smooth_side and a.transfer == "nearest" and "diffuse-color" not in R:
    ax = "xyz".index(a.smooth_side[1]); sign = 1 if a.smooth_side[0] == "+" else -1
    bbx = np.array([low.matrix_world @ mathutils.Vector(c) for c in low.bound_box])
    mid = (bbx[:, ax].max() + bbx[:, ax].min()) / 2; depth = bbx[:, ax].max() - bbx[:, ax].min()
    zmin = bbx[:, 2].min(); zcut = zmin + 0.15 * (bbx[:, 2].max() - zmin)
    PP = P.reshape(a.tex, a.tex, 4)
    d = (PP[..., ax] - mid) * sign / depth                  # 0 at the middle, 0.5 at the far edge
    w = np.clip((d - 0.05) / 0.15, 0, 1) * (PP[..., 2] > zcut) * (np.abs(PP[..., :3]).sum(-1) > 0)
    w = _box(w.astype(np.float32), 4) * 0.85                  # soft edge, keep a little detail
    nm = _px(maps["Normal"]); v = nm[..., :3] * 2 - 1
    v = v * (1 - w[..., None]) + np.array([0, 0, 1], np.float32) * w[..., None]
    v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6
    nm[..., :3] = (v + 1) / 2
    maps["Normal"].pixels.foreach_set(nm.ravel())
    log(f"smoothed normals on the {a.smooth_side} side ({(w > 0.1).mean() * 100:.0f}% of texels)")
def px(im):
    arr = np.empty(a.tex * a.tex * 4, dtype=np.float32); im.pixels.foreach_get(arr)
    return arr.reshape(a.tex, a.tex, 4)

for tag, im in maps.items():
    im.filepath_raw = os.path.join(a.out, f"{name}_{tag}.png"); im.file_format = "PNG"; im.save()

mask = np.zeros((a.tex, a.tex, 4), dtype=np.float32)
mask[..., 0] = px(maps["Metallic"])[..., 0]
mask[..., 1] = px(maps["AO"])[..., 0]
mask[..., 2] = 0.0                                   # detail mask: unused
mask[..., 3] = 1.0 - px(maps["Roughness"])[..., 0]   # smoothness
mim = bpy.data.images.new(f"{name}_MaskMap", a.tex, a.tex, alpha=True)
mim.colorspace_settings.name = "Non-Color"
mim.pixels.foreach_set(mask.ravel())
mim.filepath_raw = os.path.join(a.out, f"{name}_MaskMap.png"); mim.file_format = "PNG"; mim.save()

# ---------- 6. Final material on the low-poly, then export ----------
nt = bmat.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial"); bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
def tex(tag):
    n = nt.nodes.new("ShaderNodeTexImage"); n.image = maps[tag]; return n
nt.links.new(tex("BaseColor").outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(tex("Roughness").outputs["Color"], bsdf.inputs["Roughness"])
nt.links.new(tex("Metallic").outputs["Color"], bsdf.inputs["Metallic"])
nm = nt.nodes.new("ShaderNodeNormalMap")
nt.links.new(tex("Normal").outputs["Color"], nm.inputs["Color"])
nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
bmat.name = f"{name}_mat"

bpy.data.objects.remove(high)
bpy.ops.object.select_all(action="DESELECT"); low.select_set(True)
bpy.context.view_layer.objects.active = low
low.name = f"{name}_low"
bpy.ops.export_scene.gltf(filepath=os.path.join(a.out, f"{name}_low.glb"), use_selection=True,
                          export_format="GLB")
bpy.ops.export_scene.fbx(filepath=os.path.join(a.out, f"{name}_low.fbx"), use_selection=True,
                         apply_scale_options="FBX_SCALE_UNITS", path_mode="RELATIVE",
                         mesh_smooth_type="FACE", use_tspace=True)
log(f"done: {low_tris} tris, maps {a.tex}px -> {a.out}")
