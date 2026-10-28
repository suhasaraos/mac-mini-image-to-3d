"""Render 4 turntable views of a GLB with Blender (headless).

blender -b -P render_views.py -- model.glb out_prefix
Writes out_prefix_grid.png (2x2 grid: front, 3/4, side, back).
"""
import sys, math, bpy, mathutils

glb, out = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]

# Frame the model: bounding box of all meshes in world space.
pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
center, size = (lo + hi) / 2, max(hi - lo)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
    e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = 768
scene.render.film_transparent = False
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.8, 0.82, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0; sun.rotation_euler = (math.radians(50), 0, math.radians(30))
scene.collection.objects.link(sun)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam); scene.camera = cam
cam.data.lens = 50

frames = []
for name, az in [("front", 0), ("q34", 45), ("side", 90), ("back", 180)]:
    d = size * 2.2
    a = math.radians(az)
    # glTF imports Y-up as Z-up in Blender; orbit around Z.
    cam.location = center + mathutils.Vector((math.sin(a) * d, -math.cos(a) * d, size * 0.45))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    path = f"{out}_{name}.png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    frames.append(path)

# Stitch into a 2x2 grid.
imgs = [bpy.data.images.load(p) for p in frames]
w, h = imgs[0].size
grid = bpy.data.images.new("grid", w * 2, h * 2)
px = [0.0] * (w * 2 * h * 2 * 4)
for k, im in enumerate(imgs):
    src = list(im.pixels)
    ox, oy = (k % 2) * w, (1 - k // 2) * h
    for y in range(h):
        row = src[y * w * 4:(y + 1) * w * 4]
        start = ((oy + y) * w * 2 + ox) * 4
        px[start:start + w * 4] = row
grid.pixels = px
grid.filepath_raw = f"{out}_grid.png"; grid.file_format = "PNG"; grid.save()
print("GRID", f"{out}_grid.png")
