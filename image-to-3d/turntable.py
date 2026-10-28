"""Render a studio-lit turntable video of a GLB (for the video's hero shots).

blender -b -P turntable.py -- model.glb out.mp4 [--seconds 8] [--fps 60] [--hdri studio]
                                              [--wire] [--label "text"] [--samples 32]

Eevee, 1920x1080, HDRI lighting from Blender's bundled studio lights, a key light for
shaping, a shadow-catching floor and a soft gradient backdrop. --wire overlays the mesh
wireframe (to show triangle density).
"""
import sys, math, os, argparse, glob
import bpy, mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("glb"); ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=8)
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--hdri", default="studio")
ap.add_argument("--wire", action="store_true")
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--elev", type=float, default=18, help="camera elevation in degrees")
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.glb)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
center, size = (lo + hi) / 2, max(hi - lo)

# Spin the model on a pivot at its centre, sitting on the floor.
pivot = bpy.data.objects.new("pivot", None)
bpy.context.scene.collection.objects.link(pivot)
pivot.location = (center.x, center.y, lo.z)
bpy.context.view_layer.update()
for o in objs:
    if o.parent is None:
        o.parent = pivot
        o.matrix_parent_inverse = pivot.matrix_world.inverted()

scene = bpy.context.scene
engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
ee = scene.eevee
for attr, val in (("taa_render_samples", a.samples), ("use_shadows", True), ("use_raytracing", True)):
    if hasattr(ee, attr):
        setattr(ee, attr, val)
scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
scene.render.fps = a.fps
frames = int(a.seconds * a.fps)
scene.frame_start, scene.frame_end = 1, frames
scene.view_settings.view_transform = "AgX" if "AgX" in [v.identifier for v in bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
scene.view_settings.look = "AgX - Medium High Contrast" if scene.view_settings.view_transform == "AgX" else "Medium High Contrast"

# World: bundled HDRI for reflections and fill, kept out of the camera's view.
hdri = glob.glob(os.path.join(os.path.dirname(bpy.app.binary_path), "..", "Resources", "*", "datafiles",
                              "studiolights", "world", f"{a.hdri}.exr"))
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
nt = world.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputWorld")
mix = nt.nodes.new("ShaderNodeMixShader")
lp = nt.nodes.new("ShaderNodeLightPath")
env = nt.nodes.new("ShaderNodeTexEnvironment")
if hdri:
    env.image = bpy.data.images.load(hdri[0])
bg_light = nt.nodes.new("ShaderNodeBackground"); bg_light.inputs[1].default_value = 0.5
bg_cam = nt.nodes.new("ShaderNodeBackground"); bg_cam.inputs[0].default_value = (0.055, 0.06, 0.07, 1)
nt.links.new(env.outputs[0], bg_light.inputs[0])
nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
nt.links.new(bg_light.outputs[0], mix.inputs[1])
nt.links.new(bg_cam.outputs[0], mix.inputs[2])
nt.links.new(mix.outputs[0], out.inputs[0])

# Floor + curved backdrop in one plane bent up behind the model, dark and slightly glossy.
bpy.ops.mesh.primitive_plane_add(size=size * 80, location=(center.x, center.y, lo.z))
floor = bpy.context.active_object
fm = bpy.data.materials.new("floor"); fm.use_nodes = True
fb = fm.node_tree.nodes["Principled BSDF"]
fb.inputs["Base Color"].default_value = (0.018, 0.02, 0.025, 1)
fb.inputs["Roughness"].default_value = 0.45
floor.data.materials.append(fm)

def light(name, kind, energy, loc, size_=1.0, color=(1, 1, 1)):
    l = bpy.data.lights.new(name, kind); l.energy = energy; l.color = color
    if kind == "AREA": l.size = size_
    ob = bpy.data.objects.new(name, l); scene.collection.objects.link(ob)
    ob.location = center + mathutils.Vector(loc) * size
    ob.rotation_euler = (center - ob.location).to_track_quat("-Z", "Y").to_euler()
    return ob

light("key", "AREA", 120 * size * size, (-1.6, -1.8, 1.9), size_=size * 1.2, color=(1.0, 0.96, 0.9))
light("rim", "AREA", 200 * size * size, (1.4, 2.0, 1.4), size_=size * 0.8, color=(0.8, 0.88, 1.0))
light("fill", "AREA", 25 * size * size, (2.0, -1.2, 0.4), size_=size * 1.5)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam); scene.camera = cam
cam.data.lens = 50
d = size * 3.5; el = math.radians(a.elev)
cam.location = center + mathutils.Vector((0, -d * math.cos(el), d * math.sin(el)))
cam.rotation_euler = (center + mathutils.Vector((0, 0, -size * 0.04)) - cam.location).to_track_quat("-Z", "Y").to_euler()

if a.wire:
    for o in objs:
        m = o.modifiers.new("wire", "WIREFRAME"); m.thickness = size * 0.0007; m.use_replace = False
        wm = bpy.data.materials.new("wire"); wm.use_nodes = True
        wm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.75, 1.0, 1)
        wm.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.2, 0.75, 1.0, 1)
        wm.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.5
        o.data.materials.append(wm); m.material_offset = len(o.data.materials) - 1

pivot.rotation_euler = (0, 0, 0)
pivot.keyframe_insert("rotation_euler", index=2, frame=1)
pivot.rotation_euler = (0, 0, math.radians(360))
pivot.keyframe_insert("rotation_euler", index=2, frame=frames + 1)
for fc in pivot.animation_data.action.fcurves if hasattr(pivot.animation_data.action, "fcurves") else []:
    for k in fc.keyframe_points:
        k.interpolation = "LINEAR"
try:  # Blender 4.4+ layered actions
    for layer in pivot.animation_data.action.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                for fc in cb.fcurves:
                    for k in fc.keyframe_points:
                        k.interpolation = "LINEAR"
except AttributeError:
    pass

import tempfile, subprocess
tmp = tempfile.mkdtemp(prefix="turntable-")
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(tmp, "f_")
bpy.ops.render.render(animation=True)
subprocess.run(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i",
                os.path.join(tmp, "f_%04d.png"), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                a.out], check=True)
print("TURNTABLE", a.out, frames, "frames")
