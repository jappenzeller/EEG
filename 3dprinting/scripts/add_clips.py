"""
Blender script to add snap-fit clips and retention lip to octabolt.

Uses boolean union for proper mesh connection.
"""

import bpy
import bmesh
import math
import os

# Configuration
SOURCE_STL = "h:/QuantumPython/EEG/3dprinting/M4_Octabolt_25p1mm_hex.stl"
OUTPUT_STL = "h:/QuantumPython/EEG/3dprinting/M4_Octabolt_25p1mm_clips.stl"

# Clip dimensions
N_CLIPS = 4
CLIP_Y_TOP = 23.5       # Attach at flange underside
CLIP_Y_BOTTOM = 21.0    # Clip tip
CLIP_RADIUS_INNER = 10.5 # Inner edge (shaft is ~9.7)
CLIP_RADIUS_OUTER = 12.0 # Outer edge of clip body
CLIP_WIDTH = 3.0        # Circumferential width

# Barb - angled entry ramp + catch
BARB_RAMP_Y = 1.5       # Length of angled ramp
BARB_CATCH_Y = 0.5      # Vertical catch
BARB_TIP_RADIUS = 13.3  # Outer edge of barb tip

# Retention lip - thin horizontal ring at top
RET_Y = 29.2            # At top of hex
RET_THICKNESS = 0.8     # Vertical thickness
RET_RADIUS_INNER = 12.0 # Start inside hex for overlap
RET_RADIUS_OUTER = 14.0 # Sticks out


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_stl(filepath):
    bpy.ops.import_mesh.stl(filepath=filepath)
    return bpy.context.selected_objects[0]


def create_snap_clip(index, n_clips):
    """Create snap clip with angled barb - solid piece from flange down."""
    angle = (2 * math.pi * index) / n_clips

    bm = bmesh.new()
    hw = CLIP_WIDTH / 2

    # Y positions
    y_top = CLIP_Y_TOP
    y_ramp_start = CLIP_Y_BOTTOM + BARB_RAMP_Y + BARB_CATCH_Y
    y_catch_start = CLIP_Y_BOTTOM + BARB_CATCH_Y
    y_bottom = CLIP_Y_BOTTOM

    r_inner = CLIP_RADIUS_INNER
    r_outer = CLIP_RADIUS_OUTER

    # Top section (connects to flange)
    v0 = bm.verts.new((r_inner, y_top, -hw))
    v1 = bm.verts.new((r_outer, y_top, -hw))
    v2 = bm.verts.new((r_outer, y_top, hw))
    v3 = bm.verts.new((r_inner, y_top, hw))

    # Where ramp starts
    v4 = bm.verts.new((r_inner, y_ramp_start, -hw))
    v5 = bm.verts.new((r_outer, y_ramp_start, -hw))
    v6 = bm.verts.new((r_outer, y_ramp_start, hw))
    v7 = bm.verts.new((r_inner, y_ramp_start, hw))

    # Ramp end / catch start (barb sticks out here)
    v8 = bm.verts.new((r_inner, y_catch_start, -hw))
    v9 = bm.verts.new((BARB_TIP_RADIUS, y_catch_start, -hw))
    v10 = bm.verts.new((BARB_TIP_RADIUS, y_catch_start, hw))
    v11 = bm.verts.new((r_inner, y_catch_start, hw))

    # Bottom (barb tip)
    v12 = bm.verts.new((r_inner, y_bottom, -hw))
    v13 = bm.verts.new((BARB_TIP_RADIUS, y_bottom, -hw))
    v14 = bm.verts.new((BARB_TIP_RADIUS, y_bottom, hw))
    v15 = bm.verts.new((r_inner, y_bottom, hw))

    # Faces
    bm.faces.new((v3, v2, v1, v0))  # top cap

    # Body sides
    bm.faces.new((v0, v1, v5, v4))
    bm.faces.new((v2, v3, v7, v6))
    bm.faces.new((v1, v2, v6, v5))
    bm.faces.new((v3, v0, v4, v7))

    # Ramp sides (angled outward)
    bm.faces.new((v4, v5, v9, v8))
    bm.faces.new((v6, v7, v11, v10))
    bm.faces.new((v5, v6, v10, v9))
    bm.faces.new((v7, v4, v8, v11))

    # Catch sides (vertical)
    bm.faces.new((v8, v9, v13, v12))
    bm.faces.new((v10, v11, v15, v14))
    bm.faces.new((v9, v10, v14, v13))
    bm.faces.new((v11, v8, v12, v15))

    # Bottom cap
    bm.faces.new((v12, v13, v14, v15))

    # Rotate around Y axis
    for v in bm.verts:
        x, y, z = v.co
        v.co.x = x * math.cos(angle) - z * math.sin(angle)
        v.co.z = x * math.sin(angle) + z * math.cos(angle)

    mesh = bpy.data.meshes.new(f"Clip_{index}")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(f"Clip_{index}", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_retention_lip():
    """Create horizontal retention lip ring at top of hex."""
    bm = bmesh.new()

    # Create a ring (annulus) - many segments for smooth circle
    segments = 32

    for i in range(segments):
        angle1 = (2 * math.pi * i) / segments
        angle2 = (2 * math.pi * (i + 1)) / segments

        # Inner and outer points at top and bottom of lip
        r_in = RET_RADIUS_INNER
        r_out = RET_RADIUS_OUTER
        y_top = RET_Y
        y_bot = RET_Y - RET_THICKNESS

        # 8 vertices for this segment
        v0 = bm.verts.new((r_in * math.cos(angle1), y_top, r_in * math.sin(angle1)))
        v1 = bm.verts.new((r_out * math.cos(angle1), y_top, r_out * math.sin(angle1)))
        v2 = bm.verts.new((r_out * math.cos(angle2), y_top, r_out * math.sin(angle2)))
        v3 = bm.verts.new((r_in * math.cos(angle2), y_top, r_in * math.sin(angle2)))

        v4 = bm.verts.new((r_in * math.cos(angle1), y_bot, r_in * math.sin(angle1)))
        v5 = bm.verts.new((r_out * math.cos(angle1), y_bot, r_out * math.sin(angle1)))
        v6 = bm.verts.new((r_out * math.cos(angle2), y_bot, r_out * math.sin(angle2)))
        v7 = bm.verts.new((r_in * math.cos(angle2), y_bot, r_in * math.sin(angle2)))

        # Faces
        bm.faces.new((v0, v1, v2, v3))  # top
        bm.faces.new((v7, v6, v5, v4))  # bottom
        bm.faces.new((v1, v5, v6, v2))  # outer
        bm.faces.new((v0, v3, v7, v4))  # inner

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)

    mesh = bpy.data.meshes.new("RetentionLip")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("RetentionLip", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def main():
    print("Adding clips + retention lip with boolean union...")
    print(f"Source: {SOURCE_STL}")

    clear_scene()
    octabolt = import_stl(SOURCE_STL)
    octabolt.name = "Octabolt"
    print(f"Imported: {len(octabolt.data.vertices)} vertices")

    # Create clips
    clips = []
    for i in range(N_CLIPS):
        clip = create_snap_clip(i, N_CLIPS)
        clips.append(clip)
        print(f"Clip {i+1}/{N_CLIPS}")

    # Create retention lip
    lip = create_retention_lip()
    print("Retention lip created")

    # Boolean union all parts
    bpy.context.view_layer.objects.active = octabolt

    for clip in clips:
        mod = octabolt.modifiers.new(name="Union", type='BOOLEAN')
        mod.operation = 'UNION'
        mod.object = clip
        mod.solver = 'EXACT'
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(clip)

    mod = octabolt.modifiers.new(name="Union", type='BOOLEAN')
    mod.operation = 'UNION'
    mod.object = lip
    mod.solver = 'EXACT'
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(lip)

    print(f"After boolean: {len(octabolt.data.vertices)} vertices")

    # Export
    bpy.ops.object.select_all(action='DESELECT')
    octabolt.select_set(True)
    bpy.ops.export_mesh.stl(filepath=OUTPUT_STL, use_selection=True)

    if os.path.exists(OUTPUT_STL):
        print(f"Success: {OUTPUT_STL} ({os.path.getsize(OUTPUT_STL):,} bytes)")
    else:
        print("ERROR: Export failed")


if __name__ == "__main__":
    main()
