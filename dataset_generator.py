import bpy
import os
import random
import OpenEXR
import Imath
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys
import argparse

cwd = os.getcwd()
time_string = datetime.now().strftime('%Y%m%d%H%M%S')

def init_cycles():
    """Initialize Blender Cycles settings"""
    bpy.context.scene.render.engine = 'CYCLES'

    bpy.context.scene.cycles.use_denoising = False

    bpy.context.view_layer.use_pass_vector = True
    bpy.context.view_layer.cycles.denoising_store_passes = True

    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.cycles.tile_x = 256
    bpy.context.scene.cycles.tile_y = 256
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.cycles.use_motion_blur = False
    bpy.context.scene.camera.data.dof.use_dof = False
    bpy.context.scene.camera.data.clip_start = 0.1
    bpy.context.scene.camera.data.clip_end = 1000.0

    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.feature_set = 'SUPPORTED'

def init_compositing():
    """Sets up the compositor to output RGB, Depth, and Normals to separate EXR files"""
    scene = bpy.context.scene
    scene.cycles.film_transparent = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    cwd = os.getcwd()

    # Render Layers
    rlayers = tree.nodes.new(type='CompositorNodeRLayers')

    # RGB
    rgb_fileout = tree.nodes.new(type="CompositorNodeOutputFile")
    rgb_fileout.format.file_format = 'OPEN_EXR'
    rgb_fileout.format.color_mode = 'RGB'
    rgb_fileout.format.color_depth = '16'
    rgb_fileout.base_path = os.path.join(cwd, 'dataset', 'train', 'rgb')

    # Depth
    depth_fileout = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_fileout.format.file_format = 'OPEN_EXR'
    depth_fileout.format.color_mode = 'RGB'
    depth_fileout.format.color_depth = '16'
    depth_fileout.base_path = os.path.join(cwd, 'dataset', 'train', 'linear_depth')

    # Normals  
    normals_fileout = tree.nodes.new(type='CompositorNodeOutputFile')
    normals_fileout.format.file_format = 'OPEN_EXR'
    normals_fileout.format.color_mode = 'RGB'
    normals_fileout.format.color_depth = '16'
    normals_fileout.base_path = os.path.join(cwd, 'dataset', 'test', 'hdr_normal')

    # Albedo
    albedo_fileout = tree.nodes.new(type='CompositorNodeOutputFile')
    albedo_fileout.format.file_format = 'OPEN_EXR'
    albedo_fileout.format.color_mode = 'RGB'
    albedo_fileout.format.color_depth = '16'
    albedo_fileout.base_path = os.path.join(cwd, 'dataset', 'train', 'albedo')

    # Motion Vectors
    motion_fileout = tree.nodes.new(type="CompositorNodeOutputFile")
    motion_fileout.format.file_format = 'OPEN_EXR'
    motion_fileout.format.color_mode = 'RGBA'
    motion_fileout.format.color_depth = '16'
    motion_fileout.base_path = os.path.join(cwd, 'dataset', 'test', 'motion')

    # Link nodes
    tree.links.new(rlayers.outputs['Denoising Depth'], depth_fileout.inputs[0])
    tree.links.new(rlayers.outputs['Denoising Normal'], normals_fileout.inputs[0])
    tree.links.new(rlayers.outputs['Denoising Albedo'], albedo_fileout.inputs[0])
    tree.links.new(rlayers.outputs['Vector'], motion_fileout.inputs[0])    

    mute_aux(depth_fileout, normals_fileout, albedo_fileout, motion_fileout)

    return rgb_fileout, depth_fileout, normals_fileout, albedo_fileout, motion_fileout

def mute_aux(depth_fileout, normals_fileout, albedo_fileout, motion_fileout):
    depth_fileout.mute = True
    normals_fileout.mute = True
    albedo_fileout.mute = True
    motion_fileout.mute = True

def unmute_aux(depth_fileout, normals_fileout, albedo_fileout, motion_fileout):
    depth_fileout.mute = False
    normals_fileout.mute = False
    albedo_fileout.mute = False
    motion_fileout.mute = False

def render_gt(frame, blend_filename, rgb_fileout, depth_fileout, normals_fileout, albedo_fileout, motion_fileout, samples):
    bpy.context.scene.cycles.samples = samples

    unmute_aux(depth_fileout, normals_fileout, albedo_fileout, motion_fileout)

    rgb_fileout.file_slots[0].path = f"rgb_{blend_filename}_#####_gt"
    depth_fileout.file_slots[0].path = f"depth_{blend_filename}_#####_gt"
    normals_fileout.file_slots[0].path = f"normal_{blend_filename}_#####_gt"
    albedo_fileout.file_slots[0].path = f"albedo_{blend_filename}_#####_gt"
    motion_fileout.file_slots[0].path = f"motion_{blend_filename}_#####_gt"

    bpy.context.scene.cycles.seed = random.randint(0, 2**31 - 1)
    bpy.ops.render.render(write_still=False)
    bpy.context.scene.cycles.samples = 1

    mute_aux(depth_fileout, normals_fileout, albedo_fileout, motion_fileout)

def render_1spp(idx, blend_filename, rgb_fileout):
    rgb_fileout.file_slots[0].path = f"rgb_{blend_filename}_#####_{idx:03d}"
    bpy.context.scene.cycles.seed = random.randint(0, 2**31 - 1)
    bpy.ops.render.render(write_still=False)

def render_scene(rgb_fileout, depth_fileout, normals_fileout, albedo_fileout, motion_fileout):
    """Renders the current scene"""
    NUM_RGB_1SPP_RENDERS = 10
    RENDER_GT = True
    NUM_GT_SAMPLES = 4096
    FRAME_LIMIT = 1000

    blend_filepath = bpy.data.filepath
    blend_filename = os.path.splitext(os.path.basename(blend_filepath))[0]

    start_frame = 0
    end_frame = bpy.context.scene.frame_end

    if end_frame - start_frame > FRAME_LIMIT:
        end_frame = start_frame + FRAME_LIMIT

    for frame in tqdm(range(start_frame, end_frame), desc=f'{blend_filename}', unit='frame', file=sys.stderr, ascii=' █'):
        bpy.context.scene.frame_set(frame)
        render_gt(frame, blend_filename, rgb_fileout, depth_fileout, normals_fileout, albedo_fileout, motion_fileout, NUM_GT_SAMPLES)
        for i in range(NUM_RGB_1SPP_RENDERS):
            bpy.context.scene.cycles.seed = random.randint(0, 2**31 - 1)
            render_1spp(i, blend_filename, rgb_fileout)

def process_blend_file(filepath):    
    bpy.ops.wm.open_mainfile(filepath=filepath)
    init_cycles()
    render_scene(*init_compositing())

def main():
    old = os.dup(1)
    os.close(1)
    os.open(os.path.join(cwd, "outputs", "log", "generate", f"log_global_{time_string}.txt"), os.O_WRONLY|os.O_CREAT)

    scenes_directory = os.path.join(cwd, "scenes", "train")
    for filename in tqdm(os.listdir(scenes_directory), desc='Overall Progress', unit='file', file=sys.stderr, ascii=' █'):
        if filename.endswith(".blend"):
            blend_filepath = os.path.join(scenes_directory, filename)
            process_blend_file(blend_filepath)

    os.close(1)
    os.dup(old)
    os.close(old)

if __name__ == "__main__":
    main()