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

cwd = os.getcwd()
time_string = datetime.now().strftime('%Y%m%d%H%M%S')

def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    
    dw = exr_file.header()['dataWindow']
    isize = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    red_str = exr_file.channel("R", Imath.PixelType(Imath.PixelType.HALF))
    green_str = exr_file.channel("G", Imath.PixelType(Imath.PixelType.HALF))
    blue_str = exr_file.channel("B", Imath.PixelType(Imath.PixelType.HALF))

    red = np.frombuffer(red_str, dtype=np.float16)
    green = np.frombuffer(green_str, dtype=np.float16)
    blue = np.frombuffer(blue_str, dtype=np.float16)
    
    red.shape = (isize[1], isize[0])
    green.shape = (isize[1], isize[0])
    blue.shape = (isize[1], isize[0])
    
    return red, green, blue

def exr_to_png(exr_path, png_path):
    red, green, blue = read_exr(exr_path)

    red_8bit = np.clip((red * 255), 0, 255).astype(np.uint8)
    green_8bit = np.clip((green * 255), 0, 255).astype(np.uint8)
    blue_8bit = np.clip((blue * 255), 0, 255).astype(np.uint8)

    img = Image.merge("RGB", [Image.fromarray(red_8bit), Image.fromarray(green_8bit), Image.fromarray(blue_8bit)])

    img.save(png_path)

def init_cycles():
    """Initialize Blender Cycles settings"""
    bpy.context.scene.render.engine = 'CYCLES'

    bpy.context.scene.cycles.use_denoising = False
    bpy.context.view_layer.cycles.denoising_store_passes = True

    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
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
    rgb_fileout.base_path = os.path.join(cwd, 'dataset', 'test', 'rgb')

    # Depth
    normalize = tree.nodes.new(type='CompositorNodeNormalize')

    depth_fileout = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_fileout.format.file_format = 'OPEN_EXR'
    depth_fileout.format.color_mode = 'RGB'
    depth_fileout.format.color_depth = '16'
    depth_fileout.base_path = os.path.join(cwd, 'dataset', 'test', 'depth')

    # Normals
    multiply_node = tree.nodes.new('CompositorNodeMixRGB')
    multiply_node.blend_type = 'MULTIPLY'
    multiply_node.inputs[0].default_value = 1.0

    add_node = tree.nodes.new('CompositorNodeMixRGB')
    add_node.blend_type = 'ADD'
    add_node.inputs[0].default_value = 1.0

    rgb_half_node = tree.nodes.new('CompositorNodeRGB')
    rgb_half_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)    

    normals_fileout = tree.nodes.new(type='CompositorNodeOutputFile')
    normals_fileout.format.file_format = 'OPEN_EXR'
    normals_fileout.format.color_mode = 'RGB'
    normals_fileout.format.color_depth = '16'
    normals_fileout.base_path = os.path.join(cwd, 'dataset', 'test', 'normal')

    # Albedo
    albedo_fileout = tree.nodes.new(type='CompositorNodeOutputFile')
    albedo_fileout.format.file_format = 'OPEN_EXR'
    albedo_fileout.format.color_mode = 'RGB'
    albedo_fileout.format.color_depth = '16'
    albedo_fileout.base_path = os.path.join(cwd, 'dataset', 'test', 'albedo')

    # Link nodes
    tree.links.new(rlayers.outputs['Image'], rgb_fileout.inputs[0])
    tree.links.new(rlayers.outputs['Denoising Depth'], normalize.inputs[0])
    tree.links.new(normalize.outputs[0], depth_fileout.inputs[0])
    
    tree.links.new(rlayers.outputs['Denoising Normal'], multiply_node.inputs[1])
    tree.links.new(rgb_half_node.outputs[0], multiply_node.inputs[2])
    tree.links.new(multiply_node.outputs[0], add_node.inputs[1])
    tree.links.new(rgb_half_node.outputs[0], add_node.inputs[2])
    tree.links.new(add_node.outputs[0], normals_fileout.inputs[0])

    tree.links.new(rlayers.outputs['Denoising Albedo'], albedo_fileout.inputs[0])

    # These should only be unmuted for the GT render
    mute_aux(depth_fileout, normals_fileout, albedo_fileout)

    return rgb_fileout, depth_fileout, normals_fileout, albedo_fileout

def mute_aux(depth_fileout, normals_fileout, albedo_fileout):
    depth_fileout.mute = True
    normals_fileout.mute = True
    albedo_fileout.mute = True

def unmute_aux(depth_fileout, normals_fileout, albedo_fileout):
    depth_fileout.mute = False
    normals_fileout.mute = False
    albedo_fileout.mute = False

def render_gt(frame, blend_filename, rgb_fileout, depth_fileout, normals_fileout, albedo_fileout, samples):
    bpy.context.scene.cycles.samples = samples
    unmute_aux(depth_fileout, normals_fileout, albedo_fileout)

    rgb_fileout.file_slots[0].path = f"rgb_{blend_filename}_#####_gt"
    depth_fileout.file_slots[0].path = f"depth_{blend_filename}_#####_gt"
    normals_fileout.file_slots[0].path = f"normal_{blend_filename}_#####_gt"
    albedo_fileout.file_slots[0].path = f"albedo_{blend_filename}_#####_gt"

    rng_seed = random.randint(0, 2**31 - 1)
    bpy.context.scene.cycles.seed = rng_seed

    bpy.ops.render.render(write_still=False)

    normal_exr_path = os.path.join(cwd, "dataset", "test", "normal", f"normal_{blend_filename}_{frame:05d}_gt.exr") 
    normal_png_path = os.path.join(cwd, "dataset", "test", "normal", f"normal_{blend_filename}_{frame:05d}_gt.png") 
    exr_to_png(normal_exr_path, normal_png_path)
    os.remove(normal_exr_path)

    bpy.context.scene.cycles.samples = 1
    mute_aux(depth_fileout, normals_fileout, albedo_fileout)

def render_1spp(idx, blend_filename, rgb_fileout):
    rgb_fileout.file_slots[0].path = f"rgb_{blend_filename}_#####_{idx:03d}"

    rng_seed = random.randint(0, 2**31 - 1)
    bpy.context.scene.cycles.seed = rng_seed

    bpy.ops.render.render(write_still=False)

def render_scene(rgb_fileout, depth_fileout, normals_fileout, albedo_fileout):
    """Renders the current scene"""
    NUM_1SPP_RENDERS = 1
    RENDER_GT = True
    NUM_GT_SAMPLES = 4096
    FRAME_LIMIT = 200 

    blend_filepath = bpy.data.filepath
    blend_filename = os.path.splitext(os.path.basename(blend_filepath))[0]

    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    if end_frame - start_frame > FRAME_LIMIT:
        end_frame = start_frame + FRAME_LIMIT

    for frame in tqdm(range(start_frame, end_frame), desc=f'Rendering {blend_filename}', unit='frame', file=sys.stderr, ascii=' █'):
        bpy.context.scene.frame_set(frame)

        if RENDER_GT:
            render_gt(frame, blend_filename, rgb_fileout, depth_fileout, normals_fileout, albedo_fileout, NUM_GT_SAMPLES)   

        for i in range(NUM_1SPP_RENDERS):
            render_1spp(i, blend_filename, rgb_fileout)

             

def process_blend_file(filepath, filename):    
    bpy.ops.wm.open_mainfile(filepath=filepath)
    init_cycles()
    rgb, depth, normal, albedo = init_compositing()
    render_scene(rgb, depth, normal, albedo)

def main():
    scenes_directory = os.path.join(cwd, "scenes", "test")

    for filename in tqdm(os.listdir(scenes_directory), desc='Overall Progress', unit='file', file=sys.stderr, ascii=' █'):
        if filename.endswith(".blend"):
            blend_filepath = os.path.join(scenes_directory, filename)
            process_blend_file(blend_filepath, os.path.splitext(filename)[0])

if __name__ == "__main__":
    old = os.dup(1)
    os.close(1)
    os.open(os.path.join(cwd, "log", f"log_global_{time_string}.txt"), os.O_WRONLY|os.O_CREAT)

    main()

    os.close(1)
    os.dup(old)
    os.close(old)