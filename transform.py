import torch
import numpy as np

class RandomCrop(object):
    """Spatially crop the image sequence into (H, W) = (output_size, output_size)"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, frames):
        albedo, depth, normal, rgb_gt, rgb_1spp, motion, normal_vanilla = (frames[x] for x in ['albedo', 'depth', 'normal', 'rgb_gt', 'rgb_1spp', 'motion', 'normal_vanilla'])

        h, w = albedo[0].shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(low=0, high=(h - new_h), size=(1,)).item()
        left = torch.randint(low=0, high=(w - new_w), size=(1,)).item()

        albedo = albedo[:, top:top+new_h, left:left+new_w]
        depth = depth[:, top:top+new_h, left:left+new_w]
        normal = normal[:, top:top+new_h, left:left+new_w]
        rgb_gt = rgb_gt[:, top:top+new_h, left:left+new_w]
        rgb_1spp = rgb_1spp[:, top:top+new_h, left:left+new_w]
        motion = motion[:, top:top+new_h, left:left+new_w]
        normal_vanilla = normal_vanilla[:, top:top+new_h, left:left+new_w]

        return {'albedo': albedo, 
                'depth': depth, 
                'normal': normal, 
                'rgb_gt': rgb_gt, 
                'rgb_1spp': rgb_1spp,
                'motion': motion,
                'rgb_in': rgb_1spp.copy(),
                'normal_vanilla': normal_vanilla}
    
class RandomModulate(object):
    """Multiply RGB channels by a random value each in [a,b] | a,b >= 0. Effectively acts as a hue shift"""

    def __init__(self, modulate):
        assert isinstance(modulate, (int, tuple))
        if isinstance(modulate, int):
            assert modulate >= 0
            self.modulate = (0, modulate)
        else:
            assert len(modulate) == 2
            assert all(x >= 0 for x in modulate)
            assert modulate[0] <= modulate[1]
            self.modulate = modulate

    def __call__(self, frames):
        a, b = self.modulate

        mask = torch.rand((3,))

        r_mod = a + (b - a) * mask[0]
        g_mod = a + (b - a) * mask[1]
        b_mod = a + (b - a) * mask[2]

        datatype = np.float32

        frames['rgb_gt'] = np.clip(frames['rgb_gt'] * np.array([b_mod, g_mod, r_mod]).astype(datatype), 0, 65504)
        frames['rgb_1spp'] = np.clip(frames['rgb_1spp'] * np.array([b_mod, g_mod, r_mod]).astype(datatype), 0, 65504)

        return frames
    
class RandomRotate(object):
    """Rotate the entire image sequence by 0, 90, 180 or 270 degrees (selected randomly)"""

    def __call__(self, frames):
        k = torch.randint(low=0, high=4, size=(1,)).item()

        return {'albedo': np.rot90(frames['albedo'], k, axes=(1, 2)),
                'depth': np.rot90(frames['depth'], k, axes=(1, 2)),
                'normal': np.rot90(frames['normal'], k, axes=(1, 2)), 
                'rgb_gt': np.rot90(frames['rgb_gt'], k, axes=(1, 2)), 
                'rgb_1spp': np.rot90(frames['rgb_1spp'], k, axes=(1, 2))}
    
class AlbedoDemodulate(object):
    """
    Divide input 1spp RGB buffer by Albedo buffer. 
    Corrects NaNs and Infs to be within the min-max limits of the previously unmodulated images
    """

    def __call__(self, frames):
        albedo, rgb_1spp, rgb_in = frames['albedo'], frames['rgb_1spp'], frames['rgb_in']

        max_rgb = np.max(rgb_1spp)
        min_rgb = np.min(rgb_1spp)

        max_rgb_in = np.max(rgb_in)
        min_rgb_in = np.min(rgb_in)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            frames['rgb_in'] = np.clip(np.nan_to_num(rgb_in / albedo, posinf=max_rgb_in, neginf=min_rgb_in, nan=min_rgb_in), a_min=min_rgb_in, a_max=max_rgb_in)
            frames['rgb_1spp'] = np.clip(np.nan_to_num(rgb_1spp / albedo, posinf=max_rgb, neginf=min_rgb, nan=min_rgb), a_min=min_rgb, a_max=max_rgb)

        return frames
    
class AlbedoDemodulateBoring(object):
    def __call__(self, frames):
        albedo, rgb_1spp, rgb_in = frames['albedo'], frames['rgb_1spp'], frames['rgb_in']

        frames['rgb_1spp'] = rgb_1spp / (albedo + 1e-2) # Both HDR, i.e. same color space
        frames['rgb_in'] = rgb_in / (albedo + 1e-2) # This one is remodulated by non-normalised, non-tonemapped albedo

        return frames

class Flatten(object):
    """
    Raise RGB and Albedo buffers to the power gamma. 
    NVIDIA uses 0.2, percpetual gamma correction is 1/2.2
    """

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, frames):
        albedo, rgb_gt, rgb_1spp = frames['albedo'], frames['rgb_gt'], frames['rgb_1spp']

        rgb_gt[rgb_gt <= 0] = 0.0
        rgb_1spp[rgb_1spp <= 0] = 0.0
        albedo[albedo <= 0] = 0.0

        frames['rgb_gt'] = rgb_gt ** self.gamma
        frames['rgb_1spp'] = rgb_1spp ** self.gamma
        frames['albedo'] = albedo ** self.gamma

        return frames
    
class TonemapIsik(object):
    def __call__(self, frames):
        frames['rgb_1spp'] = np.log(1 + frames['rgb_1spp']) # Implicitly tonemaps the albedo demodulation
        return frames
        
class Normalise(object):
    """Divide image buffers by maximum value present in the sequence"""

    def __call__(self, frames):
        normal, albedo, rgb_1spp = frames['normal'], frames['albedo'], frames['rgb_1spp']

        eps = 1e-10

        frames['normal']   = (normal - normal.min()) / (normal.max() - normal.min() + eps)
        frames['albedo_f'] = (albedo - albedo.min()) / (albedo.max() - albedo.min() + eps) # Appends a normalised version of the albedo
        frames['rgb_1spp'] = (rgb_1spp - rgb_1spp.min()) / (rgb_1spp.max() - rgb_1spp.min() + eps)

        return frames
    
class ToTensor(object):
    """
    Convert 4D ndarrays into 4D tensors. 
    Concatenate input features along channel dimension into one 4D tensor. 
    Also returns Albedo tensor for remodulating model output, and target RGB buffer for loss computation
    """

    def __call__(self, frames):
        albedo = torch.from_numpy(frames['albedo'].transpose(0, 3, 1, 2).copy())
        depth = torch.from_numpy(frames['depth'].transpose(0, 3, 1, 2).copy())
        normal = torch.from_numpy(frames['normal'].transpose(0, 3, 1, 2).copy())
        rgb_gt = torch.from_numpy(frames['rgb_gt'].transpose(0, 3, 1, 2).copy())
        rgb_1spp = torch.from_numpy(frames['rgb_1spp'].transpose(0, 3, 1, 2).copy())
        motion = torch.from_numpy(frames['motion'].transpose(0, 3, 1, 2).copy())
        rgb_in = torch.from_numpy(frames['rgb_in'].transpose(0, 3, 1, 2).copy())
        normal_vanilla = torch.from_numpy(frames['normal_vanilla'].transpose(0, 3, 1, 2).copy())

        inputs = torch.cat((rgb_1spp, normal, albedo), dim=1) 

        return {'inputs': inputs,
                'targets': rgb_gt,
                'albedo': albedo,
                'motion': motion,
                'rgb_in': rgb_in,
                'depth': depth,
                'normal_vanilla': normal_vanilla} 