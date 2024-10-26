from model_parts import *
import torch.nn.init as init
import torch.nn.functional as F
from math import log2

class PointwiseNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super().__init__()

        self.u0 = EncoderBlock(in_channel, 64, 64, kernel_size)
        self.u1 = EncoderBlock(64, 64, 64, kernel_size)
        self.u2 = EncoderBlock(64, 64, 64, kernel_size)
        self.u3 = EncoderBlock(64, 80, 80, kernel_size)
        self.u4 = Bottleneck(80, 96, 96, kernel_size)
        self.u5 = DecoderBlock(96+80, 80, 80, kernel_size)
        self.u6 = DecoderBlock(80+64, 64, 64, kernel_size)
        self.u7 = DecoderBlock(64+64, 64, 64, kernel_size)
        self.u8 = OutputBlock(64+64, 32, 32, kernel_size)

    def forward(self, x):
        x, s1 = self.u0(x)
        x, s2 = self.u1(x)
        x, s3 = self.u2(x)
        x, s4 = self.u3(x)
        x = self.u4(x)
        x = self.u5(x, s4)
        x = self.u6(x, s3)
        x = self.u7(x, s2)
        x = self.u8(x, s1)

        return x

class IsikNet(nn.Module):
    def __init__(self, in_channel, embedding_dims, kernel_size, device):
        super().__init__()

        self.device = device
        self.pw = PointwiseNet(in_channel=in_channel, out_channel=embedding_dims)
        self.unet = Autoencoder(in_channel=2*embedding_dims, kernel_size=kernel_size)

        self.e_bar = None
        self.L_bar = None
        self.f_prev = None
        self.O_prev = None
        self.prev_normal = None
        self.prev_depth = None

    def forward(self, gbuffers, motion_vector, rgb_in, normal_vanilla, depth):
        """Section 3 / Figure 2
        motion_vector : N x C x H x W | Describes sample motion (t -> t-1) in screen-space
        gbuffers      : N x C x H x W | Radiance, Normals, Depth, Albedo, etc."""
        embed_map = self.pw(gbuffers)

        cur_normal = self.compute_heuristics_normals(normal_vanilla)
        cur_depth = depth.clone()
        self.disocclusion_mask = self.compute_disocclusion_mask(motion_vector, cur_normal, cur_depth)
        self.prev_normal = cur_normal.clone()
        self.prev_depth = cur_depth.clone()

        cat_embed_map = self.cat_e(embed_map, motion_vector)
        param_map = self.unet(cat_embed_map)
        lambda_map = param_map[:, 31].unsqueeze(1)
        self.compute_e_bar(embed_map, lambda_map, motion_vector)

        kernel_maps = self.compute_w(param_map)
        temporal_map = self.compute_omega(param_map, motion_vector)
        self.f_prev = param_map[:, :24].clone()

        self.compute_L_bar(rgb_in, lambda_map, motion_vector)
        rad = self.filter_L(kernel_maps)

        rad_final = self.filter_O(rad, kernel_maps, temporal_map, motion_vector)
        self.O_prev = rad_final.clone()

        return rad_final, param_map[:, 24:27], param_map[:, 30]
    
    def flow_warp_heuristics(self, prev_buffer, mv, defer):
        """
        prev_buffer : N x C x H x W
        mv          : N x 4 x H x W
        defer       : = prev_buffer"""
        fw_prev_buffer = self.flow_warp(prev_buffer, mv)
        fw_prev_buffer[self.disocclusion_mask.expand_as(fw_prev_buffer)] = 0.0

        mask = (fw_prev_buffer == 0.0).all(dim=1).unsqueeze(1)
        fw_prev_buffer[mask.expand_as(fw_prev_buffer)] = defer[mask.expand_as(defer)]

        return fw_prev_buffer
    
    def compute_disocclusion_mask(self, mv, cur_normal, cur_depth):
        """
        cur_normal : N x 3 x H x W
        cur_depth  : N x 1 x H x W"""
        if self.prev_depth is None or self.prev_normal is None: # t = 0
            return None
        
        DEPTH_TOL = 0.2 # Lower = stricter (0.0 to 1.0)
        NORM_TOL = 0.9 # Higher = stricter (0.0 to 1.0)

        fw_prev_depth = self.flow_warp(self.prev_depth, mv)
        fw_prev_normal = self.flow_warp(self.prev_normal, mv)

        mask_depth = (torch.abs(cur_depth - fw_prev_depth) > DEPTH_TOL)

        dot_norm = (fw_prev_normal * cur_normal).sum(dim=1)
        mask_normals = (dot_norm < NORM_TOL).unsqueeze(1)
        mask_nan = torch.isnan(cur_normal).any(dim=1).unsqueeze(1)

        mask_disocclusion = (mask_normals | mask_depth | mask_nan)

        return mask_disocclusion
    
    def compute_heuristics_normals(self, normal_vanilla):
        """normal_vanilla : N x 3 x H x W"""
        cur_normal = (normal_vanilla.clone() - 127.0) / 127.0
        cur_normal = cur_normal / torch.sqrt((cur_normal ** 2).sum(dim=1)).unsqueeze(1)
        return cur_normal
    
    def flow_warp(self, prev_buffer, mv):
        """Returns new tensor containing pixels in frame t-1 (prev_buffer) at their locations in frame t (proj_buffer) using flow field vectors
        mv          : N x 4 x H x W | [:2] Describes sample motion (t -> t-1) in screen-space
        prev_buffer : N x C x H x W | Pixels at t-1"""
        N, C, H, W = prev_buffer.shape

        xa = torch.arange(0, W, device=self.device) # 1 x W
        ya = torch.arange(0, H, device=self.device) # 1 x H

        xx, yy = torch.meshgrid(xa, ya, indexing='xy')

        xx = xx.unsqueeze(0).repeat(N, 1, 1) # N x H x W
        yy = yy.unsqueeze(0).repeat(N, 1, 1) # N x H x W

        xx_new = (2 * ((xx + mv[:, 0]) / (W-1)) - 1).unsqueeze(-1) # N x H x W x 1
        yy_new = (2 * ((yy - mv[:, 1]) / (H-1)) - 1).unsqueeze(-1) # N x H x W x 1
        flow_field = torch.cat((xx_new, yy_new), dim=3) # N x H x W x 2

        proj_buffer = F.grid_sample(prev_buffer, flow_field, mode='nearest', padding_mode='zeros', align_corners=True)

        return proj_buffer

    def filter_O(self, rad, kernel_maps, temporal_map, mv):
        """Equation 8
        rad          : N x 3 x H x W
        kernel_maps  : N x 1 x H x W x KW x KH (3-tuple)
        temporal_map : N x 1 x H x W x KW x KH
        mv           : N x 4 x H x W"""
        _, _, Kmap_D2 = kernel_maps

        spatial = self.filter_L_once_patched(rad, Kmap_D2, 128)

        if temporal_map is None: # t = 0
            norm_factor = 1e-10 + Kmap_D2.sum(dim=(-2, -1))
            rad_out = spatial / norm_factor
        else:
            spatial_norm = 1e-10 + Kmap_D2.sum(dim=(-2, -1))
            spatial_defer = spatial.clone() / spatial_norm
            O_prev_warped = self.flow_warp_heuristics(self.O_prev, mv, spatial_defer)

            temporal = self.filter_L_once_patched(O_prev_warped, temporal_map, 128)

            norm_factor = spatial_norm + temporal_map.sum(dim=(-2, -1))
            rad_out = (spatial + temporal) / norm_factor

        return rad_out
    
    def filter_L(self, kernel_maps):
        """Section 3.5.1
        kernel_maps : N x 1 x H x W x KW x KH (3-tuple)"""
        Kmap_D0, Kmap_D1, _ = kernel_maps

        norm_factor_D0 = 1e-10 + Kmap_D0.sum(dim=(-2, -1)).repeat(1, 3, 1, 1) # N x C x H x W
        norm_factor_D1 = 1e-10 + Kmap_D1.sum(dim=(-2, -1)).repeat(1, 3, 1, 1) # N x C x H x W

        rad1 = self.filter_L_once_patched(self.L_bar, Kmap_D0, 128) / norm_factor_D0
        rad2 = self.filter_L_once_patched(rad1, Kmap_D1, 128) / norm_factor_D1

        return rad2
    
    def filter_L_once_patched(self, rad_in, kernel_map, patch_size):
        _, C, _, _ = rad_in.shape
        N, _, H, W, KH, KW = kernel_map.shape
        KS = (KH - 1) // 2
        PS = patch_size

        rad_out = torch.zeros(N, C, H, W, device=self.device)
        rad_in_padded = F.pad(rad_in, (KS, KS, KS, KS), 'constant', 0)

        for y_ol in range(0, H, PS):
            for x_ol in range(0, W, PS):
                y_ou = min(y_ol + PS, H)
                x_ou = min(x_ol + PS, W)

                y_il = y_ol
                y_iu = y_ou + (KH - 1)
                x_il = x_ol
                x_iu = x_ou + (KW - 1)

                rad_out[..., y_ol:y_ou, x_ol:x_ou] = self.filter_L_once(rad_in_padded[..., y_il:y_iu, x_il:x_iu], kernel_map[:, :, y_ol:y_ou, x_ol:x_ou])
        
        return rad_out
    
    def filter_L_once(self, rad_in, kernel_map):
        """Equation 6. Currently filters R/G/B seperately with a C = 1 kernel and concatenates results
        rad_in_unpadded : N x 3 x H x W
        kernel_map      : N x 1 x H x W x KW x KH"""
        KH = KW = kernel_map.shape[-1]

        rad_in_unfolded = rad_in.unfold(2, KH, 1).unfold(3, KW, 1) # N x 3 x H x W x KH x KW

        rad_out = (kernel_map * rad_in_unfolded).sum(dim=(-2, -1)) # N x 3 x H x W

        return rad_out
    
    def compute_omega(self, param_map, mv):
        """Returns per-pixel temporal kernel (Equation 7). omega = 0 when t = 0
        param_map : N x 32 x H x W
        mv        : N x 4  x H x W"""
        if self.f_prev is None: # t = 0
            return None
        
        N, C, H, W = param_map.shape
        KW = KH = 11
        KS = (KW - 1) // 2
        
        f_xyt = param_map[:, 16:24]
        b_xyt = param_map[:, 30]
        f_prev_W = self.flow_warp_heuristics(self.f_prev, mv, param_map[:, :24])
        f_prev_W = F.pad(f_prev_W, (KS, KS, KS, KS), 'constant', 0)

        omega_map = torch.zeros(N, 1, H, W, KW, KH, device=self.device)

        for u in range(KW):
            for v in range(KH):
                end_u = None if u == KW-1 else u - (KW-1)
                end_v = None if v == KH-1 else v - (KH-1)

                f_uvt_prev = f_prev_W[:, 16:24, u:end_u, v:end_v]

                omega_map[:, 0, ..., u, v] = torch.exp(-b_xyt * (torch.linalg.vector_norm(f_xyt - f_uvt_prev, ord=2, dim=1) ** 2))

        return omega_map
    
    def compute_w(self, param_map):
        """Returns K = 3 per-pixel spatial kernels (Equation 4), each with increasing levels of dilation in {1, 2, 4}
        param_map : N x 32 x H x W"""
        Kmap_D0 = self.compute_kernel_map(1, 11, param_map)
        Kmap_D1 = self.compute_kernel_map(2, 11, param_map)
        Kmap_D2 = self.compute_kernel_map(4, 11, param_map)

        return Kmap_D0, Kmap_D1, Kmap_D2
    
    def compute_kernel_map(self, dilation, kernel_size, param_map):
        """Return the per-pixel kernels with a given kernel size and dilation
        dilation    : int
        kernel_size : int
        param_map   : N x 32 x H x W
        Kmap        : N x 1 x H x W x KW x KH"""
        N, C, H, W = param_map.shape
        KW = KH = (kernel_size - 1) * dilation + 1
        KS = (KH - 1) // 2

        k = int(log2(dilation))
        fl, fh = 8*k, 8*(k+1)

        f = F.pad(param_map[:, :24], (KS, KS, KS, KS), 'constant', 0)
        f_xyt = f[:, :, KS:-KS, KS:-KS]
        a_xyt = param_map[:, 24:27]
        c_xyt = param_map[:, 27:30]        

        Kmap = torch.zeros(N, 1, H, W, KW, KH, device=self.device)
        Kmap[..., KS, KS] = c_xyt[:, k].unsqueeze(1)   

        for u in range(0, KW, dilation):
            for v in range(0, KH, dilation):
                if u == KS and v == KS:
                    continue

                end_u = None if u == KW-1 else u - (KW-1)
                end_v = None if v == KH-1 else v - (KH-1)

                f_uvt = f[..., u:end_u, v:end_v]

                Kmap[:, 0, ..., u, v] = torch.exp(-a_xyt[:, k] * (torch.linalg.vector_norm(f_xyt[:, fl:fh] - f_uvt[:, fl:fh], ord=2, dim=1) ** 2))

        return Kmap
        
    def compute_L_bar(self, L, w_lambda, mv):
        """Equation 5
        L        : N x 3 x H x W
        w_lambda : N x 1 x H x W
        mv       : N x 4 x H x W"""
        if self.L_bar is None: # t = 0
            self.L_bar = L.clone()
        else:
            self.L_bar = ((1 - w_lambda) * L) + (w_lambda * self.flow_warp_heuristics(self.L_bar, mv, L))

    def compute_e_bar(self, embed_map, w_lambda, mv):
        """Equation 3
        w_lambda  : N x 1  x H x W
        embed_map : N x 32 x H x W
        mv        : N x 4  x H x W"""
        if self.e_bar is None: # t = 0
            self.e_bar = embed_map.clone()
        else:
            self.e_bar = ((1 - w_lambda) * embed_map) + (w_lambda * self.flow_warp_heuristics(self.e_bar, mv, embed_map))
    
    def cat_e(self, embed_map, mv):
        """Equation 2 (RHS)
        embed_map : N x 32 x H x W
        mv        : N x 4  x H x W"""
        if self.e_bar is None: # t = 0
            return torch.cat((embed_map, embed_map), dim=1)
        else:
            return torch.cat((embed_map, self.flow_warp_heuristics(self.e_bar, mv, embed_map)), dim=1)
        
    def clear_state(self):
        self.e_bar = None
        self.L_bar = None
        self.f_prev = None
        self.O_prev = None
        self.prev_normal = None
        self.prev_depth = None