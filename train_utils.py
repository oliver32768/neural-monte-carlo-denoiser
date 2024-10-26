import torch
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def flow_warp_utils(prev_buffer, mv, mode, device):
    """Returns new tensor containing pixels in frame t-1 (prev_buffer) at their locations in frame t (proj_buffer) using flow field vectors
    mv          : N x 4 x H x W | [:2] Describes sample motion (t -> t-1) in screen-space
    prev_buffer : N x C x H x W | Pixels at t-1"""
    N, C, H, W = prev_buffer.shape

    xa = torch.arange(0, W, device=device) # 1 x W
    ya = torch.arange(0, H, device=device) # 1 x H

    xx, yy = torch.meshgrid(xa, ya, indexing='xy')

    xx = xx.unsqueeze(0).repeat(N, 1, 1) # N x H x W
    yy = yy.unsqueeze(0).repeat(N, 1, 1) # N x H x W

    xx_new = (2.0 * ((xx + mv[:, 0]) / (W-1)) - 1.0).unsqueeze(-1) # N x H x W x 1
    yy_new = (2.0 * ((yy - mv[:, 1]) / (H-1)) - 1.0).unsqueeze(-1) # N x H x W x 1
    flow_field = torch.cat((xx_new, yy_new), dim=3) # N x H x W x 2

    proj_buffer = F.grid_sample(prev_buffer, flow_field, mode=mode, padding_mode='zeros', align_corners=True)

    return proj_buffer

def psnr(A, B):
    """A, B : N x C x H x W
    Expects A and B to already be tonemapped"""
    assert A.shape == B.shape
    N, C, H, W = A.shape

    return -10 * torch.log10(((A-B) ** 2).mean())

def trmae(A, B):
    """A, B : N x 2 x C x H x W
    M = 0 is all 0s
    M = 1 is the difference image between t and t-1"""
    assert A.shape == B.shape
    N, M, C, H, W = A.shape

    eps = 1e-2
    num = torch.abs(A-B)
    den = torch.abs(B) + eps

    return M * (num / den).mean()

def save_checkpoint_func(state_dict, checkpoint_dir, filename):
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state_dict, path)

def create_state_dict(epoch, mean_val_loss, early_stopper, model, optimizer, scheduler):
    return {
        'epoch': epoch+1,
        'loss': mean_val_loss,
        'early_stopper': early_stopper,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'random_state_np': np.random.get_state(),
        'random_state_torch': torch.get_rng_state(),
        'random_state_torch_cuda': torch.cuda.get_rng_state_all(),
    }

def reg_loss(a, b):
    """Return regularization loss between kernel parameters a, b (Equation 13)
    a : N x M x K x H x W
    b : N x M x H x W"""
    N, M, K, H, W = a.shape
    E = 1/(N*M*H*W)

    a_l2_squared = (a ** 2).sum()
    b_l2_squared = (b ** 2).sum()

    return E * (a_l2_squared + b_l2_squared)

def SMAPE(A, B):
    """Return SMAPE loss between A, B (Equation 11) for a batch of a sequence of image tensors
    A, B : N x M x C x H x W"""
    assert A.shape == B.shape
    N, M, C, H, W = A.shape

    abs_diff = torch.abs(A-B)
    abs_sum = torch.abs(A) + torch.abs(B)
    eps = 1e-10 

    return (abs_diff/(abs_sum + eps)).mean()

def log_to_file(path, msg):
    with open(path, 'a') as file:
           file.write(msg + '\n')

def finite_differencing(outputs, targets):
    assert outputs.shape == targets.shape
    N, M, C, H, W = outputs.shape

    dt_outputs = torch.zeros_like(outputs)
    dt_targets = torch.zeros_like(targets)

    for i in range(1, M):
        dt_outputs[:, i] = outputs[:, i] - outputs[:, i-1]
        dt_targets[:, i] = targets[:, i] - targets[:, i-1]

    return dt_outputs, dt_targets

def finite_differencing_single(outputs, targets):
    """Return difference between frame t and frame t-1
    outputs : N x 2 x C x H x W
    targets : N x 2 x C x H x W
    return  : N x C x H x W"""
    with torch.no_grad():
        return (outputs[:, 1] - outputs[:, 0]), (targets[:, 1] - targets[:, 0])

def save_sequence_test(inputs, targets, outputs, title, filename): 
    with torch.no_grad():
        display = np.concatenate((unrolled_inputs := unroll_sequence(inputs[:, :, :3, ...], 1), 
                                  unrolled_outputs := unroll_sequence(outputs, 1), 
                                  unrolled_targets := unroll_sequence(targets, 1)
                                  ), axis=0)
        
        display = np.clip(display, a_min=0.0, a_max=1.0)
        
        h, w, _ = display.shape
        my_dpi = 96
        plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
        plt.imshow(display)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close()

def save_sequence(inputs, targets, outputs, title, filename): 
    with torch.no_grad():
        display = np.concatenate((unrolled_inputs := unroll_sequence(inputs[:, :, :3, ...], 7), 
                                  unrolled_normals := unroll_sequence(inputs[:, :, 3:6, ...], 7),
                                  unrolled_outputs := unroll_sequence(outputs, 7), 
                                  unrolled_targets := unroll_sequence(targets, 7)
                                  ), axis=0)
        
        display = np.clip(display, a_min=0.0, a_max=1.0)
        
        h, w, _ = display.shape
        my_dpi = 96
        plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
        plt.imshow(display)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close()

def unroll_sequence(sequence, nrow):
    with torch.no_grad():
        unrolled = sequence[0].detach().clone().cpu().float()
        unrolled = torchvision.utils.make_grid(unrolled, nrow=nrow)
        unrolled = unrolled.permute(1, 2, 0)
        unrolled = unrolled.numpy()
        return unrolled

def save_local_loss_plot(train_loss_curve, filename, epoch_num):
    with torch.no_grad():
        plt.figure()
        plt.plot(train_loss_curve, label='Training Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch_num} Training Loss')
        plt.legend()
        plt.savefig(filename)
        plt.close()

def save_global_loss_plot(train_loss_curve, val_loss_curve, filename):
    with torch.no_grad():
        plt.figure()
        plt.plot(train_loss_curve, label='Training Loss')
        plt.plot(val_loss_curve, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.savefig(filename)
        plt.close()
